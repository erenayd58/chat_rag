"""One controlled load characterisation of the bounded ingest path.

Fourteen uploads arrive at the same instant (a barrier releases them) at a
server configured for two workers, a queue of four, a Deep provider budget of
three and an embedding budget of two. Each accepted job behaves like a Deep
Analysis ingest: a per-job pool of four threads makes provider calls through
the budget-limited wrapper against one shared gated provider, and then the job
embeds its chunks through the separate embedding budget. The Deep provider
answers only when the test opens its gate, so the peaks are measured while the
system is as loaded as it can be, not after the fact.

The numbers printed at the end are the characterisation; the assertions are
the bounds they must satisfy, whatever the machine.
"""

from __future__ import annotations

import io
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from fastapi.testclient import TestClient

import asgi as entrypoint
import interfaces.http as http

V1 = http.v1.PREFIX
import tempfile
from chat_rag.components.ingest import IngestManager
from chat_rag.components.ingest import jobs as J
from chat_rag.components.ingest import limits as L
from chat_rag.components.knowledgebase.manager import KnowledgeBaseManager
from chat_rag.config.ingest import IngestLimits

from ingest_doubles import GatedProvider

SUBMITTED = 14
WORKERS = 2
QUEUE = 4
BUDGET = 3
EMBEDDING_BUDGET = 2
PER_JOB_POOL = 4
CALLS_PER_JOB = 8
EMBEDDING_CALLS_PER_JOB = 2


@pytest.fixture
def staging(tmp_path, monkeypatch):
    directory = tmp_path / "staging"
    directory.mkdir()
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(directory))
    return directory


@pytest.fixture
def client(tmp_path, monkeypatch, staging):
    monkeypatch.chdir(tmp_path)
    manager = KnowledgeBaseManager(str(tmp_path / "kbs.json"))
    monkeypatch.setattr(entrypoint.services, "kb_manager", manager)
    monkeypatch.setattr(entrypoint.services, "get_pipeline", lambda *a, **k: _pipeline_stub)
    kbs = [manager.create(f"load-kb-{i}", chunker={"type": "structure_first"})["kb_id"] for i in range(SUBMITTED)]
    # One application, many callers -- as a server is. Building one per thread
    # would put the cost of composing it inside the window this measures.
    yield http.create_app(entrypoint.services), kbs


_pipeline_stub = type("Stub", (), {
    "chunker": type("C", (), {"get_name": lambda self: "StructuralChunker",
                              "chunk_text_deep": lambda self, *a, **k: None})(),
})()


class EmbeddingDouble:
    """An embedding transport that reports through the shared gate."""

    model_id = "test:embed@1"

    def __init__(self, provider: GatedProvider):
        self.provider = provider

    def embed(self, texts):
        self.provider.complete("embedding batch")
        return [[0.0, 0.0, 0.0] for _ in texts]


class DeepLikeJob:
    """What a Deep ingest costs both budgets, without a parser or a store."""

    def __init__(self, provider: GatedProvider, budget: L.ProviderBudget,
                 embedder: GatedProvider, embedding_budget: L.ProviderBudget):
        self.provider = provider
        self.budget = budget
        self.embedder = embedder
        self.embedding_budget = embedding_budget
        self.lock = threading.Lock()
        self.active = 0
        self.peak_active = 0

    def __call__(self, job: J.IngestJob) -> dict:
        with self.lock:
            self.active += 1
            self.peak_active = max(self.peak_active, self.active)
        try:
            limited = L.LimitedProvider(self.provider, self.budget, L.current_guard())
            with ThreadPoolExecutor(max_workers=PER_JOB_POOL) as pool:
                answers = list(pool.map(limited.complete, [f"{job.job_id}:{i}" for i in range(CALLS_PER_JOB)]))
            assert len(answers) == CALLS_PER_JOB
            # ... and then the chunks are embedded, against the other budget.
            embedding = L.LimitedEmbeddingTransport(
                EmbeddingDouble(self.embedder), self.embedding_budget
            )
            for _ in range(EMBEDDING_CALLS_PER_JOB):
                embedding.embed(["chunk one", "chunk two"])
            return {"success": True, "doc_id": f"doc-{job.job_id}"}
        finally:
            with self.lock:
                self.active -= 1


def test_load_characterisation(client, monkeypatch, staging):
    app, kbs = client
    budget = L.configure_budget(BUDGET)
    embedding_budget = L.configure_embedding_budget(EMBEDDING_BUDGET)
    provider = GatedProvider(expect=BUDGET)
    embedder = GatedProvider(expect=EMBEDDING_BUDGET)
    embedder.release()  # embeddings are not the thing being held here
    work = DeepLikeJob(provider, budget, embedder, embedding_budget)
    manager = IngestManager(
        IngestLimits(workers=WORKERS, queue_capacity=QUEUE, job_timeout_seconds=120), execute=work,
    )
    monkeypatch.setattr(entrypoint.services, "ingest_jobs", manager)

    barrier = threading.Barrier(SUBMITTED)
    outcomes: list[tuple[int, dict]] = []
    lock = threading.Lock()
    started = time.perf_counter()

    def submit(index: int):
        # One client per thread, over the one application.
        with TestClient(app, raise_server_exceptions=False) as test_client:
            barrier.wait(10)
            response = test_client.post(
                f"{V1}/documents",
                files={"file": (f"belge-{index}.txt",
                                io.BytesIO(f"belge {index}".encode()), "text/plain")},
                data={"knowledge_base_id": kbs[index], "methods": ["agentic"]},
            )
            with lock:
                outcomes.append((response.status_code, response.json()))

    threads = [threading.Thread(target=submit, args=(i,)) for i in range(SUBMITTED)]
    try:
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(30)
        submitted_at = time.perf_counter() - started

        accepted = [body for code, body in outcomes if code == 202]
        rejected = [body for code, body in outcomes if code == 503]
        assert len(accepted) + len(rejected) == SUBMITTED, outcomes
        # The system is now as full as it can be: the provider is at its
        # budget and every other caller is waiting for a slot.
        assert provider.full.wait(20), "the provider never reached the budget"
        peak_snapshot = manager.snapshot()
        peak_budget = budget.snapshot()
        provider.release()
        assert manager.drain(120)
        elapsed = time.perf_counter() - started
    finally:
        provider.release()
        embedder.release()
        manager.close(timeout=60)
        L.configure_budget(8)
        L.configure_embedding_budget(4)

    finished = manager.stats
    report = {
        "submitted": SUBMITTED,
        "accepted": len(accepted),
        "queued_at_peak": peak_snapshot["queued"],
        "rejected": len(rejected),
        "peak_active_jobs": max(work.peak_active, finished["peak_active"]),
        "peak_provider_calls": max(provider.peak, budget.peak),
        "provider_calls_total": provider.calls,
        "peak_embedding_calls": max(embedder.peak, embedding_budget.peak),
        "embedding_calls_total": embedder.calls,
        "succeeded": finished["succeeded"],
        "failed": finished["failed"] + finished["timed_out"] + finished["cancelled"],
        "submit_seconds": round(submitted_at, 3),
        "elapsed_seconds": round(elapsed, 3),
        "limits": {"workers": WORKERS, "queue": QUEUE, "provider_budget": BUDGET,
                   "embedding_budget": EMBEDDING_BUDGET,
                   "per_job_pool": PER_JOB_POOL, "calls_per_job": CALLS_PER_JOB},
    }
    print("\nINGEST LOAD CHARACTERISATION")
    for key, value in report.items():
        print(f"  {key:>22}: {value}")

    # -- the bounds -------------------------------------------------------
    assert report["accepted"] == WORKERS + QUEUE
    assert report["rejected"] == SUBMITTED - (WORKERS + QUEUE)
    assert report["queued_at_peak"] <= QUEUE
    assert peak_snapshot["running"] <= WORKERS
    assert report["peak_active_jobs"] == WORKERS, "more submitters did not mean more execution"
    assert report["peak_provider_calls"] == BUDGET, (
        f"{WORKERS * PER_JOB_POOL} potential callers, {BUDGET} allowed"
    )
    assert peak_budget["inflight"] == BUDGET
    assert report["succeeded"] == report["accepted"] and report["failed"] == 0
    assert report["provider_calls_total"] == report["accepted"] * CALLS_PER_JOB
    assert report["peak_embedding_calls"] <= EMBEDDING_BUDGET, (
        "embeddings are bounded by their own budget, not by Deep's"
    )
    assert report["embedding_calls_total"] == report["accepted"] * EMBEDDING_CALLS_PER_JOB
    assert budget.snapshot()["inflight"] == 0
    assert embedding_budget.snapshot()["inflight"] == 0
    assert all(body["error"]["type"] == "overloaded" for body in rejected)
    assert sorted(p.name for p in staging.rglob("*") if p.is_file()) == []
