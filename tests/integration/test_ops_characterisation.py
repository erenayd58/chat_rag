"""A controlled workload, measured through the telemetry it produced.

Phase 2's characterisation proved the limits hold. This one proves the
*instruments* do: it runs a known workload where every duration is exact --
each stage advances a clock the test owns by a fixed amount -- and then
checks that what ``/api/ops/metrics`` reports is what actually happened.
A metric that cannot be checked against a known answer is decoration.

It also watches the two things that could grow while it runs: the pipeline
cache (twenty sessions, a cache of four) and the job registry.

Numbers are printed for the report; the assertions are the properties that
must hold on any machine.
"""

from __future__ import annotations

import io
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from fastapi.testclient import TestClient

import asgi as entrypoint
import interfaces.http as http

V1 = http.v1.PREFIX
from chat_rag.application import ingest as app_ingest
from chat_rag.application import workspace as app_workspace
import tempfile
from chat_rag.components.ingest import IngestManager, PipelineCache
from chat_rag.components.ingest import jobs as J
from chat_rag.components.ingest import limits as L
from chat_rag.components.knowledgebase.manager import KnowledgeBaseManager
from chat_rag.components.observability import telemetry as T
from chat_rag.config.ingest import IngestLimits

from ingest_doubles import GatedProvider

SESSIONS = 20
WORKERS = 2
QUEUE = 6
CACHE_MAX = 4
DEEP_BUDGET = 3
EMBEDDING_BUDGET = 2
PER_JOB_POOL = 4
DEEP_CALLS = 6
EMBED_CALLS = 2

#: Exact, so the report's latency can be checked rather than eyeballed.
PARSE_SECONDS = 2.0
CHUNK_SECONDS = 1.0
INDEX_SECONDS = 0.5


class Clock:
    """A perf counter the test advances. Every stage duration is exact."""

    def __init__(self):
        self._lock = threading.Lock()
        self.now = 10_000.0

    def __call__(self):
        with self._lock:
            return self.now

    def advance(self, seconds):
        with self._lock:
            self.now += seconds


class FakeStore:
    def __init__(self):
        self.closed = 0

    def close(self):
        self.closed += 1

    def get_all_chunks(self):
        return []

    def delete_by_doc_id(self, doc_id):
        pass


class FakeRetriever:
    def __init__(self):
        self.invalidated = 0

    def build_keyword_index(self, chunks):
        pass

    def invalidate_index(self):
        self.invalidated += 1


class WorkloadPipeline:
    """One built pipeline: expensive to make, and measurable while it works."""

    built = 0

    def __init__(self, clock, deep_provider, embed_provider, budgets):
        type(self).built += 1
        self.clock = clock
        self.deep_provider = deep_provider
        self.embed_provider = embed_provider
        self.budgets = budgets
        self.settings = SimpleNamespace(deep_analysis_api_key_env="FAKE_DEEP_KEY")
        self.chunker = SimpleNamespace(
            get_name=lambda: "StructuralChunker",
            chunk_text_deep=lambda *a, **k: None,
            last_canonical_units=None, last_deep_result=None,
        )
        self.vector_db = FakeStore()
        self.hybrid_retriever = FakeRetriever()
        self.last_deep_analysis_report = None
        self.last_parse_seconds = PARSE_SECONDS

    def ingest_document_from_file(self, file_path, doc_title, deep_analysis=False):
        with T.stage(T.PARSE):
            self.clock.advance(PARSE_SECONDS)
            T.annotate(characters=4096)
        with T.stage(T.DEEP):
            self.clock.advance(CHUNK_SECONDS)
            limited = L.LimitedProvider(self.deep_provider, self.budgets["deep"],
                                        L.current_guard())
            with ThreadPoolExecutor(max_workers=PER_JOB_POOL) as pool:
                list(pool.map(limited.complete, [f"p{i}" for i in range(DEEP_CALLS)]))
        with T.stage(T.EMBED):
            embedding = L.LimitedEmbeddingTransport(self.embed_provider, self.budgets["embed"])
            for _ in range(EMBED_CALLS):
                embedding.embed(["a", "b"])
        with T.stage(T.INDEX):
            self.clock.advance(INDEX_SECONDS)
        return [SimpleNamespace(doc_id=f"doc-{os.path.basename(file_path)}")]


class EmbeddingDouble:
    model_id = "test:embed@1"

    def __init__(self, gate):
        self.gate = gate

    def embed(self, texts):
        self.gate.complete("embedding")
        return [[0.0] * 3 for _ in texts]


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    staging = tmp_path / "staging"
    staging.mkdir()
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(staging))
    manager = KnowledgeBaseManager(str(tmp_path / "kbs.json"))
    monkeypatch.setattr(entrypoint.services, "kb_manager", manager)
    monkeypatch.setattr(app_workspace, "stage_analysis", lambda *a, **k: {"status": "queued"})
    kb = manager.create("load-kb", chunker={"type": "structure_first"})
    return kb["kb_id"], staging


def test_load_characterisation(workspace, monkeypatch):
    kb_id, staging = workspace
    clock = Clock()
    monkeypatch.setattr(T.time, "perf_counter", clock)
    registry = T.MetricsRegistry(window=200)
    monkeypatch.setattr(T, "_registry", registry)

    deep_gate = GatedProvider(expect=DEEP_BUDGET)
    embed_gate = GatedProvider(expect=EMBEDDING_BUDGET)
    embed_gate.release()
    budgets = {"deep": L.configure_budget(DEEP_BUDGET),
               "embed": L.configure_embedding_budget(EMBEDDING_BUDGET)}

    WorkloadPipeline.built = 0
    cache = PipelineCache(
        build=lambda session_id, kb: WorkloadPipeline(
            clock, deep_gate, EmbeddingDouble(embed_gate), budgets),
        max_entries=CACHE_MAX, ttl_seconds=0,
    )
    monkeypatch.setattr(entrypoint.services, "pipeline_cache", cache)
    monkeypatch.setattr(entrypoint.services, "get_pipeline", lambda s, k=None: cache.get(s, k))

    manager = IngestManager(
        IngestLimits(workers=WORKERS, queue_capacity=QUEUE, job_timeout_seconds=300),
        execute=lambda job: app_ingest.execute_job(entrypoint.services, job),
    )
    monkeypatch.setattr(entrypoint.services, "ingest_jobs", manager)

    cache_before = cache.snapshot()["size"]
    rss_before = _rss_mb()

    barrier = threading.Barrier(SESSIONS)
    outcomes: list = []
    lock = threading.Lock()
    started = time.perf_counter

    def submit(index: int):
        """One session each, uploading at once.

        The session id is what selects a cached pipeline, and it is the thing
        that used to grow the cache without bound: one browser, one entry, per
        knowledge base. The cache is asked for this session's pipeline
        directly, because that is the read a request carrying a session makes
        -- this surface takes the id off the ASGI scope rather than from a
        cookie of its own, so an anonymous caller shares one entry and the
        pressure being characterised has to be put on the cache deliberately.
        """
        with TestClient(http.create_app(entrypoint.services),
                        raise_server_exceptions=False) as client:
            barrier.wait(20)
            cache.get(f"session-{index}", kb_id)
            response = client.post(
                f"{V1}/documents",
                files={"file": (f"b{index}.txt",
                                io.BytesIO(f"belge {index}".encode()), "text/plain")},
                data={"knowledge_base_id": kb_id, "methods": ["agentic"]},
            )
            with lock:
                outcomes.append((response.status_code, response.json()))

    wall_started = time.monotonic()
    threads = [threading.Thread(target=submit, args=(index,)) for index in range(SESSIONS)]
    try:
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(60)
        assert deep_gate.full.wait(30), "the Deep budget was never saturated"
        peak_capacity = manager.snapshot()
        peak_cache = cache.snapshot()
        deep_gate.release()
        assert manager.drain(120)
        elapsed = time.monotonic() - wall_started
    finally:
        deep_gate.release()
        embed_gate.release()
        manager.close(timeout=60)
        L.configure_budget(8)
        L.configure_embedding_budget(4)

    accepted = [body for code, body in outcomes if code == 202]
    rejected = [body for code, body in outcomes if code == 503]
    metrics = registry.snapshot()
    cache_after = cache.snapshot()
    rss_after = _rss_mb()

    report = {
        "submitted": SESSIONS,
        "accepted": len(accepted),
        "rejected": len(rejected),
        "queued_at_peak": peak_capacity["queued"],
        "admitted_at_peak": peak_capacity["queued"] + peak_capacity["running"],
        "peak_active_jobs": manager.stats["peak_active"],
        "peak_deep_slots": budgets["deep"].peak,
        "peak_embedding_slots": budgets["embed"].peak,
        "queue_wait_p50": metrics["jobs"]["queue_wait_seconds"]["p50"],
        "queue_wait_max": metrics["jobs"]["queue_wait_seconds"]["max"],
        "job_total_p50": metrics["jobs"]["total_seconds"]["p50"],
        "stage_parse_p50": metrics["stages"]["parse"]["p50"],
        "stage_deep_p50": metrics["stages"]["deep_analysis"]["p50"],
        "stage_index_p50": metrics["stages"]["index"]["p50"],
        "provider_wait_total": budgets["deep"].snapshot()["wait_seconds_total"],
        "pipelines_built": WorkloadPipeline.built,
        "pipeline_cache_before": cache_before,
        "pipeline_cache_peak": peak_cache["size"],
        "pipeline_cache_after": cache_after["size"],
        "pipeline_cache_max": CACHE_MAX,
        "pipelines_evicted": cache_after["stats"]["evicted_lru"],
        "jobs_retained": manager.snapshot()["retained"]["finished"],
        "succeeded": manager.stats["succeeded"],
        "failed": manager.stats["failed"] + manager.stats["timed_out"],
        "rss_mb_before": rss_before,
        "rss_mb_after": rss_after,
        "elapsed_seconds": round(elapsed, 3),
    }
    print("\nPHASE 3 LOAD CHARACTERISATION")
    for key, value in report.items():
        print(f"  {key:>24}: {value}")

    # -- the telemetry says what actually happened ------------------------
    assert report["accepted"] == WORKERS + QUEUE
    assert report["rejected"] == SESSIONS - (WORKERS + QUEUE)
    assert report["succeeded"] == report["accepted"] and report["failed"] == 0
    assert report["stage_parse_p50"] == PARSE_SECONDS, "an exact, known duration"
    assert report["stage_index_p50"] == INDEX_SECONDS
    assert metrics["stages"]["parse"]["count"] == report["accepted"]
    assert metrics["counters"]["ingest.accepted"] == report["accepted"]
    assert metrics["counters"]["ingest.rejected"] == report["rejected"]
    assert metrics["counters"]["ingest.succeeded"] == report["accepted"]
    assert metrics["errors"]["by_category"]["overloaded"] == report["rejected"]
    # Jobs that queued behind a worker waited; the ones that ran first did not.
    assert report["queue_wait_max"] > 0

    # -- the limits still held -------------------------------------------
    assert report["peak_active_jobs"] <= WORKERS
    # The bound is on what has been admitted -- running plus waiting -- not on
    # the queue alone at every instant: a worker that has not yet picked its
    # job up leaves that job counted as queued. Phase 2 states the contract
    # this way; this workload is the first to sample the moment it matters.
    assert report["admitted_at_peak"] <= WORKERS + QUEUE
    assert report["queued_at_peak"] <= WORKERS + QUEUE
    assert report["peak_deep_slots"] == DEEP_BUDGET
    assert report["peak_embedding_slots"] <= EMBEDDING_BUDGET

    # -- and nothing grew without a bound --------------------------------
    assert report["pipeline_cache_peak"] <= CACHE_MAX + WORKERS, (
        "over the bound only by pipelines actively leased by a worker"
    )
    assert report["pipeline_cache_after"] <= CACHE_MAX
    assert report["pipelines_built"] >= report["accepted"] // 2
    assert report["pipelines_evicted"] > 0, "the churn was real, not avoided"
    assert metrics["jobs"]["measured"] <= 200
    assert sorted(p.name for p in staging.rglob("*") if p.is_file()) == []


def _rss_mb():
    """Resident memory, when the platform will say. Reported, never asserted:
    a garbage-collected heap is not a stable number to gate a test on."""
    try:
        import ctypes
        import ctypes.wintypes

        class Counters(ctypes.Structure):
            _fields_ = [("cb", ctypes.wintypes.DWORD),
                        ("PageFaultCount", ctypes.wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t),
                        ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t),
                        ("PeakPagefileUsage", ctypes.c_size_t)]

        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        # The return type matters: GetCurrentProcess hands back a pseudo-handle
        # that truncates to a wrong value under the default c_int on 64-bit,
        # and the call then simply fails and reports nothing.
        kernel32 = ctypes.windll.kernel32
        kernel32.GetCurrentProcess.restype = ctypes.wintypes.HANDLE
        psapi = ctypes.windll.psapi
        psapi.GetProcessMemoryInfo.argtypes = [
            ctypes.wintypes.HANDLE, ctypes.POINTER(Counters), ctypes.wintypes.DWORD,
        ]
        if psapi.GetProcessMemoryInfo(kernel32.GetCurrentProcess(),
                                      ctypes.byref(counters), counters.cb):
            return round(counters.WorkingSetSize / (1024 * 1024), 1)
    except Exception:  # noqa: BLE001 - a missing measurement is not a failure
        pass
    try:
        import resource

        return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)
    except Exception:  # noqa: BLE001
        return None
