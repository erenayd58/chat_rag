"""What an operator sees: telemetry from real jobs, health that means
something, and metrics that stay bounded.

These drive the application's own job body, so the numbers under test are
the ones a real ingest produces -- the stages the pipeline opens, the queue
wait the manager measures, the counters the lifecycle increments. The
pipeline itself is a stub with a controllable clock, so a stage that claims
to have taken two seconds took exactly two seconds.
"""

from __future__ import annotations

import io
import json
import os
import threading
from types import SimpleNamespace

import pytest

import app as flask_app
from components.ingest import IngestManager
from components.ingest import jobs as J
from components.ingest import limits as L
from components.knowledgebase.manager import KnowledgeBaseManager
from components.observability import telemetry as T
from config.ingest import IngestLimits


class Chunker:
    def __init__(self):
        self.last_canonical_units = None
        self.last_deep_result = None
        self.chunk_text_deep = lambda *a, **k: None  # noqa: E731

    def get_name(self):
        return "StructuralChunker"


class TimedPipeline:
    """A pipeline that spends a known amount of time in each stage."""

    def __init__(self, clock, *, seconds=None, error=None, gate=None, chunks=3):
        self.settings = SimpleNamespace(deep_analysis_api_key_env="FAKE_DEEP_KEY")
        self.chunker = Chunker()
        self.vector_db = SimpleNamespace(delete_by_doc_id=lambda d: None,
                                         get_all_chunks=lambda: [])
        self.hybrid_retriever = SimpleNamespace(
            build_keyword_index=lambda chunks: None,
            invalidate_index=lambda: None,
        )
        self.last_deep_analysis_report = None
        self.last_parse_seconds = 0.1
        self.clock = clock
        self.seconds = seconds or {T.PARSE: 2.0, T.CHUNK: 1.0, T.INDEX: 0.5}
        self.error = error
        self.gate = gate
        self.chunks = chunks
        self.started = threading.Event()

    def ingest_document_from_file(self, file_path, doc_title, deep_analysis=False):
        self.started.set()
        if self.gate is not None:
            self.gate.wait(20)
        for name, seconds in self.seconds.items():
            with T.stage(name):
                self.clock.advance(seconds)
                T.annotate(units=7)
        if self.error is not None:
            raise self.error
        return [SimpleNamespace(doc_id="doc-under-test")] * self.chunks


class Clock:
    def __init__(self):
        self.now = 5000.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


@pytest.fixture
def clock(monkeypatch):
    fake = Clock()
    monkeypatch.setattr(T.time, "perf_counter", fake)
    return fake


@pytest.fixture
def registry(monkeypatch):
    fresh = T.MetricsRegistry(window=50)
    monkeypatch.setattr(T, "_registry", fresh)
    return fresh


@pytest.fixture
def client(tmp_path, monkeypatch, registry):
    monkeypatch.chdir(tmp_path)
    staging = tmp_path / "staging"
    staging.mkdir()
    monkeypatch.setattr(flask_app.tempfile, "gettempdir", lambda: str(staging))
    manager = KnowledgeBaseManager(str(tmp_path / "kbs.json"))
    monkeypatch.setattr(flask_app, "kb_manager", manager)
    monkeypatch.setattr(flask_app, "stage_viewer_analysis", lambda *a, **k: {"status": "queued"})
    flask_app.app.config.update(TESTING=True)
    kb = manager.create("ops-kb", chunker={"type": "structure_first"})
    with flask_app.app.test_client() as test_client:
        yield test_client, kb["kb_id"]


@pytest.fixture
def jobs(monkeypatch):
    managers = []

    def make(**limits):
        fields = dict(workers=1, queue_capacity=2, job_timeout_seconds=60)
        fields.update(limits)
        manager = IngestManager(IngestLimits(**fields),
                                execute=lambda job: flask_app._execute_ingest(job))
        monkeypatch.setattr(flask_app, "ingest_jobs", manager)
        managers.append(manager)
        return manager

    yield make
    for manager in managers:
        manager.close(timeout=20)


def use_pipeline(monkeypatch, pipeline):
    """Make the cache build this pipeline, rather than replacing the seam.

    Patching ``get_pipeline`` would take the cache out of the path, and the
    cache is part of what these tests are about -- a job leases the entry the
    cache holds. Building the stub *through* the cache keeps that real.
    """
    flask_app.pipeline_cache.clear()
    monkeypatch.setattr(flask_app.pipeline_cache, "_build", lambda session, kb: pipeline)
    return pipeline


def upload(test_client, kb_id, *, content=b"belge", **fields):
    data = {"file": (io.BytesIO(content), "belge.txt"), "kb_id": kb_id}
    data.update(fields)
    return test_client.post("/api/documents/upload", data=data,
                            content_type="multipart/form-data")


# ------------------------------------------------------------- job timing


def test_a_finished_job_reports_where_its_time_went(client, jobs, monkeypatch, clock):
    test_client, kb_id = client
    manager = jobs()
    use_pipeline(monkeypatch, TimedPipeline(clock))

    body = upload(test_client, kb_id).get_json()
    job = test_client.get(f"/api/ingest/jobs/{body['job_id']}").get_json()["job"]

    timing = job["timing"]
    assert timing["stages"]["parse"] == 2.0
    assert timing["stages"]["chunk"] == 1.0
    assert timing["stages"]["index"] == 0.5
    assert timing["status"] == J.SUCCEEDED
    assert timing["failed_stages"] == []
    assert "ledger" in timing["stages"], "the commit is measured too"
    # The stages run on the clock this test drives; the job's total is real
    # wall time, so it is asserted as a measurement rather than as a sum.
    assert timing["total_seconds"] > 0
    assert sum(timing["stages"].values()) >= 3.5


def test_the_queue_wait_is_measured_and_is_not_the_work(client, jobs, monkeypatch, clock):
    """The first number an operator wants when uploads feel slow: it separates
    "the system is busy" from "this document is slow"."""
    test_client, kb_id = client
    manager = jobs(workers=1, queue_capacity=2)
    gate = threading.Event()
    pipeline = use_pipeline(monkeypatch, TimedPipeline(clock, gate=gate))

    first = upload(test_client, kb_id, content=b"one", **{"async": "1"}).get_json()
    assert pipeline.started.wait(10)
    second = upload(test_client, kb_id, content=b"two", **{"async": "1"}).get_json()
    gate.set()
    assert manager.drain(20)

    waited = test_client.get(f"/api/ingest/jobs/{second['job_id']}").get_json()["job"]
    ran_first = test_client.get(f"/api/ingest/jobs/{first['job_id']}").get_json()["job"]
    assert waited["timing"]["queue_seconds"] > ran_first["timing"]["queue_seconds"]
    assert waited["timing"]["stages"]["parse"] == 2.0, "queueing is not counted as work"


def test_a_failed_job_says_which_stage_failed_and_why(client, jobs, monkeypatch, clock):
    from core.exceptions import ChunkerException

    test_client, kb_id = client
    jobs()
    use_pipeline(monkeypatch, TimedPipeline(
        clock, seconds={T.PARSE: 1.0}, error=ChunkerException("bad canonical")))

    response = upload(test_client, kb_id)
    assert response.status_code == 500
    body = response.get_json()
    job = test_client.get(f"/api/ingest/jobs/{body['job_id']}").get_json()["job"]
    assert job["status"] == J.FAILED
    assert job["error_category"] == "chunking"
    assert job["timing"]["stages"]["parse"] == 1.0, "the work before the failure is kept"


# ------------------------------------------------------------- the metrics


def test_the_metrics_endpoint_reports_real_counters_and_latency(
    client, jobs, monkeypatch, clock, registry
):
    test_client, kb_id = client
    manager = jobs(workers=1, queue_capacity=4)
    use_pipeline(monkeypatch, TimedPipeline(clock))

    for index in range(3):
        upload(test_client, kb_id, content=f"belge {index}".encode())
    assert manager.drain(20)

    metrics = test_client.get("/api/ops/metrics").get_json()
    assert metrics["success"] is True
    counters = metrics["metrics"]["counters"]
    assert counters["ingest.accepted"] == 3
    assert counters["ingest.succeeded"] == 3
    assert metrics["metrics"]["jobs"]["measured"] == 3
    assert metrics["metrics"]["stages"]["parse"]["count"] == 3
    assert metrics["metrics"]["stages"]["parse"]["p50"] == 2.0
    assert metrics["metrics"]["stages"]["parse"]["total"] == 6.0
    assert metrics["ingest"]["stats"]["succeeded"] == 3
    assert metrics["caches"]["pipelines"]["max"] >= 1


def test_overload_and_failure_counters_are_correct(client, jobs, monkeypatch, clock, registry):
    test_client, kb_id = client
    manager = jobs(workers=1, queue_capacity=0)
    gate = threading.Event()
    pipeline = use_pipeline(monkeypatch, TimedPipeline(clock, gate=gate))

    accepted = upload(test_client, kb_id, content=b"one", **{"async": "1"})
    assert accepted.status_code == 202
    assert pipeline.started.wait(10)
    refused = upload(test_client, kb_id, content=b"two", **{"async": "1"})
    assert refused.status_code == 503

    metrics = test_client.get("/api/ops/metrics").get_json()["metrics"]
    assert metrics["counters"]["ingest.rejected"] == 1
    assert metrics["counters"]["ingest.accepted"] == 1
    assert metrics["errors"]["by_category"]["overloaded"] == 1
    assert metrics["errors"]["recent_messages"]["overloaded"], "with a reason attached"

    gate.set()
    assert manager.drain(20)


def test_the_metrics_answer_cannot_grow_with_uptime(client, jobs, monkeypatch, clock, registry):
    test_client, kb_id = client
    manager = jobs(workers=1, queue_capacity=40)
    use_pipeline(monkeypatch, TimedPipeline(clock))
    for index in range(25):
        upload(test_client, kb_id, content=f"belge {index}".encode())
    assert manager.drain(60)

    small = test_client.get("/api/ops/metrics?recent=5").get_json()
    big = test_client.get("/api/ops/metrics?recent=1000").get_json()
    assert len(small["metrics"]["recent"]) == 5
    assert len(big["metrics"]["recent"]) <= 25, "capped whatever is asked for"
    assert big["metrics"]["jobs"]["measured"] == 25
    assert len(json.dumps(big)) < 60_000


def test_a_bad_recent_parameter_does_not_break_the_endpoint(client, jobs):
    test_client, _ = client
    jobs()
    assert test_client.get("/api/ops/metrics?recent=lots").status_code == 200


# -------------------------------------------------------------- the health


def test_health_is_small_and_says_what_state_the_service_is_in(client, jobs):
    test_client, _ = client
    jobs()
    body = test_client.get("/api/health").get_json()
    assert body["status"] == "healthy", "the historical field, for existing probes"
    assert body["state"] == "ok"
    assert body["ready"] is True
    assert body["reasons"] == []
    assert set(body["ingest"]) == {
        "running", "queued", "queue_capacity", "workers",
        "provider_inflight", "provider_limit", "embedding_inflight", "embedding_limit",
    }
    assert "recent" not in json.dumps(body), "no history on the liveness endpoint"
    assert len(json.dumps(body)) < 1000


def test_health_says_overloaded_while_the_queue_is_full_but_stays_ready(
    client, jobs, monkeypatch, clock
):
    """Overloaded is not unready: chat and search still work, and refusing
    traffic entirely would make the overload worse."""
    test_client, kb_id = client
    manager = jobs(workers=1, queue_capacity=0)
    gate = threading.Event()
    pipeline = use_pipeline(monkeypatch, TimedPipeline(clock, gate=gate))
    upload(test_client, kb_id, content=b"one", **{"async": "1"})
    assert pipeline.started.wait(10)

    body = test_client.get("/api/health").get_json()
    assert body["state"] == "overloaded"
    assert body["ready"] is True
    assert "queue is full" in " ".join(body["reasons"])

    gate.set()
    assert manager.drain(20)
    assert test_client.get("/api/health").get_json()["state"] == "ok"


def test_health_is_degraded_when_the_knowledge_base_records_cannot_be_read(
    client, jobs, monkeypatch
):
    test_client, _ = client
    jobs()

    def broken():
        raise OSError("the records are unreadable")

    monkeypatch.setattr(flask_app.kb_manager, "list", broken)
    body = test_client.get("/api/health").get_json()
    assert body["state"] == "degraded"
    assert "unreadable" in " ".join(body["reasons"])


def test_health_answers_while_the_workers_are_busy(client, jobs, monkeypatch, clock):
    """Bounded load must not make the liveness endpoint slow: it touches no
    store, no model and no job history."""
    test_client, kb_id = client
    manager = jobs(workers=1, queue_capacity=4)
    gate = threading.Event()
    pipeline = use_pipeline(monkeypatch, TimedPipeline(clock, gate=gate))
    for index in range(4):
        upload(test_client, kb_id, content=f"belge {index}".encode(), **{"async": "1"})
    assert pipeline.started.wait(10)

    for _ in range(20):
        assert test_client.get("/api/health").status_code == 200
        assert test_client.get("/api/ops/metrics").status_code == 200

    gate.set()
    assert manager.drain(30)


# ------------------------------------------------------- the pipeline cache


def test_an_ingest_leases_its_pipeline_so_it_cannot_be_evicted(
    client, jobs, monkeypatch, clock
):
    test_client, kb_id = client
    manager = jobs()
    gate = threading.Event()
    pipeline = use_pipeline(monkeypatch, TimedPipeline(clock, gate=gate))
    monkeypatch.setattr(flask_app.pipeline_cache, "max_entries", 1)

    upload(test_client, kb_id, **{"async": "1"})
    assert pipeline.started.wait(10)
    assert flask_app.pipeline_cache.snapshot()["leased"] == 1

    # Traffic from other sessions cannot take the running job's pipeline away.
    for index in range(4):
        flask_app.pipeline_cache.get(f"browser{index}", kb_id)
    assert flask_app.pipeline_cache.snapshot()["leased"] == 1

    gate.set()
    assert manager.drain(20)
    assert flask_app.pipeline_cache.snapshot()["leased"] == 0


def test_the_cache_state_is_visible_to_an_operator(client, jobs):
    test_client, kb_id = client
    jobs()
    flask_app.pipeline_cache.get("someone", kb_id)
    caches = test_client.get("/api/ops/metrics").get_json()["caches"]
    assert caches["pipelines"]["size"] >= 1
    assert caches["pipelines"]["max"] >= 1
    assert "loads" in caches["viewer_boundary_model"]


# ------------------------------------------------ what the endpoint may say


def test_an_exception_string_does_not_become_a_leak_surface(
    client, jobs, monkeypatch, clock, registry
):
    """The endpoint is reachable by anyone who can reach the app, so what a
    library happened to write must not be served verbatim.

    A storage failure names the data root, the account it lives under and the
    document, and none of that looks like a credential -- so passing the
    credential filter is not the same as being safe to publish. What comes
    back should say what went wrong and nothing about where.
    """
    test_client, kb_id = client
    jobs()
    secret_path = os.path.join("C:" + os.sep, "Users", "alice", "data", "Q3 board minutes.pdf")
    use_pipeline(monkeypatch, TimedPipeline(
        clock, seconds={T.PARSE: 1.0},
        error=OSError("[Errno 13] Permission denied: " + repr(secret_path)
                      + " using sk-abcdefghijklmnopqrstuvwxyz012345")))

    upload(test_client, kb_id)
    served = json.dumps(test_client.get("/api/ops/metrics").get_json())

    assert "alice" not in served
    assert "Q3 board minutes" not in served
    assert "sk-abcdefghijkl" not in served
    # And the useful half is still there.
    assert "Permission denied" in served


def test_the_metrics_body_carries_no_content_or_filesystem_path(
    client, jobs, monkeypatch, clock, registry
):
    """Aggregates, ids, categories and durations -- nothing else."""
    test_client, kb_id = client
    manager = jobs()
    use_pipeline(monkeypatch, TimedPipeline(clock))
    upload(test_client, kb_id, content="gizli sirket belgesi".encode())
    assert manager.drain(20)

    served = json.dumps(test_client.get("/api/ops/metrics?recent=25").get_json())
    assert "gizli sirket belgesi" not in served, "no document content"
    assert "belge.txt" not in served, "no uploaded filename"
    assert os.sep + "staging" not in served, "no temp path"
    assert "OPENROUTER_API_KEY" not in served and "sk-" not in served


def test_a_broken_store_does_not_publish_the_data_root(client, jobs, monkeypatch):
    """The same rule on the reason /api/health gives for being degraded."""
    test_client, _ = client
    jobs()
    root = os.path.join("C:" + os.sep, "srv", "chat_rag", "data", "kbs.json")

    def broken():
        raise OSError("[Errno 13] Permission denied: " + repr(root))

    monkeypatch.setattr(flask_app.kb_manager, "list", broken)
    body = test_client.get("/api/health").get_json()
    reasons = " ".join(body["reasons"])
    assert body["state"] == "degraded"
    assert "chat_rag" not in reasons and "srv" not in reasons
    assert "Permission denied" in reasons


# ------------------------------------------------------- health semantics


def test_alive_ready_and_state_are_three_different_answers(
    client, jobs, monkeypatch, clock, registry
):
    """The contract, asserted rather than described.

    ``status`` stays the historical liveness word for probes that predate
    this; ``ready`` says whether traffic may be sent and stays true in every
    state, because refusing traffic during an overload makes it worse; and
    ``state`` is the one an operator reads.
    """
    test_client, kb_id = client
    manager = jobs(workers=1, queue_capacity=4)
    use_pipeline(monkeypatch, TimedPipeline(
        clock, seconds={T.PARSE: 1.0}, error=OSError("the store is gone")))

    healthy = test_client.get("/api/health").get_json()
    assert (healthy["status"], healthy["state"], healthy["ready"]) == ("healthy", "ok", True)

    for index in range(flask_app.DEGRADED_AFTER_JOBS):
        upload(test_client, kb_id, content=f"belge {index}".encode())
    assert manager.drain(20)

    broken = test_client.get("/api/health").get_json()
    assert broken["state"] == "degraded"
    assert broken["ready"] is True, "a degraded process still serves reads"
    assert broken["status"] == "healthy", "liveness is not readiness is not state"
    assert "all failed" in " ".join(broken["reasons"])


def test_a_service_that_recovers_stops_calling_itself_degraded(
    client, jobs, monkeypatch, clock, registry
):
    """Degraded is a ratio over a window, not a latch."""
    test_client, kb_id = client
    manager = jobs(workers=1, queue_capacity=8)
    use_pipeline(monkeypatch, TimedPipeline(
        clock, seconds={T.PARSE: 1.0}, error=OSError("the store is gone")))
    for index in range(flask_app.DEGRADED_AFTER_JOBS):
        upload(test_client, kb_id, content=f"kotu {index}".encode())
    assert manager.drain(20)
    assert test_client.get("/api/health").get_json()["state"] == "degraded"

    use_pipeline(monkeypatch, TimedPipeline(clock))
    upload(test_client, kb_id, content=b"iyi belge")
    assert manager.drain(20)
    assert test_client.get("/api/health").get_json()["state"] == "ok"
