"""The upload route as a job, over HTTP: the asynchronous contract, the
synchronous one it keeps for existing callers, overload, duplicates, the
status endpoints, and what a stopped job leaves behind (nothing).

The pipeline is a stub -- these tests are about the route, the job and the
ledger, not about parsing -- and every test drives a job manager of its
own, so a job blocked here cannot reach another test.
"""

from __future__ import annotations

import io
import json
import threading
from types import SimpleNamespace

import pytest

import app as flask_app
from components.ingest import IngestManager
from components.ingest import jobs as J
from components.ingest.limits import JobGuard, checkpoint
from components.knowledgebase.manager import KnowledgeBaseManager
from config import paths
from config.ingest import IngestLimits


class Chunker:
    def __init__(self):
        self.last_canonical_units = None
        self.last_deep_result = None
        self.chunk_text_deep = lambda *a, **k: None  # noqa: E731 - presence is the contract

    def get_name(self):
        return "StructuralChunker"


class VectorDB:
    def __init__(self):
        self.deleted: list[str] = []

    def delete_by_doc_id(self, doc_id):
        self.deleted.append(doc_id)

    def get_all_chunks(self):
        return []


class StubPipeline:
    """Only what the job touches. ``gate`` holds the ingest open; ``error``
    makes it fail; ``checkpoints`` makes it ask the job guard like the
    real pipeline does between stages."""

    def __init__(self, *, gate=None, error=None, checkpoints=False, chunks=1):
        self.settings = SimpleNamespace(deep_analysis_api_key_env="FAKE_DEEP_KEY")
        self.chunker = Chunker()
        self.vector_db = VectorDB()
        self.last_deep_analysis_report = None
        self.last_parse_seconds = 0.1
        self.gate = gate
        self.error = error
        self.checkpoints = checkpoints
        self.chunks = chunks
        self.started = threading.Event()
        self.calls = 0

    def ingest_document_from_file(self, file_path, doc_title, deep_analysis=False):
        self.calls += 1
        self.started.set()
        if self.gate is not None:
            self.gate.wait(20)
        if self.checkpoints:
            checkpoint()
        if self.error is not None:
            raise self.error
        return [SimpleNamespace(doc_id="doc-under-test")] * self.chunks


@pytest.fixture
def staging(tmp_path, monkeypatch):
    directory = tmp_path / "staging"
    directory.mkdir()
    monkeypatch.setattr(flask_app.tempfile, "gettempdir", lambda: str(directory))
    return directory


def staged_files(staging) -> list[str]:
    return sorted(p.name for p in staging.rglob("*") if p.is_file())


@pytest.fixture
def client(tmp_path, monkeypatch, staging):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("FAKE_DEEP_KEY", "sk-placeholder")
    manager = KnowledgeBaseManager(str(tmp_path / "kbs.json"))
    monkeypatch.setattr(flask_app, "kb_manager", manager)
    monkeypatch.setattr(flask_app, "stage_viewer_analysis", lambda *a, **k: {"status": "queued"})
    flask_app.app.config.update(TESTING=True)
    kb = manager.create("jobs-kb", chunker={"type": "structure_first"})
    with flask_app.app.test_client() as test_client:
        yield test_client, kb["kb_id"]


@pytest.fixture
def jobs(monkeypatch):
    """A manager of this test's own, running the application's real job body."""
    managers = []

    def make(journal=None, **limits):
        fields = dict(workers=1, queue_capacity=1, job_timeout_seconds=60)
        fields.update(limits)
        manager = IngestManager(
            IngestLimits(**fields),
            execute=lambda job: flask_app._execute_ingest(job),
            journal=journal,
        )
        monkeypatch.setattr(flask_app, "ingest_jobs", manager)
        managers.append(manager)
        return manager

    yield make
    for manager in managers:
        manager.close(timeout=20)


def use_pipeline(monkeypatch, pipeline):
    monkeypatch.setattr(flask_app, "get_pipeline", lambda *a, **k: pipeline)
    return pipeline


def upload(test_client, kb_id, *, content=b"kucuk bir test belgesi", name="belge.txt", **fields):
    data = {"file": (io.BytesIO(content), name), "kb_id": kb_id}
    data.update(fields)
    return test_client.post("/api/documents/upload", data=data, content_type="multipart/form-data")


def poll(test_client, job_id, manager):
    """Wait for the job through the manager, then read it back over HTTP --
    the same body a browser polls for, without a polling loop in the test."""
    job = manager.get(job_id)
    assert job is not None
    assert manager.wait(job, 20)
    response = test_client.get(f"/api/ingest/jobs/{job_id}")
    assert response.status_code == 200
    return response.get_json()["job"]


def ledger():
    return json.load(open(paths.ingested_documents(), encoding="utf-8"))


# ---------------------------------------------------------------- async


def test_an_async_upload_is_accepted_at_once_and_finishes_later(client, jobs, monkeypatch, staging):
    test_client, kb_id = client
    manager = jobs()
    gate = threading.Event()
    pipeline = use_pipeline(monkeypatch, StubPipeline(gate=gate))

    response = upload(test_client, kb_id, **{"async": "1"})
    assert response.status_code == 202
    body = response.get_json()
    assert body["success"] is True and body["pending"] is True and body["attached"] is False
    assert body["job"]["status"] in (J.QUEUED, J.RUNNING)
    assert body["job"]["filename"] == "belge.txt"
    assert body["job"]["chunking_mode"] == "standard"
    assert "temp_path" not in json.dumps(body)

    assert pipeline.started.wait(10)
    running = test_client.get(f"/api/ingest/jobs/{body['job_id']}").get_json()["job"]
    assert running["status"] == J.RUNNING
    assert ledger() == {} if __import__("os").path.exists(paths.ingested_documents()) else True
    assert len(staged_files(staging)) == 1, "the file lives while the job runs"

    gate.set()
    job = poll(test_client, body["job_id"], manager)
    assert job["status"] == J.SUCCEEDED
    assert job["doc_id"] == "doc-under-test"
    assert job["result"]["chunks_created"] == 1
    assert job["result"]["success"] is True
    (record,) = ledger().values()
    assert record["doc_id"] == "doc-under-test"
    assert record["metadata"]["ingest_job_id"] == body["job_id"]
    assert staged_files(staging) == []


def test_a_sync_upload_still_answers_as_it_always_did(client, jobs, monkeypatch):
    test_client, kb_id = client
    jobs()
    use_pipeline(monkeypatch, StubPipeline())
    response = upload(test_client, kb_id)
    assert response.status_code == 200
    body = response.get_json()
    assert body["success"] is True
    assert body["doc_id"] == "doc-under-test"
    assert body["chunks_created"] == 1
    assert body["chunking_mode"] == "standard"
    assert body["job"]["status"] == J.SUCCEEDED
    assert len(ledger()) == 1


def test_a_sync_upload_that_outlasts_the_wait_hands_back_the_job(client, jobs, monkeypatch):
    test_client, kb_id = client
    manager = jobs()
    monkeypatch.setattr(flask_app.settings, "ingest_sync_wait", 0.05)
    gate = threading.Event()
    use_pipeline(monkeypatch, StubPipeline(gate=gate))
    response = upload(test_client, kb_id)
    assert response.status_code == 202
    body = response.get_json()
    assert body["pending"] is True
    gate.set()
    assert poll(test_client, body["job_id"], manager)["status"] == J.SUCCEEDED


# ------------------------------------------------------------- failures


def test_a_failed_job_is_truthful_and_registers_nothing(client, jobs, monkeypatch, staging):
    test_client, kb_id = client
    manager = jobs()
    use_pipeline(monkeypatch, StubPipeline(error=RuntimeError("parser exploded")))
    body = upload(test_client, kb_id, **{"async": "1"}).get_json()
    job = poll(test_client, body["job_id"], manager)
    assert job["status"] == J.FAILED
    assert "parser exploded" in job["error"]
    assert job["result"] is None
    assert not __import__("os").path.exists(paths.ingested_documents()) or ledger() == {}
    assert staged_files(staging) == []


def test_a_sync_failure_keeps_its_status_codes(client, jobs, monkeypatch):
    from core.exceptions import ConfigurationException, IndexIncompatibleException

    test_client, kb_id = client
    jobs()
    use_pipeline(monkeypatch, StubPipeline(error=IndexIncompatibleException("other model")))
    response = upload(test_client, kb_id)
    assert response.status_code == 409 and response.get_json()["reindex_required"] is True

    use_pipeline(monkeypatch, StubPipeline(error=ConfigurationException("no deep path")))
    response = upload(test_client, kb_id, content=b"another")
    assert response.status_code == 503 and response.get_json()["deep_analysis_unavailable"] is True

    use_pipeline(monkeypatch, StubPipeline(error=RuntimeError("boom")))
    response = upload(test_client, kb_id, content=b"third")
    assert response.status_code == 500 and response.get_json()["success"] is False


def test_a_job_past_its_deadline_ends_timed_out_and_commits_nothing(client, jobs, monkeypatch, staging):
    test_client, kb_id = client
    manager = jobs()
    monkeypatch.setattr(J.JobGuard, "for_timeout", classmethod(lambda cls, s: JobGuard(deadline=-1.0)))
    use_pipeline(monkeypatch, StubPipeline(checkpoints=True))
    response = upload(test_client, kb_id)
    assert response.status_code == 504
    body = response.get_json()
    assert body["timed_out"] is True and body["success"] is False
    job = manager.get(body["job_id"])
    assert job.status == J.TIMED_OUT
    assert not __import__("os").path.exists(paths.ingested_documents()) or ledger() == {}
    assert staged_files(staging) == []


def test_a_ledger_that_cannot_be_written_takes_the_store_rows_back(client, jobs, monkeypatch):
    from utils import document_tracker

    test_client, kb_id = client
    manager = jobs()
    pipeline = use_pipeline(monkeypatch, StubPipeline(chunks=3))
    monkeypatch.setattr(document_tracker.DocumentTracker, "mark_as_ingested", lambda self, **k: False)
    body = upload(test_client, kb_id, **{"async": "1"}).get_json()
    job = poll(test_client, body["job_id"], manager)
    assert job["status"] == J.FAILED
    assert "ledger" in job["error"]
    assert pipeline.vector_db.deleted == ["doc-under-test"], "the rows were rolled back"


# ------------------------------------------------------------- overload


def test_a_full_queue_is_refused_with_a_retry_after(client, jobs, monkeypatch, staging):
    test_client, kb_id = client
    manager = jobs(workers=1, queue_capacity=1)
    gate = threading.Event()
    pipeline = use_pipeline(monkeypatch, StubPipeline(gate=gate))

    first = upload(test_client, kb_id, content=b"one", **{"async": "1"})
    assert first.status_code == 202
    assert pipeline.started.wait(10)
    second = upload(test_client, kb_id, content=b"two", **{"async": "1"})
    assert second.status_code == 202
    assert second.get_json()["job"]["status"] == J.QUEUED
    assert second.get_json()["job"]["position"] == 1

    third = upload(test_client, kb_id, content=b"three", **{"async": "1"})
    assert third.status_code == 503
    body = third.get_json()
    assert body["overloaded"] is True and body["success"] is False
    assert body["retry_after_seconds"] >= 5
    assert third.headers["Retry-After"] == str(int(body["retry_after_seconds"]))
    assert manager.snapshot()["queued"] == 1
    assert len(staged_files(staging)) == 2, "the refused file is already gone"

    health = test_client.get("/api/health").get_json()
    assert health["ingest"]["running"] == 1 and health["ingest"]["queued"] == 1
    assert health["state"] == "overloaded", "the queue is full and uploads are refused"
    # Counters live at the operational endpoint; health stays small enough for
    # a probe to poll every few seconds.
    metrics = test_client.get("/api/ops/metrics").get_json()
    assert metrics["ingest"]["stats"]["rejected"] == 1
    assert metrics["metrics"]["counters"]["ingest.rejected"] >= 1

    gate.set()
    assert manager.drain(20)
    assert staged_files(staging) == []


# ------------------------------------------------------------ duplicates


def test_the_same_file_twice_in_flight_attaches_to_one_job(client, jobs, monkeypatch, staging):
    test_client, kb_id = client
    manager = jobs()
    gate = threading.Event()
    pipeline = use_pipeline(monkeypatch, StubPipeline(gate=gate))

    first = upload(test_client, kb_id, **{"async": "1"}).get_json()
    assert pipeline.started.wait(10)
    second = upload(test_client, kb_id, **{"async": "1"}).get_json()
    assert second["attached"] is True
    assert second["job_id"] == first["job_id"]
    assert second["job"]["attached_uploads"] == 1
    assert len(staged_files(staging)) == 1

    gate.set()
    job = poll(test_client, first["job_id"], manager)
    assert job["status"] == J.SUCCEEDED
    assert pipeline.calls == 1, "the document was ingested once"
    assert len(ledger()) == 1


def test_a_different_method_set_for_the_same_file_is_its_own_job(client, jobs, monkeypatch):
    test_client, kb_id = client
    manager = jobs(queue_capacity=2)
    gate = threading.Event()
    pipeline = use_pipeline(monkeypatch, StubPipeline(gate=gate))
    standard = upload(test_client, kb_id, methods="structure-only", **{"async": "1"}).get_json()
    assert pipeline.started.wait(10)
    deep = upload(test_client, kb_id, deep_analysis="true", **{"async": "1"}).get_json()
    assert deep["attached"] is False and deep["job_id"] != standard["job_id"]
    assert deep["job"]["chunking_mode"] == "deep_analysis"
    gate.set()
    assert manager.drain(20)


# ---------------------------------------------------------------- status


def test_an_unknown_job_is_a_404_that_says_why(client, jobs):
    """404 now means one thing only: older than the retention window. A
    restart no longer produces one, because jobs are journalled."""
    test_client, _ = client
    jobs()
    response = test_client.get("/api/ingest/jobs/nope")
    assert response.status_code == 404
    body = response.get_json()
    assert body["unknown_job"] is True
    assert "kept" in body["error"] and "restart" not in body["error"]


def test_a_job_id_still_answers_after_a_restart(client, jobs, monkeypatch, staging, tmp_path):
    """A client holding a 202 across a restart gets a truthful terminal state,
    not an unexplained 404. The ledger settles which one."""
    from components.ingest import JobJournal

    test_client, kb_id = client
    journal = JobJournal(str(tmp_path / "journal"))
    monkeypatch.setattr(flask_app.paths, "ingest_journal", lambda: str(tmp_path / "journal"))
    gate = threading.Event()
    pipeline = use_pipeline(monkeypatch, StubPipeline(gate=gate))
    first = jobs(journal=journal)

    accepted = upload(test_client, kb_id, **{"async": "1"}).get_json()
    job_id = accepted["job_id"]
    assert pipeline.started.wait(10)

    # The process restarts: a new manager over the same journal directory,
    # exactly what start-up builds.
    second = jobs(journal=journal)
    interrupted = second.recover(resolve_document=flask_app.document_of_ingest_job)
    assert [record["job_id"] for record in interrupted] == [job_id]

    response = test_client.get(f"/api/ingest/jobs/{job_id}")
    assert response.status_code == 200, "not a 404"
    job = response.get_json()["job"]
    assert job["status"] == "interrupted"
    assert job["restart_recovered"] is True
    assert "upload it again" in job["error"]

    gate.set()
    assert first.drain(20)


def test_a_job_that_finished_before_the_restart_reports_success(client, jobs, monkeypatch, tmp_path):
    """The ledger is the authority: a document carrying the job's id means the
    job completed, so the client is told so rather than told to retry."""
    from components.ingest import JobJournal

    test_client, kb_id = client
    journal = JobJournal(str(tmp_path / "journal"))
    monkeypatch.setattr(flask_app.paths, "ingest_journal", lambda: str(tmp_path / "journal"))
    use_pipeline(monkeypatch, StubPipeline())
    first = jobs(journal=journal)

    body = upload(test_client, kb_id).get_json()
    job_id = body["job_id"]
    assert first.get(job_id).status == J.SUCCEEDED
    assert len(ledger()) == 1

    # Pretend the terminal record never landed -- the process died between the
    # ledger write and the journal write, the narrowest window there is.
    journal.record({**first.get(job_id).snapshot(), "status": J.RUNNING,
                    "result": None, "finished_at": None})

    second = jobs(journal=journal)
    settled = second.recover(resolve_document=flask_app.document_of_ingest_job)
    assert len(settled) == 1
    job = test_client.get(f"/api/ingest/jobs/{job_id}").get_json()["job"]
    assert job["status"] == J.SUCCEEDED
    assert job["resolution"] == "recovered_from_ledger"
    assert job["result"]["recovered_from_ledger"] is True
    assert job["doc_id"] == "doc-under-test"


def test_the_job_list_is_filtered_by_knowledge_base_and_carries_capacity(client, jobs, monkeypatch):
    test_client, kb_id = client
    manager = jobs()
    use_pipeline(monkeypatch, StubPipeline())
    body = upload(test_client, kb_id).get_json()
    listed = test_client.get(f"/api/ingest/jobs?kb_id={kb_id}").get_json()
    assert [job["job_id"] for job in listed["jobs"]] == [body["job_id"]]
    assert listed["capacity"]["limits"]["workers"] == 1
    assert listed["capacity"]["provider"]["limit"] >= 1
    assert test_client.get("/api/ingest/jobs?kb_id=other").get_json()["jobs"] == []
    assert test_client.get(f"/api/ingest/jobs?kb_id={kb_id}&active=1").get_json()["jobs"] == []
    assert manager.get(body["job_id"]).status == J.SUCCEEDED


def test_cancelling_a_queued_job_over_http(client, jobs, monkeypatch, staging):
    test_client, kb_id = client
    manager = jobs(queue_capacity=1)
    gate = threading.Event()
    pipeline = use_pipeline(monkeypatch, StubPipeline(gate=gate))
    upload(test_client, kb_id, content=b"one", **{"async": "1"})
    assert pipeline.started.wait(10)
    waiting = upload(test_client, kb_id, content=b"two", **{"async": "1"}).get_json()
    response = test_client.delete(f"/api/ingest/jobs/{waiting['job_id']}")
    assert response.status_code == 200
    assert response.get_json()["job"]["status"] == J.CANCELLED
    assert len(staged_files(staging)) == 1
    assert test_client.delete("/api/ingest/jobs/nope").status_code == 404
    gate.set()
    assert manager.drain(20)
    assert len(ledger()) == 1, "only the job that ran was recorded"


# ------------------------------------------------------- early exits


def test_validation_failures_never_create_a_job(client, jobs, monkeypatch, staging):
    test_client, kb_id = client
    manager = jobs()
    assert upload(test_client, "").status_code == 400
    assert upload(test_client, "no-such-kb").status_code == 404
    use_pipeline(monkeypatch, StubPipeline())
    monkeypatch.setattr(flask_app, "get_pipeline", lambda *a, **k: SimpleNamespace(
        chunker=SimpleNamespace(get_name=lambda: "SemanticChunker")))
    refused = upload(test_client, kb_id, deep_analysis="true")
    assert refused.status_code == 400 and refused.get_json()["deep_analysis_unavailable"] is True
    assert manager.stats["submitted"] == 0
    assert staged_files(staging) == []
