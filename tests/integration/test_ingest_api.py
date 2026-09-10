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

from fastapi.testclient import TestClient

import asgi as entrypoint
import interfaces.http as http

V1 = http.v1.PREFIX
from chat_rag.application import documents as app_documents
from chat_rag.application import ingest as app_ingest
from chat_rag.application import workspace as app_workspace
from chat_rag.config import paths as app_paths
import tempfile
from chat_rag.components.ingest import IngestManager
from chat_rag.components.ingest import jobs as J
from chat_rag.components.ingest.limits import JobGuard, checkpoint
from chat_rag.components.knowledgebase.manager import KnowledgeBaseManager
from chat_rag.components.viewer import methods as M
from chat_rag.config import paths
from chat_rag.config.ingest import IngestLimits


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
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(directory))
    return directory


def staged_files(staging) -> list[str]:
    return sorted(p.name for p in staging.rglob("*") if p.is_file())


@pytest.fixture
def client(tmp_path, monkeypatch, staging):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("FAKE_DEEP_KEY", "sk-placeholder")
    manager = KnowledgeBaseManager(str(tmp_path / "kbs.json"))
    monkeypatch.setattr(entrypoint.services, "kb_manager", manager)
    monkeypatch.setattr(app_workspace, "stage_analysis", lambda *a, **k: {"status": "queued"})
    kb = manager.create("jobs-kb", chunker={"type": "structure_first"})
    with TestClient(http.create_app(entrypoint.services),
                    raise_server_exceptions=False) as test_client:
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
            execute=lambda job: app_ingest.execute_job(entrypoint.services, job),
            journal=journal,
        )
        monkeypatch.setattr(entrypoint.services, "ingest_jobs", manager)
        managers.append(manager)
        return manager

    yield make
    for manager in managers:
        manager.close(timeout=20)


def use_pipeline(monkeypatch, pipeline):
    monkeypatch.setattr(entrypoint.services, "get_pipeline", lambda *a, **k: pipeline)
    return pipeline


def upload(test_client, kb_id, *, content=b"kucuk bir test belgesi", name="belge.txt",
           methods=None):
    """One upload. Always 202 with the job: this contract has no synchronous
    mode, so there is no second answer shape to test."""
    data = {"knowledge_base_id": kb_id}
    if methods is not None:
        data["methods"] = methods
    return test_client.post(
        f"{V1}/documents",
        files={"file": (name, io.BytesIO(content), "text/plain")},
        data=data,
    )


def poll(test_client, job_id, manager):
    """Wait for the job through the manager, then read it back over HTTP --
    the same body a client polls for, without a polling loop in the test."""
    job = manager.get(job_id)
    assert job is not None
    assert manager.wait(job, 20)
    response = test_client.get(f"{V1}/ingest-jobs/{job_id}")
    assert response.status_code == 200, response.text
    return response.json()


def ledger():
    """The ingest ledger, keyed by upload, read through the store that owns it.

    The records are rows since Step 8; this used to open the JSON file
    ``paths.ingested_documents()`` named. Every assertion below is about what
    the ledger holds, not about where it holds it.
    """
    from chat_rag.utils import DocumentTracker

    return DocumentTracker().ingested_docs


# ---------------------------------------------------------------- async


def test_an_upload_is_accepted_at_once_and_finishes_later(client, jobs, monkeypatch, staging):
    test_client, kb_id = client
    manager = jobs()
    gate = threading.Event()
    pipeline = use_pipeline(monkeypatch, StubPipeline(gate=gate))

    response = upload(test_client, kb_id)
    assert response.status_code == 202
    body = response.json()
    assert body["status"] in (J.QUEUED, J.RUNNING)
    assert body["attached_uploads"] == 0
    assert body["name"] == "belge.txt"
    assert body["knowledge_base_id"] == kb_id
    assert "temp_path" not in json.dumps(body)
    # The job is where the client polls, and this says where.
    assert response.headers["Location"] == f"{V1}/ingest-jobs/{body['id']}"

    assert pipeline.started.wait(10)
    running = test_client.get(f"{V1}/ingest-jobs/{body['id']}").json()
    assert running["status"] == J.RUNNING
    assert ledger() == {}, "nothing is registered until the job finishes"
    assert len(staged_files(staging)) == 1, "the file lives while the job runs"

    gate.set()
    job = poll(test_client, body["id"], manager)
    assert job["status"] == J.SUCCEEDED
    assert job["document_id"] == "doc-under-test"
    assert job["result"]["document_id"] == "doc-under-test"
    assert job["result"]["chunk_count"] == 1
    assert job["result"]["chunking_mode"] == "standard"
    (record,) = ledger().values()
    assert record["doc_id"] == "doc-under-test"
    assert record["metadata"]["ingest_job_id"] == body["id"]
    assert staged_files(staging) == []


def test_a_slow_upload_is_the_same_answer_as_a_fast_one(client, jobs, monkeypatch):
    """202 and the job, whether it settles in a millisecond or a minute.

    The Flask-era surface had a second, synchronous mode -- it held the
    request thread until the job settled and answered 200 with the document --
    and it was the reason that path needed a semaphore. This contract never
    offered it, and Step 13 removed the surface that did.
    """
    test_client, kb_id = client
    manager = jobs()
    gate = threading.Event()
    use_pipeline(monkeypatch, StubPipeline(gate=gate))

    response = upload(test_client, kb_id)
    assert response.status_code == 202
    body = response.json()
    assert body["result"] is None, "nothing has been produced yet"
    gate.set()
    assert poll(test_client, body["id"], manager)["status"] == J.SUCCEEDED


# ------------------------------------------------------------- failures


def test_a_failed_job_is_truthful_and_registers_nothing(client, jobs, monkeypatch, staging):
    test_client, kb_id = client
    manager = jobs()
    use_pipeline(monkeypatch, StubPipeline(error=RuntimeError("parser exploded")))
    body = upload(test_client, kb_id).json()
    job = poll(test_client, body["id"], manager)
    assert job["status"] == J.FAILED
    assert "parser exploded" in job["error"]["message"]
    assert job["result"] is None
    assert ledger() == {}
    assert staged_files(staging) == []


def test_a_failure_keeps_its_category_on_the_job(client, jobs, monkeypatch):
    """The three an operator acts on differently. They used to be three status
    codes on the synchronous upload; the submission is always 202 now, so the
    distinction lives where the outcome does -- on the job."""
    from chat_rag.core.exceptions import ConfigurationException, IndexIncompatibleException

    test_client, kb_id = client
    manager = jobs()
    for error, category, content in (
        (IndexIncompatibleException("other model"), "index_incompatible", b"one"),
        (ConfigurationException("no deep path"), "configuration", b"two"),
        (RuntimeError("boom"), "unknown", b"three"),
    ):
        use_pipeline(monkeypatch, StubPipeline(error=error))
        submitted = upload(test_client, kb_id, content=content)
        assert submitted.status_code == 202, submitted.text
        job = poll(test_client, submitted.json()["id"], manager)
        assert job["status"] == J.FAILED, job
        assert job["error"]["type"] == category, job["error"]
        assert job["error"]["message"]


def test_a_job_past_its_deadline_ends_timed_out_and_commits_nothing(client, jobs, monkeypatch, staging):
    test_client, kb_id = client
    manager = jobs()
    monkeypatch.setattr(J.JobGuard, "for_timeout", classmethod(lambda cls, s: JobGuard(deadline=-1.0)))
    use_pipeline(monkeypatch, StubPipeline(checkpoints=True))
    response = upload(test_client, kb_id)
    assert response.status_code == 202
    job = poll(test_client, response.json()["id"], manager)
    assert job["status"] == J.TIMED_OUT
    assert job["error"]["type"] == "timed_out"
    assert ledger() == {}
    assert staged_files(staging) == []


def test_a_ledger_that_cannot_be_written_takes_the_store_rows_back(client, jobs, monkeypatch):
    from chat_rag.utils import document_tracker

    test_client, kb_id = client
    manager = jobs()
    pipeline = use_pipeline(monkeypatch, StubPipeline(chunks=3))
    monkeypatch.setattr(document_tracker.DocumentTracker, "mark_as_ingested", lambda self, **k: False)
    body = upload(test_client, kb_id).json()
    job = poll(test_client, body["id"], manager)
    assert job["status"] == J.FAILED
    assert "ledger" in job["error"]["message"]
    assert pipeline.vector_db.deleted == ["doc-under-test"], "the rows were rolled back"


# ------------------------------------------------------------- overload


def test_a_full_queue_is_refused_with_a_retry_after(client, jobs, monkeypatch, staging):
    test_client, kb_id = client
    manager = jobs(workers=1, queue_capacity=1)
    gate = threading.Event()
    pipeline = use_pipeline(monkeypatch, StubPipeline(gate=gate))

    first = upload(test_client, kb_id, content=b"one")
    assert first.status_code == 202
    assert pipeline.started.wait(10)
    second = upload(test_client, kb_id, content=b"two")
    assert second.status_code == 202
    assert second.json()["status"] == J.QUEUED
    assert second.json()["queue_position"] == 1

    third = upload(test_client, kb_id, content=b"three")
    assert third.status_code == 503
    error = third.json()["error"]
    assert error["type"] == "overloaded"
    assert error["details"]["retry_after_seconds"] >= 5
    assert third.headers["Retry-After"] == str(int(error["details"]["retry_after_seconds"]))
    assert manager.snapshot()["queued"] == 1
    assert len(staged_files(staging)) == 2, "the refused file is already gone"

    health = test_client.get(f"{V1}/health").json()
    assert health["capacity"]["ingest"]["running"] == 1
    assert health["capacity"]["ingest"]["queued"] == 1
    assert health["state"] == "overloaded", "the queue is full and uploads are refused"
    # Counters live at the operational endpoint; health stays small enough for
    # a probe to poll every few seconds.
    metrics = test_client.get("/api/ops/metrics").json()
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

    first = upload(test_client, kb_id).json()
    assert pipeline.started.wait(10)
    second = upload(test_client, kb_id).json()
    assert second["id"] == first["id"], "the same bytes joined the job in flight"
    assert second["attached_uploads"] == 1
    assert len(staged_files(staging)) == 1

    gate.set()
    job = poll(test_client, first["id"], manager)
    assert job["status"] == J.SUCCEEDED
    assert pipeline.calls == 1, "the document was ingested once"
    assert len(ledger()) == 1


def test_a_different_method_set_for_the_same_file_is_its_own_job(client, jobs, monkeypatch):
    test_client, kb_id = client
    manager = jobs(queue_capacity=2)
    gate = threading.Event()
    pipeline = use_pipeline(monkeypatch, StubPipeline(gate=gate))
    standard = upload(test_client, kb_id, methods=["structure-only"]).json()
    assert pipeline.started.wait(10)
    deep = upload(test_client, kb_id, methods=[M.DEEP]).json()
    assert deep["attached_uploads"] == 0 and deep["id"] != standard["id"]
    assert deep["methods"] == [M.DEEP], "the methods asked for, not a mode"
    gate.set()
    assert manager.drain(20)


# ---------------------------------------------------------------- status


def test_an_unknown_job_is_a_404_that_says_why(client, jobs):
    """404 now means one thing only: older than the retention window. A
    restart no longer produces one, because jobs are journalled."""
    test_client, _ = client
    jobs()
    response = test_client.get(f"{V1}/ingest-jobs/nope")
    assert response.status_code == 404
    error = response.json()["error"]
    assert error["type"] == "not_found"
    assert error["details"]["unknown_job"] is True
    assert "kept" in error["message"] and "restart" not in error["message"]


def test_a_job_id_still_answers_after_a_restart(client, jobs, monkeypatch, staging, tmp_path):
    """A client holding a 202 across a restart gets a truthful terminal state,
    not an unexplained 404. The ledger settles which one."""
    from chat_rag.components.ingest import JobJournal

    test_client, kb_id = client
    journal = JobJournal(str(tmp_path / "journal"))
    monkeypatch.setattr(app_paths, "ingest_journal", lambda: str(tmp_path / "journal"))
    gate = threading.Event()
    pipeline = use_pipeline(monkeypatch, StubPipeline(gate=gate))
    first = jobs(journal=journal)

    accepted = upload(test_client, kb_id).json()
    job_id = accepted["id"]
    assert pipeline.started.wait(10)

    # The process restarts: a new manager over the same journal directory,
    # exactly what start-up builds.
    second = jobs(journal=journal)
    interrupted = second.recover(resolve_document=lambda job_id: app_documents.of_ingest_job(entrypoint.services, job_id))
    assert [record["job_id"] for record in interrupted] == [job_id]

    response = test_client.get(f"{V1}/ingest-jobs/{job_id}")
    assert response.status_code == 200, "not a 404"
    job = response.json()
    assert job["status"] == "interrupted"
    assert job["restart_settled"] is True
    assert "upload it again" in job["error"]["message"]

    gate.set()
    assert first.drain(20)


def test_a_job_that_finished_before_the_restart_reports_success(client, jobs, monkeypatch, tmp_path):
    """The ledger is the authority: a document carrying the job's id means the
    job completed, so the client is told so rather than told to retry."""
    from chat_rag.components.ingest import JobJournal

    test_client, kb_id = client
    journal = JobJournal(str(tmp_path / "journal"))
    monkeypatch.setattr(app_paths, "ingest_journal", lambda: str(tmp_path / "journal"))
    use_pipeline(monkeypatch, StubPipeline())
    first = jobs(journal=journal)

    body = upload(test_client, kb_id).json()
    job_id = body["id"]
    assert first.wait(first.get(job_id), 20)
    assert first.get(job_id).status == J.SUCCEEDED
    assert len(ledger()) == 1

    # Pretend the terminal record never landed -- the process died between the
    # ledger write and the journal write, the narrowest window there is.
    journal.record({**first.get(job_id).snapshot(), "status": J.RUNNING,
                    "result": None, "finished_at": None})

    second = jobs(journal=journal)
    settled = second.recover(resolve_document=lambda job_id: app_documents.of_ingest_job(entrypoint.services, job_id))
    assert len(settled) == 1
    job = test_client.get(f"{V1}/ingest-jobs/{job_id}").json()
    assert job["status"] == J.SUCCEEDED
    assert job["restart_settled"] is True, "settled against the ledger, not re-run"
    assert job["document_id"] == "doc-under-test"


def test_the_job_list_is_filtered_by_knowledge_base_and_carries_capacity(client, jobs, monkeypatch):
    test_client, kb_id = client
    manager = jobs()
    use_pipeline(monkeypatch, StubPipeline())
    body = upload(test_client, kb_id).json()
    assert manager.wait(manager.get(body["id"]), 20)
    listed = test_client.get(f"{V1}/ingest-jobs?knowledge_base_id={kb_id}").json()
    assert [job["id"] for job in listed["items"]] == [body["id"]]
    assert listed["capacity"]["workers"] == 1
    assert listed["capacity"]["queue_capacity"] >= 1
    assert test_client.get(f"{V1}/ingest-jobs?knowledge_base_id=other").json()["items"] == []
    assert test_client.get(
        f"{V1}/ingest-jobs?knowledge_base_id={kb_id}&active=1").json()["items"] == []
    assert manager.get(body["id"]).status == J.SUCCEEDED


def test_cancelling_a_queued_job_over_http(client, jobs, monkeypatch, staging):
    test_client, kb_id = client
    manager = jobs(queue_capacity=1)
    gate = threading.Event()
    pipeline = use_pipeline(monkeypatch, StubPipeline(gate=gate))
    upload(test_client, kb_id, content=b"one")
    assert pipeline.started.wait(10)
    waiting = upload(test_client, kb_id, content=b"two").json()
    response = test_client.delete(f"{V1}/ingest-jobs/{waiting['id']}")
    assert response.status_code == 200
    assert response.json()["status"] == J.CANCELLED
    assert len(staged_files(staging)) == 1
    assert test_client.delete(f"{V1}/ingest-jobs/nope").status_code == 404
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
    monkeypatch.setattr(entrypoint.services, "get_pipeline", lambda *a, **k: SimpleNamespace(
        chunker=SimpleNamespace(get_name=lambda: "SemanticChunker")))
    refused = upload(test_client, kb_id, methods=[M.DEEP])
    assert refused.status_code == 400
    assert refused.json()["error"]["details"]["deep_analysis_unavailable"] is True
    assert manager.stats["submitted"] == 0
    assert staged_files(staging) == []
