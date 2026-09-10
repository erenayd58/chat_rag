"""An uploaded file exists for exactly one ingest job and not a moment longer.

Ingestion takes a path, so an upload is written to a staging directory before
anything can parse it. It used to live for one request; now the job that
reads it outlives the request, so the file belongs to the job from submission
to its terminal state -- success, failure, refusal at a full queue, attachment
to a twin already in flight -- and is deleted in every one of them. Before a
job exists, the route owns it, and every early exit deletes it itself.

These tests walk each of those exits and assert the same thing about all of
them: no ``upload_*`` file is left behind. They watch a temp directory of their
own rather than the machine's, so a real upload from a running console cannot
make them pass or fail.
"""

from __future__ import annotations

import io
import os
from types import SimpleNamespace

import pytest

from fastapi.testclient import TestClient

import asgi as entrypoint
import interfaces.http as http
from chat_rag.components.viewer import methods as M
from runtime import bootstrap

V1 = http.v1.PREFIX
from chat_rag.application import workspace as app_workspace
import tempfile
from chat_rag.components.knowledgebase.manager import KnowledgeBaseManager


class Chunker:
    def __init__(self, deep_capable: bool = True):
        self.last_canonical_units = None
        self.last_deep_result = None
        self._deep_capable = deep_capable
        if deep_capable:
            self.chunk_text_deep = lambda *a, **k: None  # noqa: E731 - presence is the contract

    def get_name(self):
        return "StructuralChunker" if self._deep_capable else "SemanticChunker"


class StubPipeline:
    """Only what the upload route touches."""

    def __init__(self, *, deep_capable: bool = True, ingest_error: Exception | None = None):
        self.settings = SimpleNamespace(deep_analysis_api_key_env="FAKE_DEEP_KEY")
        self.chunker = Chunker(deep_capable)
        self.last_deep_analysis_report = None
        self.last_parse_seconds = 0.1
        self._ingest_error = ingest_error
        self.seen_paths: list[str] = []

    def ingest_document_from_file(self, file_path, doc_title, deep_analysis=False):
        # The file has to be readable here: this is where the parser opens it.
        assert os.path.isfile(file_path), "the upload was deleted before ingestion could read it"
        self.seen_paths.append(file_path)
        if self._ingest_error is not None:
            raise self._ingest_error
        return [SimpleNamespace(doc_id="doc-under-test")]


@pytest.fixture
def temp_dir(tmp_path, monkeypatch):
    """A private staging directory, so only this test's uploads are counted."""
    staging = tmp_path / "staging"
    staging.mkdir()
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(staging))
    return staging


def leftovers(staging) -> list[str]:
    """Every *file* under the staging directory, however deep.

    The staging directory itself (``chat_rag-uploads``) is allowed to exist:
    it is the job manager's, and it is what a restart sweeps.
    """
    return sorted(p.name for p in staging.rglob("*") if p.is_file())


@pytest.fixture
def client(tmp_path, monkeypatch, temp_dir):
    monkeypatch.chdir(tmp_path)
    manager = KnowledgeBaseManager(str(tmp_path / "kbs.json"))
    monkeypatch.setattr(entrypoint.services, "kb_manager", manager)
    # Staging for the Viewer writes under the working directory; the route
    # already treats a failure there as non-fatal, and nothing here asserts on
    # it. Stubbed so a test never reaches the packaging worker.
    monkeypatch.setattr(app_workspace, "stage_analysis", lambda *a, **k: {"status": "queued"})
    kb = manager.create("kb", chunker={"type": "structure_first"})
    with TestClient(http.create_app(entrypoint.services),
                    raise_server_exceptions=False) as test_client:
        yield test_client, kb["kb_id"]


def upload(test_client, content: bytes = b"kucuk bir test belgesi", **fields):
    return test_client.post(
        f"{V1}/documents",
        files={"file": ("belge.txt", io.BytesIO(content), "text/plain")},
        data={name: value for name, value in fields.items()},
    )


def settled(test_client, response):
    """Wait for an accepted upload's job, and hand back its record.

    Every upload is asynchronous on this contract, so "what happened to the
    file" is a question about the job rather than about the response -- and
    the staging directory is only quiet once the job has finished with it.
    """
    assert response.status_code == 202, response.text
    manager = entrypoint.services.ingest_jobs
    job = manager.get(response.json()["id"])
    assert job is not None
    assert manager.wait(job, 30), "the job did not settle"
    return job.snapshot()


# ------------------------------------------------------------------ success


def test_a_successful_upload_leaves_no_temp_file(client, temp_dir, monkeypatch):
    """The path that never cleaned up at all, and the one that runs every time."""
    test_client, kb_id = client
    pipeline = StubPipeline()
    monkeypatch.setattr(entrypoint.services, "get_pipeline", lambda *a, **k: pipeline)

    job = settled(test_client, upload(test_client, knowledge_base_id=kb_id))

    assert job["status"] == "succeeded", job
    assert pipeline.seen_paths, "the job never handed a path to ingestion"
    assert leftovers(temp_dir) == []


def test_two_uploads_do_not_accumulate(client, temp_dir, monkeypatch):
    test_client, kb_id = client
    monkeypatch.setattr(entrypoint.services, "get_pipeline", lambda *a, **k: StubPipeline())

    for index in range(3):
        # Different bytes each time: the same file would attach to the job
        # already in flight rather than stage a second copy.
        job = settled(test_client, upload(test_client, f"belge {index}".encode(),
                                          knowledge_base_id=kb_id))
        assert job["status"] == "succeeded", job

    assert leftovers(temp_dir) == []


# ------------------------------------------------------------ early returns


def test_a_missing_knowledge_base_leaves_no_temp_file(client, temp_dir):
    test_client, _ = client
    response = upload(test_client)
    assert response.status_code == 400
    assert leftovers(temp_dir) == []


def test_an_unknown_knowledge_base_leaves_no_temp_file(client, temp_dir):
    test_client, _ = client
    response = upload(test_client, knowledge_base_id="no-such-kb")
    assert response.status_code == 404
    assert leftovers(temp_dir) == []


def test_a_refused_deep_analysis_leaves_no_temp_file(client, temp_dir, monkeypatch):
    test_client, kb_id = client
    monkeypatch.setattr(
        entrypoint.services, "get_pipeline", lambda *a, **k: StubPipeline(deep_capable=False)
    )

    response = upload(test_client, knowledge_base_id=kb_id, methods=M.DEEP)

    assert response.status_code == 400
    assert response.json()["error"]["details"]["deep_analysis_unavailable"] is True
    assert leftovers(temp_dir) == []


# ---------------------------------------------------------------- failures


def test_a_failed_ingest_leaves_no_temp_file(client, temp_dir, monkeypatch):
    test_client, kb_id = client
    monkeypatch.setattr(
        entrypoint.services,
        "get_pipeline",
        lambda *a, **k: StubPipeline(ingest_error=RuntimeError("parser exploded")),
    )

    job = settled(test_client, upload(test_client, knowledge_base_id=kb_id))

    assert job["status"] == "failed"
    assert leftovers(temp_dir) == []


def test_a_store_that_refuses_the_document_leaves_no_temp_file(client, temp_dir, monkeypatch):
    """An incompatible index ends the job, and the file goes with it."""
    from chat_rag.core.exceptions import IndexIncompatibleException

    test_client, kb_id = client
    monkeypatch.setattr(
        entrypoint.services,
        "get_pipeline",
        lambda *a, **k: StubPipeline(
            ingest_error=IndexIncompatibleException("another embedding model")
        ),
    )

    job = settled(test_client, upload(test_client, knowledge_base_id=kb_id))

    assert job["status"] == "failed"
    assert job["error_category"] == "index_incompatible"
    assert leftovers(temp_dir) == []


# ------------------------------------------------------- the job's own exits


def test_a_refused_upload_leaves_no_temp_file(client, temp_dir, monkeypatch):
    """A full queue refuses the upload up front; the file is gone by then."""
    from chat_rag.config.ingest import IngestLimits
    from chat_rag.components.ingest import IngestManager

    test_client, kb_id = client
    monkeypatch.setattr(entrypoint.services, "get_pipeline", lambda *a, **k: StubPipeline())
    gate = __import__("threading").Event()

    def hold(job):
        gate.wait(10)
        return {"doc_id": "held"}

    manager = IngestManager(IngestLimits(workers=1, queue_capacity=0), execute=hold)
    monkeypatch.setattr(entrypoint.services, "ingest_jobs", manager)
    try:
        first = upload(test_client, knowledge_base_id=kb_id)
        assert first.status_code == 202
        # Different bytes: the same file would attach to the running job.
        second = upload(test_client, b"baska bir belge", knowledge_base_id=kb_id)
        assert second.status_code == 503
        assert second.json()["error"]["type"] == "overloaded"
        # Only the running job's file may exist; the refused one is gone.
        assert len(leftovers(temp_dir)) == 1
    finally:
        gate.set()
        manager.close()
    assert leftovers(temp_dir) == []


def test_an_attached_duplicate_leaves_no_second_temp_file(client, temp_dir, monkeypatch):
    from chat_rag.config.ingest import IngestLimits
    from chat_rag.components.ingest import IngestManager

    test_client, kb_id = client
    monkeypatch.setattr(entrypoint.services, "get_pipeline", lambda *a, **k: StubPipeline())
    gate = __import__("threading").Event()

    def hold(job):
        gate.wait(10)
        return {"doc_id": "held"}

    manager = IngestManager(IngestLimits(workers=1, queue_capacity=2), execute=hold)
    monkeypatch.setattr(entrypoint.services, "ingest_jobs", manager)
    try:
        first = upload(test_client, knowledge_base_id=kb_id).json()
        second = upload(test_client, knowledge_base_id=kb_id).json()
        assert second["attached_uploads"] == 1
        assert second["id"] == first["id"]
        assert len(leftovers(temp_dir)) == 1, "the twin's file was discarded at once"
    finally:
        gate.set()
        manager.close()
    assert leftovers(temp_dir) == []


def test_a_restart_sweeps_what_a_previous_process_left(temp_dir, monkeypatch):
    """A job lives in memory, so a staged file with no process is nobody's."""
    from chat_rag.config import paths

    staging = os.path.join(str(temp_dir), "chat_rag-uploads")
    os.makedirs(staging)
    open(os.path.join(staging, "upload_deadbeef.pdf"), "wb").write(b"%PDF-1.7")
    assert paths.upload_staging() == staging

    bootstrap.resume_background_work(entrypoint.services)

    assert leftovers(temp_dir) == []
