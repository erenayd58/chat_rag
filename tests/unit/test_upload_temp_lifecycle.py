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

import app as flask_app
from components.knowledgebase.manager import KnowledgeBaseManager


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
    monkeypatch.setattr(flask_app.tempfile, "gettempdir", lambda: str(staging))
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
    monkeypatch.setattr(flask_app, "kb_manager", manager)
    # Staging for the Viewer writes under the working directory; the route
    # already treats a failure there as non-fatal, and nothing here asserts on
    # it. Stubbed so a test never reaches the packaging worker.
    monkeypatch.setattr(flask_app, "stage_viewer_analysis", lambda *a, **k: {"status": "queued"})
    flask_app.app.config.update(TESTING=True)
    kb = manager.create("kb", chunker={"type": "structure_first"})
    with flask_app.app.test_client() as test_client:
        yield test_client, kb["kb_id"]


def upload(test_client, content: bytes = b"kucuk bir test belgesi", **fields):
    data = {"file": (io.BytesIO(content), "belge.txt")}
    data.update(fields)
    return test_client.post(
        "/api/documents/upload", data=data, content_type="multipart/form-data"
    )


# ------------------------------------------------------------------ success


def test_a_successful_upload_leaves_no_temp_file(client, temp_dir, monkeypatch):
    """The path that never cleaned up at all, and the one that runs every time."""
    test_client, kb_id = client
    pipeline = StubPipeline()
    monkeypatch.setattr(flask_app, "get_pipeline", lambda *a, **k: pipeline)

    response = upload(test_client, kb_id=kb_id, deep_analysis="false")

    assert response.status_code == 200
    assert response.get_json()["success"] is True
    assert pipeline.seen_paths, "the route never handed a path to ingestion"
    assert leftovers(temp_dir) == []


def test_two_uploads_do_not_accumulate(client, temp_dir, monkeypatch):
    test_client, kb_id = client
    monkeypatch.setattr(flask_app, "get_pipeline", lambda *a, **k: StubPipeline())

    for _ in range(3):
        assert upload(test_client, kb_id=kb_id).status_code == 200

    assert leftovers(temp_dir) == []


# ------------------------------------------------------------ early returns


def test_a_missing_knowledge_base_leaves_no_temp_file(client, temp_dir):
    test_client, _ = client
    response = upload(test_client)
    assert response.status_code == 400
    assert leftovers(temp_dir) == []


def test_an_unknown_knowledge_base_leaves_no_temp_file(client, temp_dir):
    test_client, _ = client
    response = upload(test_client, kb_id="no-such-kb")
    assert response.status_code == 404
    assert leftovers(temp_dir) == []


def test_a_refused_deep_analysis_leaves_no_temp_file(client, temp_dir, monkeypatch):
    test_client, kb_id = client
    monkeypatch.setattr(
        flask_app, "get_pipeline", lambda *a, **k: StubPipeline(deep_capable=False)
    )

    response = upload(test_client, kb_id=kb_id, deep_analysis="true")

    assert response.status_code == 400
    assert response.get_json()["deep_analysis_unavailable"] is True
    assert leftovers(temp_dir) == []


# ---------------------------------------------------------------- failures


def test_a_failed_ingest_leaves_no_temp_file(client, temp_dir, monkeypatch):
    test_client, kb_id = client
    monkeypatch.setattr(
        flask_app,
        "get_pipeline",
        lambda *a, **k: StubPipeline(ingest_error=RuntimeError("parser exploded")),
    )

    response = upload(test_client, kb_id=kb_id)

    assert response.status_code == 500
    assert leftovers(temp_dir) == []


def test_a_store_that_refuses_the_document_leaves_no_temp_file(client, temp_dir, monkeypatch):
    """The 409 re-index path returns from the outer handler, not the inner one."""
    from core.exceptions import IndexIncompatibleException

    test_client, kb_id = client
    monkeypatch.setattr(
        flask_app,
        "get_pipeline",
        lambda *a, **k: StubPipeline(
            ingest_error=IndexIncompatibleException("another embedding model")
        ),
    )

    response = upload(test_client, kb_id=kb_id)

    assert response.status_code == 409
    assert response.get_json()["reindex_required"] is True
    assert leftovers(temp_dir) == []


# ------------------------------------------------------- the job's own exits


def test_a_refused_upload_leaves_no_temp_file(client, temp_dir, monkeypatch):
    """A full queue refuses the upload up front; the file is gone by then."""
    from config.ingest import IngestLimits
    from components.ingest import IngestManager

    test_client, kb_id = client
    monkeypatch.setattr(flask_app, "get_pipeline", lambda *a, **k: StubPipeline())
    gate = __import__("threading").Event()

    def hold(job):
        gate.wait(10)
        return {"doc_id": "held"}

    manager = IngestManager(IngestLimits(workers=1, queue_capacity=0), execute=hold)
    monkeypatch.setattr(flask_app, "ingest_jobs", manager)
    try:
        first = upload(test_client, kb_id=kb_id, **{"async": "1"})
        assert first.status_code == 202
        # Different bytes: the same file would attach to the running job.
        second = upload(test_client, b"baska bir belge", kb_id=kb_id, **{"async": "1"})
        assert second.status_code == 503
        assert second.get_json()["overloaded"] is True
        # Only the running job's file may exist; the refused one is gone.
        assert len(leftovers(temp_dir)) == 1
    finally:
        gate.set()
        manager.close()
    assert leftovers(temp_dir) == []


def test_an_attached_duplicate_leaves_no_second_temp_file(client, temp_dir, monkeypatch):
    from config.ingest import IngestLimits
    from components.ingest import IngestManager

    test_client, kb_id = client
    monkeypatch.setattr(flask_app, "get_pipeline", lambda *a, **k: StubPipeline())
    gate = __import__("threading").Event()

    def hold(job):
        gate.wait(10)
        return {"doc_id": "held"}

    manager = IngestManager(IngestLimits(workers=1, queue_capacity=2), execute=hold)
    monkeypatch.setattr(flask_app, "ingest_jobs", manager)
    try:
        first = upload(test_client, kb_id=kb_id, **{"async": "1"}).get_json()
        second = upload(test_client, kb_id=kb_id, **{"async": "1"}).get_json()
        assert second["attached"] is True
        assert second["job_id"] == first["job_id"]
        assert len(leftovers(temp_dir)) == 1, "the twin's file was discarded at once"
    finally:
        gate.set()
        manager.close()
    assert leftovers(temp_dir) == []


def test_a_restart_sweeps_what_a_previous_process_left(temp_dir, monkeypatch):
    """A job lives in memory, so a staged file with no process is nobody's."""
    from config import paths

    staging = os.path.join(str(temp_dir), "chat_rag-uploads")
    os.makedirs(staging)
    open(os.path.join(staging, "upload_deadbeef.pdf"), "wb").write(b"%PDF-1.7")
    assert paths.upload_staging() == staging

    flask_app.resume_background_work()

    assert leftovers(temp_dir) == []
