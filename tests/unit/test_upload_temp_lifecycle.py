"""An uploaded file exists for one request and not a moment longer.

Ingestion takes a path, so an upload is written to the system temp directory
before anything can parse it. Deleting it was the caller's job, and the caller
only did it from one place: the error handler. Every other exit -- a missing
knowledge base, an unknown one, a chunker with no Deep Analysis path, and above
all the successful one -- returned straight past it, so the temp directory
accumulated one copy of every document ever uploaded, indefinitely, including
whatever confidential material the documents happened to contain.

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
    return sorted(p.name for p in staging.iterdir())


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


def upload(test_client, **fields):
    data = {"file": (io.BytesIO(b"kucuk bir test belgesi"), "belge.txt")}
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


# ------------------------------------------------------- the mechanism itself


def test_the_staging_helper_deletes_on_an_exception(temp_dir):
    """``staged_upload`` is a context manager so that no caller has to remember."""
    seen = {}

    class Uploaded:
        filename = "rapor.pdf"

        def save(self, path):
            open(path, "wb").write(b"%PDF-1.7")

    with pytest.raises(ValueError):
        with flask_app.staged_upload(Uploaded()) as path:
            seen["path"] = path
            assert os.path.isfile(path)
            assert path.endswith(".pdf"), "the extension is what tells the parser what this is"
            raise ValueError("something in the middle of the request")

    assert not os.path.exists(seen["path"])
    assert leftovers(temp_dir) == []


def test_the_staging_helper_survives_a_file_deleted_under_it(temp_dir):
    """Cleanup is best effort: an already-gone file is not an error."""

    class Uploaded:
        filename = "notlar.txt"

        def save(self, path):
            open(path, "wb").write(b"x")

    with flask_app.staged_upload(Uploaded()) as path:
        os.remove(path)

    assert leftovers(temp_dir) == []
