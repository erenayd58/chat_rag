"""The workspace snapshot the Viewer reads to stay in step with the console.

The console owns knowledge bases and documents; the viewer owns the chunking
analysis. Rather than keeping a second copy of the console's state over there,
the viewer asks for this snapshot -- so a knowledge base created here shows up
there without anyone copying anything by hand.
"""

from __future__ import annotations

import pytest

import app as flask_app
from application import workspace as app_workspace


@pytest.fixture
def client():
    flask_app.app.config.update(TESTING=True)
    with flask_app.app.test_client() as test_client:
        yield test_client


class _Tracker:
    """A DocumentTracker double holding exactly the rows a test names."""

    rows: list = []

    def get_all_documents(self, kb_id=None):
        return [row for row in self.rows if kb_id is None or row.get("kb_id") == kb_id]


def _install(monkeypatch, *, knowledge_bases, documents, viewer_states=None, tmp_path=None):
    _Tracker.rows = documents
    monkeypatch.setattr(flask_app.services, "documents", _Tracker)
    monkeypatch.setattr(flask_app.services.kb_manager, "list", lambda: knowledge_bases)
    from components.viewer import analysis
    # An analysis root of the test's own, so a snapshot never reports whatever
    # this checkout happens to have packaged.
    if tmp_path is not None:
        monkeypatch.setattr(analysis, "root", lambda: tmp_path / "viewer-live")
    monkeypatch.setattr(analysis, "states", lambda: dict(viewer_states or {}))


def _doc(**overrides):
    row = {
        "doc_id": "upload_1_pdf",
        "file_name": "upload_1.pdf",
        "chunk_count": 12,
        "file_size": 4096,
        "ingested_at": "2026-08-29T10:00:00",
        "file_hash": "a" * 64,
        "kb_id": "kb1",
        "status": "indexed",
        "chunking_mode": "deep",
        "metadata": {"original_filename": "kkbfaaliyetraporu2024.pdf"},
    }
    row.update(overrides)
    return row


def test_a_knowledge_base_and_its_documents_reach_the_viewer(client, monkeypatch):
    _install(
        monkeypatch,
        knowledge_bases=[{
            "kb_id": "kb1", "name": "kkb-final-v2",
            "chunker": {"type": "structure_first"}, "retrieval_method": "bm25",
            "vector_db_provider": "chroma", "embedding_model_name": "paraphrase-multilingual-MiniLM-L12-v2",
        }],
        documents=[_doc(), _doc(doc_id="upload_2_pdf", chunk_count=8, metadata={"original_filename": "arcelik-2024.pdf"})],
    )
    body = client.get("/api/demo/workspace").get_json()

    assert body["success"] is True
    assert body["totals"] == {"knowledge_bases": 1, "documents": 2, "chunks": 20, "viewer_ready": 0}
    kb = body["knowledge_bases"][0]
    assert kb["name"] == "kkb-final-v2" and kb["chunker"] == "structure_first"
    assert kb["document_count"] == 2 and kb["chunk_count"] == 20
    # The name a person recognises is the one they uploaded, not the temp path
    # the upload landed on.
    assert [d["name"] for d in kb["documents"]] == ["kkbfaaliyetraporu2024.pdf", "arcelik-2024.pdf"]


def test_an_empty_knowledge_base_is_reported_as_empty_not_omitted(client, monkeypatch):
    _install(
        monkeypatch,
        knowledge_bases=[{"kb_id": "kb1", "name": "yeni-kb", "chunker": {"type": "structure_first"}}],
        documents=[],
    )
    kb = client.get("/api/demo/workspace").get_json()["knowledge_bases"][0]
    assert kb["name"] == "yeni-kb" and kb["documents"] == [] and kb["document_count"] == 0


def test_documents_whose_knowledge_base_is_gone_are_grouped_not_dropped(client, monkeypatch):
    """Hiding them would make the panel disagree with the console it mirrors;
    one card per vanished id would bury the live bases under a wall of hex."""
    _install(
        monkeypatch,
        knowledge_bases=[{"kb_id": "kb1", "name": "canli", "chunker": {"type": "structure_first"}}],
        documents=[_doc(kb_id="deleted-a"), _doc(doc_id="d2", kb_id="deleted-b", chunk_count=5)],
    )
    kbs = client.get("/api/demo/workspace").get_json()["knowledge_bases"]
    assert [kb["name"] for kb in kbs] == ["canli", "Bilgi tabani silinmis kayitlar"]
    orphan = kbs[-1]
    assert orphan["orphan"] is True and orphan["document_count"] == 2 and orphan["chunk_count"] == 17
    # the former id survives on the document, so nothing is actually lost
    assert sorted(d["kb_id"] for d in orphan["documents"]) == ["deleted-a", "deleted-b"]


def test_no_absolute_path_and_no_full_hash_leave_the_console(client, monkeypatch):
    _install(
        monkeypatch,
        knowledge_bases=[{"kb_id": "kb1", "name": "kb", "chunker": {"type": "structure_first"}}],
        documents=[_doc(file_name="upload_1.pdf")],
    )
    payload = client.get("/api/demo/workspace").get_data(as_text=True)
    assert "AppData" not in payload and "file_path" not in payload
    doc = client.get("/api/demo/workspace").get_json()["knowledge_bases"][0]["documents"][0]
    assert doc["file_hash"] == "a" * 16, "the short hash identifies, the full one is not needed here"


def test_a_tracker_failure_is_an_error_response_not_a_traceback(client, monkeypatch):
    class _Broken:
        def get_all_documents(self, kb_id=None):
            raise RuntimeError("tracker unavailable")

    monkeypatch.setattr(flask_app.services, "documents", _Broken)
    response = client.get("/api/demo/workspace")
    assert response.status_code == 500
    assert response.get_json()["success"] is False


# --- the Viewer's own analysis of these documents --------------------------


def test_each_document_reports_where_its_viewer_analysis_got_to(client, monkeypatch, tmp_path):
    """The two screens read one record, so they cannot disagree about whether
    a document can be opened in the Viewer."""
    from components.viewer import analysis

    _install(
        monkeypatch,
        knowledge_bases=[{"kb_id": "kb1", "name": "kb", "chunker": {"type": "structure_first"}}],
        documents=[_doc(), _doc(doc_id="upload_2_pdf", metadata={"original_filename": "b.pdf"})],
        viewer_states={
            "upload_1_pdf": {"status": analysis.STATUS_READY, "deep_source": analysis.SOURCE_INGEST,
                             "unit_count": 120, "chunk_count": {"deep": 9, "standard": 11}},
            "upload_2_pdf": {"status": analysis.STATUS_RUNNING},
        },
        tmp_path=tmp_path,
    )
    body = client.get("/api/demo/workspace").get_json()
    docs = {d["doc_id"]: d for d in body["knowledge_bases"][0]["documents"]}

    assert docs["upload_1_pdf"]["viewer"]["status"] == analysis.STATUS_READY
    assert docs["upload_1_pdf"]["viewer"]["deep_source"] == analysis.SOURCE_INGEST
    assert docs["upload_2_pdf"]["viewer"]["status"] == analysis.STATUS_RUNNING
    assert body["totals"]["viewer_ready"] == 1


def test_a_document_with_no_analysis_says_so(client, monkeypatch, tmp_path):
    _install(
        monkeypatch,
        knowledge_bases=[{"kb_id": "kb1", "name": "kb", "chunker": {"type": "structure_first"}}],
        documents=[_doc()],
        tmp_path=tmp_path,
    )
    doc = client.get("/api/demo/workspace").get_json()["knowledge_bases"][0]["documents"][0]
    from components.viewer import analysis
    assert doc["viewer"]["status"] == analysis.STATUS_MISSING


def test_the_refresh_can_queue_the_missing_analyses(client, monkeypatch, tmp_path):
    """?prepare=1 queues; it never packages inline, because the page's refresh
    must return at status speed however much work is outstanding."""
    queued = []
    monkeypatch.setattr(app_workspace, "prepare_missing",
                        lambda services: queued.append("called") or [])
    _install(monkeypatch, knowledge_bases=[], documents=[], tmp_path=tmp_path)

    client.get("/api/demo/workspace")
    assert queued == [], "a plain refresh queues nothing"
    client.get("/api/demo/workspace?prepare=1")
    assert queued == ["called"]


def test_the_payload_endpoint_is_a_clear_404_before_the_build_finishes(client, monkeypatch, tmp_path):
    from components.viewer import analysis

    monkeypatch.setattr(analysis, "root", lambda: tmp_path / "viewer-live")
    response = client.get("/api/demo/viewer-analysis/nope/payload")
    assert response.status_code == 404
    body = response.get_json()
    assert body["success"] is False and body["state"]["status"] == analysis.STATUS_MISSING
