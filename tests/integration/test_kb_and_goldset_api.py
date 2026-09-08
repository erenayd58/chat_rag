"""HTTP surface for the four debts closed before the regression CLI.

Knowledge bases could not be deleted at all, the review screen offered
retrieval methods the retriever could not serve, and confirmed answers lived
only in a browser. These tests drive the routes, with the managers pointed at a
temporary directory so the developer's own files are never touched.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from types import SimpleNamespace

import app as flask_app
from application import workspace as app_workspace
from components.goldset import GoldSetManager
from components.knowledgebase.manager import KnowledgeBaseManager
from components.retriever import BM25OnlyRetriever, NullEmbedding
from components.vectordb import PgVectorStore
from core.models import DocumentChunk


def _fill(collection, kb_id, chunk_id="c-1"):
    """One chunk in a knowledge base's collection, and the store holding it."""
    store = PgVectorStore(collection=collection, kb_id=kb_id)
    store.add_chunks(
        [DocumentChunk(chunk_id=chunk_id, content="metin", doc_id="doc-1",
                       doc_title="rapor.pdf", chunk_index=0, total_chunks=1,
                       metadata={"word_count": 1})],
        [[1.0, 0.0, 0.0]],
    )
    return store


class DenseRetriever:
    requires_document_embeddings = True

    def hybrid_search(self, *a, **k): return []
    def vector_search(self, *a, **k): return []
    def keyword_search(self, *a, **k): return []


class StubPipeline:
    def __init__(self, retriever):
        self.hybrid_retriever = retriever


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        flask_app.services, "kb_manager", KnowledgeBaseManager(str(tmp_path / "kbs.json"))
    )
    monkeypatch.setattr(
        flask_app.services, "gold_manager", GoldSetManager(str(tmp_path / "gold.json"))
    )
    # The upload tests below drive the route, not the Viewer packager. That
    # packager runs on a background thread which outlives the request and asks
    # the application for a pipeline's vector store -- something the stub
    # pipelines here deliberately do not have. Stubbed so this file exercises
    # one contract at a time.
    monkeypatch.setattr(app_workspace, "stage_analysis", lambda *a, **k: {"status": "queued"})
    flask_app.app.config.update(TESTING=True)
    with flask_app.app.test_client() as test_client:
        yield test_client


def use_retriever(monkeypatch, retriever):
    monkeypatch.setattr(
        flask_app.services, "get_pipeline", lambda *a, **k: StubPipeline(retriever)
    )


def lexical(monkeypatch):
    use_retriever(
        monkeypatch, BM25OnlyRetriever(embedding_model=NullEmbedding(), vector_db=None)
    )


# ---------------------------------------------------------- capabilities


def test_a_lexical_profile_advertises_bm25_only(client, monkeypatch):
    lexical(monkeypatch)
    body = client.get("/api/retrieval/capabilities").get_json()

    assert body["success"] is True
    assert body["default"] == "bm25"
    assert body["dense"] is False
    offered = {m["name"] for m in body["methods"] if m["available"]}
    assert offered == {"bm25"}
    for entry in body["methods"]:
        assert entry["available"] or entry["reason"]


def test_a_dense_profile_advertises_every_method(client, monkeypatch):
    use_retriever(monkeypatch, DenseRetriever())
    body = client.get("/api/retrieval/capabilities").get_json()
    assert {m["name"] for m in body["methods"] if m["available"]} == {
        "hybrid", "vector", "bm25"
    }


def test_an_unsupported_method_is_refused_with_an_explanation(client, monkeypatch):
    """It used to surface the retriever's raw exception text in the UI."""
    lexical(monkeypatch)
    response = client.post(
        "/api/experiment/search_chunks",
        json={"query": "findeks", "method": "vector", "top_k": 3},
    )
    body = response.get_json()

    assert response.status_code == 400
    assert body["success"] is False
    assert body["unsupported_method"] is True
    assert "embedding" in body["error"].lower()
    assert "RetrieverException" not in body["error"]
    assert body["capabilities"]["default"] == "bm25"


# ------------------------------------------------------------- gold set


GOLD = {
    "question": "2024 Findeks Risk Raporu sorgu adedi kac?",
    "kb_id": "kb-1",
    "document_id": "upload_abc_pdf",
    "correct_chunk_id": "doc:s-chunk-0172",
    "section": "KOSGEB",
    "pages": [36],
    "unit_ids": ["v-00808"],
    "evidence": "FINDEKS RISK RAPORU SORGU ADEDI | 11.000.144",
    "found_at_rank": 1,
}


def test_a_confirmed_answer_is_stored_and_read_back(client, tmp_path):
    saved = client.post("/api/goldset", json=GOLD).get_json()
    assert saved["success"] is True

    listed = client.get("/api/goldset?kb_id=kb-1").get_json()["entries"]
    assert len(listed) == 1
    assert listed[0]["correct_chunk_id"] == GOLD["correct_chunk_id"]
    assert listed[0]["unit_ids"] == ["v-00808"]

    # And it is really persisted, not merely echoed: a second manager, over
    # the same store, reads it back.
    stored = GoldSetManager(str(tmp_path / "gold.json")).list()
    assert [entry["question"] for entry in stored] == [GOLD["question"]]


def test_marking_the_same_question_again_updates_one_entry(client):
    client.post("/api/goldset", json=GOLD)
    client.post("/api/goldset", json={**GOLD, "correct_chunk_id": "doc:s-chunk-0002"})

    entries = client.get("/api/goldset").get_json()["entries"]
    assert len(entries) == 1
    assert entries[0]["correct_chunk_id"] == "doc:s-chunk-0002"


def test_an_entry_without_a_locator_is_rejected(client):
    response = client.post("/api/goldset", json={"question": "s", "kb_id": "kb-1"})
    assert response.status_code == 400
    assert "locator" in response.get_json()["error"]


def test_an_entry_can_be_removed(client):
    entry = client.post("/api/goldset", json=GOLD).get_json()["entry"]
    assert client.delete(f"/api/goldset/{entry['entry_id']}").status_code == 200
    assert client.get("/api/goldset").get_json()["entries"] == []
    assert client.delete(f"/api/goldset/{entry['entry_id']}").status_code == 404


def test_entries_are_filtered_by_knowledge_base(client):
    client.post("/api/goldset", json=GOLD)
    client.post("/api/goldset", json={**GOLD, "kb_id": "kb-2"})
    assert len(client.get("/api/goldset?kb_id=kb-1").get_json()["entries"]) == 1
    assert len(client.get("/api/goldset").get_json()["entries"]) == 2


# -------------------------------------------------------- kb lifecycle


def test_a_knowledge_base_can_be_created_and_deleted(client):
    created = client.post("/api/kb", json={"name": "kkb-final"}).get_json()
    assert created["success"] is True
    kb_id = created["kb"]["kb_id"]

    # Which collection this knowledge base's chunks are in, asked of the same
    # resolver the deletion route asks -- not rebuilt from a literal that
    # happens to match today's default.
    collection = flask_app.services.kb_manager.collection(kb_id)
    store = _fill(collection, kb_id)
    assert store.count() == 1

    body = client.delete(f"/api/kb/{kb_id}").get_json()

    assert body["success"] is True
    assert body["vectors_removed"] == 1
    assert body["vector_collection"] == collection
    assert store.count() == 0
    assert client.get("/api/kb").get_json()["knowledge_bases"] == []


def test_creating_a_duplicate_name_is_refused(client):
    assert client.post("/api/kb", json={"name": "kkb-final"}).get_json()["success"]
    response = client.post("/api/kb", json={"name": "kkb-final"})
    body = response.get_json()

    # A rejected payload is a client error, not a server fault.
    assert response.status_code == 400
    assert body["success"] is False
    assert "already exists" in body["error"]
    assert len(client.get("/api/kb").get_json()["knowledge_bases"]) == 1


def test_an_invalid_chunker_is_a_client_error_and_creates_nothing(client):
    response = client.post("/api/kb", json={"name": "kb", "chunker": {"type": "nope"}})
    assert response.status_code == 400
    assert client.get("/api/kb").get_json()["knowledge_bases"] == []


def test_a_failed_clearance_keeps_the_record(client, monkeypatch):
    """The record must not outlive its corpus, nor the corpus its record.

    A directory that could not be removed used to be the way this happened --
    on Windows an open Chroma handle was enough. It is one transaction now, so
    the failure has to be provoked from inside it; what is asserted is
    unchanged: 409, and both halves still there.
    """
    from sqlalchemy.exc import OperationalError

    from storage.repositories import ChunkVectorRepository

    created = client.post("/api/kb", json={"name": "locked"}).get_json()["kb"]
    collection = flask_app.services.kb_manager.collection(created["kb_id"])
    _fill(collection, created["kb_id"])

    def refuse(*_args, **_kwargs):
        raise OperationalError("DELETE FROM vector_collections", {}, Exception("in use"))

    monkeypatch.setattr(ChunkVectorRepository, "delete_collection", refuse)
    response = client.delete(f"/api/kb/{created['kb_id']}")

    assert response.status_code == 409
    assert "in use" in response.get_json()["error"]
    assert len(client.get("/api/kb").get_json()["knowledge_bases"]) == 1
    assert PgVectorStore(collection=collection).count() == 1


def test_deleting_an_unknown_knowledge_base_is_a_404(client):
    assert client.delete("/api/kb/nope").status_code == 404


def test_deleting_one_knowledge_base_leaves_anothers_vectors_alone(client):
    """Two knowledge bases could once be pointed at one directory, and
    deleting either had to decide whether the other still needed it. A
    collection is named by the knowledge base's own id, so the question cannot
    be asked -- which is what this pins."""
    first = client.post("/api/kb", json={"name": "one"}).get_json()["kb"]
    second = client.post("/api/kb", json={"name": "two"}).get_json()["kb"]
    _fill(flask_app.services.kb_manager.collection(first["kb_id"]), first["kb_id"])
    kept = _fill(flask_app.services.kb_manager.collection(second["kb_id"]),
                 second["kb_id"])

    body = client.delete(f"/api/kb/{first['kb_id']}").get_json()

    assert body["vectors_removed"] == 1
    assert kept.count() == 1


def test_deleting_drops_the_cached_pipeline(client, monkeypatch):
    """A live Chroma client holds the store's sqlite open."""
    created = client.post("/api/kb", json={"name": "kb"}).get_json()["kb"]
    kb_id = created["kb_id"]
    monkeypatch.setattr(flask_app.services.pipeline_cache, "_build",
                        lambda session_id, kb: SimpleNamespace(vector_db=None))
    flask_app.services.pipeline_cache.get("global", kb_id)
    assert kb_id in flask_app.services.pipeline_cache.snapshot()["knowledge_bases"]

    client.delete(f"/api/kb/{kb_id}")

    assert kb_id not in flask_app.services.pipeline_cache.snapshot()["knowledge_bases"]


def test_the_store_holds_no_handle_that_could_outlive_a_deletion():
    """The store that replaced Chroma has nothing to close.

    Chroma held the store's sqlite file and hnsw index open, and on Windows
    that alone stopped a knowledge base's directory from ever being deleted --
    which is why the pipeline cache closes a store before dropping it, and why
    ``close`` is an optional capability rather than a required one. This one
    borrows a connection per call from the process's pooled engine and returns
    it, so it offers no ``close`` at all and the cache skips it.
    """
    store = PgVectorStore(collection="documents")
    store.add_chunks(
        [DocumentChunk(chunk_id="a", content="metin", doc_id="d", doc_title="t",
                       chunk_index=0, total_chunks=1, metadata={"word_count": 1})],
        [],
    )
    assert store.count() == 1
    assert not hasattr(store, "close")


# ------------------------------------------------------- ingest snapshot


class StubParser:
    NORMALIZATION_VERSION = "v7-as-it-ran"
    RECONSTRUCT_VISUAL_GRIDS = True
    RUNNING_HEADER_MIN_PAGES = 3
    parser_backend = "pymupdf4llm-layout"

    def get_name(self):
        return "StructuredPDFParser-stub"


class StubFactory:
    def get_parser(self, path):
        return StubParser()


class IngestingPipeline:
    """A pipeline that ingests, or refuses to."""

    retrieval_profile = "bm25_only"

    def __init__(self, chunks=None, error=None):
        self.hybrid_retriever = BM25OnlyRetriever(
            embedding_model=NullEmbedding(), vector_db=None
        )
        self.parser_factory = StubFactory()
        self._chunks = chunks or []
        self._error = error

    def ingest_document_from_file(self, file_path, doc_title, deep_analysis=False):
        if self._error:
            raise self._error
        return self._chunks


class IngestedChunk:
    doc_id = "upload_test_pdf"


def upload(client, monkeypatch, pipeline):
    import io

    monkeypatch.setattr(flask_app.services, "get_pipeline", lambda *a, **k: pipeline)
    kb = json.loads(client.post("/api/kb", json={
        "name": "ingest-kb", "chunker": {"type": "structure_first"},
    }).data)
    return client.post(
        "/api/documents/upload",
        data={"file": (io.BytesIO(b"%PDF-1.7 pretend"), "rapor.pdf"),
              "kb_id": kb["kb"]["kb_id"]},
        content_type="multipart/form-data",
    )


def tracked():
    """Every document in the ledger this configuration writes to.

    ``DocumentTracker`` resolves that itself, through ``config.paths``. The
    test does not know the path and must not: it once took a ``tmp_path`` it
    ignored, which read as a promise that the ledger was under it.
    """
    from utils import DocumentTracker

    return DocumentTracker().get_all_documents()


def test_a_successful_ingest_records_how_the_pipeline_was_configured(
    client, monkeypatch
):
    response = upload(
        client, monkeypatch, IngestingPipeline(chunks=[IngestedChunk()])
    )
    assert response.status_code == 200

    documents = tracked()
    assert len(documents) == 1
    snapshot = documents[0]["pipeline_snapshot"]
    assert snapshot is not None
    assert snapshot["pipeline"]["chunker"] == "structure_first"
    assert snapshot["pipeline"]["retriever"] == "BM25OnlyRetriever"
    assert snapshot["pipeline"]["parser"] == "StructuredPDFParser-stub"
    assert snapshot["pipeline"]["normalization_version"] == "v7-as-it-ran"
    assert snapshot["features"]["visual_grid"] is True
    assert snapshot["versions"]["important_dependencies"]
    assert snapshot["document_sha256"] == documents[0]["file_hash"]


def test_a_failed_ingest_leaves_no_document_and_no_snapshot(
    client, monkeypatch
):
    response = upload(
        client, monkeypatch, IngestingPipeline(error=RuntimeError("parser blew up"))
    )

    assert response.status_code == 500
    assert tracked() == []
