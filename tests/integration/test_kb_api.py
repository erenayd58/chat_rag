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

from fastapi.testclient import TestClient

import asgi as entrypoint
import interfaces.http as http

V1 = http.v1.PREFIX
from chat_rag.application import workspace as app_workspace
from chat_rag.components.goldset import GoldSetManager
from chat_rag.components.knowledgebase.manager import KnowledgeBaseManager
from chat_rag.components.retriever import BM25OnlyRetriever, NullEmbedding
from chat_rag.components.vectordb import PgVectorStore
from chat_rag.core.models import DocumentChunk


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
        entrypoint.services, "kb_manager", KnowledgeBaseManager(str(tmp_path / "kbs.json"))
    )
    monkeypatch.setattr(
        entrypoint.services, "gold_manager", GoldSetManager(str(tmp_path / "gold.json"))
    )
    # The upload tests below drive the route, not the Viewer packager. That
    # packager runs on a background thread which outlives the request and asks
    # the application for a pipeline's vector store -- something the stub
    # pipelines here deliberately do not have. Stubbed so this file exercises
    # one contract at a time.
    monkeypatch.setattr(app_workspace, "stage_analysis", lambda *a, **k: {"status": "queued"})
    with TestClient(http.create_app(entrypoint.services),
                    raise_server_exceptions=False) as test_client:
        yield test_client


def use_retriever(monkeypatch, retriever):
    monkeypatch.setattr(
        entrypoint.services, "get_pipeline", lambda *a, **k: StubPipeline(retriever)
    )


def lexical(monkeypatch):
    use_retriever(
        monkeypatch, BM25OnlyRetriever(embedding_model=NullEmbedding(), vector_db=None)
    )


# ---------------------------------------------------------- capabilities


def test_a_lexical_profile_advertises_bm25_only(client, monkeypatch):
    lexical(monkeypatch)
    body = client.get(f"{V1}/meta/retrieval-methods").json()

    assert body["default"] == "bm25"
    offered = {m["name"] for m in body["items"] if m["available"]}
    assert offered == {"bm25"}
    for entry in body["items"]:
        assert entry["available"] or entry["unavailable_reason"]


def test_a_dense_profile_advertises_every_method(client, monkeypatch):
    use_retriever(monkeypatch, DenseRetriever())
    body = client.get(f"{V1}/meta/retrieval-methods").json()
    assert {m["name"] for m in body["items"] if m["available"]} == {
        "hybrid", "vector", "bm25"
    }


def test_an_unsupported_method_is_refused_with_an_explanation(client, monkeypatch):
    """It used to surface the retriever's raw exception text in the UI."""
    lexical(monkeypatch)
    response = client.post(
        f"{V1}/searches",
        json={"query": "findeks", "method": "vector", "limit": 3},
    )
    error = response.json()["error"]

    assert response.status_code == 400
    assert error["type"] == "invalid_request"
    assert error["details"]["unsupported_method"] is True
    assert "embedding" in error["message"].lower()
    assert "RetrieverException" not in error["message"]
    assert error["details"]["capabilities"]["default"] == "bm25"


#: The gold set had three routes on the Flask-era surface -- list, upsert and
#: delete -- and no ``/api/v1`` answer, deliberately: it is an offline input to
#: ``python -m cli`` rather than a product operation (``docs/api-v1.md``,
#: *Not here, on purpose*). Step 13 removed those routes with the rest of that
#: surface, and the use case that translated for them. What the entries have
#: to do is unchanged and is driven where it lives, over the manager itself:
#: ``tests/unit/test_goldset_manager.py``.


# -------------------------------------------------------- kb lifecycle


def test_a_knowledge_base_can_be_created_and_deleted(client):
    created = client.post(f"{V1}/knowledge-bases", json={"name": "kkb-final"})
    assert created.status_code == 201, created.text
    kb_id = created.json()["id"]

    # Which collection this knowledge base's chunks are in, asked of the same
    # resolver the deletion route asks -- not rebuilt from a literal that
    # happens to match today's default.
    collection = entrypoint.services.kb_manager.collection(kb_id)
    store = _fill(collection, kb_id)
    assert store.count() == 1

    assert client.delete(f"{V1}/knowledge-bases/{kb_id}").status_code == 204

    # The corpus went with the record, in the same transaction. The counts are
    # read from the store rather than from a response body: the contract
    # answers 204, and how many rows a deletion took is not a promise it makes.
    assert store.count() == 0
    assert client.get(f"{V1}/knowledge-bases").json()["items"] == []


def test_creating_a_duplicate_name_is_refused(client):
    assert client.post(f"{V1}/knowledge-bases",
                       json={"name": "kkb-final"}).status_code == 201
    response = client.post(f"{V1}/knowledge-bases", json={"name": "kkb-final"})

    # A rejected payload is a client error, not a server fault.
    assert response.status_code == 400
    assert "already exists" in response.json()["error"]["message"]
    assert client.get(f"{V1}/knowledge-bases").json()["page"]["total"] == 1


def test_an_invalid_chunker_is_a_client_error_and_creates_nothing(client):
    response = client.post(f"{V1}/knowledge-bases",
                           json={"name": "kb", "chunker": {"type": "nope"}})
    assert response.status_code == 400
    assert client.get(f"{V1}/knowledge-bases").json()["items"] == []


def test_a_failed_clearance_keeps_the_record(client, monkeypatch):
    """The record must not outlive its corpus, nor the corpus its record.

    A directory that could not be removed used to be the way this happened --
    on Windows an open Chroma handle was enough. It is one transaction now, so
    the failure has to be provoked from inside it; what is asserted is
    unchanged: 409, and both halves still there.
    """
    from sqlalchemy.exc import OperationalError

    from chat_rag.storage.repositories import ChunkVectorRepository

    created = client.post(f"{V1}/knowledge-bases", json={"name": "locked"}).json()
    collection = entrypoint.services.kb_manager.collection(created["id"])
    _fill(collection, created["id"])

    def refuse(*_args, **_kwargs):
        raise OperationalError("DELETE FROM vector_collections", {}, Exception("in use"))

    monkeypatch.setattr(ChunkVectorRepository, "delete_collection", refuse)
    response = client.delete(f"{V1}/knowledge-bases/{created['id']}")

    assert response.status_code == 409
    assert "in use" in response.json()["error"]["message"]
    assert client.get(f"{V1}/knowledge-bases").json()["page"]["total"] == 1
    assert PgVectorStore(collection=collection).count() == 1


def test_deleting_an_unknown_knowledge_base_is_a_404(client):
    assert client.delete(f"{V1}/knowledge-bases/nope").status_code == 404


def test_deleting_one_knowledge_base_leaves_anothers_vectors_alone(client):
    """Two knowledge bases could once be pointed at one directory, and
    deleting either had to decide whether the other still needed it. A
    collection is named by the knowledge base's own id, so the question cannot
    be asked -- which is what this pins."""
    first = client.post(f"{V1}/knowledge-bases", json={"name": "one"}).json()
    second = client.post(f"{V1}/knowledge-bases", json={"name": "two"}).json()
    gone = _fill(entrypoint.services.kb_manager.collection(first["id"]), first["id"])
    kept = _fill(entrypoint.services.kb_manager.collection(second["id"]), second["id"])

    assert client.delete(f"{V1}/knowledge-bases/{first['id']}").status_code == 204

    assert gone.count() == 0
    assert kept.count() == 1


def test_deleting_drops_the_cached_pipeline(client, monkeypatch):
    """A live Chroma client holds the store's sqlite open."""
    kb_id = client.post(f"{V1}/knowledge-bases", json={"name": "kb"}).json()["id"]
    monkeypatch.setattr(entrypoint.services.pipeline_cache, "_build",
                        lambda session_id, kb: SimpleNamespace(vector_db=None))
    entrypoint.services.pipeline_cache.get("global", kb_id)
    assert kb_id in entrypoint.services.pipeline_cache.snapshot()["knowledge_bases"]

    client.delete(f"{V1}/knowledge-bases/{kb_id}")

    assert kb_id not in entrypoint.services.pipeline_cache.snapshot()["knowledge_bases"]


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
    """One upload, settled. The answer is the job, so the ingest is finished
    where the job is, not where the response is."""
    import io

    monkeypatch.setattr(entrypoint.services, "get_pipeline", lambda *a, **k: pipeline)
    kb = client.post(f"{V1}/knowledge-bases", json={
        "name": "ingest-kb", "chunker": {"type": "structure_first"},
    }).json()
    response = client.post(
        f"{V1}/documents",
        files={"file": ("rapor.pdf", io.BytesIO(b"%PDF-1.7 pretend"), "application/pdf")},
        data={"knowledge_base_id": kb["id"]},
    )
    assert response.status_code == 202, response.text
    manager = entrypoint.services.ingest_jobs
    job = manager.get(response.json()["id"])
    assert manager.wait(job, 30), "the job did not settle"
    return job.snapshot()


def tracked():
    """Every document in the ledger this configuration writes to.

    ``DocumentTracker`` resolves that itself, through ``config.paths``. The
    test does not know the path and must not: it once took a ``tmp_path`` it
    ignored, which read as a promise that the ledger was under it.
    """
    from chat_rag.utils import DocumentTracker

    return DocumentTracker().get_all_documents()


def test_a_successful_ingest_records_how_the_pipeline_was_configured(
    client, monkeypatch
):
    job = upload(client, monkeypatch, IngestingPipeline(chunks=[IngestedChunk()]))
    assert job["status"] == "succeeded", job

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
    job = upload(client, monkeypatch,
                 IngestingPipeline(error=RuntimeError("parser blew up")))

    assert job["status"] == "failed"
    assert tracked() == []
