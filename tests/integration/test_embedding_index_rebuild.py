"""Rebuilding a knowledge base's embedding index, end to end on pgvector.

``POST /api/v1/knowledge-bases/<kb_id>/embedding-index/rebuild`` is the one
long write the API does inline, and the operator action that gets a knowledge
base out of the state the manifest exists to detect: vectors written by a
model that is no longer the configured one. Step 9 changed what it writes --
rows and a manifest row, rather than a Chroma collection and a JSON file
beside it -- and changed nothing a caller sees.

So this drives the real route, over the real store, with the real pipeline,
and pins both halves: the endpoint's contract (status, shape, the fields the
console reads) and the effect (every chunk re-embedded, the text untouched,
the manifest telling the truth afterwards, and dense retrieval working again).

The only double is the embedding transport -- a deterministic in-process one,
so the test computes vectors without a provider.
"""

from __future__ import annotations

import hashlib
import re
from types import SimpleNamespace

import numpy as np
import pytest

from components.embedding import OpenAICompatibleEmbedding
from components.embedding.index_manifest import (
    STATE_COMPATIBLE, STATE_REINDEX_REQUIRED, build_manifest,
)
from components.llm.base import BaseLLM
from components.vectordb import PgVectorStore
from config import Settings
from pipeline.rag_pipeline import RAGPipeline

KEY = "REBUILD_TEST_KEY"
UNITS = [
    {"unit_id": f"u-{i:04d}", "order": i, "type": "paragraph",
     "text": f"Risk Merkezi {i} numarali kayit hakkinda ayrintili aciklama metni. "
             + "kelime " * 40,
     "section_path": ("1. RISK MERKEZI",), "page": 1 + i // 3}
    for i in range(9)
]


class FakeTransport:
    """Deterministic bag-of-words vectors; counts what was asked for."""

    #: The transport's own identity, read by the embedding's per-text cache
    #: and by the ingest budget wrapper around it.
    model_id = "test/embedding"

    def __init__(self, dimension=16):
        self.dimension = dimension
        self.calls = 0
        self.texts = []

    def embed(self, texts):
        self.calls += 1
        self.texts.extend(texts)
        rows = []
        for text in texts:
            vector = [0.0] * self.dimension
            for word in re.findall("[a-z0-9]+", text.casefold()):
                index = int(hashlib.sha1(word.encode()).hexdigest(), 16) % self.dimension
                vector[index] += 1.0
            norm = sum(value * value for value in vector) ** 0.5 or 1.0
            rows.append([value / norm for value in vector])
        return np.asarray(rows, dtype=float)


class FakeAnswer(BaseLLM):
    def generate(self, messages, **kwargs):
        return "[S1] cevap"

    def is_available(self):
        return True

    def get_name(self):
        return "fake"

    def get_model_name(self):
        return "test/answer"


@pytest.fixture
def kb(tmp_path, monkeypatch, client_app):
    """A knowledge base with a real corpus in it, and the pipeline that wrote
    it wired into the container the routes read."""
    monkeypatch.setenv(KEY, "not-a-real-key")
    services, client = client_app

    record = services.kb_manager.create(name="rebuild-me")
    kb_id = record["kb_id"]

    transport = FakeTransport()
    embedding = OpenAICompatibleEmbedding(
        "test/embedding", api_key_env=KEY, cache_dir=str(tmp_path / "cache"),
        provider=transport,
    )
    settings = Settings()
    settings.retrieval_profile = "hybrid_rrf"
    settings.chunker_type = "structure_first"
    settings.embedding_provider = "openai_compatible"
    settings.embedding_model_name = "test/embedding"
    settings.embedding_api_key_env = KEY
    settings.vector_collection = kb_id
    settings.vector_kb_id = kb_id

    pipeline = RAGPipeline(
        llm_model=FakeAnswer(), embedding_model=embedding,
        vector_db=PgVectorStore(collection=kb_id, kb_id=kb_id), settings=settings,
    )
    pipeline.ingest_document("", doc_id="doc-1", doc_title="Rapor",
                             parsed_units=UNITS)
    monkeypatch.setattr(services.pipeline_cache, "_build",
                        lambda session_id, kb: pipeline)
    services.pipeline_cache.clear()
    return SimpleNamespace(kb_id=kb_id, client=client, pipeline=pipeline,
                           transport=transport, embedding=embedding)


@pytest.fixture
def client_app():
    """The FastAPI surface over the process container."""
    from fastapi.testclient import TestClient

    import app as flask_app
    from interfaces.http import v1

    with TestClient(v1.create_app(flask_app.services),
                    raise_server_exceptions=False) as client:
        yield flask_app.services, client


V1 = "/api/v1"


# ------------------------------------------------------------- the contract
def test_the_endpoint_answers_the_shape_it_always_did(kb):
    response = kb.client.post(
        f"{V1}/knowledge-bases/{kb.kb_id}/embedding-index/rebuild")

    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"result", "index"}
    assert body["result"]["chunks"] == kb.pipeline.vector_db.count()
    assert body["result"]["dimension"] == 16
    assert body["index"]["state"] == STATE_COMPATIBLE
    assert body["index"]["dense_available"] is True


def test_an_unknown_knowledge_base_is_still_a_404(kb):
    assert kb.client.post(
        f"{V1}/knowledge-bases/nope/embedding-index/rebuild").status_code == 404


def test_the_report_names_no_credential(kb):
    body = kb.client.post(
        f"{V1}/knowledge-bases/{kb.kb_id}/embedding-index/rebuild").text
    assert KEY not in body and "api_key" not in body.casefold()


# --------------------------------------------------------------- the effect
def test_a_rebuild_recovers_a_knowledge_base_whose_index_went_stale(kb):
    """The state the endpoint exists for: somebody's vectors were written by
    another model, the dense leg is refused, and one call fixes it."""
    kb.pipeline.vector_db.write_manifest(build_manifest(
        {"provider": "openai_compatible", "model": "other/model",
         "fingerprint": "deadbeefdeadbeef"}, dimension=16, chunk_count=9))
    assert kb.pipeline.embedding_index_status()["state"] == STATE_REINDEX_REQUIRED

    body = kb.client.post(
        f"{V1}/knowledge-bases/{kb.kb_id}/embedding-index/rebuild").json()

    assert body["index"]["state"] == STATE_COMPATIBLE
    assert kb.pipeline.embedding_index_status()["dense_available"] is True
    hits = kb.pipeline.hybrid_retriever.vector_search("Risk Merkezi kaydi", top_k=3)
    assert hits, "dense retrieval is working again"


def test_a_rebuild_rewrites_vectors_and_nothing_else(kb):
    before = {chunk.chunk_id: chunk for chunk in kb.pipeline.vector_db.get_all_chunks()}
    ids = sorted(before)
    vectors_before = {
        chunk_id: kb.pipeline.vector_db.get_chunk_by_id(chunk_id)["embedding"]
        for chunk_id in ids
    }
    # Move the store into another space so the rebuild has something to undo.
    kb.pipeline.vector_db.replace_all(
        [before[chunk_id] for chunk_id in ids],
        [[1.0] + [0.0] * 15 for _ in ids],
    )

    kb.client.post(f"{V1}/knowledge-bases/{kb.kb_id}/embedding-index/rebuild")

    after = {chunk.chunk_id: chunk for chunk in kb.pipeline.vector_db.get_all_chunks()}
    assert sorted(after) == ids, "a rebuild added or lost a chunk"
    for chunk_id in ids:
        assert after[chunk_id].content == before[chunk_id].content
        assert after[chunk_id].metadata == before[chunk_id].metadata
        assert kb.pipeline.vector_db.get_chunk_by_id(chunk_id)["embedding"] == \
            pytest.approx(vectors_before[chunk_id]), "the vectors did not come back"


def test_the_manifest_afterwards_describes_the_model_that_ran(kb):
    kb.client.post(f"{V1}/knowledge-bases/{kb.kb_id}/embedding-index/rebuild")

    manifest = kb.pipeline.vector_db.read_manifest()
    assert manifest["embedding_model"] == "test/embedding"
    assert manifest["embedding_dimension"] == 16
    assert manifest["embedding_fingerprint"] == kb.embedding.fingerprint
    assert manifest["chunk_count"] == kb.pipeline.vector_db.count()


def test_a_rebuild_of_one_knowledge_base_touches_no_other(kb):
    other = PgVectorStore(collection="untouched")
    other.add_chunks(kb.pipeline.vector_db.get_all_chunks()[:2],
                     [[1.0] + [0.0] * 15, [0.0, 1.0] + [0.0] * 14])

    kb.client.post(f"{V1}/knowledge-bases/{kb.kb_id}/embedding-index/rebuild")

    assert other.count() == 2
    assert other.get_chunk_by_id(other.get_all_chunks()[0].chunk_id)["embedding"][0] \
        in (0.0, 1.0)
