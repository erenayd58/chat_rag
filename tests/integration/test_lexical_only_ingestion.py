"""Ingestion must not compute document vectors when the retriever has no dense leg.

Regression cover for the upload failure where ``ingest_document`` called
``embedding_model.encode`` unconditionally, so the bm25_only profile -- whose
NullEmbedding deliberately raises -- returned 500 after chunking had already
succeeded.

The branch is on the retriever's ``requires_document_embeddings`` capability,
not on the profile name, so these tests assert the capability contract.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest

from chat_rag.components.retriever import BM25OnlyRetriever, NullEmbedding
from chat_rag.core.exceptions import RetrieverException
from chat_rag.core.models import DocumentChunk
from chat_rag.pipeline.rag_pipeline import RAGPipeline


class RecordingEmbedding:
    """Dense embedder that records every call."""

    def __init__(self) -> None:
        self.calls = 0

    def encode(self, texts, convert_to_tensor: bool = False, **kwargs):
        self.calls += 1
        return np.asarray([0.1, 0.2, 0.3], dtype=float)

    def encode_documents(self, texts):
        self.calls += 1
        return np.asarray([[0.1, 0.2, 0.3] for _ in texts], dtype=float)

    def get_name(self):
        return "RecordingEmbedding"

    def get_dimension(self):
        return 3


class InMemoryVectorDB:
    def __init__(self) -> None:
        self.chunks: list[DocumentChunk] = []
        self.embeddings_seen: list = []

    def add_chunks(self, chunks, embeddings, **kwargs):
        self.chunks.extend(chunks)
        self.embeddings_seen.append(embeddings)

    def get_all_chunks(self):
        return list(self.chunks)


class StubChunker:
    def chunk_text(self, text, doc_id, doc_title, document_summary=None, **kwargs):
        return [
            DocumentChunk(
                chunk_id=f"{doc_id}-{index}",
                doc_id=doc_id,
                content=content,
                chunk_index=index,
                total_chunks=2,
                doc_title=doc_title,
                metadata={"created_at": datetime.now().isoformat()},
            )
            for index, content in enumerate(
                ["TBB Risk Merkezi 28 Haziran 2013 tarihinde faaliyete gecti.",
                 "KKB 2024 yil sonunda 709 calisani ile hizmet vermektedir."]
            )
        ]

    def get_name(self):
        return "StubChunker"


class StubContextualEnhancer:
    """Only the two hooks ingestion touches."""

    def generate_document_summary(self, text, title):
        return ""

    def enrich_chunk_with_context(self, chunk):
        return chunk.content


def _pipeline(retriever, embedding, vector_db, profile):
    pipeline = RAGPipeline.__new__(RAGPipeline)
    pipeline.contextual_enhancer = StubContextualEnhancer()
    pipeline.retrieval_profile = profile
    pipeline.chunker = StubChunker()
    pipeline.embedding_model = embedding
    pipeline.vector_db = vector_db
    pipeline.hybrid_retriever = retriever
    return pipeline


def test_lexical_only_ingest_never_calls_encode():
    embedding = NullEmbedding()
    db = InMemoryVectorDB()
    retriever = BM25OnlyRetriever(embedding, db)
    assert retriever.requires_document_embeddings is False

    pipeline = _pipeline(retriever, embedding, db, "bm25_only")
    chunks = pipeline.ingest_document(
        document_text="irrelevant", doc_id="doc", doc_title="doc.pdf"
    )

    assert len(chunks) == 2
    # NullEmbedding raises on any call, so completing at all proves it was
    # never invoked; assert the store saw no vectors either.
    assert db.embeddings_seen == [[]]
    assert all(chunk.embedding is None for chunk in chunks)


def test_lexical_only_ingest_completes_and_is_searchable():
    embedding = NullEmbedding()
    db = InMemoryVectorDB()
    retriever = BM25OnlyRetriever(embedding, db)
    pipeline = _pipeline(retriever, embedding, db, "bm25_only")

    pipeline.ingest_document(
        document_text="irrelevant", doc_id="doc", doc_title="doc.pdf"
    )
    assert len(db.chunks) == 2

    hits = retriever.hybrid_search("Risk Merkezi ne zaman faaliyete gecti", top_k=2)
    assert hits
    assert hits[0].retrieval_method == "bm25_only"
    assert hits[0].dense_rank is None
    assert "Risk Merkezi" in hits[0].chunk.content


def test_dense_profiles_still_embed_every_chunk():
    embedding = RecordingEmbedding()
    db = InMemoryVectorDB()

    class DenseRetriever:
        requires_document_embeddings = True

        def build_keyword_index(self, chunks):
            pass

    pipeline = _pipeline(DenseRetriever(), embedding, db, "benchmark_aligned")
    chunks = pipeline.ingest_document(
        document_text="irrelevant", doc_id="doc", doc_title="doc.pdf"
    )

    assert embedding.calls >= 1
    assert len(db.embeddings_seen[0]) == len(chunks)
    assert all(chunk.embedding is not None for chunk in chunks)


def test_retriever_without_the_flag_defaults_to_embedding():
    """An older retriever with no capability attribute must keep embedding."""
    embedding = RecordingEmbedding()
    db = InMemoryVectorDB()

    class LegacyRetriever:
        def build_keyword_index(self, chunks):
            pass

    pipeline = _pipeline(LegacyRetriever(), embedding, db, "benchmark_aligned")
    pipeline.ingest_document(
        document_text="irrelevant", doc_id="doc", doc_title="doc.pdf"
    )
    assert embedding.calls >= 1
    assert db.embeddings_seen[0]


def test_null_embedding_still_refuses_direct_use():
    with pytest.raises(RetrieverException):
        NullEmbedding().encode(["x"])
