"""Ingest a document and retrieve it back, for each shipped indexing chunker.

The end-to-end shape the unit tests do not reach: real chunker, real store,
real retriever, on the frozen ``benchmark_aligned`` profile -- which is the
one that must run without a single model call, without query expansion,
reranking or contextualisation, and write no document summary.
"""

from __future__ import annotations

import hashlib
from types import SimpleNamespace

import numpy as np
import pytest

from amsc.document.models import EmbeddingBatch, SemanticEmbeddingProvenance

from chat_rag.components.chunker import FrozenV4Chunker, StructuralChunker
from chat_rag.components.llm import BaseLLM
from chat_rag.components.retriever import BenchmarkAlignedEmbedding
from chat_rag.components.vectordb import PgVectorStore
from chat_rag.pipeline import RAGPipeline


class FakeLLM(BaseLLM):
    def __init__(self):
        self.calls = 0

    def generate(self, messages, temperature=0.3, max_tokens=200, **kwargs):
        self.calls += 1
        return "Deterministic answer."

    def get_name(self):
        return "FakeLLM"

    def get_model_name(self):
        return "fake"


def _vector(text: str) -> np.ndarray:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    vector = np.asarray([digest[0] + 1, digest[1] + 1, digest[2] + 1], dtype=float)
    return vector / np.linalg.norm(vector)


class DeterministicBoundaryEmbedder:
    model_id = "test:deterministic-boundary@1"
    prefix_policy = "symmetric_query"
    model_input_limit = 512
    cache_namespace = "test-deterministic-boundary"

    def embed_units(self, texts):
        vectors = np.vstack([_vector(text) for text in texts])
        provenance = tuple(
            SemanticEmbeddingProvenance(
                model_id=self.model_id,
                prefix_policy=self.prefix_policy,
                prefix="query: ",
                model_input_limit=self.model_input_limit,
                semantic_fragment_count=1,
                semantic_pooling="token_weighted_mean",
            )
            for _ in texts
        )
        return EmbeddingBatch(vectors=vectors, provenance=provenance)


class FakeFrozenRetrievalEmbedder:
    model_id = "fake-e5@frozen"

    @staticmethod
    def _encode(texts):
        return np.vstack([_vector(text) for text in texts]).astype(np.float32)

    def embed_documents(self, texts):
        return self._encode(texts), None

    def embed_queries(self, texts):
        return self._encode(texts), None


def _chunker(chunker_type: str):
    if chunker_type == "v4":
        return FrozenV4Chunker(boundary_embedder=DeterministicBoundaryEmbedder())
    return StructuralChunker()


@pytest.mark.parametrize("chunker_type", ["structure_first", "v4"])
def test_benchmark_aligned_profile_ingests_and_retrieves_without_context_or_rerank(
    chunker_type, tmp_path
):
    fake_llm = FakeLLM()
    settings = SimpleNamespace(
        retrieval_profile="benchmark_aligned",
        default_top_k=5,
    )
    pipeline = RAGPipeline(
        llm_model=fake_llm,
        embedding_model=BenchmarkAlignedEmbedding(embedder=FakeFrozenRetrievalEmbedder()),
        vector_db=PgVectorStore(collection=f"benchmark-{chunker_type}"),
        chunker=_chunker(chunker_type),
        settings=settings,
    )

    chunks = pipeline.ingest_document(
        "Revenue increased during the year. Customer growth remained strong.",
        doc_id=f"benchmark-{chunker_type}",
        doc_title="Benchmark Smoke",
        additional_metadata={"parser": "TextParser"},
    )
    results, metadata = pipeline.retrieve("revenue growth", top_k=3)

    assert chunks
    assert results
    assert results[0].chunk.doc_id == f"benchmark-{chunker_type}"
    # Not one model call reaches the provider on this path.
    assert fake_llm.calls == 0
    assert all(chunk.document_summary == "" for chunk in chunks)
    assert metadata["retrieval_profile"] == "benchmark_aligned"
    assert metadata["query_expansion"] is False
    assert metadata["reranking"] is False
    assert metadata["contextualization"] is False
