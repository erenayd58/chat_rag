from __future__ import annotations

import hashlib
from types import SimpleNamespace

import numpy as np
import pytest

from amsc.models import EmbeddingBatch, SemanticEmbeddingProvenance

from components.chunker import FrozenV4Chunker, SemanticChunker
from components.embedding import BaseEmbedding
from components.llm import BaseLLM
from components.reranker import BaseReranker
from components.retriever import BenchmarkAlignedEmbedding
from components.vectordb import FaissVectorDB
from core.models import RetrievalResult
from pipeline import RAGPipeline


class FakeLLM(BaseLLM):
    def __init__(self):
        self.calls = 0

    def generate(self, messages, temperature=0.3, max_tokens=200, **kwargs):
        self.calls += 1
        return "Deterministic document summary."

    def get_name(self):
        return "FakeLLM"

    def get_model_name(self):
        return "fake"


class HashEmbedding(BaseEmbedding):
    def encode(self, texts, convert_to_tensor=False, **kwargs):
        if isinstance(texts, list):
            return np.vstack([self._one(text) for text in texts])
        return self._one(texts)

    @staticmethod
    def _one(text: str) -> np.ndarray:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        vector = np.asarray([digest[0] + 1, digest[1] + 1, digest[2] + 1], dtype=float)
        return vector / np.linalg.norm(vector)

    def get_name(self):
        return "HashEmbedding"

    def get_dimension(self):
        return 3


class DeterministicBoundaryEmbedder:
    model_id = "test:deterministic-boundary@1"
    prefix_policy = "symmetric_query"
    model_input_limit = 512
    cache_namespace = "test-deterministic-boundary"

    def embed_units(self, texts):
        vectors = np.vstack([HashEmbedding._one(text) for text in texts])
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


class PassthroughReranker(BaseReranker):
    def rerank(self, query, results, top_k=5):
        return results[:top_k]

    def get_name(self):
        return "PassthroughReranker"


def _settings():
    return SimpleNamespace(
        enable_conversation=False,
        max_conversation_history=0,
        reranker_type="llm",
        vector_weight=0.7,
        bm25_weight=0.3,
        include_vector_results_n=0,
        include_bm25_results_n=0,
    )


class FakeFrozenRetrievalEmbedder:
    model_id = "fake-e5@frozen"

    @staticmethod
    def _encode(texts):
        vectors = []
        for text in texts:
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            vector = np.asarray(
                [digest[0] + 1, digest[1] + 1, digest[2] + 1], dtype=np.float32
            )
            vectors.append(vector / np.linalg.norm(vector))
        return np.vstack(vectors)

    def embed_documents(self, texts):
        return self._encode(texts), None

    def embed_queries(self, texts):
        return self._encode(texts), None


@pytest.mark.parametrize("chunker_type", ["legacy", "v4"])
def test_document_ingestion_vector_index_and_retrieval_smoke(chunker_type, tmp_path):
    if chunker_type == "legacy":
        chunker = SemanticChunker(
            chunk_size=300,
            chunk_overlap=60,
            min_chunk_size=50,
            use_semantic_segmentation=False,
        )
    else:
        chunker = FrozenV4Chunker(
            boundary_embedder=DeterministicBoundaryEmbedder()
        )

    vector_db = FaissVectorDB(
        path=str(tmp_path / chunker_type), rebuild_bm25_on_load=False
    )
    pipeline = RAGPipeline(
        llm_model=FakeLLM(),
        embedding_model=HashEmbedding(),
        vector_db=vector_db,
        chunker=chunker,
        reranker=PassthroughReranker(),
        settings=_settings(),
    )
    # BM25 is outside this smoke's scope; vector indexing/retrieval is exercised.
    pipeline.hybrid_retriever.build_keyword_index = lambda chunks: None

    chunks = pipeline.ingest_document(
        "Revenue increased during the year.\n\nCustomer growth remained strong.",
        doc_id=f"smoke-{chunker_type}",
        doc_title="Smoke Test",
        additional_metadata={"parser": "TextParser", "file_name": "smoke.txt"},
    )
    results = pipeline.hybrid_retriever.vector_search("revenue growth", top_k=3)

    assert chunks
    assert vector_db.count() == len(chunks)
    assert results
    assert all(isinstance(item, RetrievalResult) for item in results)
    assert results[0].chunk.doc_id == f"smoke-{chunker_type}"


@pytest.mark.parametrize("chunker_type", ["legacy", "v4"])
def test_benchmark_aligned_profile_ingests_and_retrieves_without_context_or_rerank(
    chunker_type, tmp_path
):
    if chunker_type == "legacy":
        chunker = SemanticChunker(
            chunk_size=300,
            chunk_overlap=60,
            min_chunk_size=1,
            use_semantic_segmentation=False,
        )
    else:
        chunker = FrozenV4Chunker(
            boundary_embedder=DeterministicBoundaryEmbedder()
        )
    fake_llm = FakeLLM()
    embedding = BenchmarkAlignedEmbedding(embedder=FakeFrozenRetrievalEmbedder())
    settings = _settings()
    settings.retrieval_profile = "benchmark_aligned"
    settings.default_top_k = 5
    vector_db = FaissVectorDB(
        path=str(tmp_path / f"benchmark-{chunker_type}"),
        rebuild_bm25_on_load=False,
    )
    pipeline = RAGPipeline(
        llm_model=fake_llm,
        embedding_model=embedding,
        vector_db=vector_db,
        chunker=chunker,
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
    assert fake_llm.calls == 0
    assert all(chunk.document_summary == "" for chunk in chunks)
    assert metadata["retrieval_profile"] == "benchmark_aligned"
    assert metadata["query_expansion"] is False
    assert metadata["reranking"] is False
    assert metadata["contextualization"] is False
