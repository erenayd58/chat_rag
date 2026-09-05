"""Adapter for the retrieval pipeline frozen at the Phase 4/5 checkpoint."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import List, Sequence

import numpy as np
import yaml

from amsc.retrieval_pipeline import (
    DeterministicHybridIndex,
    E5RetrievalEmbedder,
    RetrievalDocument,
)

from components.embedding import BaseEmbedding
from components.vectordb import BaseVectorDB
from core.exceptions import ConfigurationException, RetrieverException
from core.models import DocumentChunk, RetrievalResult


FROZEN_RETRIEVAL_COMMIT = "1e7f7186c13729c739ccb3170da0892f7350cb27"
FROZEN_RETRIEVAL_CONFIG = {
    "profile": "benchmark_aligned",
    "source": {
        "repository": "https://github.com/erenayd58/chunk.git",
        "commit": FROZEN_RETRIEVAL_COMMIT,
        "tag": "phase5-holdout-validation",
    },
    "retrieval_embedding": {
        "model": "intfloat/multilingual-e5-base",
        "revision": None,
        "device": "auto",
        "local_files_only": True,
        "query_prefix": "query: ",
        "document_prefix": "passage: ",
        "model_input_limit": 512,
        "overlength_strategy": "sentence_fragment_token_weighted_pooling",
        "normalize_embeddings": True,
        "cache_queries": False,
        "batch_size": 16,
        "cache_dir": ".cache/retrieval-benchmark-v1",
    },
    "bm25": {
        "tokenizer": "unicode_word_casefold_v1",
        "k1": 1.5,
        "b": 0.75,
    },
    "rrf": {
        "rank_constant": 60,
        "dense_weight": 1.0,
        "bm25_weight": 1.0,
        "candidate_pool_size": 100,
    },
    "pipeline": {
        "query_expansion": False,
        "reranking": False,
        "contextualization": False,
    },
}


def load_benchmark_aligned_config(path: Path | None = None) -> dict:
    config_path = path or (
        Path(__file__).resolve().parents[2] / "config" / "benchmark_aligned_retrieval.yaml"
    )
    loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if loaded != FROZEN_RETRIEVAL_CONFIG:
        raise ConfigurationException(
            "benchmark_aligned retrieval config drifted from Phase 4/5"
        )
    return loaded


def _frozen_config(config: dict | None) -> dict:
    resolved = load_benchmark_aligned_config() if config is None else config
    if resolved != FROZEN_RETRIEVAL_CONFIG:
        raise ConfigurationException(
            "benchmark_aligned retrieval config drifted from Phase 4/5"
        )
    return deepcopy(resolved)


class BenchmarkAlignedEmbedding(BaseEmbedding):
    """Role-aware BaseEmbedding bridge backed by the frozen E5 implementation."""

    def __init__(self, config: dict | None = None, embedder=None) -> None:
        self.config = _frozen_config(config)
        embedding = self.config["retrieval_embedding"]
        self.embedder = embedder or E5RetrievalEmbedder.from_pretrained(
            embedding["model"],
            revision=embedding["revision"],
            device=embedding["device"],
            local_files_only=embedding["local_files_only"],
            query_prefix=embedding["query_prefix"],
            document_prefix=embedding["document_prefix"],
            model_input_limit=embedding["model_input_limit"],
            batch_size=embedding["batch_size"],
            cache_dir=embedding["cache_dir"],
            normalize_embeddings=embedding["normalize_embeddings"],
            cache_queries=embedding["cache_queries"],
        )

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray:
        vectors, _ = self.embedder.embed_documents(list(texts))
        return vectors

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        vectors, _ = self.embedder.embed_queries(list(texts))
        return vectors

    def encode(self, texts, convert_to_tensor=False, **kwargs):
        values = [texts] if isinstance(texts, str) else list(texts)
        vectors = self.encode_documents(values)
        return vectors[0] if isinstance(texts, str) else vectors

    def get_name(self) -> str:
        return self.embedder.model_id

    def get_dimension(self) -> int:
        model = getattr(self.embedder, "_model", None)
        if model is None or not hasattr(model, "get_sentence_embedding_dimension"):
            raise RuntimeError("Retrieval embedder does not expose its dimension")
        return int(model.get_sentence_embedding_dimension())


class BenchmarkAlignedRetriever:
    """Maps chat_rag chunks to the immutable Phase 4/5 hybrid index."""

    def __init__(
        self,
        embedding_model: BenchmarkAlignedEmbedding,
        vector_db: BaseVectorDB,
        config: dict | None = None,
    ) -> None:
        self.embedding_model = embedding_model
        self.vector_db = vector_db
        self.config = _frozen_config(config)
        self.chunks_list: List[DocumentChunk] = []
        self._chunks_by_id: dict[str, DocumentChunk] = {}
        self._index: DeterministicHybridIndex | None = None

    @staticmethod
    def _retrieval_document(chunk: DocumentChunk) -> RetrievalDocument:
        metadata = chunk.metadata or {}
        return RetrievalDocument(
            chunk_id=chunk.chunk_id,
            text=chunk.content,
            unit_ids=tuple(metadata.get("unit_ids") or ()),
            pages=tuple(metadata.get("pages") or ()),
            token_count=int(metadata.get("token_count") or 0),
        )

    def build_keyword_index(self, chunks: List[DocumentChunk]) -> None:
        """Compatibility hook: the frozen index builds dense and BM25 together."""
        self.build_index(chunks)

    def invalidate_index(self) -> None:
        """Forget the lexical index so the next search rebuilds it.

        Called when the knowledge base changed underneath this pipeline --
        another session ingested or deleted a document. The pipeline that did
        the writing rebuilds its own index directly; every other pipeline for
        that knowledge base is told here, and rebuilds lazily on its next
        query rather than eagerly for a user who may never come back.
        """
        self._index = None
        self.chunks_list = []
        self._chunks_by_id = {}

    def build_index(
        self,
        chunks: List[DocumentChunk],
        document_embeddings: np.ndarray | None = None,
    ) -> None:
        ordered = sorted(chunks, key=lambda chunk: chunk.chunk_id)
        self.chunks_list = ordered
        self._chunks_by_id = {chunk.chunk_id: chunk for chunk in ordered}
        if not ordered:
            self._index = None
            return
        embeddings = (
            self.embedding_model.encode_documents([chunk.content for chunk in ordered])
            if document_embeddings is None
            else np.asarray(document_embeddings, dtype=np.float32)
        )
        bm25 = self.config["bm25"]
        rrf = self.config["rrf"]
        self._index = DeterministicHybridIndex(
            documents=[self._retrieval_document(chunk) for chunk in ordered],
            document_embeddings=embeddings,
            bm25_k1=bm25["k1"],
            bm25_b=bm25["b"],
            rrf_rank_constant=rrf["rank_constant"],
            dense_weight=rrf["dense_weight"],
            bm25_weight=rrf["bm25_weight"],
            candidate_pool_size=rrf["candidate_pool_size"],
        )

    def ensure_index(self) -> None:
        if self._index is None:
            self.build_index(self.vector_db.get_all_chunks())

    def hybrid_search(self, query: str, top_k: int = 5, *args, **kwargs):
        try:
            self.ensure_index()
            if self._index is None:
                return []
            query_vector = self.embedding_model.encode_queries([query])[0]
            hits = self._index.search(query, query_vector, top_k=top_k)
            results: List[RetrievalResult] = []
            for hit in hits:
                result = RetrievalResult(
                    chunk=self._chunks_by_id[hit.chunk_id],
                    score=hit.rrf_score,
                    retrieval_method="benchmark_aligned_rrf",
                    rank=hit.rank - 1,
                )
                result.dense_rank = hit.dense_rank
                result.bm25_rank = hit.bm25_rank
                result.dense_score = hit.dense_score
                result.bm25_score = hit.bm25_score
                results.append(result)
            return results
        except Exception as exc:
            raise RetrieverException(f"Benchmark-aligned retrieval failed: {exc}") from exc
