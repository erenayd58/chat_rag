"""The product's final retrieval profile: dense + lexical, fused by RRF.

Dense retrieval runs against the vectors the store already holds, embedded
at ingest by the configured OpenAI-compatible model (``qwen/qwen3-embedding-8b``
in the demo) and queried with the same model; the lexical leg is the frozen
deterministic BM25 with the Turkish diacritic fold the ``bm25_only`` profile
validated; the two rank lists are combined by reciprocal-rank fusion with
deterministic tie-breaking on chunk id. Standard and Deep Analysis documents
go through exactly this path -- only their chunk partition differs.

Safety before availability: the retriever reads the store's embedding
manifest and refuses the dense leg when the stored vectors were not produced
by the current model (or are a lexical profile's placeholders). It never
compares across spaces; it falls back to the lexical leg alone and says so in
``last_stats``, and the console shows that state with a re-index action.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import numpy as np

from amsc.retrieval_pipeline import DeterministicBM25

from components.embedding.index_manifest import (
    STATE_COMPATIBLE,
    STATE_EMPTY,
    index_status,
    read_manifest,
    write_manifest,
)
from core.exceptions import EmbeddingException, RetrieverException
from core.models import DocumentChunk, RetrievalResult

from .bm25_only_retriever import BM25_B, BM25_K1, fold_turkish

RRF_RANK_CONSTANT = 60
CANDIDATE_POOL_SIZE = 50


def _identity(embedding_model: Any) -> Dict[str, Any]:
    describe = getattr(embedding_model, "describe", None)
    if callable(describe):
        return dict(describe())
    name = embedding_model.get_name() if hasattr(embedding_model, "get_name") else str(embedding_model)
    dimension = None
    try:
        dimension = int(embedding_model.get_dimension())
    except Exception:
        pass
    from components.embedding.openai_compatible_embedding import embedding_fingerprint

    return {
        "provider": "local",
        "model": name,
        "endpoint": "",
        "dimension": dimension,
        "fingerprint": embedding_fingerprint("local", name, "", dimension),
    }


class HybridRRFRetriever:
    """Dense (stored vectors) + BM25 (frozen, folded) + RRF."""

    requires_document_embeddings = True

    def __init__(
        self,
        embedding_model: Any,
        vector_db: Any,
        *,
        store_path: Optional[str] = None,
        rank_constant: int = RRF_RANK_CONSTANT,
        candidate_pool_size: int = CANDIDATE_POOL_SIZE,
    ) -> None:
        self.embedding_model = embedding_model
        self.vector_db = vector_db
        self.store_path = store_path
        self.rank_constant = rank_constant
        self.candidate_pool_size = candidate_pool_size
        self.chunks_list: List[DocumentChunk] = []
        self._chunks_by_id: Dict[str, DocumentChunk] = {}
        self._chunks_by_position: Dict[tuple, DocumentChunk] = {}
        self._bm25: Optional[DeterministicBM25] = None
        self._index_status: Optional[Dict[str, Any]] = None
        self.last_stats: Dict[str, Any] = {}

    # ------------------------------------------------------------- config
    @property
    def config(self) -> dict:
        identity = _identity(self.embedding_model)
        return {
            "profile": "hybrid_rrf",
            "dense": {
                "provider": identity.get("provider"),
                "model": identity.get("model"),
                "dimension": identity.get("dimension"),
                "fingerprint": identity.get("fingerprint"),
                "space": "cosine",
            },
            "bm25": {"k1": BM25_K1, "b": BM25_B, "fold": "turkish_diacritics_v1"},
            "rrf": {
                "rank_constant": self.rank_constant,
                "candidate_pool_size": self.candidate_pool_size,
                "dense_weight": 1.0,
                "bm25_weight": 1.0,
            },
        }

    # ------------------------------------------------------ index status
    def refresh_index_status(self) -> Dict[str, Any]:
        """Re-read the manifest and the store; the answer is cached until
        the next build or write."""
        stored_dimension = None
        probe = getattr(self.vector_db, "_stored_dimension", None)
        if callable(probe):
            try:
                stored_dimension = probe()
            except Exception:
                stored_dimension = None
        try:
            stored_count = int(self.vector_db.count())
        except Exception:
            stored_count = len(self.chunks_list)
        self._index_status = index_status(
            manifest=read_manifest(self.store_path),
            identity=_identity(self.embedding_model),
            stored_dimension=stored_dimension,
            stored_count=stored_count,
        )
        return self._index_status

    @property
    def index_status(self) -> Dict[str, Any]:
        if self._index_status is None:
            self.refresh_index_status()
        return self._index_status or {}

    @property
    def dense_available(self) -> bool:
        return bool(self.index_status.get("dense_available"))

    def record_index(self, dimension: int) -> Dict[str, Any]:
        """The store was just written by the current model: say so."""
        if not self.store_path:
            self._index_status = None
            return {}
        try:
            count = int(self.vector_db.count())
        except Exception:
            count = len(self.chunks_list)
        manifest = write_manifest(
            self.store_path, _identity(self.embedding_model), dimension=dimension, chunk_count=count
        )
        self._index_status = None
        return manifest

    def can_accept_dense_writes(self) -> tuple:
        """``(ok, reason)`` -- whether new vectors may be added to the store
        without mixing embedding spaces."""
        status = self.refresh_index_status()
        if status["state"] in (STATE_EMPTY, STATE_COMPATIBLE):
            return True, ""
        return False, status["reason"]

    # ------------------------------------------------------------ indexes
    def build_index(self, chunks: List[DocumentChunk], *args: Any, **kwargs: Any) -> None:
        ordered = sorted(chunks, key=lambda chunk: chunk.chunk_id)
        self.chunks_list = ordered
        self._chunks_by_id = {chunk.chunk_id: chunk for chunk in ordered}
        self._chunks_by_position = {
            (chunk.doc_id, int(chunk.chunk_index)): chunk for chunk in ordered
        }
        self._bm25 = (
            DeterministicBM25([fold_turkish(chunk.content) for chunk in ordered], k1=BM25_K1, b=BM25_B)
            if ordered else None
        )
        self._index_status = None

    def build_keyword_index(self, chunks: List[DocumentChunk]) -> None:
        self.build_index(chunks)

    def ensure_index(self) -> None:
        if self._bm25 is None and not self.chunks_list:
            self.build_index(self.vector_db.get_all_chunks())

    def neighbor(self, chunk: DocumentChunk, offset: int) -> Optional[DocumentChunk]:
        """The adjacent chunk of the same document, if indexed."""
        self.ensure_index()
        return self._chunks_by_position.get((chunk.doc_id, int(chunk.chunk_index) + offset))

    def chunk_by_id(self, chunk_id: str) -> Optional[DocumentChunk]:
        self.ensure_index()
        return self._chunks_by_id.get(chunk_id)

    # ------------------------------------------------------------- search
    def _chunk_from_hit(self, hit: Dict[str, Any]) -> DocumentChunk:
        known = self._chunks_by_id.get(hit["chunk_id"])
        if known is not None:
            return known
        metadata = hit.get("metadata") or {}
        return DocumentChunk(
            chunk_id=hit["chunk_id"],
            content=hit.get("content", ""),
            doc_id=metadata.get("doc_id", ""),
            doc_title=metadata.get("doc_title", ""),
            chunk_index=metadata.get("chunk_index", 0),
            total_chunks=metadata.get("total_chunks", 0),
            section_title=metadata.get("section_title"),
            document_summary=metadata.get("document_summary"),
            metadata=metadata,
        )

    def vector_search(self, query: str, top_k: int = 10, **kwargs: Any) -> List[RetrievalResult]:
        if not self.dense_available:
            raise RetrieverException(
                "Dense retrieval is unavailable for this knowledge base: "
                + str(self.index_status.get("reason"))
            )
        try:
            self.ensure_index()
            vector = self.embedding_model.encode_queries([query])[0]
            hits = self.vector_db.query(np.asarray(vector, dtype=float).tolist(), top_k=top_k)
        except EmbeddingException:
            raise
        except Exception as exc:
            raise RetrieverException(f"Dense retrieval failed: {exc}") from exc
        results: List[RetrievalResult] = []
        for rank, hit in enumerate(hits):
            distance = hit.get("distance")
            score = 1.0 - float(distance) if distance is not None else 0.0
            result = RetrievalResult(
                chunk=self._chunk_from_hit(hit), score=score, retrieval_method="dense", rank=rank
            )
            result.dense_rank = rank + 1
            result.dense_score = score
            result.bm25_rank = None
            result.bm25_score = None
            results.append(result)
        return results

    def keyword_search(self, query: str, top_k: int = 10, **kwargs: Any) -> List[RetrievalResult]:
        try:
            self.ensure_index()
            if self._bm25 is None:
                return []
            scores = self._bm25.scores(fold_turkish(query))
            order = sorted(
                range(len(self.chunks_list)),
                key=lambda i: (-float(scores[i]), self.chunks_list[i].chunk_id),
            )
            results: List[RetrievalResult] = []
            for rank, index in enumerate(order[:top_k]):
                if float(scores[index]) <= 0.0:
                    break
                chunk = self.chunks_list[index]
                result = RetrievalResult(
                    chunk=chunk, score=float(scores[index]), retrieval_method="bm25", rank=rank
                )
                result.bm25_rank = rank + 1
                result.bm25_score = float(scores[index])
                result.dense_rank = None
                result.dense_score = None
                results.append(result)
            return results
        except RetrieverException:
            raise
        except Exception as exc:
            raise RetrieverException(f"BM25 retrieval failed: {exc}") from exc

    def hybrid_search(self, query: str, top_k: int = 5, *args: Any, **kwargs: Any) -> List[RetrievalResult]:
        """RRF over the dense and lexical candidate lists.

        Deterministic: equal fused scores are ordered by chunk id. When the
        dense leg is unavailable the lexical ranking is returned and
        ``last_stats`` records why -- the caller decides how to tell the user.
        """
        started = time.perf_counter()
        pool = max(int(self.candidate_pool_size), int(top_k))
        status = self.index_status
        dense_hits: List[RetrievalResult] = []
        dense_error: Optional[str] = None
        if status.get("dense_available"):
            try:
                dense_hits = self.vector_search(query, top_k=pool)
            except EmbeddingException as exc:
                # The embedding provider is down. Never substitute another
                # space; run lexical-only and report it.
                dense_error = str(exc)
        bm25_hits = self.keyword_search(query, top_k=pool)

        fused: Dict[str, Dict[str, Any]] = {}

        def add(hits: List[RetrievalResult], leg: str) -> None:
            for hit in hits:
                entry = fused.setdefault(
                    hit.chunk.chunk_id,
                    {"chunk": hit.chunk, "score": 0.0, "dense_rank": None, "bm25_rank": None,
                     "dense_score": None, "bm25_score": None},
                )
                entry["score"] += 1.0 / (self.rank_constant + (hit.rank + 1))
                if leg == "dense":
                    entry["dense_rank"], entry["dense_score"] = hit.dense_rank, hit.dense_score
                else:
                    entry["bm25_rank"], entry["bm25_score"] = hit.bm25_rank, hit.bm25_score

        add(dense_hits, "dense")
        add(bm25_hits, "bm25")
        ordered = sorted(fused.values(), key=lambda entry: (-entry["score"], entry["chunk"].chunk_id))

        results: List[RetrievalResult] = []
        for rank, entry in enumerate(ordered[:top_k]):
            result = RetrievalResult(
                chunk=entry["chunk"], score=float(entry["score"]),
                retrieval_method="hybrid_rrf" if dense_hits else "bm25_only",
                rank=rank,
            )
            result.dense_rank = entry["dense_rank"]
            result.bm25_rank = entry["bm25_rank"]
            result.dense_score = entry["dense_score"]
            result.bm25_score = entry["bm25_score"]
            results.append(result)

        self.last_stats = {
            "dense_used": bool(dense_hits),
            "dense_available": bool(status.get("dense_available")),
            "dense_unavailable_reason": (
                dense_error if dense_error else (None if status.get("dense_available") else status.get("reason"))
            ),
            "index_state": status.get("state"),
            "dense_hits": len(dense_hits),
            "bm25_hits": len(bm25_hits),
            "fused_candidates": len(fused),
            "returned": len(results),
            "candidate_pool_size": pool,
            "rrf_rank_constant": self.rank_constant,
            "latency_ms": round((time.perf_counter() - started) * 1000.0, 1),
        }
        return results
