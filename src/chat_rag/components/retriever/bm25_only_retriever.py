"""BM25-only retrieval profile.

Validated as the strongest low-cost configuration on the holdout document and
on an independent validation set drawn from a region that gold barely covered.
It loads no embedding model and stores no vectors, so ingestion and search cost
drop to the lexical index alone.

The BM25 implementation is the frozen deterministic one from ``amsc``; nothing
about tokenisation or scoring is re-tuned here.
"""

from __future__ import annotations

from typing import Any, List

import numpy as np

from amsc.retrieval.pipeline import DeterministicBM25

from chat_rag.core.exceptions import RetrieverException
from chat_rag.core.models import DocumentChunk, RetrievalResult

BM25_K1 = 1.5
BM25_B = 0.75

# Search-representation only. Stored chunk text is never modified; the same
# fold is applied to documents at index time and to the query at search time,
# so a query typed without Turkish diacritics still matches. Deliberately not
# stemming: this is a reversible character fold, nothing morphological.
_TURKISH_FOLD = str.maketrans({
    "ç": "c", "Ç": "c",
    "ğ": "g", "Ğ": "g",
    "ı": "i", "I": "i", "İ": "i", "i": "i",
    "ö": "o", "Ö": "o",
    "ş": "s", "Ş": "s",
    "ü": "u", "Ü": "u",
    "â": "a", "Â": "a", "î": "i", "Î": "i", "û": "u", "Û": "u",
})


def fold_turkish(text: str) -> str:
    """Diacritic-insensitive search representation for Turkish text."""
    return text.translate(_TURKISH_FOLD).lower()


class NullEmbedding:
    """Placeholder embedding that must never be used.

    The BM25-only profile is defined by the absence of a dense leg. Any call
    here means something silently reintroduced embeddings, so it fails loudly
    instead of quietly loading a model.
    """

    def get_name(self) -> str:
        return "NullEmbedding"

    def get_dimension(self) -> int:
        return 0

    def encode(self, texts, convert_to_tensor: bool = False, **kwargs):
        raise RetrieverException(
            "bm25_only profile does not compute embeddings; "
            "switch RETRIEVAL_PROFILE to use a dense leg"
        )

    encode_documents = encode
    encode_queries = encode


class BM25OnlyRetriever:
    """Lexical-only retriever over the stored chunks."""

    #: This retriever has no dense leg, so ingestion must not compute or store
    #: document vectors for it. Callers branch on this capability rather than
    #: on the profile name.
    requires_document_embeddings = False

    def __init__(self, embedding_model: Any, vector_db: Any) -> None:
        self.embedding_model = embedding_model
        self.vector_db = vector_db
        self.chunks_list: List[DocumentChunk] = []
        self._chunks_by_id: dict[str, DocumentChunk] = {}
        self._bm25: DeterministicBM25 | None = None

    @property
    def config(self) -> dict:
        return {
            "bm25": {"k1": BM25_K1, "b": BM25_B, "fold": "turkish_diacritics_v1"},
            "dense": None,
        }

    def build_index(self, chunks: List[DocumentChunk], *args, **kwargs) -> None:
        ordered = sorted(chunks, key=lambda chunk: chunk.chunk_id)
        self.chunks_list = ordered
        self._chunks_by_id = {chunk.chunk_id: chunk for chunk in ordered}
        if not ordered:
            self._bm25 = None
            return
        self._bm25 = DeterministicBM25(
            [fold_turkish(chunk.content) for chunk in ordered],
            k1=BM25_K1,
            b=BM25_B,
        )

    def build_keyword_index(self, chunks: List[DocumentChunk]) -> None:
        self.build_index(chunks)

    def ensure_index(self) -> None:
        if self._bm25 is None:
            self.build_index(self.vector_db.get_all_chunks())

    def invalidate_index(self) -> None:
        """Forget the lexical index so the next search rebuilds it.

        Called when the knowledge base changed underneath this pipeline --
        another session ingested or deleted a document. The pipeline that did
        the writing rebuilds its own index directly; every other pipeline for
        that knowledge base is told here, and rebuilds lazily on its next
        query rather than eagerly for a user who may never come back.
        """
        self._bm25 = None
        self.chunks_list = []
        self._chunks_by_id = {}

    def hybrid_search(self, query: str, top_k: int = 5, *args, **kwargs) -> List[RetrievalResult]:
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
                chunk = self.chunks_list[index]
                result = RetrievalResult(
                    chunk=chunk,
                    score=float(scores[index]),
                    retrieval_method="bm25_only",
                    rank=rank,
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
            raise RetrieverException(f"BM25-only retrieval failed: {exc}") from exc

    # Compatibility shims for callers that expect the legacy retriever surface.
    def keyword_search(self, query: str, top_k: int = 5, **kwargs) -> List[RetrievalResult]:
        return self.hybrid_search(query, top_k=top_k)

    def vector_search(self, *args, **kwargs) -> List[RetrievalResult]:
        raise RetrieverException("bm25_only profile has no dense leg")
