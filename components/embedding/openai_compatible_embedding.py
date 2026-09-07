"""Dense embeddings from an OpenAI-compatible ``/embeddings`` endpoint.

The product's final retrieval embedding is ``qwen/qwen3-embedding-8b``
reached through OpenRouter, but nothing here names a vendor: the model, the
endpoint and the *name* of the environment variable holding the key are
configuration. The transport and the per-text cache are amsc's own
(``amsc.retrieval.embeddings``), so the console and the research viewer embed with
one implementation.

Two rules this class exists to keep:

* documents and queries are embedded by the same model in the same space --
  ``encode_documents`` and ``encode_queries`` are one call path;
* an index is only ever compared against vectors from the model that built
  it. ``fingerprint`` identifies the space; the store's manifest records it
  (see :mod:`components.embedding.index_manifest`).

The key is read by the provider at request time and never held here.
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

from amsc.retrieval.embeddings import (
    DEFAULT_EMBEDDING_ENDPOINT,
    CachedEmbeddings,
    OpenAICompatibleEmbeddingProvider,
)

from core.exceptions import EmbeddingException

from .base import BaseEmbedding

PROVIDER_KIND = "openai_compatible"


def embedding_fingerprint(provider: str, model: str, endpoint: str = "", dimensions: Optional[int] = None) -> str:
    """A short, stable id of an embedding *space*: provider, model, endpoint
    and any requested output size. Two stores with equal fingerprints hold
    comparable vectors; nothing else does."""
    raw = f"{provider}|{model}|{endpoint}|{dimensions or ''}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


class ResilientBatches:
    """Batch an embedding transport and survive a rejected batch.

    Gateways occasionally refuse one request out of a long series (an
    HTTP 400 that the same texts do not reproduce a minute later). Rather
    than failing a whole re-index for it, a rejected batch is retried once
    after a pause and then embedded text by text, so a genuinely bad text is
    named and everything else still gets its vector. Batches are sent one
    after another; a re-index is not latency-critical and a burst of
    parallel requests is what the gateway objected to.
    """

    #: Wait before retrying a rejected batch (class-level so tests can zero it).
    pause_seconds: float = 1.5

    def __init__(self, provider: Any, *, batch_size: int = 32, pause_seconds: Optional[float] = None) -> None:
        self.provider = provider
        self.batch_size = max(1, int(batch_size))
        if pause_seconds is not None:
            self.pause_seconds = pause_seconds
        self.batch_retries = 0
        self.single_fallbacks = 0

    @property
    def model_id(self) -> str:
        return self.provider.model_id

    @property
    def calls(self) -> int:
        return int(getattr(self.provider, "calls", 0))

    def _embed_batch(self, texts: List[str]) -> np.ndarray:
        try:
            return self.provider.embed(texts)
        except RuntimeError as first:
            if len(texts) == 1:
                raise
            import time

            time.sleep(self.pause_seconds)
            self.batch_retries += 1
            try:
                return self.provider.embed(texts)
            except RuntimeError:
                pass
            self.single_fallbacks += 1
            rows = []
            for index, text in enumerate(texts):
                try:
                    rows.append(self.provider.embed([text]))
                except RuntimeError as exc:
                    raise RuntimeError(
                        f"{exc} (text {index + 1} of a batch of {len(texts)}, "
                        f"{len(text)} chars; batch error: {first})"
                    ) from exc
            return np.vstack(rows)

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        texts = list(texts)
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        return np.vstack([
            self._embed_batch(texts[start:start + self.batch_size])
            for start in range(0, len(texts), self.batch_size)
        ])


class OpenAICompatibleEmbedding(BaseEmbedding):
    """``BaseEmbedding`` over amsc's OpenAI-compatible embedding provider."""

    provider_kind = PROVIDER_KIND

    def __init__(
        self,
        model: str,
        *,
        endpoint: str = DEFAULT_EMBEDDING_ENDPOINT,
        api_key_env: str = "OPENROUTER_API_KEY",
        batch_size: int = 32,
        timeout_seconds: float = 120.0,
        dimensions: Optional[int] = None,
        concurrency: int = 1,
        cache_dir: Optional[str] = None,
        provider: Any = None,
    ) -> None:
        if not model:
            raise EmbeddingException("EMBEDDING_MODEL is required for the openai_compatible embedding provider")
        self.model = model
        self.endpoint = endpoint
        self.api_key_env = api_key_env
        self.requested_dimensions = dimensions
        transport = provider or OpenAICompatibleEmbeddingProvider(
            model,
            endpoint=endpoint,
            api_key_env=api_key_env,
            batch_size=batch_size,
            timeout_seconds=timeout_seconds,
            dimensions=dimensions,
            concurrency=max(1, int(concurrency)),
        )
        # Every request this makes counts against the process-wide embedding
        # budget (EMBEDDING_MAX_INFLIGHT). These are provider-facing calls
        # exactly like Deep Analysis's, and they multiply from more places:
        # every ingest embeds its chunks, a re-index embeds a whole knowledge
        # base from a request thread, and a query embeds itself. The wrapper
        # sits on the transport rather than on this class because that is the
        # level at which one call is one HTTP request -- ResilientBatches
        # below hands it exactly one batch at a time.
        from components.ingest.limits import LimitedEmbeddingTransport, embedding_budget

        self._transport = transport
        self._limited = LimitedEmbeddingTransport(transport, embedding_budget())
        # Batching and the rejected-batch fallback live here; the transport
        # underneath only ever sees one batch at a time.
        self._provider = ResilientBatches(self._limited, batch_size=batch_size)
        self._cache = CachedEmbeddings(self._provider, cache_dir)
        self._dimension: Optional[int] = dimensions
        self.last_usage = None

    # ------------------------------------------------------------ identity
    @property
    def fingerprint(self) -> str:
        return embedding_fingerprint(self.provider_kind, self.model, self.endpoint, self.requested_dimensions)

    def describe(self) -> Dict[str, Any]:
        """Provider, model, dimension and fingerprint. Never the key."""
        return {
            "provider": self.provider_kind,
            "model": self.model,
            "endpoint": self.endpoint,
            "api_key_env": self.api_key_env,
            "dimension": self._dimension,
            "fingerprint": self.fingerprint,
            # This provider is remote, so its calls are budgeted. A local
            # sentence-transformers model reports no budget because it makes
            # no request; its cost is CPU on the worker that asked for it.
            "budgeted": True,
        }

    def get_name(self) -> str:
        return f"{self.provider_kind}:{self.model}"

    def get_dimension(self) -> int:
        """The vector width; discovered from the first embedding when the
        model does not declare one (one small request, cached)."""
        if self._dimension is None:
            self._embed(["dimension probe"])
        return int(self._dimension or 0)

    # ------------------------------------------------------------ encoding
    def _embed(self, texts: Sequence[str]) -> np.ndarray:
        try:
            vectors = self._cache.embed(list(texts))
        except RuntimeError as exc:
            raise EmbeddingException(f"Embedding request failed: {exc}") from exc
        except EmbeddingException:
            raise
        except Exception as exc:
            raise EmbeddingException(f"Embedding failed: {exc}") from exc
        self.last_usage = self._cache.last_usage
        if vectors.ndim == 2 and vectors.shape[0]:
            self._dimension = int(vectors.shape[1])
        return vectors

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray:
        return self._embed(texts)

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        # Same model, same space, same text treatment as the documents: a
        # query instruction prefix would move queries into a different
        # region than the chunks and silently cost recall.
        return self._embed(texts)

    def encode(
        self,
        texts: Union[str, List[str]],
        convert_to_tensor: bool = False,
        **kwargs: Any,
    ) -> Union[np.ndarray, List[np.ndarray]]:
        if isinstance(texts, str):
            return self._embed([texts])[0]
        return self._embed(list(texts))
