"""Asking a question of a document's *analysis arms*, one method at a time.

This is the one thing the Viewer does that no other screen does. Everywhere
else a question searches a knowledge base's corpus -- one chunker, the one its
knowledge base was ingested with. Here the same question is put to the same
document chunked several ways, side by side, and what comes back is a
comparison of the chunkers rather than of the documents:

    document -> analysis arms (one per chunking method)
             -> one index per arm, built from the packaged rows
             -> the same question, the same retriever, the same context budget

Only the chunker differs between arms; everything downstream is shared. That
is the whole claim the comparison makes, and it is why the engine is the
library's (:mod:`amsc.viewer.chat.session`) rather than the console's own
retriever: the rows it indexes are the rows the frozen benchmark indexes, and
the BM25 fold, the fusion and the tie-breaks are the frozen ones.

What this module owns is the console side of that: which models the engine
answers with, when an arm has to be registered again, and the bounds the work
runs under. The models are the deployment's own -- the same embedding instance
and the same answer chain the console's chat uses, adapted to the engine's two
small protocols -- so a Viewer answer and a chat answer are not two
configurations quietly diverging.

Registration is by analysis identity, not by document id. A document whose
analysis was rebuilt, or that gained a method, has different rows under the
same id; re-registering drops the indexes built from the old ones rather than
answering out of them.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Optional, Sequence

import numpy as np

from amsc.viewer.chat.context import ContextSettings
from amsc.viewer.chat.index import RetrievalSettings
from amsc.viewer.chat.session import Catalog, ChatEngine

from components.query import query_scope
from components.viewer import methods as viewer_methods
from config import paths as data_paths
from core.exceptions import LLMException

from . import workspace
from .errors import InvalidRequest, NotFound

logger = logging.getLogger("RAG.analysis_query")

#: The pipeline key the Viewer's models are resolved under. Fixed rather than
#: per browser session: the engine outlives any one request, and a per-session
#: key would build a pipeline per browser to read two objects off it.
VIEWER_SESSION = "viewer-query"

#: What the measurement calls this traffic, so a Viewer comparison is
#: distinguishable from chat and from the Lab in the query telemetry.
QUERY_MODE = "viewer"

#: What one arm returns when the caller does not choose.
DEFAULT_TOP_K = 5


# --------------------------------------------------------------- the models
class _Embedder:
    """The console's embedding component, as the engine's provider protocol.

    Adapted rather than rebuilt, so that exactly one copy of a local model is
    resident in this process (``components.embedding`` shares it) and a
    gateway embedder honours the one configured endpoint, batch size and
    concurrency. ``model_id`` names the vector cache directory, so it is the
    configured model name and not a class name.
    """

    def __init__(self, embedding: Any, model_id: str):
        self._embedding = embedding
        self._model_id = model_id

    @property
    def model_id(self) -> str:
        return self._model_id

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        try:
            vectors = self._embedding.encode(list(texts))
        except Exception as error:  # noqa: BLE001
            # The engine's contract for "no dense leg right now" is
            # RuntimeError: it then falls back to BM25 for this index or this
            # question and says so on the response, which is what a reader
            # should get when the embedding endpoint is down.
            raise RuntimeError(str(error)) from error
        return np.asarray(vectors, dtype=np.float32)


class _Answerer:
    """The console's configured answer chain, as the engine's chat protocol."""

    def __init__(self, llm: Any):
        self._llm = llm

    @property
    def model_id(self) -> str:
        return self._llm.get_model_name()

    def chat(self, system: str, user: str) -> tuple[str, dict[str, Any]]:
        try:
            text = self._llm.generate(
                [{"role": "system", "content": system},
                 {"role": "user", "content": user}],
                temperature=0.0,
                max_tokens=900,
            )
        except LLMException as error:
            raise RuntimeError(str(error)) from error
        except Exception as error:  # noqa: BLE001
            raise RuntimeError(f"{type(error).__name__}: {error}") from error
        # The engine records usage verbatim; this chain reports none, and an
        # empty mapping is the truthful answer to "how many tokens".
        return text or "", {}


# --------------------------------------------------------------- the engine
_engine: ChatEngine | None = None
#: What is registered, and out of which analysis: ``doc_id -> (content_id, methods)``.
_registered: dict[str, tuple[str, tuple[str, ...]]] = {}
_lock = threading.RLock()


def _build_engine(services) -> ChatEngine:
    settings = services.settings
    pipeline = services.get_pipeline(VIEWER_SESSION, None)
    embedder = None
    if (getattr(settings, "retrieval_profile", "") or "") != "bm25_only":
        from amsc.retrieval.embeddings import CachedEmbeddings

        embedder = CachedEmbeddings(
            _Embedder(pipeline.embedding_model, settings.embedding_model_name),
            data_paths.embedding_cache(),
        )
    return ChatEngine(
        catalog=Catalog(documents={}),
        retrieval=RetrievalSettings.from_mapping({"top_k": DEFAULT_TOP_K}),
        context=ContextSettings.from_mapping({
            "max_context_tokens": settings.context_max_tokens,
            "max_sources": settings.context_max_sources,
            "expansion": {"enabled": settings.context_expand_neighbors},
        }),
        embedder=embedder,
        answerer=_Answerer(pipeline.llm_model),
    )


def engine(services) -> ChatEngine:
    """The one engine this process holds, built on first use."""
    global _engine
    with _lock:
        if _engine is None:
            _engine = _build_engine(services)
        return _engine


def reset() -> None:
    """Drop the engine and everything registered in it.

    For tests, and for an operator who has reconfigured the models: the
    indexes are derived data, and rebuilding one costs what it cost the first
    time and nothing else.
    """
    global _engine
    with _lock:
        _engine = None
        _registered.clear()


# ------------------------------------------------------------ registration
def _register(services, document_id: str) -> tuple[ChatEngine, list[str]]:
    """Index this document's ready arms, unless they are already indexed.

    Keyed by the analysis's own identity -- its content id and the exact set
    of methods that are ready -- so a rebuilt or extended analysis is
    registered again, and the indexes built from the previous rows are
    dropped rather than answered out of.
    """
    built = engine(services)
    found = workspace.chunk_rows(document_id)
    arms = found.get("arms") or {}
    if not arms:
        raise NotFound(f"no packaged chunks for {document_id!r}")
    identity = (str(found.get("key") or ""), tuple(sorted(arms)))
    with _lock:
        if _registered.get(document_id) != identity:
            built.register_live(document_id, str(found.get("label") or document_id), arms)
            _registered[document_id] = identity
    return built, [method for method in viewer_methods.ORDER if method in arms]


def available_methods(services, document_id: str) -> list[str]:
    """The methods this document can be asked about now, in product order."""
    return _register(services, document_id)[1]


# ------------------------------------------------------------- the use case
def ask(services, *, document_id: str, question: str, session_id: str,
        methods: Optional[Sequence[str]] = None, top_k: int = DEFAULT_TOP_K,
        answer: bool = True) -> dict:
    """One question, over one document, through each named chunking method.

    Validation happens before admission on purpose: a question that is not a
    question must not consume a slot somebody else could have used.

    The result is the engine's comparison shape whether one method was named
    or four. A comparison of one is still a comparison, and a client that
    branches on the count has two rendering paths where it needs one.
    """
    question = (question or "").strip()
    # The question is the user's text; the log carries its size, not it.
    logger.info(f"Analysis query over {document_id}: {len(question)} chars")
    if not question:
        raise InvalidRequest("Question is required")

    unknown = [m for m in (methods or ()) if m not in viewer_methods.METHODS]
    if unknown:
        raise InvalidRequest(
            f"unknown chunking method {unknown[0]!r}",
            details={"supported": list(viewer_methods.ORDER)},
        )

    built, ready = _register(services, document_id)
    chosen = [m for m in ready if m in set(methods)] if methods else ready
    if not chosen:
        # Named methods, none of them built for this upload. Not the same
        # refusal as an unknown name, and the difference is what tells a
        # client to wait rather than to ask for something else.
        raise NotFound(
            f"none of the requested methods are ready for {document_id!r}",
            details={"ready_methods": ready},
        )

    with query_scope(
        services.query_admission, timeout_seconds=services.settings.query_timeout,
        kb_id=None, mode=QUERY_MODE, session_id=session_id,
    ) as scope:
        result = built.compare(document_id, question,
                               arms=chosen, top_k=top_k, answers=answer)
    result["timing"] = scope.timing()
    result["methods"] = chosen
    return result
