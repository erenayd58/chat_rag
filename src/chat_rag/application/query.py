"""Answering a question, and everything else that runs under the query bounds.

Three bounds, all of them owned by :mod:`components.query.limits` and none of
them re-implemented here:

* **admission** -- at most ``QUERY_MAX_ACTIVE`` callers may be inside a query
  at once, and one more is refused immediately rather than queued, because a
  queued query holds the very thread the bound exists to keep free;
* **the answer budget** -- at most ``ANSWER_MAX_INFLIGHT`` answer-model calls
  in flight process-wide, waited for only as long as the query has;
* **the deadline** -- ``QUERY_TIMEOUT`` seconds in total, checked at every
  outbound seam.

:func:`bounded` is the same three (minus the answer budget, which retrieval
never needs) for the read-only searches: they do the front half of a query on
the caller's thread -- embed the question, search the store, build the lexical
index if this pipeline has not built it yet -- so under no limit at all they
would be a way around the bound rather than a lighter path.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any, Iterator, Optional

from chat_rag.components.observability import telemetry as T
from chat_rag.components.query import query_scope
from chat_rag.core.exceptions import LLMException

from .errors import ApplicationError, InvalidRequest, ProcessingFailed, Unavailable
from .services import in_engine

logger = logging.getLogger("RAG.query")


@in_engine
def answer(services, *, question: str, session_id: str, kb_id: Optional[str] = None,
           top_k: int = 5, temperature: float = 0.3, max_tokens: int = 500) -> dict:
    """Answer one question over one knowledge base.

    Validation happens **before** admission on purpose: a question that is not
    a question must not consume a slot somebody else could have used.
    """
    question = (question or '').strip()
    # The question is the user's text; the log carries its size, not it.
    logger.info(f"Received query request: {len(question)} chars")
    if not question:
        logger.warning("Empty question received")
        raise InvalidRequest('Question is required')

    try:
        with query_scope(
            services.query_admission, timeout_seconds=services.settings.query_timeout,
            kb_id=kb_id, mode=services.settings.retrieval_profile, session_id=session_id,
        ) as scope:
            # Leased, not merely fetched: a burst of other sessions must not
            # be able to evict this pipeline and close its store mid-query.
            with services.lease_pipeline(session_id, kb_id) as pipeline:
                logger.info("Processing query through RAG pipeline")
                result = pipeline.query(question=question, top_k=top_k,
                                        temperature=temperature, max_tokens=max_tokens)
    except LLMException as missing:
        # Retrieval and ingestion do not need a language model, so a missing
        # one is a temporarily unavailable feature rather than a broken
        # application. Converted outside the scope, which has already counted
        # it as a provider failure under its own name.
        logger.warning(f"Generation unavailable: {missing}")
        raise Unavailable(str(missing), details={'generation_unavailable': True}) from missing

    logger.info("Query processed successfully")
    logger.debug(
        f"Answer length: {len(result['answer'])} chars, Sources: {len(result['sources'])}"
    )
    metadata = dict(result['metadata'] or {})
    metadata['query'] = scope.timing()
    return {'answer': result['answer'], 'sources': result['sources'], 'metadata': metadata}


class Bounded:
    """What a bounded read-only search is handed: its pipeline and its scope."""

    def __init__(self, pipeline: Any, scope: Any):
        self.pipeline = pipeline
        self.scope = scope


@contextmanager
def bounded(services, *, mode: str, session_id: str,
            kb_id: Optional[str] = None) -> Iterator[Bounded]:
    """Run a read-only retrieval under the query path's bounds.

    Three things, and exactly the three the query path has: admission (the
    same counter, so the bound is on threads doing retrieval whichever entry
    point asked), the query deadline (which is what bounds the embedding-slot
    wait, since the embedding wrapper reads the same guard), and the
    measurement under this caller's own ``mode``, so Lab traffic is
    distinguishable from chat. No answer budget, because no answer-model call
    is made here.

    A refusal raised inside the block leaves the scope *cleanly* and is
    re-raised after it: an invalid query is not a failed query, and counting
    it as one would make the metrics lie. A server fault is marked failed
    first, for the same reason in the other direction.
    """
    refusal: list[BaseException] = []
    with services.activate(), query_scope(
        services.query_admission, timeout_seconds=services.settings.query_timeout,
        kb_id=kb_id, mode=mode, session_id=session_id,
    ) as scope:
        with services.lease_pipeline(session_id, kb_id) as pipeline:
            # These callers are retrieval from end to end, so the whole block
            # is the retrieve stage and their traffic lands in the same
            # per-stage summary.
            with T.stage(T.RETRIEVE):
                try:
                    yield Bounded(pipeline, scope)
                except ApplicationError as refused:
                    refusal.append(refused)
        if refusal and isinstance(refusal[0], ProcessingFailed):
            scope.failed(f"the endpoint answered 500: {refusal[0]}")
    if refusal:
        raise refusal[0]
