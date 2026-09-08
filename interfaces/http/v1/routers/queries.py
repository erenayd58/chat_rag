"""`/api/v1/queries` and `/api/v1/searches` -- asking, and looking.

Two verbs that are one shape apart. A **query** retrieves and then asks the
answer model for a cited answer; a **search** stops after retrieval and returns
the ranked chunks. They are separate resources rather than one endpoint with a
flag because they cost different things and refuse for different reasons: only
a query can run out of answer-model capacity, and only a query can come back
ungrounded.

Both run under the same bounds (``application.query``): admission, so a burst
of retrieval cannot take every request thread, and the query deadline. A
refusal is **503** ``overloaded`` with a ``Retry-After``, or **504**
``timeout``; neither is queued, and neither is decided here --
``interfaces.http.v1.errors`` holds the whole table.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Body

from application import chunks as search_use_case
from application import query as use_case
from application.errors import InvalidRequest

from ..dependencies import Container, FreshSessionId
from ..schemas import Answer, QueryRequest, ScoredChunk, SearchRequest, SearchResults

router = APIRouter(tags=["queries"])

#: The retrieval methods a search may name. Which of them a given knowledge
#: base can actually serve is a capability question, answered by
#: ``/api/v1/meta/retrieval-methods`` and enforced by the use case.
SEARCH_METHODS = ("hybrid", "bm25", "vector")
#: What a search does when the caller does not choose.
DEFAULT_SEARCH_METHOD = SEARCH_METHODS[0]
#: The most rows one search returns, for the reason a page has a ceiling.
MAX_SEARCH_RESULTS = 200


@router.post("/queries", response_model=Answer,
             summary="Answer one question over one knowledge base")
def ask(services: Container, session: FreshSessionId,
        payload: Annotated[QueryRequest, Body(default_factory=QueryRequest)]) -> Answer:
    """The answer carries its citations -- the sources it was given, each
    saying whether the answer actually used it -- and ``grounded``, which is
    false when the model answered without citing any of them. A client that
    shows the answer without those two cannot tell an answer from a guess."""
    result = use_case.answer(
        services,
        question=payload.question,
        session_id=session,
        kb_id=payload.knowledge_base_id,
        top_k=payload.top_k,
        temperature=payload.temperature,
        max_tokens=payload.max_tokens,
    )
    return Answer.of(result, knowledge_base_id=payload.knowledge_base_id)


@router.post("/searches", response_model=SearchResults,
             summary="Retrieval without an answer: the ranked chunks")
def search(services: Container, session: FreshSessionId,
           payload: Annotated[SearchRequest, Body(default_factory=SearchRequest)]
           ) -> SearchResults:
    """A method the configured retriever cannot serve is refused with **400**
    and the reason, rather than surfacing a retriever exception as a fault --
    a lexical-only profile has no vectors, and saying so is the answer.

    The refusal is raised here rather than declared as a schema enum because
    it carries the supported list, which is what a picker needs and what a
    validation error does not have.
    """
    method = (payload.method or DEFAULT_SEARCH_METHOD).strip().lower()
    if method not in SEARCH_METHODS:
        raise InvalidRequest(
            f"unknown retrieval method {method!r}",
            details={"supported": list(SEARCH_METHODS)},
        )

    limit = max(1, min(MAX_SEARCH_RESULTS, payload.limit))
    found = search_use_case.experiment_search(
        services,
        query=payload.query,
        method=method,
        kb_id=payload.knowledge_base_id,
        session_id=session,
        top_k=limit,
    )
    items = [ScoredChunk.ranked(row, method=method) for row in found["chunks"]]
    return SearchResults.of(items, offset=0, limit=limit, total=len(items),
                            method=found["retrieval_method"],
                            knowledge_base_id=payload.knowledge_base_id)
