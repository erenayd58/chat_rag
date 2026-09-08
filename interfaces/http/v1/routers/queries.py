"""`/api/v1/queries`, `/api/v1/searches` and `/api/v1/analysis-queries`.

Asking, looking, and comparing.

The first two are one shape apart. A **query** retrieves and then asks the
answer model for a cited answer; a **search** stops after retrieval and returns
the ranked chunks. They are separate resources rather than one endpoint with a
flag because they cost different things and refuse for different reasons: only
a query can run out of answer-model capacity, and only a query can come back
ungrounded.

The third asks a different question of a different corpus. Both of the first
two search a knowledge base -- one document set, chunked the one way its
knowledge base ingests. An **analysis query** searches one document's *analysis
arms*: the same document chunked several ways, one index per chunking method,
with only the chunker differing. It is therefore the only thing on this
contract that compares chunkers, and it is what the Viewer runs on.

All three run under the same bounds: admission, so a burst of retrieval cannot
take every request thread, and the query deadline. A refusal is **503**
``overloaded`` with a ``Retry-After``, or **504** ``timeout``; neither is
queued, and neither is decided here -- ``interfaces.http.v1.errors`` holds the
whole table.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Body

from application import analysis_query as analysis_use_case
from application import chunks as search_use_case
from application import query as use_case
from application.errors import InvalidRequest

from ..dependencies import Container, FreshSessionId
from ..schemas import (
    AnalysisQueryRequest, AnalysisQueryResult, Answer, QueryRequest, ScoredChunk,
    SearchRequest, SearchResults,
)

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


@router.post("/analysis-queries", response_model=AnalysisQueryResult, tags=["analysis"],
             summary="One question, over one document, through each chunking method")
def analysis_query(services: Container, session: FreshSessionId,
                   payload: Annotated[AnalysisQueryRequest,
                                      Body(default_factory=AnalysisQueryRequest)]
                   ) -> AnalysisQueryResult:
    """The comparison a knowledge-base query cannot make.

    ``/queries`` searches a corpus that was chunked one way -- the way its
    knowledge base ingests. This puts the same question to the same document
    chunked several ways, over indexes built from the analysis's own packaged
    rows, with only the chunker differing between arms. What comes back is
    therefore a comparison of chunkers, and it is the Viewer's reason to
    exist.

    It runs under the query path's own admission and deadline: retrieval and
    an answer-model call per arm is not a lighter thing than a query, and
    leaving it outside the bound would make it the way around it.

    An arm that could not be answered does not fail the request -- it comes
    back with its ``status`` and its sources -- because in a comparison the
    other arms are still the answer.
    """
    found = analysis_use_case.ask(
        services,
        document_id=payload.document_id,
        question=payload.question,
        session_id=session,
        methods=_methods(payload.methods),
        top_k=max(1, min(MAX_SEARCH_RESULTS, payload.top_k)),
        answer=payload.answer,
    )
    return AnalysisQueryResult.of(found, document_id=payload.document_id)


def _methods(selection) -> list[str] | None:
    """The methods a request named, from either spelling.

    A list, or one comma-separated string: both are what a registry's keys
    arrive as, and the same reading as ``AnalysisMethods`` on the upload path.
    ``None`` means "every ready one", which is not the same as an empty list.
    """
    if selection is None:
        return None
    if isinstance(selection, str):
        return [part.strip() for part in selection.replace(";", ",").split(",") if part.strip()]
    return [str(part).strip() for part in selection if str(part).strip()]
