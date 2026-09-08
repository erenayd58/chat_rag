"""Asking, and looking.

A **query** retrieves and then asks the answer model for a cited answer; a
**search** stops after retrieval and returns the ranked chunks. They are
separate resources rather than one endpoint with a flag because they cost
different things and refuse for different reasons: only a query can run out of
answer-model capacity, and only a query can come back ungrounded.

Both are POSTs with a request body, and both create nothing. A question is
user text of unbounded length, it should not land in an access log, and it
should not be cached by anything in between -- which is what a query string
would invite.
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import Field

from .common import Schema


class QueryRequest(Schema):
    """One question, over one knowledge base."""

    question: str = ""
    knowledge_base_id: Optional[str] = None
    top_k: int = 5
    temperature: float = 0.3
    max_tokens: int = 500


class SearchRequest(Schema):
    """Retrieval without an answer.

    ``method`` is validated against what this deployment offers rather than by
    a schema enum, so a rejected value is answered with the supported list in
    ``details`` -- which is what a picker needs and what an enum's validation
    error does not carry.
    """

    query: str = ""
    knowledge_base_id: Optional[str] = None
    method: Optional[str] = None
    limit: int = 20


class Citation(Schema):
    """One source an answer was allowed to use, and whether it used it.

    ``label`` is what the answer cites (``[S1]``), ``used`` says whether the
    text actually cited it, and ``content`` is the chunk verbatim so a client
    can show the passage rather than a summary of it.
    """

    label: Optional[str] = None
    chunk_id: Optional[str] = None
    document_id: Optional[str] = None
    document: Optional[str] = None
    section: Optional[str] = None
    pages: list[Any] = Field(default_factory=list)
    chunking_mode: Optional[str] = None
    used: bool = False
    score: Optional[float] = None
    content: str = ""

    @classmethod
    def of(cls, source: dict) -> "Citation":
        return cls(
            label=source.get("label"),
            chunk_id=source.get("chunk_id"),
            document_id=source.get("doc_id"),
            document=source.get("document"),
            section=source.get("heading") or source.get("section"),
            pages=source.get("pages") or [],
            chunking_mode=source.get("chunking_mode"),
            used=bool(source.get("used")),
            score=source.get("score"),
            content=source.get("content") or source.get("content_preview") or "",
        )


class AnswerTiming(Schema):
    """What the query cost, stage by stage."""

    query_id: Optional[str] = None
    total_seconds: Optional[float] = None
    stages: dict[str, Any] = Field(default_factory=dict)


class Answer(Schema):
    """One answered question.

    ``grounded`` is true when the answer actually cited at least one of the
    sources it was given. A client showing the answer without it cannot tell
    an answer from a guess.

    ``diagnostics`` is the pipeline's own metadata, passed through and
    explicitly not contractual: it is how a retrieval change is investigated,
    and freezing it would freeze the internals this contract exists to leave
    free.
    """

    answer: str = ""
    citations: list[Citation] = Field(default_factory=list)
    knowledge_base_id: Optional[str] = None
    retrieval_method: Optional[str] = None
    grounded: bool = False
    timing: AnswerTiming
    diagnostics: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def of(cls, result: dict, *, knowledge_base_id: Optional[str]) -> "Answer":
        metadata: dict[str, Any] = result.get("metadata") or {}
        generation = metadata.get("answer") or {}
        timing = metadata.get("query") or {}
        return cls(
            answer=result.get("answer") or "",
            citations=[Citation.of(source) for source in (result.get("sources") or [])],
            knowledge_base_id=knowledge_base_id,
            retrieval_method=metadata.get("retrieval_method"),
            grounded=bool(generation.get("grounded")),
            timing=AnswerTiming(
                query_id=timing.get("query_id"),
                total_seconds=timing.get("total_seconds"),
                stages=timing.get("stages") or {},
            ),
            diagnostics=metadata,
        )
