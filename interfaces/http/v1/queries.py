"""`/api/v1/queries` and `/api/v1/searches` -- asking, and looking.

Two verbs that are one shape apart. A **query** retrieves and then asks the
answer model for a cited answer; a **search** stops after retrieval and returns
the ranked chunks. They are separate resources rather than one endpoint with a
flag because they cost different things and refuse for different reasons: only
a query can run out of answer-model capacity, and only a query can come back
ungrounded.

Both are POSTs, and both create nothing. A question is a request body, not a
query string: it is user text of unbounded length, it should not land in an
access log, and it should not be cached by anything in between.

Both run under the same bounds (``application.query``): admission, so a burst
of retrieval cannot take every request thread, and the query deadline. A
refusal is **503** ``overloaded`` with a ``Retry-After``, or **504**
``timeout``; neither is queued.
"""

from __future__ import annotations

from flask import Blueprint, request

from application import chunks as search_use_case
from application import query as use_case
from application.errors import InvalidRequest

from ..context import fresh_session_id, services
from . import resources
from .envelope import collection, install, resource

bp = Blueprint('v1_queries', __name__)
install(bp)

#: The retrieval methods a search may name. Which of them a given knowledge
#: base can actually serve is a capability question, answered by
#: ``/api/v1/meta/retrieval-methods`` and enforced by the use case.
SEARCH_METHODS = ("hybrid", "bm25", "vector")


@bp.route('/queries', methods=['POST'])
def ask():
    """Answer one question over one knowledge base.

    The answer carries its citations -- the sources it was given, each saying
    whether the answer actually used it -- and ``grounded``, which is false
    when the model answered without citing any of them. A client that shows
    the answer without those two cannot tell an answer from a guess.
    """
    body = request.get_json(silent=True) or {}
    kb_id = body.get("knowledge_base_id")
    result = use_case.answer(
        services(),
        question=body.get("question", ""),
        session_id=fresh_session_id(),
        kb_id=kb_id,
        top_k=body.get("top_k", 5),
        temperature=body.get("temperature", 0.3),
        max_tokens=body.get("max_tokens", 500),
    )
    return resource(resources.answer(result, knowledge_base_id=kb_id))


@bp.route('/searches', methods=['POST'])
def search():
    """Retrieval without an answer: the ranked chunks, and why they ranked.

    ``method`` is one of ``hybrid``, ``bm25`` or ``vector``. A method the
    configured retriever cannot serve is refused with **400** and the reason,
    rather than surfacing a retriever exception as a fault -- a lexical-only
    profile has no vectors, and saying so is the answer.
    """
    body = request.get_json(silent=True) or {}
    method = (body.get("method") or "hybrid").strip().lower()
    if method not in SEARCH_METHODS:
        raise InvalidRequest(
            f"unknown retrieval method {method!r}",
            details={"supported": list(SEARCH_METHODS)},
        )

    kb_id = body.get("knowledge_base_id")
    limit = max(1, min(200, int(body.get("limit", 20))))
    found = search_use_case.experiment_search(
        services(),
        query=body.get("query", ""),
        method=method,
        kb_id=kb_id,
        session_id=fresh_session_id(),
        top_k=limit,
    )
    items = [resources.scored_chunk(row, method=method) for row in found["chunks"]]
    return collection(items, offset=0, limit=limit, total=len(items),
                      method=found["retrieval_method"], knowledge_base_id=kb_id)
