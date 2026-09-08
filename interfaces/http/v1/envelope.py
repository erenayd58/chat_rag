"""The shapes every `/api/v1` answer has, and the one place a refusal becomes
a status code.

Three shapes, and no fourth:

* **a resource** -- the object itself, at the top level. No envelope, no
  ``success`` flag: the status line already says whether it worked, and a
  client that has to unwrap every answer to reach the thing it asked for is
  one that will unwrap wrongly somewhere.
* **a collection** -- ``{"items": [...], "page": {...}}``. Always both, even
  when everything fits on one page, so a client never has to branch on which
  kind of list it received.
* **a refusal** -- ``{"error": {"type", "message", "details"}}``. ``type`` is
  the machine-readable name; it is what a client branches on, and it does not
  change when a message is reworded or a status is reconsidered.

The type names are the product's refusal taxonomy, and they map onto the
:mod:`application.errors` classes one for one. That mapping is the contract a
FastAPI port has to reproduce -- not this file.
"""

from __future__ import annotations

import logging

from flask import jsonify, request

from application.errors import (
    ApplicationError, Conflict, InvalidRequest, NotFound, NotReady, ProcessingFailed,
    Unavailable,
)
from core.exceptions import IngestOverloaded, QueryOverloaded, QueryTimeout

from ..context import services

logger = logging.getLogger("RAG.api.v1")

#: How many items a collection returns when the caller does not say.
DEFAULT_LIMIT = 50
#: The most a caller may ask for in one page. A ceiling, not a suggestion:
#: without one, `?limit=100000` is a way to make any list endpoint expensive.
MAX_LIMIT = 200

#: What each refusal is called on the wire, and what it answers with. The
#: names are the contract; the numbers are HTTP's opinion of them.
REFUSALS: dict[type, tuple[str, int]] = {
    InvalidRequest: ("invalid_request", 400),
    NotFound: ("not_found", 404),
    NotReady: ("not_ready", 409),
    Conflict: ("conflict", 409),
    Unavailable: ("unavailable", 503),
    ProcessingFailed: ("internal", 500),
}


def resource(payload: dict, status: int = 200, headers: dict | None = None):
    response = jsonify(payload)
    response.status_code = status
    for name, value in (headers or {}).items():
        response.headers[name] = value
    return response


def collection(items: list, *, offset: int, limit: int, total: int, **extra):
    """A page of a list, with the numbers a client needs to ask for the next."""
    return jsonify({
        "items": items,
        "page": {"offset": offset, "limit": limit, "total": total},
        **extra,
    })


def page_request() -> tuple[int, int]:
    """``offset`` and ``limit`` from the query string, clamped, never raising.

    A page is a presentation decision, so a client that sends nonsense gets
    the default rather than a 400: refusing ``?limit=abc`` teaches nobody
    anything and breaks a link somebody pasted.
    """
    def _number(name: str, fallback: int) -> int:
        try:
            return int(request.args.get(name, fallback))
        except (TypeError, ValueError):
            return fallback

    offset = max(0, _number("offset", 0))
    limit = max(1, min(MAX_LIMIT, _number("limit", DEFAULT_LIMIT)))
    return offset, limit


def _error(kind: str, message: str, status: int, *, details: dict | None = None,
           headers: dict | None = None):
    body: dict = {"type": kind, "message": message}
    if details:
        body["details"] = details
    response = jsonify({"error": body})
    response.status_code = status
    for name, value in (headers or {}).items():
        response.headers[name] = value
    return response


def refused(error: ApplicationError):
    kind, status = next(
        ((k, s) for cls, (k, s) in REFUSALS.items() if isinstance(error, cls)),
        ("internal", 500),
    )
    details = dict(error.details)
    if isinstance(error, NotReady) and error.state:
        # A not-ready answer carries where the work got to, because a client
        # polling for a build needs "becoming ready" to look different from
        # "not here".
        details["state"] = error.state
    return _error(kind, str(error), status, details=details)


def overloaded(error):
    """No capacity, refused rather than queued. ``reason`` names the limit --
    'raise QUERY_MAX_ACTIVE' and 'raise ANSWER_MAX_INFLIGHT' are different
    decisions, and the caller's retry should not have to guess which."""
    logger.warning(f"v1 refused, overloaded: {error}")
    details = {"retry_after_seconds": error.retry_after_seconds}
    if isinstance(error, QueryOverloaded):
        details["reason"] = error.reason
    return _error("overloaded", str(error), 503, details=details,
                  headers={"Retry-After": str(int(error.retry_after_seconds))})


def timed_out(error: QueryTimeout):
    logger.warning(f"v1 query timed out: {error}")
    seconds = services().settings.query_timeout
    return _error(
        "timeout",
        f"The request could not be completed within {seconds:.0f} seconds. "
        "Try again, or ask a narrower question.",
        504, details={"timeout_seconds": seconds},
    )


def failed(error: Exception):
    """Anything nobody decided about: logged with its traceback, answered
    without one."""
    logger.error(f"v1 request failed: {error}", exc_info=True)
    return _error("internal", str(error), 500)


def install(blueprint) -> None:
    blueprint.register_error_handler(ApplicationError, refused)
    blueprint.register_error_handler(QueryOverloaded, overloaded)
    blueprint.register_error_handler(IngestOverloaded, overloaded)
    blueprint.register_error_handler(QueryTimeout, timed_out)
    blueprint.register_error_handler(Exception, failed)
