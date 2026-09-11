"""The one place a refusal becomes a status code.

Every handler on this surface is registered here and nowhere else. A router
raises what the use case raised -- it does not catch it, does not translate it
and does not know a number -- so the taxonomy is a table that can be read in
one screen rather than a habit spread over five files, and adding a refusal
means adding a row.

The type names are the product's refusal taxonomy and they map onto the
:mod:`application.errors` classes one for one. ``type`` is what a client
branches on; it does not change when a message is reworded or a status is
reconsidered.

Two refusals are not application errors and are handled beside them, because
they are the same thing to a client: overload (**503** with a ``Retry-After``,
refused rather than queued) and a passed deadline (**504**). Both are raised
by the subsystem that owns the limit, which is why they live in
:mod:`core.exceptions` and not in the application's own six.

FastAPI's own validation failure is mapped here too. Its default is a **422**
with a ``detail`` list, which is a different taxonomy from this one; a
malformed body has always been **400** ``invalid_request`` on this surface and
it stays that, with the field-level report carried in ``details`` where a
client that wants it can find it.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException

from chat_rag.application.errors import (
    ApplicationError, Conflict, InvalidRequest, NotFound, NotReady, ProcessingFailed,
    Unavailable,
)
from chat_rag.core.exceptions import IngestOverloaded, QueryOverloaded, QueryTimeout

from .schemas import ApiError, ErrorResponse

logger = logging.getLogger("RAG.api.v1")

#: What each refusal answers with, and what it is called on the wire. The
#: numbers are HTTP's opinion; the names are the contract.
REFUSALS: dict[type, tuple[int, str]] = {
    InvalidRequest: (400, "invalid_request"),
    NotFound: (404, "not_found"),
    NotReady: (409, "not_ready"),
    Conflict: (409, "conflict"),
    Unavailable: (503, "unavailable"),
    ProcessingFailed: (500, "internal"),
    IngestOverloaded: (503, "overloaded"),
    QueryOverloaded: (503, "overloaded"),
    QueryTimeout: (504, "timeout"),
}

#: What an unhandled exception is called. Never a message a client can act on,
#: because nobody decided this one.
INTERNAL = (500, "internal")
#: What an unhandled exception says on the wire. The exception's own text is
#: for the log: a database error names the host, the port and the database it
#: could not reach, a provider error can quote a URL, and none of that belongs
#: in a body served to whoever asked.
INTERNAL_MESSAGE = "The server failed to handle this request; the log has the details."


def _response(status: int, kind: str, message: str, *, details: dict | None = None,
              headers: dict | None = None) -> JSONResponse:
    body = ErrorResponse(error=ApiError(type=kind, message=message, details=details or None))
    return JSONResponse(body.body(), status_code=status, headers=headers or None)


def _classify(error: Exception) -> tuple[int, str]:
    for kind, answer in REFUSALS.items():
        if isinstance(error, kind):
            return answer
    return INTERNAL


async def refused(_request: Request, error: ApplicationError) -> JSONResponse:
    """A use case said no, for a reason the caller can act on."""
    status, kind = _classify(error)
    details = dict(error.details)
    if isinstance(error, NotReady) and error.state:
        # A not-ready answer carries where the work got to, because a client
        # polling for a build needs "becoming ready" to look different from
        # "not here".
        details["state"] = error.state
    return _response(status, kind, str(error), details=details)


async def overloaded(_request: Request, error: Exception) -> JSONResponse:
    """No capacity, refused rather than queued.

    ``reason`` names the limit -- 'raise QUERY_MAX_ACTIVE' and 'raise
    ANSWER_MAX_INFLIGHT' are different decisions, and the caller's retry
    should not have to guess which.
    """
    logger.warning(f"v1 refused, overloaded: {error}")
    seconds = getattr(error, "retry_after_seconds", 30.0)
    details: dict = {"retry_after_seconds": seconds}
    if isinstance(error, QueryOverloaded):
        details["reason"] = error.reason
    status, kind = _classify(error)
    return _response(status, kind, str(error), details=details,
                     headers={"Retry-After": str(int(seconds))})


async def timed_out(request: Request, error: QueryTimeout) -> JSONResponse:
    """A query ran past its deadline and was stopped at the next seam."""
    logger.warning(f"v1 query timed out: {error}")
    seconds = request.app.state.services.settings.query_timeout
    status, kind = _classify(error)
    return _response(
        status, kind,
        f"The request could not be completed within {seconds:.0f} seconds. "
        "Try again, or ask a narrower question.",
        details={"timeout_seconds": seconds},
    )


async def invalid_payload(_request: Request, error: RequestValidationError) -> JSONResponse:
    """A body, a form or a path that this surface could not read.

    Answered as the product's own **400** ``invalid_request`` rather than
    FastAPI's **422**: a client on this contract branches on ``type``, and a
    second vocabulary for "your request was wrong" is a second thing every
    caller has to learn.
    """
    return _response(400, "invalid_request", "the request could not be read",
                     details={"fields": _fields(error)})


def _fields(error: RequestValidationError) -> list[dict]:
    """The validation report, flattened to what a client can show a user."""
    report = []
    for problem in error.errors():
        report.append({
            "field": ".".join(str(part) for part in problem.get("loc", ())),
            "message": problem.get("msg", ""),
            "type": problem.get("type", ""),
        })
    return report


async def http_error(_request: Request, error: HTTPException) -> JSONResponse:
    """Starlette's own refusals -- a wrong method, an unroutable path -- said
    in this surface's vocabulary rather than in Starlette's."""
    kind = {400: "invalid_request", 404: "not_found", 405: "invalid_request",
            409: "conflict", 503: "unavailable", 504: "timeout"}.get(
                error.status_code, "internal")
    return _response(error.status_code, kind, error.detail or "",
                     headers=getattr(error, "headers", None))


async def failed(request: Request, error: Exception) -> JSONResponse:
    """Anything nobody decided about: logged with its traceback, answered
    without one -- and without its text, which is the log's and not the
    client's (see :data:`INTERNAL_MESSAGE`)."""
    logger.error(f"v1 request failed: {request.method} {request.url.path}: "
                 f"{type(error).__name__}: {error}", exc_info=True)
    status, kind = INTERNAL
    return _response(status, kind, INTERNAL_MESSAGE)


def install(app: FastAPI) -> None:
    """Give one application the whole table.

    Registered on the application rather than per router, because a refusal
    means the same thing whichever resource raised it -- which is the half of
    this that a per-handler ``try/except`` always gets wrong eventually.
    """
    app.add_exception_handler(ApplicationError, refused)
    app.add_exception_handler(QueryOverloaded, overloaded)
    app.add_exception_handler(IngestOverloaded, overloaded)
    app.add_exception_handler(QueryTimeout, timed_out)
    app.add_exception_handler(RequestValidationError, invalid_payload)
    app.add_exception_handler(HTTPException, http_error)
    app.add_exception_handler(Exception, failed)
