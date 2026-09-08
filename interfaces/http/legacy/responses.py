"""Turning what the application said into what HTTP says.

This is the whole translation table, in one file, because the distinctions
themselves are the contract (``tests/migration/test_http_surface.py`` pins
them): a malformed request, an unknown resource, a resource in use, an
unavailable capability, a passed deadline and a server fault are six different
answers, and the console's JavaScript and the Viewer's relay branch on them.
Collapsing any of them into 500 -- the usual failure of a framework port -- is
what these handlers exist to prevent.

Nothing above this file knows a status code, and nothing below it builds a
response.
"""

from __future__ import annotations

import logging

from flask import jsonify

from application.errors import (
    ApplicationError, Conflict, InvalidRequest, NotFound, NotReady, ProcessingFailed,
    Unavailable,
)
from core.exceptions import IngestOverloaded, QueryOverloaded, QueryTimeout

from ..context import services

logger = logging.getLogger("RAG.http")

#: The one place an application refusal becomes a status code.
STATUS = {
    InvalidRequest: 400,
    NotFound: 404,
    NotReady: 404,
    Conflict: 409,
    Unavailable: 503,
    ProcessingFailed: 500,
}


def ok(**payload):
    """The shape every successful answer has had: ``success`` and the body."""
    return jsonify({'success': True, **payload})


def _status_for(error: ApplicationError) -> int:
    for kind, code in STATUS.items():
        if isinstance(error, kind):
            return code
    return 500


def refused(error: ApplicationError):
    """An application refusal, as its status and its body.

    ``details`` carries the flags a client branches on -- ``unknown_job``,
    ``deep_analysis_unavailable``, ``generation_unavailable`` -- and a
    :class:`~application.errors.NotReady` also carries the state it got to, so
    a caller polling for a build can tell "becoming ready" from "not here".
    """
    body = {'success': False, 'error': str(error), **error.details}
    if isinstance(error, NotReady):
        body['state'] = error.state
    return jsonify(body), _status_for(error)


def overloaded(error):
    """The one answer an overloaded path gives, wherever it is refused.

    Deterministic and immediate: no slot, no waiting on this thread. A query's
    refusal also names *which* limit refused it, because "raise
    QUERY_MAX_ACTIVE" and "raise ANSWER_MAX_INFLIGHT" are different decisions.
    """
    logger.warning(f"Refused, overloaded: {error}")
    body = {
        'success': False,
        'overloaded': True,
        'retry_after_seconds': error.retry_after_seconds,
        'error': str(error),
    }
    if isinstance(error, QueryOverloaded):
        body['reason'] = error.reason
    response = jsonify(body)
    response.headers['Retry-After'] = str(int(error.retry_after_seconds))
    return response, 503


def timed_out(_error: QueryTimeout):
    """The one answer a query past its deadline gives."""
    logger.warning(f"Query timed out: {_error}")
    seconds = services().settings.query_timeout
    return jsonify({
        'success': False,
        'timed_out': True,
        'timeout_seconds': seconds,
        'error': (
            f"The request could not be completed within {seconds:.0f} seconds. "
            "Try again, or ask a narrower question."
        ),
    }), 504


def failed(error: Exception):
    """Anything nobody decided about. Logged with its traceback, answered
    without one: the message is for a developer reading the log, and the
    status is what the client acts on."""
    logger.error(f"Request failed: {error}", exc_info=True)
    return jsonify({'success': False, 'error': str(error)}), 500


def install(blueprint) -> None:
    """Give one blueprint the whole table.

    Per blueprint rather than per application so that the page routes keep
    Flask's own error behaviour -- a template problem should still reach the
    developer as a traceback, not as JSON.
    """
    blueprint.register_error_handler(ApplicationError, refused)
    blueprint.register_error_handler(QueryOverloaded, overloaded)
    blueprint.register_error_handler(IngestOverloaded, overloaded)
    blueprint.register_error_handler(QueryTimeout, timed_out)
    blueprint.register_error_handler(Exception, failed)
