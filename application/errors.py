"""What a use case refuses, said without HTTP.

Six meanings, because six is what the product actually distinguishes and a
caller actually branches on. Overload, deadline and interruption are
deliberately *not* here: they already exist as ``IngestOverloaded``,
``QueryOverloaded``, ``QueryTimeout`` and ``IngestInterrupted`` in
:mod:`core.exceptions`, raised by the subsystem that owns the limit, and a
second name for the same refusal is a second thing to keep in step.

``details`` carries the flags a client branches on -- ``generation_unavailable``,
``deep_analysis_unavailable`` -- because those are product distinctions rather
than transport ones, and an adapter that had to infer them from the message
would be guessing.
"""

from __future__ import annotations

from typing import Any, Optional


class ApplicationError(Exception):
    """A use case refused for a reason the caller can act on."""

    def __init__(self, message: str, *, details: Optional[dict[str, Any]] = None):
        super().__init__(message)
        self.details: dict[str, Any] = dict(details or {})


class InvalidRequest(ApplicationError):
    """The inputs are wrong. Nothing was created and nothing changed."""


class NotFound(ApplicationError):
    """The named resource is not here."""


class Conflict(ApplicationError):
    """The resource is here, and its current state refuses this."""


class Unavailable(ApplicationError):
    """A capability or dependency this deployment cannot provide right now.

    Not a fault of the request: the same request would succeed with the
    provider configured, so the caller is told to try later rather than to
    change what it asked for.
    """


class NotReady(ApplicationError):
    """It exists but is not finished, and ``state`` says where it got to.

    Separate from :class:`NotFound` because the answer carries that state: a
    caller polling for a build needs to know it is *becoming* ready rather
    than absent.
    """

    def __init__(self, message: str, *, state: Optional[dict[str, Any]] = None,
                 details: Optional[dict[str, Any]] = None):
        super().__init__(message, details=details)
        self.state: dict[str, Any] = state or {}


class ProcessingFailed(ApplicationError):
    """The work was attempted and failed. A server fault, not a bad request."""
