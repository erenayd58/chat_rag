"""The server process itself: where it listens and how many threads it has.

Phase 2 gave ingest one place to read its limits, Phase 4 did the same for
queries, and both then sized themselves against ``WAITRESS_THREADS`` -- which
was read in three places, with the default ``8`` written out in each and two
different ideas of what a bad value means. ``WAITRESS_THREADS=-4`` produced a
server with eight threads and limits computed against minus four. This module
is the one owner of those numbers, so the server and the limits can no longer
disagree about how many threads there are.

What it owns
------------

``FLASK_HOST`` / ``FLASK_PORT``
    Where the server listens. ``0.0.0.0`` because a container's port mapping
    only reaches a process listening on every interface.
``WAITRESS_THREADS``
    Request threads, and the number every other limit is sized against: at
    most ``INGEST_SYNC_WAITERS`` of them may wait on a synchronous upload and
    at most ``QUERY_MAX_ACTIVE`` may be inside a query, so the rest stay free
    for ``/api/v1/health`` and job polling. Under uvicorn this sizes the
    thread pool Starlette runs the synchronous handlers in (``asgi.py``).
``WAITRESS_CHANNEL_TIMEOUT``
    How long a request may be held. ``INGEST_SYNC_WAIT`` must stay below it:
    an upload that waits longer than the server will hold the connection
    answers into a socket the client has already lost.

**The four names are the deployment's, not this module's.** Flask and waitress
are gone -- ``asgi.py`` is uvicorn -- and these variables are still spelled the
way every ``.env``, compose file and container image already spells them.
Renaming a setting is a change to somebody's deployment, and a silent one: an
old name simply stops being read and the default takes over. It is left to
whoever decides to make it, with a migration note, rather than folded into a
cleanup.

What it does not own
--------------------

Logging. ``utils/logger.py`` resolves ``LOG_LEVEL``, ``LOG_FILE_LEVEL`` and
the rotation limits, and it does so with **fail-safe** semantics rather than
the fail-fast ones here: an unrecognised level falls back to INFO instead of
refusing to start, because the risky direction is DEBUG (it writes document
text to disk) and a typo must never be the thing that turns it on. That
difference is deliberate and is the one place the two categories diverge; see
``docs/configuration.md``. The values are reported at start-up either way, so
a fallback is visible rather than silent.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Mapping


def _number(env: Mapping[str, str], name: str, default, kind):
    """One environment number, parsed or refused by name.

    ``default`` is the value itself, not a string of it, so the dataclass
    field below stays the single place a default is written down.
    """
    raw = (env.get(name) or "").strip()
    if not raw:
        return kind(default)
    try:
        return kind(raw)
    except ValueError:
        raise ValueError(
            f"{name}={raw!r} is not a {'whole number' if kind is int else 'number'}"
        ) from None


def _text(env: Mapping[str, str], name: str, default: str) -> str:
    return (env.get(name) or "").strip() or default


@dataclass(frozen=True)
class RuntimeLimits:
    """The production server's own configuration. One default each, here."""

    #: Every interface: a container's port mapping only reaches a process
    #: listening on all of them.
    host: str = "0.0.0.0"
    port: int = 5005
    #: One thread serves one request. Eight is enough for a console with a
    #: handful of users, and small enough that eight concurrent ingests cannot
    #: exhaust the machine. Request concurrency is this, plus the ingest
    #: workers and the one Viewer packaging thread, in one address space --
    #: which is what lets the provider budgets be plain semaphores.
    request_threads: int = 8
    #: Long enough for an upload that parses a large PDF on the request
    #: thread, short enough that a dead connection is not held forever. It has
    #: to stay above ``INGEST_SYNC_WAIT``; :func:`cross_check` enforces that.
    #: Declared rather than handed to the server: uvicorn has no equivalent
    #: knob, so what this bounds today is the relationship below.
    channel_timeout_seconds: int = 900

    def validate(self) -> "RuntimeLimits":
        problems = []
        if not self.host:
            problems.append("FLASK_HOST must not be empty")
        if not (1 <= self.port <= 65535):
            problems.append("FLASK_PORT must be a port number between 1 and 65535")
        if self.request_threads < 1:
            problems.append("WAITRESS_THREADS must be at least 1")
        if self.channel_timeout_seconds <= 0:
            problems.append("WAITRESS_CHANNEL_TIMEOUT must be a positive number of seconds")
        if problems:
            raise ValueError("invalid runtime configuration: " + "; ".join(problems))
        return self

    def to_dict(self) -> dict[str, Any]:
        return {
            "host": self.host,
            "port": self.port,
            "request_threads": self.request_threads,
            "channel_timeout_seconds": self.channel_timeout_seconds,
        }


_DEFAULTS = RuntimeLimits()


def runtime_from_env(env: Mapping[str, str] | None = None) -> RuntimeLimits:
    """Read and validate the server's own settings from an environment."""
    env = os.environ if env is None else env
    return RuntimeLimits(
        host=_text(env, "FLASK_HOST", _DEFAULTS.host),
        port=_number(env, "FLASK_PORT", _DEFAULTS.port, int),
        request_threads=_number(env, "WAITRESS_THREADS", _DEFAULTS.request_threads, int),
        channel_timeout_seconds=_number(
            env, "WAITRESS_CHANNEL_TIMEOUT", _DEFAULTS.channel_timeout_seconds, int
        ),
    ).validate()


def cross_check(runtime: RuntimeLimits, ingest, query) -> list[str]:
    """Check the relationships *between* the three groups, once.

    Each group validates its own values; only a rule that spans two of them
    belongs here, so no rule is stated twice and there is one place to look
    for "which combinations are impossible".

    Raises for a combination that cannot work whatever the operator intended.
    Returns warning lines for one that is merely unusual -- an explicit
    ``QUERY_MAX_ACTIVE`` that leaves no free request thread is a deliberate
    choice this application has always allowed and reported rather than
    refused, and that stays true.
    """
    problems = []
    if ingest.sync_wait_seconds >= runtime.channel_timeout_seconds:
        problems.append(
            f"INGEST_SYNC_WAIT={ingest.sync_wait_seconds:.0f} is not below "
            f"WAITRESS_CHANNEL_TIMEOUT={runtime.channel_timeout_seconds}: a "
            "synchronous upload would still be waiting when the server closes "
            "the connection it would answer on"
        )
    if problems:
        raise ValueError("invalid runtime configuration: " + "; ".join(problems))

    warnings = []
    free = runtime.request_threads - query.max_active - ingest.sync_waiters
    if free < 1:
        warnings.append(
            f"QUERY_MAX_ACTIVE={query.max_active} + INGEST_SYNC_WAITERS="
            f"{ingest.sync_waiters} leaves {free} of {runtime.request_threads} "
            "request threads free; health checks and job polling are not "
            "guaranteed a thread under load"
        )
    if ingest.provider_max_inflight < ingest.deep_concurrency:
        warnings.append(
            f"DEEP_ANALYSIS_CONCURRENCY={ingest.deep_concurrency} is above the "
            f"process-wide PROVIDER_MAX_INFLIGHT={ingest.provider_max_inflight}; "
            "a single Deep job can never reach its own concurrency"
        )
    return warnings
