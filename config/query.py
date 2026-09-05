"""The numbers that bound the query path, validated once and early.

Phase 2 bounded ingest; this bounds the other runtime path, chat. A query
runs on the request thread that received it -- retrieval, context assembly
and the answer-model call all happen there, because a chat answer is
synchronous -- so what multiplies under load is request threads held for the
length of a provider call. These knobs bound that, and are read here and
nowhere else.

``QUERY_MAX_ACTIVE``
    How many request threads may be inside a query at once. A question that
    arrives when every slot is taken is refused at once with **503** and a
    ``Retry-After``; it is never queued, because a queued query would hold
    the very thread the limit exists to keep free. The default leaves at
    least one request thread that neither queries nor synchronous uploads can
    take: ``WAITRESS_THREADS - INGEST_SYNC_WAITERS - 1``. That is what keeps
    ``/api/health`` and job polling answerable under any burst of either.
``ANSWER_MAX_INFLIGHT``
    Process-wide cap on answer-model calls in flight, shared by every session.
    Separate from the Deep Analysis and embedding caps (see the module
    docstring of ``components/query/limits.py`` for why).
``QUERY_TIMEOUT``
    Seconds a query may take in total. Cooperative, with the network boundary
    enforced: checked before every outbound call and while waiting for an
    answer slot, and each provider attempt's socket timeout is clamped to the
    time left. Work already inside a stage -- a lexical index rebuild, a
    store lookup, a local model's forward pass -- runs to the end of that
    stage.
``ANSWER_SLOT_WAIT``
    Longest a query waits for an answer slot before it is refused as
    ``answer_capacity`` overload, whatever its deadline still allows. Short
    on purpose: a person is watching, and "busy, try again" after a few
    seconds is a better answer than one two minutes late.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Mapping

from .ingest import _number


@dataclass(frozen=True)
class QueryLimits:
    max_active: int = 3
    answer_max_inflight: int = 4
    timeout_seconds: float = 180.0
    answer_wait_seconds: float = 30.0
    #: What the default ``max_active`` was derived from, kept for the
    #: start-up check that says whether a free thread is actually guaranteed.
    request_threads: int = 8
    sync_waiters: int = 4

    def validate(self) -> "QueryLimits":
        problems = []
        if self.max_active < 1:
            problems.append("QUERY_MAX_ACTIVE must be at least 1")
        if self.answer_max_inflight < 1:
            problems.append("ANSWER_MAX_INFLIGHT must be at least 1")
        if self.timeout_seconds <= 0:
            problems.append("QUERY_TIMEOUT must be a positive number of seconds")
        if self.answer_wait_seconds <= 0:
            problems.append("ANSWER_SLOT_WAIT must be a positive number of seconds")
        if problems:
            raise ValueError("invalid query configuration: " + "; ".join(problems))
        return self

    @property
    def free_threads(self) -> int:
        """Request threads that no query and no synchronous upload can hold.

        Below one, the guarantee that health stays answerable no longer holds
        by construction; the process says so at start-up rather than failing,
        because an operator may have sized it that way on purpose.
        """
        return self.request_threads - self.max_active - self.sync_waiters

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_active": self.max_active,
            "answer_max_inflight": self.answer_max_inflight,
            "timeout_seconds": self.timeout_seconds,
            "answer_wait_seconds": self.answer_wait_seconds,
            "request_threads": self.request_threads,
            "free_threads": self.free_threads,
            "deadline_semantics": "cooperative-with-clamped-calls",
        }


def query_limits_from_env(env: Mapping[str, str] | None = None) -> QueryLimits:
    """Read and validate the query limits from an environment mapping."""
    env = os.environ if env is None else env
    threads = _number(env, "WAITRESS_THREADS", "8", int)
    sync_waiters = _number(env, "INGEST_SYNC_WAITERS", str(max(1, threads // 2)), int)
    return QueryLimits(
        max_active=_number(env, "QUERY_MAX_ACTIVE", str(max(1, threads - sync_waiters - 1)), int),
        answer_max_inflight=_number(env, "ANSWER_MAX_INFLIGHT", "4", int),
        timeout_seconds=_number(env, "QUERY_TIMEOUT", "180", float),
        answer_wait_seconds=_number(env, "ANSWER_SLOT_WAIT", "30", float),
        request_threads=threads,
        sync_waiters=sync_waiters,
    ).validate()
