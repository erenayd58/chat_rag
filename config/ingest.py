"""The numbers that bound ingest, validated once and early.

Every limit the ingest path honours is read here and nowhere else, so a
deployment sizing itself for its hardware has one place to look and a bad
value fails at start-up with the variable's name rather than at the first
upload with a traceback. Nothing here starts a thread or touches state.

The knobs, and what each one bounds:

``INGEST_WORKERS``
    How many ingest jobs run at once, process-wide. This is the bound on
    everything an ingest does locally -- parsing, chunking, a local embedding
    model -- because each job is one of these threads.
``INGEST_QUEUE_CAPACITY``
    How many accepted jobs may wait for a worker. A submission that finds the
    queue full is refused with an explicit overload response; it is never
    queued anyway. Zero means "run or refuse".
``INGEST_JOB_TIMEOUT``
    Seconds a job may run once it has started. Checked at stage boundaries and
    before every provider call, never inside a store write, so a job that
    passes it ends as ``timed_out`` with nothing committed.
``INGEST_SYNC_WAIT``
    Seconds a synchronous upload waits for its job before answering 202 with
    the job to poll. Kept below the server's channel timeout on purpose.
``INGEST_JOB_RETENTION``
    Seconds a finished job stays queryable.
``INGEST_SYNC_WAITERS``
    How many request threads may block waiting for a job at once. Beyond it a
    synchronous upload is still accepted and still runs -- it is answered 202
    with its job instead of being waited on -- so uploads can never occupy
    every request thread and lock out ``/api/health`` and job polling.
``PROVIDER_MAX_INFLIGHT``
    Process-wide cap on Deep Analysis proposer and verifier calls in flight,
    shared by every running Deep job. Standard, Markdown and Viewer packaging
    make no such call and hold no slot.
``DEEP_ANALYSIS_CONCURRENCY``
    One Deep job's own pool. A job never has more than this many calls in
    flight, and all jobs together never have more than the global cap.
``PIPELINE_CACHE_MAX`` / ``PIPELINE_CACHE_TTL``
    How many built pipelines may be held, and how long an unused one is kept.
    A pipeline holds an embedding model, a store handle and the knowledge
    base's whole lexical index, so this is the largest single memory dial in
    the process. Eviction never touches a pipeline that is in use.
``EMBEDDING_MAX_INFLIGHT``
    Process-wide cap on embedding requests in flight, when the embedding
    provider is a remote endpoint. It is separate from the Deep cap because
    the two reach different services and one must not starve the other, and
    because embeddings multiply from places Deep Analysis does not run: every
    ingest, a whole-knowledge-base re-index, and every query.

The complete resource model, so nothing external is unaccounted for:

===================================  =====================================
outbound or expensive work           what bounds it
===================================  =====================================
Deep proposer / verifier calls       ``PROVIDER_MAX_INFLIGHT`` (global),
                                     ``DEEP_ANALYSIS_CONCURRENCY`` (per job)
Embedding requests (remote)          ``EMBEDDING_MAX_INFLIGHT`` (global)
Embedding batches (local model)      ``INGEST_WORKERS`` -- CPU on the
                                     worker that asked for them
Parsing, chunking, indexing          ``INGEST_WORKERS``
Built pipelines (models, stores,     ``PIPELINE_CACHE_MAX`` /
lexical indexes)                     ``PIPELINE_CACHE_TTL``
Viewer packaging (no provider call)  its single worker thread
Answer model at query time           ``ANSWER_MAX_INFLIGHT`` (global); the
                                     request threads a query may hold are
                                     bounded by ``QUERY_MAX_ACTIVE`` and the
                                     whole query by ``QUERY_TIMEOUT``
                                     (config/query.py)
===================================  =====================================
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Mapping

#: The number parser and the request-thread count both live in
#: :mod:`config.runtime`, which owns the server's own settings. Re-exported
#: here because ``config.query`` and this module were both written against
#: ``ingest._number`` before that module existed.
from .runtime import _number, runtime_from_env  # noqa: F401  (re-export)


@dataclass(frozen=True)
class IngestLimits:
    workers: int = 2
    queue_capacity: int = 8
    job_timeout_seconds: float = 1800.0
    sync_wait_seconds: float = 840.0
    sync_waiters: int = 2
    job_retention_seconds: float = 3600.0
    provider_max_inflight: int = 8
    deep_concurrency: int = 8
    embedding_max_inflight: int = 4
    pipeline_cache_max: int = 8
    pipeline_cache_ttl_seconds: float = 1800.0

    def validate(self) -> "IngestLimits":
        problems = []
        if self.workers < 1:
            problems.append("INGEST_WORKERS must be at least 1")
        if self.queue_capacity < 0:
            problems.append("INGEST_QUEUE_CAPACITY must be 0 or more")
        if self.job_timeout_seconds <= 0:
            problems.append("INGEST_JOB_TIMEOUT must be a positive number of seconds")
        if self.sync_wait_seconds <= 0:
            problems.append("INGEST_SYNC_WAIT must be a positive number of seconds")
        if self.sync_waiters < 0:
            problems.append("INGEST_SYNC_WAITERS must be 0 or more")
        if self.job_retention_seconds < 0:
            problems.append("INGEST_JOB_RETENTION must be 0 or more seconds")
        if self.provider_max_inflight < 1:
            problems.append("PROVIDER_MAX_INFLIGHT must be at least 1")
        if self.deep_concurrency < 1:
            problems.append("DEEP_ANALYSIS_CONCURRENCY must be at least 1")
        if self.embedding_max_inflight < 1:
            problems.append("EMBEDDING_MAX_INFLIGHT must be at least 1")
        if self.pipeline_cache_max < 1:
            problems.append("PIPELINE_CACHE_MAX must be at least 1")
        if self.pipeline_cache_ttl_seconds < 0:
            problems.append("PIPELINE_CACHE_TTL must be 0 or more seconds")
        if problems:
            raise ValueError("invalid ingest configuration: " + "; ".join(problems))
        return self

    @property
    def admission_capacity(self) -> int:
        """Jobs that can be accepted at once: running plus waiting."""
        return self.workers + self.queue_capacity

    def to_dict(self) -> dict[str, Any]:
        return {
            "workers": self.workers,
            "queue_capacity": self.queue_capacity,
            "job_timeout_seconds": self.job_timeout_seconds,
            "sync_wait_seconds": self.sync_wait_seconds,
            "sync_waiters": self.sync_waiters,
            "job_retention_seconds": self.job_retention_seconds,
            "provider_max_inflight": self.provider_max_inflight,
            "deep_concurrency": self.deep_concurrency,
            "embedding_max_inflight": self.embedding_max_inflight,
            "pipeline_cache_max": self.pipeline_cache_max,
            "pipeline_cache_ttl_seconds": self.pipeline_cache_ttl_seconds,
            "deadline_semantics": "cooperative-with-clamped-calls",
        }


#: The defaults, read off the dataclass rather than written out again as
#: strings. One place says what a default is, and it is the field above.
_DEFAULTS = IngestLimits()


def default_sync_waiters(request_threads: int) -> int:
    """Half the request threads, so a burst of synchronous uploads can never
    hold more than half of them and status traffic always has somewhere to
    land."""
    return max(1, request_threads // 2)


def limits_from_env(env: Mapping[str, str] | None = None) -> IngestLimits:
    """Read and validate the limits from an environment mapping."""
    env = os.environ if env is None else env
    # The thread count comes from the one module that owns it, so a bad value
    # is refused once, by name, rather than meaning different things here and
    # in the server.
    threads = runtime_from_env(env).request_threads
    return IngestLimits(
        workers=_number(env, "INGEST_WORKERS", _DEFAULTS.workers, int),
        queue_capacity=_number(env, "INGEST_QUEUE_CAPACITY", _DEFAULTS.queue_capacity, int),
        job_timeout_seconds=_number(env, "INGEST_JOB_TIMEOUT", _DEFAULTS.job_timeout_seconds, float),
        sync_wait_seconds=_number(env, "INGEST_SYNC_WAIT", _DEFAULTS.sync_wait_seconds, float),
        sync_waiters=_number(
            env, "INGEST_SYNC_WAITERS", default_sync_waiters(threads), int
        ),
        job_retention_seconds=_number(
            env, "INGEST_JOB_RETENTION", _DEFAULTS.job_retention_seconds, float
        ),
        provider_max_inflight=_number(
            env, "PROVIDER_MAX_INFLIGHT", _DEFAULTS.provider_max_inflight, int
        ),
        deep_concurrency=_number(
            env, "DEEP_ANALYSIS_CONCURRENCY", _DEFAULTS.deep_concurrency, int
        ),
        embedding_max_inflight=_number(
            env, "EMBEDDING_MAX_INFLIGHT", _DEFAULTS.embedding_max_inflight, int
        ),
        pipeline_cache_max=_number(env, "PIPELINE_CACHE_MAX", _DEFAULTS.pipeline_cache_max, int),
        pipeline_cache_ttl_seconds=_number(
            env, "PIPELINE_CACHE_TTL", _DEFAULTS.pipeline_cache_ttl_seconds, float
        ),
    ).validate()
