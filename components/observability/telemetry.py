"""One way to measure a job, so nobody has to invent a second one.

Phase 2 made ingest bounded. The question this answers is the next one an
operator asks: *where did the time go, and how much capacity is left?* The
temptation is a timer at every interesting line, and the result of that is
five timers that disagree. So there is exactly one abstraction here.

A **trace** belongs to one job and carries ordered **stages**. A stage is
opened with a context manager and closed by leaving it, whatever happens
inside; a stage that raises is recorded as failed with its error category,
not silently dropped. The trace is reachable from a thread-local, the same
way ``JobGuard`` is, so the pipeline measures itself by writing
``with stage("parse"):`` rather than by being handed a telemetry object
through six call signatures. Code running outside a job finds no trace and
the stage is a no-op -- a CLI ingest and a test measure nothing and cost
nothing.

Counters that outlive a job live in the **registry**: how many jobs were
accepted, rejected, failed; how long jobs waited in the queue; what each
stage usually costs. It keeps a bounded window of recent traces (a
``deque`` with a fixed length) and computes summaries from it on demand,
because an operator wants "what is p95 parse time lately", not a database.
Nothing here grows without a bound, and that is asserted in the tests.

What this is not: a tracing platform. There are no spans crossing
processes, no exporters, no sampling. One process, one queue, one kind of
job -- the smallest thing that answers the question.
"""

from __future__ import annotations

import threading
import time
from collections import Counter, deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator, Optional

#: The stages an ingest job passes through, in the order they happen. Naming
#: them here rather than at each call site is what stops two spellings of
#: "chunking" appearing in the metrics.
QUEUE_WAIT = "queue_wait"
PARSE = "parse"
CHUNK = "chunk"
DEEP = "deep_analysis"
EMBED = "embed"
INDEX = "index"
LEDGER = "ledger"
VIEWER = "viewer_stage"

STAGES = (QUEUE_WAIT, PARSE, CHUNK, DEEP, EMBED, INDEX, LEDGER, VIEWER)

#: How many finished traces the registry keeps for its summaries. Each is a
#: handful of floats and short strings; two hundred of them is a few tens of
#: kilobytes and cannot grow past that.
RECENT_TRACES = 200

#: How many distinct error messages to keep per category. A failing provider
#: must not be able to fill memory with variations of its own error text.
MESSAGES_PER_CATEGORY = 5


def categorise(error: BaseException | None) -> str:
    """One word for what went wrong, stable enough to count.

    Categories are chosen to match what an operator would do next: a
    configuration error needs a setting changed, a provider error needs the
    gateway looked at, a storage error needs the disk or the store looked at.
    Unknown is honest rather than a guess.
    """
    if error is None:
        return "none"
    from core.exceptions import (
        ChunkerException, ConfigurationException, EmbeddingException,
        IndexIncompatibleException, IngestInterrupted, IngestOverloaded,
        LLMException, VectorDBException,
    )

    if isinstance(error, IngestInterrupted):
        return error.kind  # timed_out | cancelled
    if isinstance(error, IngestOverloaded):
        return "overloaded"
    if isinstance(error, IndexIncompatibleException):
        return "index_incompatible"
    if isinstance(error, ConfigurationException):
        return "configuration"
    if isinstance(error, EmbeddingException):
        return "embedding"
    if isinstance(error, (LLMException,)):
        return "provider"
    if isinstance(error, ChunkerException):
        return "chunking"
    if isinstance(error, VectorDBException):
        return "storage"
    # Before OSError: TimeoutError is a subclass of it, and "the gateway did
    # not answer" is a different thing to look at than "the disk did not".
    if isinstance(error, TimeoutError):
        return "timeout"
    if isinstance(error, (OSError, IOError)):
        return "storage"
    return "unknown"


@dataclass
class Stage:
    name: str
    seconds: float
    ok: bool = True
    error_category: str = "none"
    #: Small, countable facts about what the stage did: units parsed, chunks
    #: written, calls made. Never content.
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass
class JobTrace:
    """Everything measured about one job, and nothing about its content."""

    job_id: str
    kb_id: str = ""
    mode: str = ""
    stages: list[Stage] = field(default_factory=list)
    #: Provider work, accumulated by the budget wrappers rather than by a
    #: stage, because those calls happen inside a pool several layers down.
    provider_calls: int = 0
    provider_seconds: float = 0.0
    provider_wait_seconds: float = 0.0
    embedding_calls: int = 0
    embedding_seconds: float = 0.0
    embedding_wait_seconds: float = 0.0
    queue_seconds: float = 0.0
    total_seconds: float = 0.0
    status: str = ""
    error_category: str = "none"
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add(self, stage: Stage) -> None:
        with self._lock:
            self.stages.append(stage)

    def record_provider(self, *, seconds: float, wait_seconds: float) -> None:
        with self._lock:
            self.provider_calls += 1
            self.provider_seconds += seconds
            self.provider_wait_seconds += wait_seconds

    def record_embedding(self, *, seconds: float, wait_seconds: float) -> None:
        with self._lock:
            self.embedding_calls += 1
            self.embedding_seconds += seconds
            self.embedding_wait_seconds += wait_seconds

    def seconds_for(self, name: str) -> float:
        """Total time in one stage, summed over however many times it ran."""
        with self._lock:
            return round(sum(s.seconds for s in self.stages if s.name == name), 4)

    def as_dict(self) -> dict[str, Any]:
        """The shape the API and the logs both use. Counts and seconds only."""
        with self._lock:
            stages = {
                name: round(sum(s.seconds for s in self.stages if s.name == name), 4)
                for name in dict.fromkeys(s.name for s in self.stages)
            }
            failed = [s.name for s in self.stages if not s.ok]
        return {
            "job_id": self.job_id,
            "kb_id": self.kb_id,
            "mode": self.mode,
            "status": self.status,
            "error_category": self.error_category,
            "queue_seconds": round(self.queue_seconds, 4),
            "total_seconds": round(self.total_seconds, 4),
            "stages": stages,
            "failed_stages": failed,
            "provider": {
                "calls": self.provider_calls,
                "seconds": round(self.provider_seconds, 4),
                "wait_seconds": round(self.provider_wait_seconds, 4),
            },
            "embedding": {
                "calls": self.embedding_calls,
                "seconds": round(self.embedding_seconds, 4),
                "wait_seconds": round(self.embedding_wait_seconds, 4),
            },
        }


# ------------------------------------------------------------- the context
_current = threading.local()


@contextmanager
def use_trace(trace: Optional[JobTrace]) -> Iterator[Optional[JobTrace]]:
    """Make ``trace`` the current job's, on this thread, for the block."""
    previous = getattr(_current, "trace", None)
    _current.trace = trace
    try:
        yield trace
    finally:
        _current.trace = previous


def current_trace() -> Optional[JobTrace]:
    return getattr(_current, "trace", None)


@contextmanager
def stage(name: str, **detail: Any) -> Iterator[None]:
    """Measure one stage of the current job. A no-op outside a job.

    A stage that raises is still recorded -- with ``ok=False`` and the error's
    category -- because "parse took 40 seconds and then failed" is the single
    most useful line in an incident, and a bare ``try/finally`` timer would
    have thrown that away.
    """
    trace = current_trace()
    if trace is None:
        yield
        return
    pending: dict[str, Any] = dict(detail)
    _open_stages().append(pending)
    started = time.perf_counter()
    try:
        yield
    except BaseException as error:
        _open_stages().pop()
        trace.add(Stage(name, time.perf_counter() - started, ok=False,
                        error_category=categorise(error), detail=pending))
        raise
    else:
        _open_stages().pop()
        trace.add(Stage(name, time.perf_counter() - started, detail=pending))


def _open_stages() -> list:
    """The stages open on this thread, innermost last."""
    stack = getattr(_current, "open_stages", None)
    if stack is None:
        stack = _current.open_stages = []
    return stack


def annotate(**detail: Any) -> None:
    """Attach countable facts to the stage being measured.

    Call sites are inside the stage -- the number is usually only known once
    the work is done but before the block closes -- so this writes into the
    innermost open stage. Outside a stage it updates the one that finished
    last, and outside a job it does nothing at all.
    """
    trace = current_trace()
    if trace is None:
        return
    stack = _open_stages()
    if stack:
        stack[-1].update(detail)
        return
    with trace._lock:
        if trace.stages:
            trace.stages[-1].detail.update(detail)


# ------------------------------------------------------------- the registry
class MetricsRegistry:
    """Process-wide counters and a bounded window of recent job traces."""

    def __init__(self, window: int = RECENT_TRACES):
        self._lock = threading.Lock()
        self._recent: deque[JobTrace] = deque(maxlen=window)
        self._counters: Counter[str] = Counter()
        self._errors: Counter[str] = Counter()
        self._messages: dict[str, deque[str]] = {}
        self.started_at = time.time()

    # -- counting
    def count(self, name: str, amount: int = 1) -> None:
        with self._lock:
            self._counters[name] += amount

    def record_error(self, category: str, message: str = "") -> None:
        """Count one failure, and keep a redacted example of what it said.

        The message is whatever a library chose to write, and these examples
        are served by ``/api/ops/metrics``. Redaction happens here rather than
        at the endpoint so there is one place where an arbitrary exception
        string becomes safe to keep: a stored message has already lost any
        credential shape and any absolute path, so nothing downstream has to
        remember to strip them.
        """
        from .events import redact_message

        with self._lock:
            self._errors[category] += 1
            short = redact_message(message)
            if short:
                seen = self._messages.setdefault(category, deque(maxlen=MESSAGES_PER_CATEGORY))
                if short not in seen:
                    seen.append(short)

    def finish(self, trace: JobTrace) -> None:
        with self._lock:
            self._recent.append(trace)

    # -- reading
    def _percentile(self, values: list[float], fraction: float) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        index = min(len(ordered) - 1, int(round(fraction * (len(ordered) - 1))))
        return round(ordered[index], 4)

    def _summary(self, values: list[float]) -> dict[str, Any]:
        if not values:
            return {"count": 0}
        return {
            "count": len(values),
            "p50": self._percentile(values, 0.50),
            "p95": self._percentile(values, 0.95),
            "max": round(max(values), 4),
            "total": round(sum(values), 4),
        }

    def snapshot(self, *, recent: int = 10) -> dict[str, Any]:
        """Counters, latency summaries and the last few jobs.

        ``recent`` is capped so this endpoint cannot become a history dump;
        the window itself is already bounded by ``maxlen``.
        """
        with self._lock:
            traces = list(self._recent)
            counters = dict(self._counters)
            errors = dict(self._errors)
            messages = {k: list(v) for k, v in self._messages.items()}
        per_stage: dict[str, list[float]] = {}
        for trace in traces:
            for name in STAGES:
                seconds = trace.seconds_for(name)
                if seconds:
                    per_stage.setdefault(name, []).append(seconds)
        return {
            "uptime_seconds": round(time.time() - self.started_at, 1),
            "counters": counters,
            "errors": {"by_category": errors, "recent_messages": messages},
            "jobs": {
                "measured": len(traces),
                "window": self._recent.maxlen,
                "queue_wait_seconds": self._summary([t.queue_seconds for t in traces]),
                "total_seconds": self._summary([t.total_seconds for t in traces]),
            },
            "stages": {name: self._summary(values) for name, values in per_stage.items()},
            # ``[-0:]`` is the whole list, which is the opposite of what
            # asking for none should do.
            "recent": [t.as_dict() for t in (traces[-min(recent, 25):] if recent > 0 else [])],
        }

    def recent_outcomes(self) -> tuple[int, int]:
        """``(measured, failed)`` over the bounded window of recent jobs.

        What "healthy but broken" looks like: the process is serving, the
        queue is not full, and every job it has finished lately has failed.
        A ratio over a fixed window rather than a total, so a service that
        recovers stops reporting a problem it no longer has.
        """
        with self._lock:
            traces = list(self._recent)
        return len(traces), sum(1 for t in traces if t.status == "failed")

    def reset(self) -> None:
        """For tests: forget everything measured so far."""
        with self._lock:
            self._recent.clear()
            self._counters.clear()
            self._errors.clear()
            self._messages.clear()
            self.started_at = time.time()


_registry_lock = threading.Lock()
_registry: Optional[MetricsRegistry] = None


def metrics() -> MetricsRegistry:
    global _registry
    with _registry_lock:
        if _registry is None:
            _registry = MetricsRegistry()
        return _registry
