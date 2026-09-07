"""What bounds an ingest while it runs: the outbound budgets and the deadline.

``ProviderBudget`` is a process-wide cap on outbound calls of one kind. Two
exist: the **Deep Analysis** budget (``PROVIDER_MAX_INFLIGHT``), taken around
each proposer and verifier call, and the **embedding** budget
(``EMBEDDING_MAX_INFLIGHT``), taken around each embedding request. Each is one
semaphore shared by every caller in the process, so a Deep job whose pool has
eight threads still holds only as many slots as its budget allows, and two
Deep jobs together never hold more than the configured maximum. Standard,
Markdown and the Viewer packager take no Deep slot because they make no such
call. Everything that can happen inside a call -- success, an HTTP error, a
timeout -- releases the slot on the way out, which is why a slot is a context
manager and not a pair of methods.

**Boundedness, not fairness.** A budget guarantees only that no more than
``limit`` calls are in flight at once. Nothing here promises that waiters are
served in arrival order: ``threading.Semaphore`` makes no such guarantee and
this code does not add one, because ordering is not what the product needs.
What it does need is that no waiter can be stuck forever, and that comes from
the deadline instead -- every wait for a slot is bounded by the job's own
remaining time (see :class:`LimitedProvider`), so a job that loses every race
fails truthfully as ``timed_out`` rather than hanging.

``JobGuard`` is one job's deadline and cancellation flag. The pipeline asks it
``check()`` at stage boundaries and the provider wrapper asks it before every
call, so a job out of time stops at the next seam with nothing half-committed.
A running stage is never interrupted from outside; Python has no safe way to
do that, and a store write must not be.

**The deadline is cooperative, with one enforced boundary.** Between seams,
work that has started runs to its end: a parse, a chunking pass, a local
embedding batch. The one place where the overshoot would otherwise be
unbounded is a network call, so :class:`LimitedProvider` clamps the
transport's own socket timeout to the time the job has left. A job therefore
overshoots its deadline by at most the longest single uninterruptible stage,
not by a whole provider timeout on top of it. ``JOB_DEADLINE_SEMANTICS``
states this in one line for anything that wants to quote it.

The guard is carried on a thread-local so the pipeline needs no new
parameters: the worker sets it for the job's duration, and a pipeline used
outside a job (a test, the CLI) finds no guard and runs unbounded, as before.

Nothing here knows what a job is; that is ``jobs.py``.
"""

from __future__ import annotations

import copy
import threading
import time
from contextlib import contextmanager
from typing import Any, Iterator, Optional

from components.observability import telemetry as T
from core.exceptions import IngestInterrupted

#: What "deadline" means here, in one sentence, for anything that reports it.
JOB_DEADLINE_SEMANTICS = (
    "cooperative: checked at stage boundaries and before every outbound call, "
    "and enforced on the network by clamping each call's socket timeout to the "
    "time the job has left; work already inside a stage runs to the end of that "
    "stage"
)


class ProviderSlotTimeout(RuntimeError):
    """No provider slot became free before the caller's deadline."""


class ProviderBudget:
    """A counting semaphore that also knows its own high-water mark.

    ``peak`` is what a test asserts on: the most slots ever held at once,
    which must never exceed ``limit``. ``inflight`` is what the status
    endpoint shows.
    """

    def __init__(self, limit: int):
        if limit < 1:
            raise ValueError("a provider budget needs at least one slot")
        self.limit = int(limit)
        self._slots = threading.Semaphore(self.limit)
        self._lock = threading.Lock()
        self.inflight = 0
        self.peak = 0
        self.acquired_total = 0
        self.refused_total = 0
        #: How long callers have spent queueing for a slot, in total. The
        #: number that says whether the limit is the bottleneck.
        self.wait_seconds_total = 0.0

    @contextmanager
    def slot(self, timeout: Optional[float] = None) -> Iterator[None]:
        """Hold one slot for the duration of the block, whatever happens in it.

        ``timeout`` bounds the wait for a free slot; ``None`` waits. A wait
        that runs out raises :class:`ProviderSlotTimeout` *before* the block,
        so a caller that never got a slot never releases one.
        """
        if timeout is not None and timeout <= 0:
            with self._lock:
                self.refused_total += 1
            raise ProviderSlotTimeout("no time left to wait for a provider slot")
        waited = time.perf_counter()
        acquired = self._slots.acquire(timeout=timeout)
        wait_seconds = time.perf_counter() - waited
        if not acquired:
            with self._lock:
                self.refused_total += 1
            raise ProviderSlotTimeout(
                f"no provider slot free within {timeout:.1f}s (limit {self.limit})"
            )
        with self._lock:
            self.inflight += 1
            self.acquired_total += 1
            self.peak = max(self.peak, self.inflight)
            self.wait_seconds_total += wait_seconds
        try:
            yield wait_seconds
        finally:
            with self._lock:
                self.inflight -= 1
            self._slots.release()

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return {
                "limit": self.limit,
                "inflight": self.inflight,
                "peak": self.peak,
                "acquired_total": self.acquired_total,
                "refused_total": self.refused_total,
                "wait_seconds_total": round(self.wait_seconds_total, 3),
            }


class JobGuard:
    """One job's deadline and cancellation, asked at every seam."""

    def __init__(self, deadline: Optional[float] = None, *, clock=time.monotonic):
        self._clock = clock
        self.deadline = deadline
        self._cancelled = threading.Event()
        self.cancel_reason = ""

    @classmethod
    def for_timeout(cls, seconds: Optional[float], *, clock=time.monotonic) -> "JobGuard":
        deadline = None if not seconds else clock() + float(seconds)
        return cls(deadline, clock=clock)

    def cancel(self, reason: str = "cancelled") -> None:
        self.cancel_reason = reason
        self._cancelled.set()

    @property
    def cancelled(self) -> bool:
        return self._cancelled.is_set()

    def remaining(self) -> Optional[float]:
        """Seconds until the deadline; ``None`` without one; never negative."""
        if self.deadline is None:
            return None
        return max(0.0, self.deadline - self._clock())

    @property
    def expired(self) -> bool:
        return self.deadline is not None and self._clock() >= self.deadline

    def check(self) -> None:
        """Raise if this job must stop here. Cancellation is checked first:
        an operator's decision outranks the clock."""
        if self.cancelled:
            raise IngestInterrupted("cancelled", self.cancel_reason or "the job was cancelled")
        if self.expired:
            raise IngestInterrupted("timed_out", "the job's deadline passed before it finished")


# ---------------------------------------------------------------- context
_current = threading.local()


@contextmanager
def use_guard(guard: Optional[JobGuard]) -> Iterator[None]:
    """Make ``guard`` the current job's for the block, on this thread only."""
    previous = getattr(_current, "guard", None)
    _current.guard = guard
    try:
        yield
    finally:
        _current.guard = previous


def current_guard() -> Optional[JobGuard]:
    return getattr(_current, "guard", None)


def checkpoint() -> None:
    """The pipeline's stage boundary: stop here if the current job must.

    Called where nothing is half-written: after parsing, after chunking,
    before the store write. A no-op when no job is running on this thread.
    """
    guard = current_guard()
    if guard is not None:
        guard.check()


def deadline_timeout(configured: Optional[float]) -> Optional[float]:
    """A transport's socket timeout, shortened to what the current guard has
    left. ``configured`` unchanged when no guard is set on this thread.

    For code that runs on the thread that owns the guard -- the answer model
    on a request thread, unlike Deep's pool -- this is all the clamping there
    is to do: read the remaining time right before the call and hand it to
    the socket. Never below a millisecond, so a call made with no time left
    fails fast rather than waiting forever on a zero.
    """
    guard = current_guard()
    if guard is None:
        return configured
    remaining = guard.remaining()
    if remaining is None:
        return configured
    if configured is None:
        return max(0.001, remaining)
    return max(0.001, min(float(configured), remaining))


# ----------------------------------------------------------- the budgets
#
# Two, because they bound two different external services and one must not be
# able to starve the other: a re-index saturating the embedding endpoint must
# still leave Deep Analysis able to call the chat endpoint, and the reverse.
_budget_lock = threading.Lock()
_budget: Optional[ProviderBudget] = None
_embedding_budget: Optional[ProviderBudget] = None


def configure_budget(limit: int) -> ProviderBudget:
    """Install the process-wide Deep Analysis budget. The entrypoints call
    this once from settings; a test calls it with a small number of its own."""
    global _budget
    with _budget_lock:
        _budget = ProviderBudget(limit)
        return _budget


def provider_budget() -> ProviderBudget:
    """The process-wide Deep Analysis budget, built from the configured limit
    on first use if nothing installed one explicitly."""
    global _budget
    with _budget_lock:
        if _budget is None:
            from config.ingest import limits_from_env

            _budget = ProviderBudget(limits_from_env().provider_max_inflight)
        return _budget


def configure_embedding_budget(limit: int) -> ProviderBudget:
    global _embedding_budget
    with _budget_lock:
        _embedding_budget = ProviderBudget(limit)
        return _embedding_budget


def embedding_budget() -> ProviderBudget:
    """The process-wide embedding budget.

    Separate from the Deep budget on purpose. Embedding requests multiply for
    reasons Deep Analysis does not: every ingest embeds its chunks, a
    re-index embeds a whole knowledge base from a request thread, and a query
    embeds itself on the way in.
    """
    global _embedding_budget
    with _budget_lock:
        if _embedding_budget is None:
            from config.ingest import limits_from_env

            _embedding_budget = ProviderBudget(limits_from_env().embedding_max_inflight)
        return _embedding_budget


def budgets() -> dict:
    """Both budgets' counters, for the status endpoint and the load report."""
    return {
        "deep_analysis": provider_budget().snapshot(),
        "embedding": embedding_budget().snapshot(),
    }


# --------------------------------------------------------- the wrapper
class _Clamped:
    """A per-call view of a transport whose socket timeout is the job's.

    A copy rather than a mutation: the transport is shared by every thread in
    a Deep pool, and they do not all have the same amount of time left at the
    same moment. The copy is shallow and the amsc transports hold only plain
    configuration, so this costs an object and no connection state.
    """

    @staticmethod
    def wrap(inner: Any, seconds: Optional[float]) -> Any:
        if seconds is None:
            return inner
        configured = getattr(inner, "timeout_seconds", None)
        if not isinstance(configured, (int, float)):
            # A transport that does not expose its timeout cannot be clamped;
            # the deadline stays cooperative for it, which the caller reports.
            return inner
        clone = copy.copy(inner)
        clone.timeout_seconds = max(0.001, min(float(configured), float(seconds)))
        return clone


class LimitedProvider:
    """A provider that takes a budget slot around each call, under the deadline.

    Wraps anything with ``complete(prompt) -> str`` and ``model_id``; the Deep
    pipeline sees the same interface. Each call, in order:

    1. asks the job guard whether to go on at all (cancelled or out of time =
       no call is made and no slot is taken);
    2. waits for a budget slot no longer than the job has left -- so a waiter
       that keeps losing races fails as a refusal rather than hanging;
    3. clamps the transport's socket timeout to what remains, so a call that
       has started cannot run past the deadline by a whole provider timeout.

    A refused or failed call raises, and the pipeline's own policy turns that
    into ``provider_error`` for that section -- the deterministic partition,
    recorded as such -- which is exactly what a call that never happened
    should look like.

    ``clamped`` counts the calls whose timeout was actually shortened, and
    ``unclampable`` counts transports that had no timeout to clamp, so the
    deadline claim is measurable rather than assumed.

    **Both the guard and the trace are carried on the object, not looked up
    per call.** ``amsc.research.agentic.chunker.collect_votes`` runs every proposer and
    verifier call on a ``ThreadPoolExecutor``, and a thread-local set on the
    ingest worker does not exist on a pool thread. The guard was already
    passed in for that reason; the trace is captured here at construction --
    which happens on the worker thread, inside the job's Deep stage -- so the
    measurement of a call follows the job that asked for it rather than the
    thread that happened to make it. Two Deep jobs running at once therefore
    each count their own calls, whichever pool thread served them.
    """

    def __init__(self, inner: Any, budget: ProviderBudget, guard: Optional[JobGuard] = None,
                 trace: Optional["T.JobTrace"] = None):
        self.inner = inner
        self.budget = budget
        self.guard = guard
        #: The job this provider belongs to, bound now because ``complete``
        #: will be called from threads that never had it.
        self.trace = trace if trace is not None else T.current_trace()
        self.calls = 0
        self.refused = 0
        self.clamped = 0
        self.unclampable = 0
        self._lock = threading.Lock()

    @property
    def model_id(self) -> Any:
        return getattr(self.inner, "model_id", getattr(self.inner, "model", None))

    def complete(self, prompt: str) -> str:
        if self.guard is not None:
            self.guard.check()
        timeout = self.guard.remaining() if self.guard is not None else None
        try:
            with self.budget.slot(timeout=timeout) as waited:
                # Taken *after* the slot: the wait for it spends the job's
                # time too, and the call must not be given the whole of what
                # was left before that wait began.
                remaining = self.guard.remaining() if self.guard is not None else None
                if remaining is not None and remaining <= 0:
                    self.guard.check()
                transport = _Clamped.wrap(self.inner, remaining)
                with self._lock:
                    self.calls += 1
                    if transport is not self.inner:
                        self.clamped += 1
                    elif remaining is not None:
                        self.unclampable += 1
                started = time.perf_counter()
                try:
                    return transport.complete(prompt)
                finally:
                    # Recorded on the job's trace, not on a stage: these calls
                    # happen on a pool several layers below the stage that
                    # opened, and "how much of Deep was the gateway" is the
                    # question they answer. The trace is the one bound at
                    # construction, because this line often runs on a pool
                    # thread where the thread-local is empty.
                    trace = self.trace or T.current_trace()
                    if trace is not None:
                        trace.record_provider(seconds=time.perf_counter() - started,
                                              wait_seconds=waited)
        except ProviderSlotTimeout:
            with self._lock:
                self.refused += 1
            raise


class LimitedEmbeddingTransport:
    """One embedding request, one slot of the embedding budget.

    Wraps an embedding *transport* -- the object with ``embed(texts) ->
    ndarray`` that actually posts to the endpoint -- rather than the
    ``BaseEmbedding`` above it, because that is the level at which one call is
    one outbound request. The caller above it (``ResilientBatches``) already
    splits work into single batches, so a slot held here is one HTTP request
    in flight, which is what the budget is counting.

    A local model is not wrapped: it makes no request, and its cost is CPU on
    a thread that an ingest worker already accounts for.

    Unlike :class:`LimitedProvider`, this looks the guard and the trace up per
    call rather than binding them, and that difference is deliberate. This
    wrapper is built once per embedding object and an embedding object belongs
    to a cached pipeline, so it outlives any one job and is shared by every job
    and every query that uses that pipeline; binding a trace here would
    attribute one job's embeddings to another. It can afford the lookup because
    ``embed`` is always called on the thread that asked for it: ``ResilientBatches``
    above hands down one batch at a time, so the transport below never reaches
    its own pool, and there is no thread boundary between the caller and here.
    """

    def __init__(self, inner: Any, budget: ProviderBudget, guard_provider=None):
        self.inner = inner
        self.budget = budget
        self._guard_provider = guard_provider or current_guard
        self.slots_taken = 0
        self._lock = threading.Lock()

    def __getattr__(self, name: str) -> Any:
        # model_id, calls, prompt_tokens, batch_size ... stay the transport's.
        return getattr(self.inner, name)

    def embed(self, texts):
        guard = self._guard_provider()
        timeout = guard.remaining() if guard is not None else None
        if guard is not None:
            guard.check()
        with self.budget.slot(timeout=timeout) as waited:
            with self._lock:
                self.slots_taken += 1
            transport = _Clamped.wrap(self.inner, timeout)
            started = time.perf_counter()
            try:
                return transport.embed(texts)
            finally:
                trace = T.current_trace()
                if trace is not None:
                    trace.record_embedding(seconds=time.perf_counter() - started,
                                           wait_seconds=waited)
