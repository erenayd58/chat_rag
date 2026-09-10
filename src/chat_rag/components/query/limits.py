"""What bounds a query while it runs, and how each bound is measured.

A query is the other runtime path. Ingest became jobs in Phase 2 -- bounded
workers, a queue, budgets around every outbound call. A query cannot become a
job: a person is waiting on the request thread for the answer, so retrieval,
context assembly and the answer-model call all run on that thread. What can
multiply under load is therefore *request threads held for the length of a
provider call*, and that is what everything here bounds.

Three mechanisms, each the smallest that closes one failure:

**Admission** (:class:`QueryAdmission`, ``QUERY_MAX_ACTIVE``). How many
request threads may be inside a query at once. A query that finds no slot is
refused *immediately* -- ``503``, ``overloaded: true``, a ``Retry-After`` --
never queued, because a queued query would hold the very thread the limit is
there to keep free. The default is derived so that queries and synchronous
uploads together can never take every request thread, which is what keeps
``/api/health`` and job polling answerable under any burst of either. The
proof is ``tests/integration/test_query_starvation.py``, on a real server.

**The answer budget** (``ANSWER_MAX_INFLIGHT``). One :class:`ProviderBudget`
for answer-model calls, shared by every session in the process, taken around
each ``generate`` by :class:`LimitedAnswerModel`. It is a *third* budget
rather than a share of the Deep Analysis one, and that is a deliberate
reading of the topology. In the demo all three roles -- Deep proposer and
verifier, embeddings, answers -- go through one gateway with one key, so a
single cap on "OpenRouter calls" is the obvious alternative. It is wrong for
this product for two reasons. First, the calls differ in kind: a Deep call is
one short vote among dozens made in bulk from a background worker, and an
answer call is one long interactive completion (a reasoning model, 1200+
tokens) that a person is watching. Sharing a semaphore would let a Deep job
holding eight slots make every chat wait for an ingest to finish, and the
reverse -- a busy afternoon of chat would stall the ingest queue. Second, the
two are not always the same service: the fallback answer model is a local
Ollama, whose cost is CPU and memory on this host, and a cap on it must not
be spent by Deep calls to a remote gateway. Separate budgets guarantee each
path a floor of capacity whatever the other is doing. A gateway-level ceiling
across all three, if a deployment ever needed one, would be a *fourth*
semaphore wrapped around these, not a merger of them.

**The deadline** (:class:`QueryGuard`, ``QUERY_TIMEOUT``). The ingest guard,
reused: carried on the same thread-local, so the embedding transport already
clamps a query's embedding call and refuses one there is no time for, with
no new plumbing. What is honestly claimed is in ``QUERY_DEADLINE_SEMANTICS``:
the deadline is checked before every outbound call and while waiting for a
slot; each answer attempt's socket timeout is clamped to the time left, and
a retry is skipped when there is no time for one; but a stage already
running -- a lexical index rebuild on a request thread, a store lookup, a
local model's forward pass, an Ollama call whose client timeout is fixed at
construction -- runs to its end. A query can overshoot its deadline by the
longest such stage, and never by a whole provider timeout on top.

Capacity is released on every exit. Admission is a context manager, the
budget slot is a context manager, and the trace is closed in a ``finally``;
a query that raises anywhere gives everything back on the way out.

Measurement reuses Phase 3's single abstraction: a query is a ``JobTrace``
of kind ``query`` with its own stages (retrieve, context, answer),
and the answer wrapper records provider seconds and slot wait on it exactly
as the Deep wrapper does. Nothing here is a second telemetry system.
"""

from __future__ import annotations

import threading
import time
import uuid
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional

from chat_rag.components.ingest.limits import (
    JobGuard, ProviderBudget, ProviderSlotTimeout, current_guard, use_guard,
)
from chat_rag.components.llm.base import BaseLLM
from chat_rag.components.observability import events
from chat_rag.components.observability import telemetry as T
from chat_rag.core.exceptions import LLMException, QueryOverloaded, QueryTimeout

#: What "deadline" means for a query, in one sentence, for anything that
#: reports it.
QUERY_DEADLINE_SEMANTICS = (
    "cooperative: checked before every outbound call and while waiting for an "
    "answer slot; each answer attempt's socket timeout is clamped to the time "
    "left and a retry is skipped when there is none; a stage already running "
    "(index rebuild, store lookup, local model, an Ollama call) runs to its end"
)

#: Bounds on the Retry-After a refused query is given: long enough to mean
#: something, short enough that a person will still be there.
RETRY_AFTER_MIN = 2.0
RETRY_AFTER_MAX = 30.0
RETRY_AFTER_DEFAULT = 5.0


class QueryGuard(JobGuard):
    """One query's deadline. The ingest guard's clock and seams, a query's
    own error: what stops a query is a :class:`QueryTimeout`, never an
    ``IngestInterrupted`` that a log reader would go looking for a job for."""

    def check(self) -> None:
        if self.cancelled:
            raise QueryTimeout(self.cancel_reason or "the query was cancelled")
        if self.expired:
            raise QueryTimeout()


# ----------------------------------------------------------- the budget
_budget_lock = threading.Lock()
_answer_budget: Optional[ProviderBudget] = None


def configure_answer_budget(limit: int) -> ProviderBudget:
    """Install the process-wide answer-model budget. The entrypoint calls
    this once from settings; a test calls it with a small number of its own."""
    global _answer_budget
    with _budget_lock:
        _answer_budget = ProviderBudget(limit)
        return _answer_budget


def answer_budget() -> ProviderBudget:
    """The process-wide answer-model budget, built from the configured limit
    on first use if nothing installed one explicitly."""
    global _answer_budget
    with _budget_lock:
        if _answer_budget is None:
            from chat_rag.config.query import query_limits_from_env

            _answer_budget = ProviderBudget(query_limits_from_env().answer_max_inflight)
        return _answer_budget


# --------------------------------------------------------- the wrapper
class LimitedAnswerModel(BaseLLM):
    """An answer model that takes a budget slot around each call, under the
    query's deadline.

    Wraps any ``BaseLLM`` -- the ``FallbackLLM`` chain in production, a fake
    in a test -- and is what the pipeline calls for generated text. Each
    call, in order:

    1. asks the query guard whether to go on at all (out of time = no call
       is made and no slot is taken);
    2. waits for an answer slot no longer than ``wait_seconds`` and no longer
       than the query has left -- a wait that runs out is an
       ``answer_capacity`` overload, refused rather than hung;
    3. makes the call. The transports clamp their own socket timeouts to the
       deadline (they run on this thread, so they can read the guard
       themselves), which is why there is no per-call copy here as the Deep
       wrapper needs.

    The wrapper holds no state of its own beyond its configuration, so it is
    safe to build one per access: the pipeline exposes it as a property over
    ``llm_model`` and a test that swaps the model underneath is honoured.
    Everything the pipeline reads off the model -- ``last_call``,
    ``last_usage``, ``describe``, ``provider_id`` -- is the inner model's.
    """

    def __init__(self, inner: BaseLLM, budget: Optional[ProviderBudget] = None,
                 *, wait_seconds: Optional[float] = None):
        self.inner = inner
        self.budget = budget if budget is not None else answer_budget()
        self.wait_seconds = wait_seconds

    def __getattr__(self, name: str) -> Any:
        # Only reached for names this class does not define: last_call,
        # last_usage, describe, provider_id, primary, fallback ...
        return getattr(self.inner, name)

    def generate(self, messages: List[Dict[str, str]], temperature: float = 0.3,
                 max_tokens: int = 200, **kwargs: Any) -> str:
        guard = current_guard()
        if guard is not None:
            guard.check()
        timeout = self.wait_seconds
        if guard is not None:
            remaining = guard.remaining()
            if remaining is not None:
                timeout = remaining if timeout is None else min(timeout, remaining)
        try:
            with self.budget.slot(timeout=timeout) as waited:
                # The wait spent the query's time too.
                if guard is not None:
                    guard.check()
                started = time.perf_counter()
                try:
                    return self.inner.generate(messages, temperature, max_tokens, **kwargs)
                finally:
                    trace = T.current_trace()
                    if trace is not None:
                        trace.record_provider(seconds=time.perf_counter() - started,
                                              wait_seconds=waited)
        except ProviderSlotTimeout as refused:
            # Out of time altogether is a deadline, not a capacity problem.
            if guard is not None and guard.expired:
                raise QueryTimeout() from refused
            raise QueryOverloaded(
                "the answer model is at capacity; no slot came free within "
                f"{(timeout or 0):.0f}s (limit {self.budget.limit})",
                reason="answer_capacity",
                retry_after_seconds=_retry_after(),
            ) from refused

    def get_name(self) -> str:
        return self.inner.get_name()

    def get_model_name(self) -> str:
        return self.inner.get_model_name()


# --------------------------------------------------------- admission
class QueryAdmission:
    """How many request threads may be inside a query at once.

    Non-blocking by design: a query either enters now or is refused now.
    ``peak`` is what a test asserts on; ``active`` is what health shows.
    """

    def __init__(self, limit: int):
        if limit < 1:
            raise ValueError("query admission needs at least one slot")
        self.limit = int(limit)
        self._lock = threading.Lock()
        self.active = 0
        self.peak = 0
        self.accepted_total = 0
        self.rejected_total = 0

    def try_enter(self) -> bool:
        with self._lock:
            if self.active >= self.limit:
                self.rejected_total += 1
                return False
            self.active += 1
            self.accepted_total += 1
            self.peak = max(self.peak, self.active)
            return True

    def leave(self) -> None:
        with self._lock:
            self.active = max(0, self.active - 1)

    @contextmanager
    def enter(self) -> Iterator[None]:
        """Hold a slot for the block, or raise :class:`QueryOverloaded`
        before it -- a caller that never entered never leaves."""
        if not self.try_enter():
            raise QueryOverloaded(
                f"every query slot is in use ({self.limit} at once); try again shortly",
                reason="admission",
                retry_after_seconds=_retry_after(),
            )
        try:
            yield
        finally:
            self.leave()

    @property
    def saturated(self) -> bool:
        with self._lock:
            return self.active >= self.limit

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "limit": self.limit,
                "active": self.active,
                "peak": self.peak,
                "accepted_total": self.accepted_total,
                "rejected_total": self.rejected_total,
            }


def _retry_after() -> float:
    """When to try again: the median recent query time, within bounds, so the
    number tracks what the deployment actually does. A fixed default until
    there is anything to measure."""
    summary = T.metrics().query_latency()
    p50 = summary.get("p50")
    if not p50:
        return RETRY_AFTER_DEFAULT
    return round(max(RETRY_AFTER_MIN, min(RETRY_AFTER_MAX, float(p50))), 1)


# ---------------------------------------------------------- the scope
class QueryScope:
    """Everything a query's lifetime carries: its id, trace and guard, and
    the outcome it ended with. Filled in by :func:`query_scope`."""

    def __init__(self, query_id: str, trace: T.JobTrace, guard: QueryGuard):
        self.query_id = query_id
        self.trace = trace
        self.guard = guard
        self.status = ""
        self.error_category = "none"

    def failed(self, message: str, *, category: str = "unknown") -> None:
        """Mark this query failed from inside the block.

        For a caller that handles its own errors and answers with a status
        code rather than raising -- the Lab's retrieval endpoints do, and a
        scope that saw a clean exit would otherwise count a 500 as a
        success. The category is ``unknown`` by default because the caller
        swallowed the exception and nothing here can categorise what it was.
        """
        self.status = "failed"
        self.error_category = category
        T.metrics().record_error(category, message)

    def timing(self) -> dict[str, Any]:
        """What the response carries so a slow answer can be correlated with
        the metrics: stage seconds, provider seconds and wait. No content."""
        summary = self.trace.as_dict()
        return {
            "query_id": self.query_id,
            "stages": summary["stages"],
            "provider": summary["provider"],
            "embedding": summary["embedding"],
            "total_seconds": summary["total_seconds"],
        }


@contextmanager
def query_scope(admission: QueryAdmission, *, timeout_seconds: Optional[float],
                kb_id: Optional[str], mode: str = "",
                session_id: str = "") -> Iterator[QueryScope]:
    """Run one query under admission, a deadline and a trace.

    Raises :class:`QueryOverloaded` *before* the block when no slot is free.
    Inside the block the guard and trace are current on this thread, so the
    pipeline's stages and the budget wrappers measure themselves. On the way
    out -- whatever happened -- admission is released, the trace is closed
    into the registry, the outcome is counted and one event line is written.
    """
    metrics = T.metrics()
    if not admission.try_enter():
        _refuse(admission, kb_id)
    query_id = uuid.uuid4().hex[:12]
    trace = T.JobTrace(job_id=query_id, kb_id=kb_id or "default", mode=mode, kind=T.KIND_QUERY)
    guard = QueryGuard.for_timeout(timeout_seconds)
    scope = QueryScope(query_id, trace, guard)
    metrics.count("query.accepted")
    metrics.begin_query()
    events.emit("query.started", query_id=query_id, kb_id=trace.kb_id, mode=mode,
                session=(session_id or "")[:8], active=admission.active)
    started = time.perf_counter()
    try:
        with use_guard(guard), T.use_trace(trace):
            yield scope
        # ``or``, not a plain assignment: a caller that answered 500 without
        # raising has already said so through :meth:`QueryScope.failed`.
        scope.status = scope.status or "succeeded"
    except QueryTimeout as stop:
        scope.status, scope.error_category = "timed_out", "timed_out"
        _fail(metrics, scope, stop)
        raise
    except QueryOverloaded as full:
        scope.status, scope.error_category = "rejected", "overloaded"
        _fail(metrics, scope, full)
        raise
    except LLMException as unavailable:
        scope.status, scope.error_category = "failed", "provider"
        _fail(metrics, scope, unavailable)
        raise
    except BaseException as error:
        scope.status, scope.error_category = "failed", T.categorise(error)
        _fail(metrics, scope, error)
        raise
    finally:
        admission.leave()
        metrics.end_query()
        seconds = time.perf_counter() - started
        trace.total_seconds = seconds
        trace.status = scope.status
        trace.error_category = scope.error_category
        metrics.finish(trace)
        metrics.count(f"query.{scope.status}")
        summary = trace.as_dict()
        fields = {
            "query_id": query_id, "kb_id": trace.kb_id, "mode": mode,
            "seconds": round(seconds, 3),
            "provider_calls": summary["provider"]["calls"],
            "provider_seconds": summary["provider"]["seconds"],
            "provider_wait_seconds": summary["provider"]["wait_seconds"],
        }
        fields.update({f"t_{name}": value for name, value in summary["stages"].items()})
        if scope.status == "succeeded":
            events.emit("query.succeeded", **fields)
        else:
            events.warn(f"query.{scope.status}", error_category=scope.error_category, **fields)


def _refuse(admission: QueryAdmission, kb_id: Optional[str]) -> None:
    metrics = T.metrics()
    metrics.count("query.rejected")
    metrics.record_error("overloaded", "every query slot is in use")
    events.warn("query.rejected", kb_id=kb_id or "default", reason="admission",
                active=admission.active, limit=admission.limit)
    raise QueryOverloaded(
        f"every query slot is in use ({admission.limit} at once); try again shortly",
        reason="admission",
        retry_after_seconds=_retry_after(),
    )


def _fail(metrics: T.MetricsRegistry, scope: QueryScope, error: BaseException) -> None:
    metrics.record_error(scope.error_category, str(error) or type(error).__name__)
