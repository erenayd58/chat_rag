"""The limits every query runs under: admission, the answer budget, the
deadline -- and the truth of what each one claims.

Pinned here: the number of answer-model calls in flight across the process
never exceeds the configured maximum, whichever sessions ask; a slot is
given back on every exit -- success, provider failure, timeout; a wait for a
slot is bounded by the query's deadline and by its own cap; the transports
really shorten their socket timeouts under a deadline and skip a retry
there is no time for; admission never blocks. Nothing here sleeps: every
ordering is proved by holding and releasing events, and the two waits that
must run out do so against a clock the test owns or a bound of a fraction
of a second.
"""

from __future__ import annotations

import threading
import time
import urllib.error
import urllib.request

import pytest

from chat_rag.components.ingest import limits as L
from chat_rag.components.llm import FallbackLLM, OpenAICompatibleLLM
from chat_rag.components.observability import telemetry as T
from chat_rag.components.query import limits as Q
from chat_rag.config.query import QueryLimits, query_limits_from_env
from chat_rag.core.exceptions import LLMException, QueryOverloaded, QueryTimeout

from query_doubles import FailingAnswerModel, GatedAnswerModel


# ---------------------------------------------------------------- config


def test_the_defaults_validate_and_leave_a_thread_free():
    limits = query_limits_from_env({})
    assert limits.answer_max_inflight == 4
    assert limits.timeout_seconds == 180.0
    # Eight threads, four synchronous-upload waiters: three for queries,
    # one that neither can take.
    assert limits.max_active == 3
    assert limits.free_threads == 1


def test_the_admission_default_follows_the_thread_count_and_the_upload_ration():
    assert query_limits_from_env({"WAITRESS_THREADS": "16"}).max_active == 7
    assert query_limits_from_env({"WAITRESS_THREADS": "2"}).max_active == 1
    assert query_limits_from_env({"WAITRESS_THREADS": "8", "INGEST_SYNC_WAITERS": "1"}).max_active == 6
    explicit = query_limits_from_env({"QUERY_MAX_ACTIVE": "7"})
    assert explicit.max_active == 7 and explicit.free_threads == -3, "an explicit choice is kept, and named"


@pytest.mark.parametrize("variable, value, wording", [
    ("QUERY_MAX_ACTIVE", "0", "QUERY_MAX_ACTIVE must be at least 1"),
    ("ANSWER_MAX_INFLIGHT", "0", "ANSWER_MAX_INFLIGHT must be at least 1"),
    ("QUERY_TIMEOUT", "0", "QUERY_TIMEOUT must be a positive"),
    ("ANSWER_SLOT_WAIT", "-1", "ANSWER_SLOT_WAIT must be a positive"),
    ("QUERY_MAX_ACTIVE", "many", "QUERY_MAX_ACTIVE='many' is not a whole number"),
])
def test_a_bad_value_is_refused_by_name(variable, value, wording):
    with pytest.raises(ValueError) as refused:
        query_limits_from_env({variable: value})
    assert wording in str(refused.value)


def test_settings_fail_at_construction(monkeypatch):
    from chat_rag.config import Settings

    monkeypatch.setenv("ANSWER_MAX_INFLIGHT", "0")
    with pytest.raises(ValueError):
        Settings.from_env()


def test_the_limits_describe_themselves():
    described = QueryLimits().to_dict()
    assert described["deadline_semantics"] == "cooperative-with-clamped-calls"
    assert set(described) >= {"max_active", "answer_max_inflight", "timeout_seconds", "free_threads"}


# ---------------------------------------------------------------- budget
@pytest.fixture
def budget(monkeypatch):
    fresh = L.ProviderBudget(2)
    monkeypatch.setattr(Q, "_answer_budget", fresh)
    return fresh


def _generate(model, results, index, guard=None):
    try:
        with L.use_guard(guard):
            results[index] = model.generate([{"role": "user", "content": f"q{index}"}])
    except BaseException as error:  # noqa: BLE001 - reported by the test
        results[index] = error


def test_answer_calls_in_flight_never_exceed_the_budget(budget):
    """Six callers, a budget of two: two inside the model, four waiting."""
    shared = GatedAnswerModel(expect=2)
    model = Q.LimitedAnswerModel(shared, budget)
    results = {}
    threads = [threading.Thread(target=_generate, args=(model, results, i)) for i in range(6)]
    for thread in threads:
        thread.start()
    assert shared.full.wait(10), "the model never reached two calls in flight"
    assert shared.inflight == 2 and budget.snapshot()["inflight"] == 2
    shared.release()
    for thread in threads:
        thread.join(30)
    assert all(isinstance(r, str) for r in results.values()), results
    assert shared.peak == 2 == budget.peak
    assert budget.snapshot()["inflight"] == 0
    assert budget.acquired_total == 6 == shared.calls


def test_sessions_share_one_budget(budget):
    """Two pipelines' worth of wrappers over two different models, one
    budget between them: the cap is on the process, not on a session."""
    first, second = GatedAnswerModel(expect=1), GatedAnswerModel(expect=1)
    wrappers = [Q.LimitedAnswerModel(first, budget), Q.LimitedAnswerModel(second, budget),
                Q.LimitedAnswerModel(first, budget), Q.LimitedAnswerModel(second, budget)]
    results = {}
    threads = [threading.Thread(target=_generate, args=(w, results, i)) for i, w in enumerate(wrappers)]
    for thread in threads:
        thread.start()
    assert first.full.wait(10) and second.full.wait(10)
    # Both models have a call inside; the budget is full; nobody else is in.
    assert budget.snapshot()["inflight"] == 2
    assert first.inflight + second.inflight == 2
    first.release()
    second.release()
    for thread in threads:
        thread.join(30)
    # The bound is on the process, so it is ``budget.peak`` that carries it:
    # two calls in flight at once, whichever models they were made on. Adding
    # the two models' own peaks does not measure that -- they can happen at
    # different moments (both of `first`'s calls may hold the two slots while
    # `second` holds none), which is legal under a shared budget and used to
    # fail this test under load.
    assert budget.peak == 2
    assert first.peak <= 2 and second.peak <= 2
    assert budget.snapshot()["inflight"] == 0


def test_a_failed_call_gives_its_slot_back(budget):
    model = Q.LimitedAnswerModel(FailingAnswerModel(), budget)
    for _ in range(5):
        with pytest.raises(LLMException):
            model.generate([{"role": "user", "content": "x"}])
    assert budget.snapshot()["inflight"] == 0
    assert budget.acquired_total == 5
    # ... and the budget is still usable afterwards, by anyone.
    ok = Q.LimitedAnswerModel(GatedAnswerModel(), budget)
    ok.inner.release()
    assert ok.generate([]) == ok.inner.reply


def test_a_call_that_raises_something_else_still_releases(budget):
    class Broken(GatedAnswerModel):
        def generate(self, *a, **k):
            raise RuntimeError("not even an LLMException")

    with pytest.raises(RuntimeError):
        Q.LimitedAnswerModel(Broken(), budget).generate([])
    assert budget.snapshot()["inflight"] == 0


# ------------------------------------------------------------- deadline
class Clock:
    def __init__(self, now=1000.0):
        self.now = now

    def __call__(self):
        return self.now


def test_an_expired_query_makes_no_call_and_takes_no_slot(budget):
    clock = Clock()
    guard = Q.QueryGuard(deadline=clock.now - 1, clock=clock)
    inner = GatedAnswerModel()
    with L.use_guard(guard):
        with pytest.raises(QueryTimeout):
            Q.LimitedAnswerModel(inner, budget).generate([])
    assert inner.calls == 0
    assert budget.acquired_total == 0 and budget.refused_total == 0


def test_the_slot_wait_is_bounded_by_the_query_deadline(monkeypatch):
    """The budget is held by someone else and the query has 0.2 s left: the
    wait ends when the deadline does, as a timeout -- never hangs."""
    one = L.ProviderBudget(1)
    holder = GatedAnswerModel(expect=1)
    held = Q.LimitedAnswerModel(holder, one)
    results = {}
    thread = threading.Thread(target=_generate, args=(held, results, "holder"))
    thread.start()
    assert holder.full.wait(10)
    try:
        guard = Q.QueryGuard.for_timeout(0.2)
        waiting = Q.LimitedAnswerModel(GatedAnswerModel(), one, wait_seconds=30.0)
        started = time.perf_counter()
        with L.use_guard(guard):
            with pytest.raises(QueryTimeout):
                waiting.generate([])
        assert time.perf_counter() - started < 5.0, "bounded by the deadline, not by the 30 s cap"
        assert one.refused_total == 1
        assert waiting.inner.calls == 0, "never got in, so never called"
    finally:
        holder.release()
        thread.join(10)
    assert one.snapshot()["inflight"] == 0


def test_the_slot_wait_is_bounded_by_its_own_cap_and_is_an_overload():
    """Plenty of deadline left but the answer cap is short: the query is
    refused as answer_capacity overload with a retry hint, so a person is
    told "busy" in seconds rather than waiting out a deadline."""
    one = L.ProviderBudget(1)
    holder = GatedAnswerModel(expect=1)
    results = {}
    thread = threading.Thread(target=_generate, args=(Q.LimitedAnswerModel(holder, one), results, "h"))
    thread.start()
    assert holder.full.wait(10)
    try:
        guard = Q.QueryGuard.for_timeout(60.0)
        waiting = Q.LimitedAnswerModel(GatedAnswerModel(), one, wait_seconds=0.1)
        with L.use_guard(guard):
            with pytest.raises(QueryOverloaded) as refused:
                waiting.generate([])
        assert refused.value.reason == "answer_capacity"
        assert refused.value.retry_after_seconds >= Q.RETRY_AFTER_MIN
    finally:
        holder.release()
        thread.join(10)


def test_the_openai_compatible_transport_clamps_its_socket_timeout(monkeypatch):
    monkeypatch.setenv("ANSWER_TEST_KEY", "sk-test")
    seen = []

    class _Response:
        def __init__(self, body):
            self.body = body

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return self.body

    def fake_urlopen(request, timeout):
        seen.append(timeout)
        return _Response(b'{"choices":[{"message":{"content":"ok"},"finish_reason":"stop"}],"usage":{}}')

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    llm = OpenAICompatibleLLM("m", api_key_env="ANSWER_TEST_KEY", timeout_seconds=120.0)
    # No guard: the configured timeout, as before.
    llm.generate([{"role": "user", "content": "x"}])
    assert seen == [120.0]
    # Under a deadline with five seconds left: five seconds, not 120.
    clock = Clock()
    with L.use_guard(Q.QueryGuard(deadline=clock.now + 5.0, clock=clock)):
        llm.generate([{"role": "user", "content": "x"}])
    assert seen[-1] == pytest.approx(5.0)
    assert llm.last_usage["timeout_seconds"] == pytest.approx(5.0)


def test_a_retry_there_is_no_time_for_is_skipped(monkeypatch):
    """Two attempts are configured and the first fails; with half a second
    left the transport reports the failure instead of sleeping a second."""
    monkeypatch.setenv("ANSWER_TEST_KEY", "sk-test")
    attempts = []

    def unreachable(request, timeout):
        attempts.append(timeout)
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(urllib.request, "urlopen", unreachable)
    llm = OpenAICompatibleLLM("m", api_key_env="ANSWER_TEST_KEY", retries=2)
    clock = Clock()
    started = time.perf_counter()
    with L.use_guard(Q.QueryGuard(deadline=clock.now + 0.5, clock=clock)):
        with pytest.raises(LLMException):
            llm.generate([{"role": "user", "content": "x"}])
    assert len(attempts) == 1
    assert time.perf_counter() - started < 0.5, "no back-off sleep was spent"


def test_the_fallback_is_not_tried_with_no_time_left():
    class Primary(GatedAnswerModel):
        def generate(self, *a, **k):
            self.calls += 1
            raise LLMException("down")

    primary, fallback = Primary(), GatedAnswerModel()
    fallback.release()
    chain = FallbackLLM(primary, fallback)
    # With time: the fallback answers, as before.
    assert chain.generate([]) == fallback.reply and fallback.calls == 1
    # Out of time between the two: the deadline, not a second model.
    clock = Clock()
    with L.use_guard(Q.QueryGuard(deadline=clock.now - 1, clock=clock)):
        with pytest.raises(QueryTimeout):
            chain.generate([])
    assert fallback.calls == 1, "not called again"


def test_the_embedding_transport_honours_a_query_guard():
    """The ingest embedding wrapper reads the current guard; a query's
    guard is one, so a query out of time embeds nothing."""
    from query_doubles import FakeEmbeddingTransport

    transport = FakeEmbeddingTransport()
    limited = L.LimitedEmbeddingTransport(transport, L.ProviderBudget(1))
    clock = Clock()
    with L.use_guard(Q.QueryGuard(deadline=clock.now - 1, clock=clock)):
        with pytest.raises(QueryTimeout):
            limited.embed(["soru"])
    assert transport.calls == 0


def test_deadline_timeout_is_the_smaller_of_configured_and_remaining():
    assert L.deadline_timeout(120.0) == 120.0, "no guard: unchanged"
    clock = Clock()
    with L.use_guard(Q.QueryGuard(deadline=clock.now + 30.0, clock=clock)):
        assert L.deadline_timeout(120.0) == pytest.approx(30.0)
        assert L.deadline_timeout(10.0) == pytest.approx(10.0)
        assert L.deadline_timeout(None) == pytest.approx(30.0)
    with L.use_guard(Q.QueryGuard(deadline=clock.now - 1, clock=clock)):
        assert L.deadline_timeout(120.0) == 0.001, "never a zero the socket would wait forever on"
    with L.use_guard(Q.QueryGuard(deadline=None, clock=clock)):
        assert L.deadline_timeout(120.0) == 120.0, "a guard without a deadline clamps nothing"


# ------------------------------------------------------------- admission
def test_admission_never_blocks_and_counts_what_it_refused():
    admission = Q.QueryAdmission(2)
    assert admission.try_enter() and admission.try_enter()
    assert admission.saturated
    started = time.perf_counter()
    assert admission.try_enter() is False
    assert time.perf_counter() - started < 0.1
    admission.leave()
    assert not admission.saturated and admission.try_enter()
    admission.leave()
    admission.leave()
    assert admission.snapshot() == {"limit": 2, "active": 0, "peak": 2,
                                    "accepted_total": 3, "rejected_total": 1}


def test_admission_as_a_context_manager_releases_on_error():
    admission = Q.QueryAdmission(1)
    with pytest.raises(RuntimeError):
        with admission.enter():
            assert admission.active == 1
            raise RuntimeError
    assert admission.active == 0
    with admission.enter():
        with pytest.raises(QueryOverloaded) as refused:
            with admission.enter():
                pass
        assert refused.value.reason == "admission"
    assert admission.active == 0


def test_a_refused_query_never_leaves(monkeypatch):
    """Whoever never entered must not decrement on the way out."""
    admission = Q.QueryAdmission(1)
    with admission.enter():
        with pytest.raises(QueryOverloaded):
            with Q.query_scope(admission, timeout_seconds=10, kb_id="kb"):
                pass
        assert admission.active == 1
    assert admission.active == 0


def test_the_wrapper_is_a_transparent_view_of_the_model(budget):
    inner = GatedAnswerModel()
    inner.release()
    inner.last_call = {"provider": "openrouter"}
    model = Q.LimitedAnswerModel(inner, budget)
    assert model.get_name() == "GatedAnswer" and model.get_model_name() == "test/gated-answer"
    assert model.provider_id == "openrouter" and model.last_call == {"provider": "openrouter"}
    assert model.generate([]) == inner.reply
    assert model.last_usage == inner.last_usage
