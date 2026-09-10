"""Two claims that were softer than they read: the deadline, and what is
counted as an outbound call.

**The deadline.** "A job stops when its time is up" was true at stage
boundaries and false in between: a provider call that had already started
ran on its own socket timeout, so a job could overshoot by a whole
``DEEP_ANALYSIS_TIMEOUT`` after passing its last seam. These tests pin what
is actually true now -- the call's socket timeout is clamped to what the job
has left, so the overshoot is bounded by the stage, not by the provider.

**The embedding budget.** ``PROVIDER_MAX_INFLIGHT`` covered Deep Analysis
only. Embeddings are provider-facing too whenever the embedding provider is
a remote endpoint (the demo's default), and they multiply from more places
than Deep does. These tests pin that they are counted, and against their own
budget rather than Deep's.
"""

from __future__ import annotations

import threading

import numpy as np
import pytest

from chat_rag.components.ingest import limits as L
from chat_rag.core.exceptions import IngestInterrupted

from ingest_doubles import GatedProvider


class TimedTransport:
    """A transport that reports the timeout it was handed, per call."""

    model_id = "test:timed@1"

    def __init__(self, timeout_seconds: float = 120.0):
        self.timeout_seconds = timeout_seconds
        self.seen: list[float] = []
        self.lock = threading.Lock()

    def complete(self, prompt: str) -> str:
        with self.lock:
            self.seen.append(self.timeout_seconds)
        return "{}"

    def embed(self, texts):
        with self.lock:
            self.seen.append(self.timeout_seconds)
        return np.zeros((len(texts), 3), dtype=np.float32)


class TimelessTransport:
    """A transport with no timeout to clamp: the deadline stays cooperative."""

    model_id = "test:timeless@1"

    def __init__(self):
        self.calls = 0

    def complete(self, prompt: str) -> str:
        self.calls += 1
        return "{}"


# ---------------------------------------------------------------- deadline


def test_a_call_is_given_only_the_time_the_job_has_left():
    clock = {"now": 0.0}
    guard = L.JobGuard.for_timeout(30, clock=lambda: clock["now"])
    transport = TimedTransport(timeout_seconds=120.0)
    limited = L.LimitedProvider(transport, L.ProviderBudget(2), guard)

    limited.complete("first")
    assert transport.seen == [30.0], "not the transport's own 120s"

    clock["now"] = 25.0  # five seconds of the job left
    limited.complete("second")
    assert transport.seen[-1] == 5.0
    assert limited.clamped == 2
    assert transport.timeout_seconds == 120.0, "the shared transport is not mutated"


def test_a_call_is_never_given_more_time_than_the_transport_allows():
    clock = {"now": 0.0}
    guard = L.JobGuard.for_timeout(3600, clock=lambda: clock["now"])
    transport = TimedTransport(timeout_seconds=60.0)
    L.LimitedProvider(transport, L.ProviderBudget(1), guard).complete("x")
    assert transport.seen == [60.0], "the configured timeout is still the ceiling"


def test_the_time_spent_waiting_for_a_slot_is_taken_off_the_call():
    """A call that waited must not then be handed the whole of what was left
    before it waited."""
    clock = {"now": 0.0}
    guard = L.JobGuard.for_timeout(100, clock=lambda: clock["now"])
    transport = TimedTransport(timeout_seconds=1000.0)
    budget = L.ProviderBudget(1)
    limited = L.LimitedProvider(transport, budget, guard)

    holder_in = threading.Event()
    release = threading.Event()

    def hold():
        with budget.slot():
            holder_in.set()
            release.wait(10)

    holder = threading.Thread(target=hold)
    holder.start()
    assert holder_in.wait(10)

    def advance_then_release():
        clock["now"] = 40.0  # the wait cost the job forty seconds
        release.set()

    threading.Timer(0.05, advance_then_release).start()
    limited.complete("after waiting")
    holder.join(10)
    assert transport.seen == [60.0], "what was left after the wait, not before it"


def test_a_job_with_no_time_left_after_waiting_makes_no_call():
    clock = {"now": 0.0}
    guard = L.JobGuard.for_timeout(10, clock=lambda: clock["now"])
    transport = TimedTransport()
    limited = L.LimitedProvider(transport, L.ProviderBudget(1), guard)

    class ExpiringBudget(L.ProviderBudget):
        def slot(self, timeout=None):
            clock["now"] = 11.0  # the wait consumed the rest of the job
            return super().slot(timeout=timeout)

    limited.budget = ExpiringBudget(1)
    with pytest.raises(IngestInterrupted) as stopped:
        limited.complete("too late")
    assert stopped.value.kind == "timed_out"
    assert transport.seen == []


def test_a_transport_without_a_timeout_is_counted_as_cooperative():
    """No claim is made that cannot be kept: a transport with no timeout to
    clamp is recorded as such rather than silently treated as bounded."""
    guard = L.JobGuard.for_timeout(30)
    transport = TimelessTransport()
    limited = L.LimitedProvider(transport, L.ProviderBudget(1), guard)
    limited.complete("x")
    assert transport.calls == 1
    assert limited.clamped == 0 and limited.unclampable == 1


def test_without_a_deadline_nothing_is_clamped():
    transport = TimedTransport(timeout_seconds=120.0)
    L.LimitedProvider(transport, L.ProviderBudget(1)).complete("x")
    assert transport.seen == [120.0]


def test_the_semantics_are_stated_where_they_can_be_quoted():
    assert "cooperative" in L.JOB_DEADLINE_SEMANTICS
    assert "clamping" in L.JOB_DEADLINE_SEMANTICS


# --------------------------------------------------------------- embeddings


def test_an_embedding_request_takes_a_slot_of_the_embedding_budget():
    budget = L.ProviderBudget(2)
    transport = TimedTransport()
    limited = L.LimitedEmbeddingTransport(transport, budget)
    vectors = limited.embed(["a", "b"])
    assert vectors.shape == (2, 3)
    assert budget.acquired_total == 1
    assert budget.snapshot()["inflight"] == 0
    assert limited.model_id == "test:timed@1", "the transport's identity is preserved"


def test_the_embedding_budget_bounds_concurrent_requests():
    budget = L.ProviderBudget(2)
    provider = GatedProvider(expect=2)

    class EmbeddingDouble:
        model_id = "test:embed@1"

        def embed(self, texts):
            provider.complete("embedding")
            return np.zeros((len(texts), 3), dtype=np.float32)

    limited = L.LimitedEmbeddingTransport(EmbeddingDouble(), budget)
    threads = [threading.Thread(target=lambda: limited.embed(["x"])) for _ in range(6)]
    for thread in threads:
        thread.start()
    assert provider.full.wait(20)
    assert provider.inflight == 2
    provider.release()
    for thread in threads:
        thread.join(20)
    assert provider.peak == 2 == budget.peak
    assert budget.acquired_total == 6


def test_the_two_budgets_are_separate_so_neither_starves_the_other():
    deep = L.configure_budget(1)
    embedding = L.configure_embedding_budget(1)
    try:
        assert L.provider_budget() is deep
        assert L.embedding_budget() is embedding
        assert deep is not embedding
        with deep.slot():
            # A saturated Deep budget does not block an embedding request.
            with embedding.slot():
                assert embedding.snapshot()["inflight"] == 1
                assert deep.snapshot()["inflight"] == 1
        assert set(L.budgets()) == {"deep_analysis", "embedding"}
    finally:
        L.configure_budget(8)
        L.configure_embedding_budget(4)


def test_the_remote_embedding_provider_routes_its_calls_through_the_budget():
    """The wiring, through the real ``OpenAICompatibleEmbedding``: one HTTP
    request, one slot -- which is the assumption the budget's number rests on."""
    from chat_rag.components.embedding.openai_compatible_embedding import OpenAICompatibleEmbedding

    budget = L.configure_embedding_budget(3)
    try:
        transport = TimedTransport()
        embedding = OpenAICompatibleEmbedding(
            "test/model", batch_size=2, provider=transport, cache_dir=None,
        )
        vectors = embedding.encode_documents(["a", "b", "c", "d", "e"])
        assert vectors.shape == (5, 3)
        # Five texts, batches of two: three requests, three slots. If the
        # transport ever fanned out internally this count would disagree.
        assert len(transport.seen) == 3
        assert budget.acquired_total == 3
        assert embedding.describe()["budgeted"] is True
    finally:
        L.configure_embedding_budget(4)


def test_a_local_embedding_model_reports_no_budget():
    """It makes no request; its cost is CPU on a worker already accounted for."""
    from chat_rag.components.embedding.sentence_transformer_embedding import SentenceTransformerEmbedding

    assert not hasattr(SentenceTransformerEmbedding, "budgeted")
