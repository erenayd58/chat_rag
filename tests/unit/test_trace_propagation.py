"""Provider work happens on a pool; the measurement must still find its job.

Deep Analysis does not call the gateway from the ingest worker. ``amsc``'s
``collect_votes`` hands every planned proposer and verifier call to a
``ThreadPoolExecutor``, so ``provider.complete()`` runs on a thread that was
never inside ``use_trace`` or ``use_guard``. A ``threading.local`` does not
follow work onto a pool thread -- that is the whole point of one -- so
anything read from a thread-local at call time is simply absent there, and a
count taken that way would be silently lost rather than visibly wrong.

These tests pin the mechanism that makes it right: :class:`LimitedProvider`
binds the guard *and* the trace when it is constructed, on the worker thread
inside the job's Deep stage, and uses those bound objects rather than looking
anything up per call. What is proved below is the property that matters in
production: two jobs running at once, both fanning out onto pools, each
finishing with exactly its own calls counted and none of the other's.

The pool here is a real ``ThreadPoolExecutor`` driven the same way
``collect_votes`` drives one, and a barrier -- not a sleep -- guarantees the
two jobs' calls genuinely overlap, so the test would fail if attribution were
by thread rather than by job.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from components.ingest import limits as L
from components.observability import telemetry as T


class Transport:
    """A provider double that records which thread served each call."""

    model_id = "test/model"
    timeout_seconds = 60.0

    def __init__(self, gate=None):
        self.gate = gate
        self.threads = set()
        self._lock = threading.Lock()

    def complete(self, prompt):
        if self.gate is not None:
            self.gate.wait(timeout=10)
        with self._lock:
            self.threads.add(threading.get_ident())
        return f"answer to {prompt}"


def fan_out(provider, prompts, *, workers=4):
    """Call the provider the way ``amsc.collect_votes`` does: on a pool."""
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(provider.complete, prompts))


def test_calls_made_on_pool_threads_land_on_the_originating_trace():
    """The trace is bound at construction, so the pool cannot lose it."""
    trace = T.JobTrace(job_id="job-1", kb_id="kb-1", mode="deep")
    transport = Transport()

    with T.use_trace(trace):
        # Constructed here, on the "worker" thread, exactly as
        # deep_analysis.limited_providers constructs it.
        provider = L.LimitedProvider(transport, L.ProviderBudget(4))

    # ... and called from somewhere with no trace of its own at all.
    assert T.current_trace() is None
    fan_out(provider, [f"prompt {index}" for index in range(8)])

    assert trace.provider_calls == 8
    # Really other threads: if the work had run inline this would prove
    # nothing about propagation.
    assert transport.threads != {threading.get_ident()}
    assert len(transport.threads) > 1


def test_two_concurrent_jobs_each_count_only_their_own_calls():
    """No cross-contamination, with both jobs' pools in flight at once."""
    traces = {name: T.JobTrace(job_id=name, kb_id=name) for name in ("a", "b")}
    counts = {"a": 5, "b": 3}
    budget = L.ProviderBudget(8)

    # Every call blocks until all eight have arrived, so the two jobs' pool
    # threads are provably interleaved rather than running one after the other.
    gate = threading.Barrier(sum(counts.values()) + 1)
    transports = {name: Transport(gate) for name in traces}
    providers = {}
    for name, trace in traces.items():
        with T.use_trace(trace):
            providers[name] = L.LimitedProvider(transports[name], budget)

    errors = []

    def run(name):
        try:
            fan_out(providers[name], [f"{name}-{i}" for i in range(counts[name])],
                    workers=counts[name])
        except BaseException as error:  # noqa: BLE001 - surfaced below
            errors.append(error)

    runners = [threading.Thread(target=run, args=(name,)) for name in traces]
    for runner in runners:
        runner.start()
    gate.wait(timeout=10)
    for runner in runners:
        runner.join(timeout=10)

    assert not errors
    assert traces["a"].provider_calls == 5
    assert traces["b"].provider_calls == 3
    # The pools were genuinely distinct sets of threads, and the barrier means
    # they overlapped, so neither count can have come from thread affinity.
    assert not transports["a"].threads & transports["b"].threads


def test_a_provider_built_outside_a_job_still_measures_nothing():
    """No trace to bind, no trace to write to, and no crash either."""
    provider = L.LimitedProvider(Transport(), L.ProviderBudget(2))
    assert provider.trace is None
    assert fan_out(provider, ["one", "two"]) == ["answer to one", "answer to two"]


def test_the_deadline_also_survives_the_pool():
    """The guard is bound the same way, and a pool thread honours it."""
    guard = L.JobGuard.for_timeout(60)
    provider = L.LimitedProvider(Transport(), L.ProviderBudget(2), guard)
    guard.cancel("operator stopped the job")

    with ThreadPoolExecutor(max_workers=2) as pool:
        with pytest.raises(Exception) as raised:
            pool.submit(provider.complete, "prompt").result(timeout=5)
    assert T.categorise(raised.value) == "cancelled"


def test_deep_analysis_binds_the_trace_when_it_builds_the_providers(monkeypatch):
    """The wiring, not just the wrapper: the real construction site binds it."""
    from components.chunker import deep_analysis

    transport = Transport()
    monkeypatch.setattr(deep_analysis, "build_transports",
                        lambda settings: (transport, Transport()))
    configuration = deep_analysis.DeepAnalysisConfiguration(
        settings=deep_analysis.DeepAnalysisSettings(
            proposer_model="m", endpoint="http://gateway.invalid", api_key_env="FAKE_KEY",
        ),
        missing=(),
    )
    monkeypatch.setattr(type(configuration), "llm_available", property(lambda self: True))

    trace = T.JobTrace(job_id="job-9")
    with T.use_trace(trace):
        proposer, verifier = deep_analysis.limited_providers(configuration)

    assert proposer.trace is trace
    assert verifier.trace is trace
    fan_out(proposer, ["a", "b", "c"])
    assert trace.provider_calls == 3


def test_the_embedding_wrapper_deliberately_does_not_bind(monkeypatch):
    """It is shared by every job, so binding one trace would misattribute.

    The embedding wrapper lives on a cached pipeline and outlives any single
    job, which is why it looks the trace up per call instead. That is only
    safe because ``embed`` is called on the thread that asked for it, and this
    asserts both halves: two jobs sharing one wrapper each get their own
    count, and neither gets the other's.
    """
    class EmbedTransport:
        timeout_seconds = 30.0

        def embed(self, texts):
            return [[0.0] for _ in texts]

    wrapper = L.LimitedEmbeddingTransport(EmbedTransport(), L.ProviderBudget(4))
    first, second = T.JobTrace(job_id="first"), T.JobTrace(job_id="second")

    with T.use_trace(first):
        wrapper.embed(["one"])
        wrapper.embed(["two"])
    with T.use_trace(second):
        wrapper.embed(["three"])

    assert first.embedding_calls == 2
    assert second.embedding_calls == 1
