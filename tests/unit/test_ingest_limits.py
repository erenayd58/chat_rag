"""The limits every ingest runs under: the provider budget and the job guard.

What is pinned here is the contract the whole phase rests on: the number of
provider calls in flight across the process never exceeds the configured
maximum, a slot is given back on every exit -- success, error, timeout --
and nothing that makes no provider call (Standard, Markdown, the Viewer
packager) ever holds one. Nothing here sleeps: every ordering is proved by
holding and releasing events.
"""

from __future__ import annotations

import threading

import pytest

from components.chunker import deep_analysis
from components.ingest import limits as L
from config.ingest import IngestLimits, limits_from_env
from core.exceptions import IngestInterrupted

from ingest_doubles import FailingProvider, GatedProvider

# ---------------------------------------------------------------- config


def test_the_defaults_validate():
    assert limits_from_env({}).workers == 2
    assert limits_from_env({}).provider_max_inflight == 8


@pytest.mark.parametrize("variable, value, wording", [
    ("INGEST_WORKERS", "0", "INGEST_WORKERS must be at least 1"),
    ("INGEST_QUEUE_CAPACITY", "-1", "INGEST_QUEUE_CAPACITY must be 0 or more"),
    ("INGEST_JOB_TIMEOUT", "0", "INGEST_JOB_TIMEOUT must be a positive"),
    ("PROVIDER_MAX_INFLIGHT", "0", "PROVIDER_MAX_INFLIGHT must be at least 1"),
    ("DEEP_ANALYSIS_CONCURRENCY", "0", "DEEP_ANALYSIS_CONCURRENCY must be at least 1"),
    ("INGEST_WORKERS", "two", "INGEST_WORKERS='two' is not a whole number"),
    ("INGEST_JOB_TIMEOUT", "soon", "INGEST_JOB_TIMEOUT='soon' is not a number"),
])
def test_a_bad_value_is_refused_by_name(variable, value, wording):
    with pytest.raises(ValueError) as refused:
        limits_from_env({variable: value})
    assert wording in str(refused.value)


def test_settings_fail_at_construction_not_at_first_upload(monkeypatch):
    from config import Settings

    monkeypatch.setenv("INGEST_WORKERS", "0")
    with pytest.raises(ValueError):
        Settings()


def test_admission_capacity_is_workers_plus_queue():
    assert IngestLimits(workers=2, queue_capacity=8).admission_capacity == 10


# ---------------------------------------------------------------- budget


def test_the_budget_never_hands_out_more_slots_than_it_has():
    """Eight threads want a slot; three exist. Exactly three are inside at
    once, the other five wait, and when the three leave the rest get in."""
    budget = L.ProviderBudget(3)
    gate = threading.Event()
    inside = threading.Condition()
    count = {"inside": 0, "done": 0}

    def hold():
        with budget.slot():
            with inside:
                count["inside"] += 1
                inside.notify_all()
            gate.wait(10)
            with inside:
                count["inside"] -= 1
        with inside:
            count["done"] += 1
            inside.notify_all()

    threads = [threading.Thread(target=hold) for _ in range(8)]
    for thread in threads:
        thread.start()
    with inside:
        assert inside.wait_for(lambda: count["inside"] == 3, timeout=10)
    assert budget.snapshot()["inflight"] == 3
    assert budget.peak == 3
    gate.set()
    with inside:
        assert inside.wait_for(lambda: count["done"] == 8, timeout=10)
    for thread in threads:
        thread.join(5)
    assert budget.peak == 3, "the limit held from first to last"
    assert budget.snapshot()["inflight"] == 0
    assert budget.acquired_total == 8


def test_a_failing_call_gives_its_slot_back():
    budget = L.ProviderBudget(1)
    with pytest.raises(ConnectionError):
        with budget.slot():
            raise ConnectionError("gateway down")
    assert budget.snapshot()["inflight"] == 0
    with budget.slot():  # the slot is free again
        assert budget.snapshot()["inflight"] == 1


def test_a_slot_that_does_not_come_in_time_is_refused_without_a_release():
    budget = L.ProviderBudget(1)
    with budget.slot():
        with pytest.raises(L.ProviderSlotTimeout):
            with budget.slot(timeout=0.05):
                pass  # pragma: no cover - never reached
        assert budget.snapshot()["inflight"] == 1, "the refused caller released nothing"
    assert budget.refused_total == 1
    assert budget.snapshot()["inflight"] == 0


def test_no_time_left_means_no_wait_at_all():
    budget = L.ProviderBudget(1)
    with pytest.raises(L.ProviderSlotTimeout):
        with budget.slot(timeout=0):
            pass  # pragma: no cover


# ----------------------------------------------------------------- guard


def test_the_guard_stops_a_job_at_a_seam_when_its_deadline_has_passed():
    clock = {"now": 100.0}
    guard = L.JobGuard.for_timeout(5, clock=lambda: clock["now"])
    guard.check()  # within time
    clock["now"] = 106.0
    with pytest.raises(IngestInterrupted) as stopped:
        guard.check()
    assert stopped.value.kind == "timed_out"
    assert guard.remaining() == 0.0


def test_cancellation_outranks_the_clock():
    guard = L.JobGuard.for_timeout(None)
    guard.cancel("operator asked")
    with pytest.raises(IngestInterrupted) as stopped:
        guard.check()
    assert stopped.value.kind == "cancelled"
    assert "operator asked" in str(stopped.value)


def test_the_checkpoint_is_a_no_op_outside_a_job():
    assert L.current_guard() is None
    L.checkpoint()  # nothing to stop


def test_the_guard_travels_with_the_thread_not_the_process():
    guard = L.JobGuard()
    seen = {}

    def other():
        seen["other"] = L.current_guard()

    with L.use_guard(guard):
        assert L.current_guard() is guard
        thread = threading.Thread(target=other)
        thread.start()
        thread.join(5)
    assert seen["other"] is None
    assert L.current_guard() is None


# --------------------------------------------------------------- wrapper


def test_the_wrapper_refuses_before_touching_the_provider_when_the_job_is_over():
    inner = GatedProvider()
    budget = L.ProviderBudget(4)
    guard = L.JobGuard()
    guard.cancel()
    limited = L.LimitedProvider(inner, budget, guard)
    with pytest.raises(IngestInterrupted):
        limited.complete("anything")
    assert inner.calls == 0
    assert budget.acquired_total == 0


def test_the_wrapper_releases_the_slot_when_the_provider_fails():
    budget = L.ProviderBudget(2)
    limited = L.LimitedProvider(FailingProvider(), budget)
    with pytest.raises(ConnectionError):
        limited.complete("prompt")
    assert budget.snapshot()["inflight"] == 0
    assert limited.calls == 1


def test_the_wrapper_waits_no_longer_than_the_job_has_left():
    budget = L.ProviderBudget(1)
    clock = {"now": 0.0}
    guard = L.JobGuard.for_timeout(0.05, clock=lambda: clock["now"])
    limited = L.LimitedProvider(GatedProvider(), budget, guard)
    with budget.slot():  # someone else holds the only slot
        with pytest.raises(L.ProviderSlotTimeout):
            limited.complete("prompt")
    assert limited.refused == 1
    assert budget.snapshot()["inflight"] == 0


def test_the_wrapper_keeps_the_model_id_the_report_records():
    limited = L.LimitedProvider(GatedProvider(), L.ProviderBudget(1))
    assert limited.model_id == "test:gated@1"


# ------------------------------------------------- wiring into Deep Analysis


def _configuration(monkeypatch, *, use_llm=True):
    from types import SimpleNamespace

    from components.chunker.structural_chunker import (
        HARD_MAX_TOKENS, MIN_TOKENS, SOFT_MAX_TOKENS, TARGET_TOKENS,
    )

    monkeypatch.setenv("FAKE_DEEP_KEY", "sk-placeholder")
    app_settings = SimpleNamespace(
        deep_analysis_model="test/proposer" if use_llm else "",
        deep_analysis_verifier_model="",
        deep_analysis_endpoint="",
        deep_analysis_api_key_env="FAKE_DEEP_KEY",
        deep_analysis_use_llm=use_llm,
        deep_analysis_verify=True,
        deep_analysis_timeout=5.0,
        deep_analysis_concurrency=4,
    )
    return deep_analysis.build_configuration(app_settings, deep_analysis.deep_config(
        min_tokens=MIN_TOKENS, target_tokens=TARGET_TOKENS,
        soft_max_tokens=SOFT_MAX_TOKENS, hard_max_tokens=HARD_MAX_TOKENS,
    ))


def test_both_transports_are_wrapped_and_share_the_one_budget(monkeypatch):
    proposer, verifier = GatedProvider(), GatedProvider()
    monkeypatch.setattr(deep_analysis, "build_transports", lambda settings: (proposer, verifier))
    budget = L.configure_budget(3)
    guard = L.JobGuard()
    with L.use_guard(guard):
        limited_proposer, limited_verifier = deep_analysis.limited_providers(_configuration(monkeypatch))
    assert isinstance(limited_proposer, L.LimitedProvider)
    assert isinstance(limited_verifier, L.LimitedProvider)
    assert limited_proposer.inner is proposer and limited_verifier.inner is verifier
    assert limited_proposer.budget is budget and limited_verifier.budget is budget
    assert limited_proposer.guard is guard and limited_verifier.guard is guard


def test_without_a_model_run_no_transport_is_built_and_no_slot_is_taken(monkeypatch):
    def never(settings):  # pragma: no cover - the assertion is that it is not called
        raise AssertionError("no transport may be built for a deterministic run")

    monkeypatch.setattr(deep_analysis, "build_transports", never)
    monkeypatch.delenv("FAKE_DEEP_KEY", raising=False)
    configuration = _configuration(monkeypatch, use_llm=False)
    assert deep_analysis.limited_providers(configuration) == (None, None)


# ------------------------------------------- who never takes a slot at all


@pytest.fixture
def budget_probe(monkeypatch):
    """A fresh budget whose counters start at zero, and a transport factory
    that fails the test if anything tries to build one."""
    budget = L.configure_budget(4)

    def never(settings):  # pragma: no cover
        raise AssertionError("a provider transport was built on a path that makes no model call")

    monkeypatch.setattr(deep_analysis, "build_transports", never)
    yield budget
    L.configure_budget(4)


def test_the_standard_path_takes_no_provider_slot(budget_probe):
    from components.chunker.structural_chunker import StructuralChunker

    from ingest_doubles import deep_corpus, deep_text

    units = deep_corpus(sections=2, document_id="std-doc")
    chunks = StructuralChunker().chunk_text(deep_text(units), "std-doc", "Standard", parsed_units=units)
    assert chunks
    assert budget_probe.acquired_total == 0


def test_the_markdown_and_standard_viewer_variants_take_no_provider_slot(budget_probe):
    from components.viewer import analysis
    from components.viewer import methods as M

    from ingest_doubles import deep_corpus

    from amsc.models import RawDocumentUnit

    units = [RawDocumentUnit.model_validate(u) for u in deep_corpus(sections=2)]
    for method in (M.MARKDOWN, M.STANDARD):
        assert analysis._chunk_rows(method, units)
    assert budget_probe.acquired_total == 0
