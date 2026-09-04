"""Several Deep Analysis documents at once share one provider budget.

This is the end-to-end claim, through the real ``amsc.deep_pipeline`` and
the real ``StructuralChunker.chunk_text_deep``: three documents, each with
a per-job pool of four, run concurrently against a budget of two, and the
provider -- one gated double every document shares -- never sees more than
two calls inside it at once. The proposer *and* the verifier are counted,
because the double answers in the way that makes the verifier run.

No network: the transport factory is replaced, the key variable holds a
placeholder, and the double is the only thing ever called.
"""

from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

from components.chunker import deep_analysis
from components.chunker.structural_chunker import (
    HARD_MAX_TOKENS, MIN_TOKENS, SOFT_MAX_TOKENS, TARGET_TOKENS, StructuralChunker,
)
from components.ingest import limits as L

from ingest_doubles import GatedProvider, deep_corpus, deep_text, forbidding_answer


@pytest.fixture
def configuration(monkeypatch):
    monkeypatch.setenv("FAKE_DEEP_KEY", "sk-placeholder")
    app_settings = SimpleNamespace(
        deep_analysis_model="test/proposer",
        deep_analysis_verifier_model="",
        deep_analysis_endpoint="http://127.0.0.1:9/never",
        deep_analysis_api_key_env="FAKE_DEEP_KEY",
        deep_analysis_use_llm=True,
        deep_analysis_verify=True,
        deep_analysis_timeout=5.0,
        deep_analysis_concurrency=4,
    )
    return deep_analysis.build_configuration(app_settings, deep_analysis.deep_config(
        min_tokens=MIN_TOKENS, target_tokens=TARGET_TOKENS,
        soft_max_tokens=SOFT_MAX_TOKENS, hard_max_tokens=HARD_MAX_TOKENS,
    ))


@pytest.fixture
def budget():
    yield L.configure_budget(2)
    L.configure_budget(8)


def _run_document(index, configuration, results, errors):
    units = deep_corpus(document_id=f"doc-{index}")
    try:
        provider, verifier = deep_analysis.limited_providers(configuration)
        chunks, report = StructuralChunker().chunk_text_deep(
            deep_text(units), f"doc-{index}", f"Belge {index}",
            configuration=configuration, provider=provider, verifier_provider=verifier,
            parsed_units=units,
        )
        results[index] = (chunks, report, provider, verifier)
    except Exception as error:  # noqa: BLE001 - reported by the test
        errors[index] = error


def test_three_deep_documents_never_exceed_a_budget_of_two(configuration, budget, monkeypatch):
    shared = GatedProvider(expect=2, answer=forbidding_answer)
    monkeypatch.setattr(deep_analysis, "build_transports", lambda settings: (shared, shared))
    assert configuration.llm_available
    assert configuration.settings.concurrency == 4, "each document alone could have four in flight"

    results, errors = {}, {}
    threads = [
        threading.Thread(target=_run_document, args=(index, configuration, results, errors))
        for index in range(3)
    ]
    for thread in threads:
        thread.start()
    # The budget is full: two calls inside the provider, everyone else --
    # up to ten more threads across three documents -- waiting for a slot.
    assert shared.full.wait(20), "the provider never reached two calls in flight"
    assert shared.inflight == 2
    assert budget.snapshot()["inflight"] == 2
    shared.release()
    for thread in threads:
        thread.join(60)
    assert not errors, errors
    assert len(results) == 3

    assert shared.peak == 2, "twelve potential callers, two ever inside"
    assert budget.peak == 2
    assert budget.snapshot()["inflight"] == 0
    for chunks, report, provider, verifier in results.values():
        assert chunks
        assert report["status"] == "ok"
        assert report["proposer"]["call_count"] == 6
        assert report["verifier"]["group_count"] == 6
        assert provider.calls == 6, "every proposer call went through the wrapper"
        assert verifier.calls == 12, "and both orders of every verifier comparison"
    assert shared.calls == 3 * (6 + 12)
    assert budget.acquired_total == shared.calls


def test_a_larger_budget_lets_more_through_but_never_more_than_itself(configuration, monkeypatch):
    budget = L.configure_budget(3)
    try:
        shared = GatedProvider(expect=3, answer="{}")
        monkeypatch.setattr(deep_analysis, "build_transports", lambda settings: (shared, shared))
        results, errors = {}, {}
        threads = [
            threading.Thread(target=_run_document, args=(index, configuration, results, errors))
            for index in range(2)
        ]
        for thread in threads:
            thread.start()
        assert shared.full.wait(20)
        shared.release()
        for thread in threads:
            thread.join(60)
        assert not errors, errors
        assert shared.peak == 3 == budget.peak
    finally:
        L.configure_budget(8)


def test_a_provider_that_fails_every_call_holds_no_slot_afterwards(configuration, monkeypatch):
    from ingest_doubles import FailingProvider

    budget = L.configure_budget(2)
    try:
        failing = FailingProvider()
        monkeypatch.setattr(deep_analysis, "build_transports", lambda settings: (failing, failing))
        results, errors = {}, {}
        _run_document(0, configuration, results, errors)
        assert not errors, errors
        chunks, report, provider, verifier = results[0]
        assert chunks, "the deterministic partition still indexes"
        assert report["status"] == "fallback_provider_error"
        assert failing.calls == 6
        assert budget.snapshot()["inflight"] == 0
        assert budget.acquired_total == 6
    finally:
        L.configure_budget(8)


def test_a_job_that_is_over_makes_no_call_at_all(configuration, budget, monkeypatch):
    shared = GatedProvider(expect=1, answer="{}")
    shared.release()
    monkeypatch.setattr(deep_analysis, "build_transports", lambda settings: (shared, shared))
    guard = L.JobGuard()
    guard.cancel("stopped before the model ran")
    with L.use_guard(guard):
        results, errors = {}, {}
        _run_document(0, configuration, results, errors)
    assert not errors, errors
    chunks, report, provider, verifier = results[0]
    assert chunks
    assert shared.calls == 0, "the wrapper refused every call before the transport"
    assert report["status"] == "fallback_provider_error"
    assert budget.acquired_total == 0
