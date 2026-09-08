"""A query measured through the same instrument as an ingest, on the real
pipeline, and the rule that keeps the operational log free of content.

The pipeline here is the product's: hybrid_rrf over a Chroma store, a fake
embedding transport that never leaves the process and an answer model the
test controls. What is pinned: a query records its stages; a failed answer
is a failed stage with a provider category, not a dropped trace; two
sessions' queries running at once each finish with exactly their own
provider call counted; queries and ingest jobs are filed in separate
windows so failed chat cannot make the service call itself degraded; and
neither the question, the retrieved text nor the answer reaches a log line
at the default level.
"""

from __future__ import annotations

import logging
import threading

import pytest

from components.ingest import limits as L
from components.observability import telemetry as T
from components.query import limits as Q
from core.exceptions import LLMException

from query_doubles import (
    SECRET_ANSWER, SECRET_CHUNK, SECRET_QUESTION, FailingAnswerModel, GatedAnswerModel,
    make_pipeline,
)


@pytest.fixture
def registry(monkeypatch):
    fresh = T.MetricsRegistry(window=20)
    monkeypatch.setattr(T, "_registry", fresh)
    return fresh


@pytest.fixture
def key(monkeypatch):
    monkeypatch.setenv("QUERY_TEST_KEY", "sk-or-test-secret-value-never-persisted")


@pytest.fixture
def admission():
    return Q.QueryAdmission(4)


def run_query(pipeline, admission, question=SECRET_QUESTION, **kwargs):
    with Q.query_scope(admission, timeout_seconds=60, kb_id="kb-test", mode="hybrid_rrf") as scope:
        result = pipeline.query(question, top_k=4, **kwargs)
    return scope, result


# -------------------------------------------------------------- stages
def test_a_query_reports_where_its_time_went(tmp_path, key, registry, admission):
    model = GatedAnswerModel()
    model.release()
    pipeline = make_pipeline(tmp_path, model)
    scope, result = run_query(pipeline, admission)

    assert result["answer"] == SECRET_ANSWER
    timing = scope.timing()
    assert set(timing["stages"]) == {T.RETRIEVE, T.CONTEXT, T.ANSWER}
    assert timing["provider"]["calls"] == 1
    assert timing["embedding"]["calls"] >= 1, "the query embedded itself through the budgeted transport"
    assert scope.status == "succeeded"
    detail = {s.name: s.detail for s in scope.trace.stages}
    assert detail[T.RETRIEVE]["dense_used"] is True and detail[T.RETRIEVE]["bm25_hits"] >= 1
    assert detail[T.CONTEXT]["selected"] >= 1
    snapshot = registry.snapshot()["queries"]
    assert snapshot["measured"] == 1 and snapshot["outcomes"] == {"succeeded": 1}
    # The summary keeps only stages that took measurable time (a fake model
    # answers in microseconds); the trace itself carries every stage.
    assert set(snapshot["stages"]) <= {T.RETRIEVE, T.CONTEXT, T.ANSWER}
    assert set(snapshot["recent"][0]["stages"]) == {T.RETRIEVE, T.CONTEXT, T.ANSWER}
    assert snapshot["provider"]["calls"] == 1
    assert snapshot["recent"][0]["kind"] == "query"


def test_a_failed_answer_is_a_failed_stage_with_a_category(tmp_path, key, registry, admission):
    pipeline = make_pipeline(tmp_path, FailingAnswerModel())
    with pytest.raises(LLMException):
        run_query(pipeline, admission)
    recent = registry.snapshot()["queries"]["recent"]
    assert len(recent) == 1
    assert recent[0]["status"] == "failed" and recent[0]["error_category"] == "provider"
    assert recent[0]["failed_stages"] == [T.ANSWER]
    assert registry.snapshot()["errors"]["by_category"] == {"provider": 1}
    assert admission.active == 0, "admission released on failure"


def test_concurrent_queries_are_attributed_to_their_own_traces(tmp_path, key, registry, admission):
    """Two sessions -- two pipelines over one store -- both inside the
    answer model at the same moment, each finishing with its own single
    provider call on its own trace and nothing of the other's."""
    shared = GatedAnswerModel(expect=2)
    budget = L.ProviderBudget(2)
    first = make_pipeline(tmp_path, shared)
    second = make_pipeline(tmp_path, shared, ingest=False)
    outcomes = {}

    def ask(index, pipeline):
        try:
            with Q.query_scope(admission, timeout_seconds=60, kb_id="kb-test") as scope:
                pipeline.query(SECRET_QUESTION, top_k=3)
            outcomes[index] = scope
        except BaseException as error:  # noqa: BLE001
            outcomes[index] = error

    for pipeline in (first, second):
        pipeline.llm_model = Q.LimitedAnswerModel(shared, budget)
    threads = [threading.Thread(target=ask, args=(i, p)) for i, p in enumerate((first, second))]
    for thread in threads:
        thread.start()
    assert shared.full.wait(20), "both queries never overlapped inside the model"
    assert registry.snapshot()["queries"]["active"] == 2
    shared.release()
    for thread in threads:
        thread.join(30)

    assert all(isinstance(o, Q.QueryScope) for o in outcomes.values()), outcomes
    for scope in outcomes.values():
        assert scope.trace.provider_calls == 1
        assert scope.trace.seconds_for(T.ANSWER) > 0
    assert {s.query_id for s in outcomes.values()}.__len__() == 2
    queries = registry.snapshot()["queries"]
    assert queries["measured"] == 2 and queries["peak_active"] == 2 and queries["active"] == 0
    assert queries["provider"]["calls"] == 2


# ------------------------------------------------------------ windows
def test_failed_queries_do_not_make_the_service_degraded(registry):
    """``recent_outcomes`` decides 'degraded' from ingest jobs alone."""
    for index in range(6):
        trace = T.JobTrace(job_id=f"q{index}", kind=T.KIND_QUERY, status="failed")
        registry.finish(trace)
    assert registry.recent_outcomes() == (0, 0)
    registry.finish(T.JobTrace(job_id="j1", status="succeeded"))
    assert registry.recent_outcomes() == (1, 0)
    snapshot = registry.snapshot()
    assert snapshot["jobs"]["measured"] == 1 and snapshot["queries"]["measured"] == 6


def test_the_query_window_cannot_grow_with_uptime(registry):
    for index in range(200):
        registry.finish(T.JobTrace(job_id=f"q{index}", kind=T.KIND_QUERY, total_seconds=1.0))
    snapshot = registry.snapshot(recent=1000)["queries"]
    assert snapshot["measured"] == 20 and len(snapshot["recent"]) == 20
    assert registry.query_latency()["p50"] == 1.0


def test_query_errors_are_categorised():
    from core.exceptions import QueryOverloaded, QueryTimeout

    assert T.categorise(QueryTimeout()) == "timed_out"
    assert T.categorise(QueryOverloaded("full")) == "overloaded"


# -------------------------------------------------------------- logs
def test_the_operational_log_of_a_query_carries_no_content(tmp_path, key, registry, admission, caplog):
    """The question is the user's text, the chunks are the corpus and the
    answer is generated from both. None of them belongs in a log line at
    the default level, and this proves none arrives there."""
    model = GatedAnswerModel()
    model.release()
    pipeline = make_pipeline(tmp_path, model)
    caplog.set_level(logging.INFO, logger="RAG")
    with caplog.at_level(logging.INFO, logger="RAG"):
        run_query(pipeline, admission)
    lines = [record.getMessage() for record in caplog.records if record.levelno >= logging.INFO]
    assert lines, "the query did log something operational"
    text = "\n".join(lines)
    for forbidden in (SECRET_QUESTION, SECRET_CHUNK, SECRET_ANSWER, "ornitorenk", "7770"):
        assert forbidden not in text, forbidden
    assert any("event=query.started" in line for line in lines)
    assert any("event=query.succeeded" in line and "provider_calls=1" in line for line in lines)
