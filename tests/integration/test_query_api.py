"""The chat route under its limits: what a client is told, and what health
and metrics show, when a query is refused, times out, fails or succeeds.

The pipeline behind the route is a stub built *through* the pipeline cache
(one per session, as in production) whose answer model is the real budgeted
wrapper over a gated fake, so the route, the admission, the budget, the
lease and the telemetry are all real. No provider is reached.
"""

from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

import app as flask_app
from components.ingest import limits as L
from components.observability import telemetry as T
from components.query import limits as Q
from core.exceptions import LLMException, QueryTimeout

from query_doubles import FailingAnswerModel, GatedAnswerModel


class StubPipeline:
    """The shape of a query as the route sees it, with real stages and the
    real answer wrapper, and no store or model behind it."""

    def __init__(self, llm, budget, *, wait_seconds=30.0, fail_retrieval=None):
        self.retrieval_profile = "hybrid_rrf"
        self.vector_db = SimpleNamespace(close=lambda: None, get_all_chunks=lambda: [])
        self.hybrid_retriever = SimpleNamespace(invalidate_index=lambda: None)
        self.llm_model = llm
        self.answer_model = Q.LimitedAnswerModel(llm, budget, wait_seconds=wait_seconds)
        self.fail_retrieval = fail_retrieval

    def query(self, question, **kwargs):
        with T.stage(T.RETRIEVE):
            if self.fail_retrieval is not None:
                raise self.fail_retrieval
        with T.stage(T.CONTEXT):
            pass
        with T.stage(T.ANSWER):
            answer = self.answer_model.generate([{"role": "user", "content": question}])
        return {"answer": answer, "sources": [{"label": "S1"}],
                "metadata": {"retrieval_method": "hybrid_rrf"}}


@pytest.fixture
def registry(monkeypatch):
    fresh = T.MetricsRegistry(window=50)
    monkeypatch.setattr(T, "_registry", fresh)
    return fresh


@pytest.fixture
def budget(monkeypatch):
    fresh = L.ProviderBudget(2)
    monkeypatch.setattr(Q, "_answer_budget", fresh)
    return fresh


@pytest.fixture
def admission(monkeypatch):
    fresh = Q.QueryAdmission(2)
    monkeypatch.setattr(flask_app.services, "query_admission", fresh)
    return fresh


@pytest.fixture
def app(tmp_path, monkeypatch, registry):
    from components.knowledgebase.manager import KnowledgeBaseManager

    monkeypatch.chdir(tmp_path)
    manager = KnowledgeBaseManager(str(tmp_path / "kbs.json"))
    monkeypatch.setattr(flask_app.services, "kb_manager", manager)
    flask_app.app.config.update(TESTING=True)
    kb = manager.create("query-kb", chunker={"type": "structure_first"})
    return SimpleNamespace(kb_id=kb["kb_id"])


def use_pipelines(monkeypatch, factory):
    """Build one stub per session through the cache, as production does."""
    flask_app.services.pipeline_cache.clear()
    monkeypatch.setattr(flask_app.services.pipeline_cache, "_build", lambda session, kb: factory())


def ask(client, kb_id, question="soru"):
    return client.post("/api/query", json={"question": question, "kb_id": kb_id})


# -------------------------------------------------------------- success
def test_a_successful_answer_keeps_its_shape_and_gains_timing(app, monkeypatch, budget, admission):
    model = GatedAnswerModel()
    model.release()
    use_pipelines(monkeypatch, lambda: StubPipeline(model, budget))
    with flask_app.app.test_client() as client:
        response = ask(client, app.kb_id)
    assert response.status_code == 200
    body = response.get_json()
    assert body["success"] is True and body["answer"] == model.reply
    assert body["sources"] == [{"label": "S1"}]
    assert body["metadata"]["retrieval_method"] == "hybrid_rrf", "existing metadata untouched"
    timing = body["metadata"]["query"]
    assert set(timing["stages"]) == {"retrieve", "context", "answer"}
    assert timing["provider"]["calls"] == 1 and timing["query_id"]
    assert admission.active == 0 and budget.snapshot()["inflight"] == 0


def test_an_empty_question_is_still_a_400_and_takes_no_slot(app, admission):
    with flask_app.app.test_client() as client:
        response = client.post("/api/query", json={"question": "  ", "kb_id": app.kb_id})
    assert response.status_code == 400
    assert admission.snapshot()["accepted_total"] == 0


# ------------------------------------------------------------- overload
def test_when_every_query_slot_is_taken_the_next_question_is_refused_at_once(
    app, monkeypatch, budget, admission, registry,
):
    model = GatedAnswerModel(expect=2)
    use_pipelines(monkeypatch, lambda: StubPipeline(model, budget))
    outcomes = []

    def blocked():
        with flask_app.app.test_client() as client:
            outcomes.append(ask(client, app.kb_id).get_json())

    threads = [threading.Thread(target=blocked) for _ in range(2)]
    for thread in threads:
        thread.start()
    assert model.full.wait(10), "two queries never got inside the model"
    assert admission.saturated

    with flask_app.app.test_client() as client:
        refused = ask(client, app.kb_id)
        health = client.get("/api/health").get_json()
        metrics = client.get("/api/ops/metrics").get_json()
    assert refused.status_code == 503
    body = refused.get_json()
    assert body == {
        "success": False, "overloaded": True, "reason": "admission",
        "retry_after_seconds": body["retry_after_seconds"], "error": body["error"],
    }
    assert body["retry_after_seconds"] >= Q.RETRY_AFTER_MIN
    assert refused.headers["Retry-After"] == str(int(body["retry_after_seconds"]))
    assert "query slot" in body["error"]

    # Health answers while both slots are busy, and says so.
    assert health["state"] == "overloaded" and health["ready"] is True
    assert "every query slot is in use" in health["reasons"][0]
    assert health["query"] == {"active": 2, "max_active": 2, "answer_inflight": 2, "answer_limit": 2}
    assert metrics["query"]["admission"]["rejected_total"] == 1
    assert metrics["metrics"]["queries"]["active"] == 2
    assert metrics["metrics"]["counters"]["query.rejected"] == 1
    assert metrics["metrics"]["errors"]["by_category"] == {"overloaded": 1}

    model.release()
    for thread in threads:
        thread.join(20)
    assert all(o["success"] for o in outcomes)
    assert admission.active == 0 and admission.peak == 2
    with flask_app.app.test_client() as client:
        assert client.get("/api/health").get_json()["state"] == "ok"
        # And the refused client, trying again, is served.
        assert ask(client, app.kb_id).status_code == 200


def test_when_the_answer_model_is_at_capacity_the_query_is_refused_not_hung(
    app, monkeypatch, admission, registry,
):
    """Admission has room but the answer budget is full and the wait cap is
    short: answer_capacity overload, with the reason named."""
    one = L.ProviderBudget(1)
    monkeypatch.setattr(Q, "_answer_budget", one)
    holder = GatedAnswerModel(expect=1)
    use_pipelines(monkeypatch, lambda: StubPipeline(holder, one, wait_seconds=0.1))
    outcome = []

    def blocked():
        with flask_app.app.test_client() as client:
            outcome.append(ask(client, app.kb_id).status_code)

    thread = threading.Thread(target=blocked)
    thread.start()
    assert holder.full.wait(10)
    with flask_app.app.test_client() as client:
        refused = ask(client, app.kb_id)
    assert refused.status_code == 503
    body = refused.get_json()
    assert body["overloaded"] is True and body["reason"] == "answer_capacity"
    assert "Retry-After" in refused.headers
    holder.release()
    thread.join(20)
    assert outcome == [200]
    assert one.snapshot()["inflight"] == 0 and one.refused_total == 1
    assert registry.snapshot()["queries"]["outcomes"] == {"rejected": 1, "succeeded": 1}


# -------------------------------------------------------------- timeout
def test_a_query_past_its_deadline_is_a_504(app, monkeypatch, budget, admission, registry):
    monkeypatch.setattr(flask_app.settings, "query_timeout", 0.0)
    monkeypatch.setattr(Q.QueryGuard, "for_timeout",
                        classmethod(lambda cls, seconds, clock=None: cls(deadline=-1.0)))
    model = GatedAnswerModel()
    model.release()
    use_pipelines(monkeypatch, lambda: StubPipeline(model, budget))
    with flask_app.app.test_client() as client:
        response = ask(client, app.kb_id)
    assert response.status_code == 504
    body = response.get_json()
    assert body["success"] is False and body["timed_out"] is True
    assert model.calls == 0, "out of time: no call was made"
    assert admission.active == 0 and budget.snapshot()["inflight"] == 0
    assert registry.snapshot()["queries"]["outcomes"] == {"timed_out": 1}


# -------------------------------------------------------------- failure
def test_a_provider_failure_releases_everything_and_is_still_a_503(app, monkeypatch, budget, admission):
    use_pipelines(monkeypatch, lambda: StubPipeline(FailingAnswerModel(), budget))
    with flask_app.app.test_client() as client:
        for _ in range(3):
            response = ask(client, app.kb_id)
            assert response.status_code == 503
            assert response.get_json()["generation_unavailable"] is True
    assert admission.active == 0 and budget.snapshot()["inflight"] == 0
    assert admission.snapshot()["accepted_total"] == 3


def test_a_retrieval_failure_is_a_500_that_releases_capacity(app, monkeypatch, budget, admission, registry):
    model = GatedAnswerModel()
    use_pipelines(monkeypatch, lambda: StubPipeline(model, budget, fail_retrieval=RuntimeError("store broke")))
    with flask_app.app.test_client() as client:
        response = ask(client, app.kb_id)
    assert response.status_code == 500
    assert admission.active == 0
    recent = registry.snapshot()["queries"]["recent"][0]
    assert recent["status"] == "failed" and recent["failed_stages"] == ["retrieve"]


def test_the_pipeline_is_leased_for_the_length_of_the_query(app, monkeypatch, budget, admission):
    """A burst of other sessions cannot evict this query's pipeline and
    close its store underneath it."""
    model = GatedAnswerModel(expect=1)
    use_pipelines(monkeypatch, lambda: StubPipeline(model, budget))
    outcome = []

    def blocked():
        with flask_app.app.test_client() as client:
            outcome.append(ask(client, app.kb_id).status_code)

    thread = threading.Thread(target=blocked)
    thread.start()
    assert model.full.wait(10)
    assert flask_app.services.pipeline_cache.snapshot()["leased"] == 1
    model.release()
    thread.join(20)
    assert outcome == [200]
    assert flask_app.services.pipeline_cache.snapshot()["leased"] == 0


def test_the_metrics_endpoint_describes_the_query_limits(app):
    with flask_app.app.test_client() as client:
        body = client.get("/api/ops/metrics").get_json()
    query = body["query"]
    assert set(query) == {"admission", "answer_budget", "limits", "deadline_semantics"}
    assert query["limits"]["deadline_semantics"] == "cooperative-with-clamped-calls"
    assert "clamped" in query["deadline_semantics"]
    assert "local_models" in body["caches"]
