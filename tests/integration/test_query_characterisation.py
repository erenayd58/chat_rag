"""One controlled load characterisation of the bounded query path.

Twelve questions from twelve browser sessions arrive at the same instant (a
barrier releases them) at an application configured for four query slots
and an answer budget of two. Each admitted query runs through the route --
its own pipeline built through the cache, the real admission, the real
budgeted answer wrapper, the real telemetry -- against one shared answer
model that answers only when the test opens its gate, so the peaks are
measured while the system is as loaded as it can be, not after the fact.
A second wave of the refused questions, sent after the first wave drains,
shows that a refusal is a moment and not a state.

The numbers printed at the end are the characterisation; the assertions are
the bounds they must satisfy, whatever the machine.
"""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import pytest

import app as flask_app
from components.ingest import limits as L
from components.knowledgebase.manager import KnowledgeBaseManager
from components.observability import telemetry as T
from components.query import limits as Q

from query_doubles import GatedAnswerModel

SUBMITTED = 12
MAX_ACTIVE = 4
ANSWER_BUDGET = 2


class Pipeline:
    def __init__(self, model, budget):
        self.answer_model = Q.LimitedAnswerModel(model, budget, wait_seconds=30.0)
        self.vector_db = SimpleNamespace(close=lambda: None)
        self.hybrid_retriever = SimpleNamespace(invalidate_index=lambda: None)

    def query(self, question, **kwargs):
        with T.stage(T.RETRIEVE):
            pass
        with T.stage(T.CONTEXT):
            pass
        with T.stage(T.ANSWER):
            answer = self.answer_model.generate([{"role": "user", "content": question}])
        return {"answer": answer, "sources": [], "metadata": {}}


@pytest.fixture
def app(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    manager = KnowledgeBaseManager(str(tmp_path / "kbs.json"))
    monkeypatch.setattr(flask_app, "kb_manager", manager)
    flask_app.app.config.update(TESTING=True)
    registry = T.MetricsRegistry(window=100)
    monkeypatch.setattr(T, "_registry", registry)
    budget = L.ProviderBudget(ANSWER_BUDGET)
    monkeypatch.setattr(Q, "_answer_budget", budget)
    admission = Q.QueryAdmission(MAX_ACTIVE)
    monkeypatch.setattr(flask_app, "query_admission", admission)
    model = GatedAnswerModel(expect=ANSWER_BUDGET)
    flask_app.pipeline_cache.clear()
    monkeypatch.setattr(flask_app.pipeline_cache, "_build", lambda session, kb: Pipeline(model, budget))
    kb = manager.create("load-kb", chunker={"type": "structure_first"})
    return SimpleNamespace(kb_id=kb["kb_id"], registry=registry, budget=budget,
                           admission=admission, model=model)


def wave(app, count):
    barrier = threading.Barrier(count)
    outcomes = {}

    def ask(index):
        with flask_app.app.test_client() as client:
            barrier.wait(timeout=10)
            started = time.perf_counter()
            response = client.post("/api/query", json={"question": f"soru {index}", "kb_id": app.kb_id})
            outcomes[index] = (response.status_code, response.get_json(), time.perf_counter() - started)

    threads = [threading.Thread(target=ask, args=(i,)) for i in range(count)]
    for thread in threads:
        thread.start()
    return threads, outcomes


def test_a_burst_of_questions_is_bounded_refused_deterministically_and_measured(app):
    threads, outcomes = wave(app, SUBMITTED)
    assert app.model.full.wait(20), "the answer budget was never reached"
    # Measured at the peak, with the gate closed: the admitted queries are
    # inside the route, two of them inside the model, the rest waiting for
    # an answer slot, and every other question already refused.
    snapshot_at_peak = app.admission.snapshot()
    active_at_peak = app.registry.snapshot()["queries"]["active"]
    app.model.release()
    for thread in threads:
        thread.join(60)

    accepted = [o for o in outcomes.values() if o[0] == 200]
    rejected = [o for o in outcomes.values() if o[0] == 503]
    assert len(accepted) + len(rejected) == SUBMITTED, outcomes
    assert len(accepted) == MAX_ACTIVE, "exactly the slots, no more and no fewer"
    assert all(o[1]["overloaded"] and o[1]["reason"] == "admission" for o in rejected)
    assert app.model.peak == ANSWER_BUDGET == app.budget.peak
    assert snapshot_at_peak["active"] == MAX_ACTIVE == app.admission.peak
    assert active_at_peak == MAX_ACTIVE
    assert app.admission.active == 0 and app.budget.snapshot()["inflight"] == 0
    assert max(o[2] for o in rejected) < min(o[2] for o in accepted), "a refusal is immediate"

    # The second wave: the refused questions, asked again once the first
    # wave has drained, are all served.
    app.model.gate.clear()
    app.model.full.clear()
    second, again = wave(app, len(rejected))
    assert app.model.full.wait(20)
    app.model.release()
    for thread in second:
        thread.join(60)
    served_again = [o for o in again.values() if o[0] == 200]
    assert len(served_again) == min(len(rejected), MAX_ACTIVE)

    metrics = app.registry.snapshot(recent=25)
    queries = metrics["queries"]
    assert queries["peak_active"] == MAX_ACTIVE and queries["active"] == 0
    assert queries["outcomes"]["succeeded"] == len(accepted) + len(served_again)
    assert queries["outcomes"]["rejected"] == SUBMITTED - len(accepted) + len(rejected) - len(served_again)
    assert queries["provider"]["calls"] == len(accepted) + len(served_again)
    wait = queries["provider"]["wait_seconds"]
    assert wait["count"] == queries["provider"]["calls"]
    assert all(trace["kind"] == "query" for trace in queries["recent"])
    served = [trace for trace in queries["recent"] if trace["status"] == "succeeded"]
    assert all(set(trace["stages"]) == {"retrieve", "context", "answer"} for trace in served)
    assert all(trace["provider"]["calls"] == 1 for trace in served), "one call per query, attributed to it"

    print()
    print("Query load characterisation")
    print(f"  submitted (wave 1)         {SUBMITTED}")
    print(f"  accepted / refused         {len(accepted)} / {len(rejected)}  "
          f"(QUERY_MAX_ACTIVE={MAX_ACTIVE}, refusal is immediate: max {max(o[2] for o in rejected) * 1000:.1f} ms)")
    print(f"  wave 2 (retries)           {len(rejected)} submitted, {len(served_again)} served")
    print(f"  peak active queries        {queries['peak_active']}")
    print(f"  peak answer-provider calls {app.model.peak}  (ANSWER_MAX_INFLIGHT={ANSWER_BUDGET})")
    print(f"  provider wait p50 / max    {wait.get('p50', 0):.3f} s / {wait.get('max', 0):.3f} s")
    total = queries["total_seconds"]
    print(f"  total latency p50 / max    {total['p50']:.3f} s / {total['max']:.3f} s "
          f"(gated by the test; the max is the gate)")
    print(f"  failures / timeouts        {queries['outcomes'].get('failed', 0)} / "
          f"{queries['outcomes'].get('timed_out', 0)}")
    print(f"  admission                  {app.admission.snapshot()}")
    print(f"  answer budget              {app.budget.snapshot()}")
