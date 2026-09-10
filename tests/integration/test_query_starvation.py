"""Can a burst of questions lock the server out of answering anything?

Not a simulation: the real server -- uvicorn, the real application, a real
socket -- with three request threads, and queries that block inside the answer
model until the test lets them finish. Every handler on this surface is a
synchronous ``def``, so Starlette runs it in a worker thread and the pool is
sized from ``WAITRESS_THREADS`` exactly as ``asgi.py`` sizes it; three blocked
questions is therefore three held threads, the same condition waitress used to
put the application in. While questions are being answered, can
``GET /api/v1/health`` still be served?

Both answers are here. The first test lets as many queries in as there are
request threads and shows the failure: three blocked questions take all three
threads and health goes unanswered. The second leaves the product's own
admission in place -- derived so that queries never take every thread -- and
shows that health, status and the metrics answer throughout, while the
question that found no slot was refused at once with a Retry-After rather
than held.

Nothing here reaches a provider.
"""

from __future__ import annotations

import json
import socket
import threading
import urllib.error
import urllib.request
from types import SimpleNamespace

import pytest

import asgi as entrypoint
import interfaces.http as http
from chat_rag.components.ingest import limits as L
from chat_rag.components.knowledgebase.manager import KnowledgeBaseManager
from chat_rag import runtime
from chat_rag.components.observability import telemetry as T
from chat_rag.components.query import limits as Q

from query_doubles import GatedAnswerModel

V1 = http.v1.PREFIX
THREADS = 3
HEALTH_TIMEOUT = 3.0
#: How long to wait for the server to bind and answer once, at start-up.
READY_TIMEOUT = 30.0


class BlockingPipeline:
    """A query that runs until the test lets it finish."""

    def __init__(self, model, budget):
        self.model = model
        self.answer_model = Q.LimitedAnswerModel(model, budget)
        self.vector_db = SimpleNamespace(close=lambda: None)

    def query(self, question, **kwargs):
        with T.stage(T.ANSWER):
            answer = self.answer_model.generate([{"role": "user", "content": question}])
        return {"answer": answer, "sources": [], "metadata": {}}


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def _serve(application, port, threads):
    """One uvicorn server on a daemon thread, with a bounded worker pool.

    The pool is what makes this a starvation test: a synchronous handler runs
    in it, so ``threads`` of them held is the whole server held. It is sized
    from inside the lifespan because that is the only place there is a running
    loop to hold the limiter -- which is exactly what ``asgi.py`` does.
    """
    import uvicorn

    instance = uvicorn.Server(uvicorn.Config(
        application, host="127.0.0.1", port=port, log_level="warning",
        lifespan="on", access_log=False,
    ))
    thread = threading.Thread(target=instance.run, name="uvicorn-under-test", daemon=True)
    thread.start()
    return instance, thread


def _sized_application(threads):
    def on_start(_services):
        import anyio.to_thread

        anyio.to_thread.current_default_thread_limiter().total_tokens = threads

    return http.create_app(entrypoint.services, on_start=on_start)


def _wait_until_serving(port):
    import time

    deadline = time.monotonic() + READY_TIMEOUT
    while time.monotonic() < deadline:
        try:
            return get(port, f"{V1}/health")
        except (urllib.error.URLError, OSError, ValueError):
            time.sleep(0.1)
    raise AssertionError("the server never answered")


@pytest.fixture
def server(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    kb_manager = KnowledgeBaseManager(str(tmp_path / "kbs.json"))
    monkeypatch.setattr(entrypoint.services, "kb_manager", kb_manager)
    registry = T.MetricsRegistry(window=20)
    monkeypatch.setattr(runtime.current(), "metrics", registry)

    model = GatedAnswerModel(expect=THREADS)
    budget = L.ProviderBudget(THREADS)
    monkeypatch.setattr(runtime.current(), "answer_budget", budget)
    pipeline = BlockingPipeline(model, budget)
    monkeypatch.setattr(entrypoint.services, "get_pipeline", lambda *a, **k: pipeline)

    port = free_port()
    instance, thread = _serve(_sized_application(THREADS), port, THREADS)
    kb = kb_manager.create("starvation-kb", chunker={"type": "structure_first"})
    _wait_until_serving(port)

    yield SimpleNamespace(port=port, model=model, kb_id=kb["kb_id"])

    model.release()
    instance.should_exit = True
    thread.join(20)


def ask(port, kb_id, *, timeout=60.0):
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}{V1}/queries", method="POST",
        data=json.dumps({"question": "soru", "knowledge_base_id": kb_id}).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, json.loads(response.read().decode()), dict(response.headers)
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read().decode()), dict(error.headers)


def get(port, path, timeout=HEALTH_TIMEOUT):
    with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=timeout) as answer:
        return answer.status, json.loads(answer.read().decode())


def start_query(server) -> tuple[threading.Thread, list]:
    outcome: list = []
    thread = threading.Thread(target=lambda: outcome.append(ask(server.port, server.kb_id)), daemon=True)
    thread.start()
    return thread, outcome


def test_without_admission_a_burst_of_questions_takes_the_whole_server(server, monkeypatch):
    """The failure this closes, on the real server: as many queries admitted
    as there are request threads, and nothing else is served."""
    monkeypatch.setattr(entrypoint.services, "query_admission", Q.QueryAdmission(THREADS))
    threads = [start_query(server) for _ in range(THREADS)]
    assert server.model.full.wait(20), "three queries never got inside the model"

    with pytest.raises((urllib.error.URLError, socket.timeout, TimeoutError)):
        get(server.port, f"{V1}/health")

    server.model.release()
    for thread, _ in threads:
        thread.join(30)
    # The threads were held, not broken.
    assert get(server.port, f"{V1}/health")[0] == 200
    assert [o[0][0] for _, o in threads] == [200] * THREADS


def test_with_the_products_admission_health_and_status_are_always_served(server, monkeypatch):
    """The fix: fewer query slots than request threads. The question that
    finds no slot is refused at once, and everything else keeps answering."""
    admission = Q.QueryAdmission(THREADS - 1)
    monkeypatch.setattr(entrypoint.services, "query_admission", admission)
    server.model.expect = THREADS - 1

    running = [start_query(server) for _ in range(THREADS - 1)]
    assert server.model.full.wait(20)

    # One more question: refused immediately, with the reason and a hint.
    status, body, headers = ask(server.port, server.kb_id, timeout=HEALTH_TIMEOUT)
    assert status == 503
    assert body["error"]["type"] == "overloaded"
    assert body["error"]["details"]["reason"] == "admission"
    # Header names are case-insensitive and this server sends them lowercase.
    assert "retry-after" in {name.lower() for name in headers}

    # Which is the point: the server still answers everything else.
    code, health = get(server.port, f"{V1}/health")
    assert code == 200
    assert health["state"] == "overloaded" and health["ready"] is True
    assert health["capacity"]["query"]["active"] == THREADS - 1
    code, metrics = get(server.port, "/api/ops/metrics")
    assert code == 200
    assert metrics["metrics"]["queries"]["active"] == THREADS - 1
    assert metrics["query"]["admission"]["rejected_total"] == 1
    assert get(server.port, f"{V1}/knowledge-bases")[0] == 200
    assert get(server.port, f"{V1}/ingest-jobs")[0] == 200

    server.model.release()
    for thread, _ in running:
        thread.join(30)
    assert [o[0][0] for _, o in running] == [200] * (THREADS - 1)
    assert admission.active == 0 and admission.peak == THREADS - 1
    assert get(server.port, f"{V1}/health")[1]["state"] == "ok"


def test_the_default_leaves_a_thread_free_for_status():
    """``QUERY_MAX_ACTIVE`` is derived so that questions can never take every
    request thread, and ``INGEST_SYNC_WAITERS`` is subtracted from it. The
    synchronous upload that ration was sized for is gone with the Flask-era
    surface; the reservation is kept, because removing it would raise the
    default number of concurrent questions -- a change to what a deployment
    does, not a cleanup (``docs/legacy-removal.md``)."""
    from chat_rag.config.query import query_limits_from_env

    for threads in (2, 4, 8, 16):
        limits = query_limits_from_env({"WAITRESS_THREADS": str(threads)})
        assert limits.max_active + limits.sync_waiters < threads or threads < 3, threads
        assert limits.max_active >= 1
