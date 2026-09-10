"""Search is not a way around the query limits.

``POST /api/v1/searches`` does the front half of a query on the request
thread, exactly as ``POST /api/v1/queries`` does -- embed the question, search
the store, build the knowledge base's lexical index if this pipeline has not
built it yet -- and it once ran under no limit at all. A burst of searches
could hold every request thread, each waiting an unbounded time for an
embedding slot, which is the starvation ``QUERY_MAX_ACTIVE`` exists to
prevent: the bound was on one endpoint rather than on the work.

There were four such endpoints when this was written -- the Flask-era Lab had
a route per retrieval leg -- and they are one resource with a ``method`` now
(``docs/legacy-removal.md``). Every method is driven below, because the
question is what the *work* costs and a dense leg costs an embedding call that
a lexical one does not.

A search makes no answer-model call, so it is given no answer budget. What it
is given is the three things the query path already had, and these tests pin
each: the same admission counter (so the bound is on request threads doing
retrieval, whichever route asked), the query deadline (which is what bounds
the embedding-slot wait), and the measurement, under its own ``mode``.

The pipeline is a stub built *through* the cache, so the lease is real. No
provider is reached.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from fastapi.testclient import TestClient

import asgi as entrypoint
import interfaces.http as http

V1 = http.v1.PREFIX
from chat_rag.components.ingest import limits as L
from chat_rag.components.knowledgebase.manager import KnowledgeBaseManager
from chat_rag import runtime
from chat_rag.components.observability import telemetry as T
from chat_rag.components.query import limits as Q
from chat_rag.core.models import DocumentChunk, RetrievalResult

from query_doubles import FakeEmbeddingTransport, GatedAnswerModel

QUESTION = "ornitorenk tarifesi"


def chunk(index: int) -> DocumentChunk:
    return DocumentChunk(chunk_id=f"c{index}", content=f"parca {index}", doc_id="d1",
                         doc_title="Rapor", chunk_index=index, total_chunks=3,
                         metadata={"doc_id": "d1"})


class Retriever:
    requires_document_embeddings = True

    def __init__(self, watch=None):
        self.watch = watch
        self.calls = 0

    def _hits(self):
        self.calls += 1
        if self.watch is not None:
            self.watch()
        return [RetrievalResult(chunk=chunk(i), score=1.0 - i / 10,
                                retrieval_method="bm25", rank=i) for i in range(3)]

    def keyword_search(self, query, top_k=10, **kwargs):
        return self._hits()

    def hybrid_search(self, query, top_k=5, *args, **kwargs):
        return self._hits()

    def vector_search(self, query, top_k=10, **kwargs):
        return self._hits()


class Store:
    def get_name(self):
        return "StubStore"

    def close(self):
        pass

    def get_all_chunks(self):
        return []

    def query(self, query_embedding, top_k=10, **kwargs):
        return [{"chunk_id": f"c{i}", "content": f"parca {i}",
                 "metadata": {"doc_id": "d1"}, "distance": 0.1 * i} for i in range(3)]


class Embedding:
    """The query's own embedding call, through the real budgeted transport --
    which is what makes the deadline test prove something."""

    def __init__(self, transport, budget):
        self.limited = L.LimitedEmbeddingTransport(transport, budget)

    def encode(self, text, **kwargs):
        return self.limited.embed([text])[0]

    def get_name(self):
        return "StubEmbedding"


class LabPipeline:
    def __init__(self, transport, budget, watch=None):
        self.settings = SimpleNamespace(embedding_model_name="test/embedding")
        self.retrieval_profile = "hybrid_rrf"
        self.hybrid_retriever = Retriever(watch)
        self.vector_db = Store()
        self.embedding_model = Embedding(transport, budget)


@pytest.fixture
def registry(monkeypatch):
    fresh = T.MetricsRegistry(window=50)
    monkeypatch.setattr(runtime.current(), "metrics", fresh)
    return fresh


@pytest.fixture
def lab(tmp_path, monkeypatch, registry):
    monkeypatch.chdir(tmp_path)
    manager = KnowledgeBaseManager(str(tmp_path / "kbs.json"))
    monkeypatch.setattr(entrypoint.services, "kb_manager", manager)

    admission = Q.QueryAdmission(2)
    monkeypatch.setattr(entrypoint.services, "query_admission", admission)
    answers = L.ProviderBudget(2)
    monkeypatch.setattr(runtime.current(), "answer_budget", answers)
    embeddings = L.ProviderBudget(2)
    transport = FakeEmbeddingTransport()
    watched = {}

    # Built through the cache, not around it: the lease the endpoint takes
    # is then a real lease on a real entry.
    entrypoint.services.pipeline_cache.clear()
    monkeypatch.setattr(
        entrypoint.services.pipeline_cache, "_build",
        lambda session, kb: LabPipeline(transport, embeddings, watch=watched.get("watch")),
    )
    kb = manager.create("lab-kb", chunker={"type": "structure_first"})
    return SimpleNamespace(kb_id=kb["kb_id"], admission=admission, answers=answers,
                           embeddings=embeddings, transport=transport, registry=registry,
                           watched=watched)


#: The retrieval methods one search resource offers. Every one of them
#: reaches retrieval; ``vector`` is the one that also embeds the question.
METHODS = ("bm25", "hybrid", "vector")

#: The one ``mode`` every search is measured under. It used to be one per
#: endpoint, because there was an endpoint per leg.
SEARCH_MODE = "lab.experiment_search"


def api():
    """One client over the process's own application."""
    return TestClient(http.create_app(entrypoint.services),
                      raise_server_exceptions=False)


def post(client, lab, method):
    return client.post(f"{V1}/searches", json={
        "query": QUESTION, "knowledge_base_id": lab.kb_id,
        "method": method, "limit": 5,
    })


# ------------------------------------------------------------- admitted
@pytest.mark.parametrize("method", METHODS)
def test_a_search_is_admitted_measured_and_takes_no_answer_slot(lab, method):
    with api() as client:
        response = post(client, lab, method)
    assert response.status_code == 200, response.json()

    assert lab.admission.snapshot()["accepted_total"] == 1
    assert lab.admission.active == 0, "released on the way out"
    # No answer-model call is made here, which is why these endpoints are
    # given no answer budget -- only admission, the deadline and the trace.
    assert lab.answers.snapshot()["acquired_total"] == 0

    queries = lab.registry.snapshot()["queries"]
    assert queries["measured"] == 1
    assert queries["recent"][0]["mode"] == SEARCH_MODE, (
        "search traffic is distinguishable from a question")
    assert queries["recent"][0]["status"] == "succeeded"
    assert queries["outcomes"] == {"succeeded": 1}


def test_the_pipeline_is_leased_for_the_length_of_a_search(lab):
    """A burst of other sessions must not evict this pipeline and close its
    store while the search is reading it."""
    seen = []
    lab.watched["watch"] = lambda: seen.append(entrypoint.services.pipeline_cache.snapshot()["leased"])
    with api() as client:
        assert post(client, lab, "bm25").status_code == 200
    assert seen == [1], "leased while the retriever was running"
    assert entrypoint.services.pipeline_cache.snapshot()["leased"] == 0


# ------------------------------------------------------------- refused
@pytest.mark.parametrize("method", METHODS)
def test_a_search_is_refused_when_every_query_slot_is_in_use(lab, method):
    """The same 503 a question gets, for the same reason, with the same
    Retry-After: one shape, whichever method was asked for."""
    assert lab.admission.try_enter() and lab.admission.try_enter()
    try:
        with api() as client:
            response = post(client, lab, method)
    finally:
        lab.admission.leave()
        lab.admission.leave()

    assert response.status_code == 503
    error = response.json()["error"]
    assert error["type"] == "overloaded" and error["details"]["reason"] == "admission"
    assert error["details"]["retry_after_seconds"] >= Q.RETRY_AFTER_MIN
    assert response.headers["Retry-After"] == str(int(error["details"]["retry_after_seconds"]))
    assert lab.registry.snapshot()["counters"]["query.rejected"] == 1


def test_searches_and_questions_share_one_bound(lab, monkeypatch):
    """The bound is on request threads doing retrieval, not on one route.
    With the only slot held by a search, a question is refused too -- and the
    reverse -- which is what makes the starvation guarantee hold across the
    whole retrieval surface rather than on one endpoint."""
    single = Q.QueryAdmission(1)
    monkeypatch.setattr(entrypoint.services, "query_admission", single)

    assert single.try_enter(), "stand in for a Lab search in flight"
    with api() as client:
        question = client.post(f"{V1}/queries",
                               json={"question": "soru", "knowledge_base_id": lab.kb_id})
        assert question.status_code == 503
        assert question.json()["error"]["details"]["reason"] == "admission"
        # ... and health still answers, and says why.
        health = client.get(f"{V1}/health").json()
        assert health["state"] == "overloaded" and health["ready"] is True
        assert "every query slot is in use" in health["reasons"][0]
        assert health["capacity"]["query"]["active"] == 1
    single.leave()

    # The reverse direction: a question in flight refuses a Lab search.
    assert single.try_enter()
    with api() as client:
        refused = post(client, lab, "hybrid")
    single.leave()
    assert refused.status_code == 503
    assert refused.json()["error"]["details"]["reason"] == "admission"


# ------------------------------------------------------------- deadline
def test_the_query_deadline_reaches_a_search(lab, monkeypatch):
    """A dense search's embedding call goes through the same wrapper a
    question's does, so the same guard stops it: 504, nothing embedded."""
    monkeypatch.setattr(Q.QueryGuard, "for_timeout",
                        classmethod(lambda cls, seconds, clock=None: cls(deadline=-1.0)))
    with api() as client:
        response = post(client, lab, "vector")

    assert response.status_code == 504
    assert response.json()["error"]["type"] == "timeout"
    assert lab.transport.calls == 0, "out of time: no embedding request was made"
    assert lab.admission.active == 0, "and the slot came back"
    assert lab.registry.snapshot()["queries"]["outcomes"] == {"timed_out": 1}


def test_a_lexical_search_runs_to_the_end_of_its_stage(lab, monkeypatch):
    """The honest half of the same claim. A lexical search makes no outbound
    call, so there is no seam for the deadline to act on and the stage runs
    to its end -- exactly what QUERY_DEADLINE_SEMANTICS says. It is still
    admitted, still measured, and still bounded by the admission count."""
    monkeypatch.setattr(Q.QueryGuard, "for_timeout",
                        classmethod(lambda cls, seconds, clock=None: cls(deadline=-1.0)))
    with api() as client:
        response = post(client, lab, "bm25")
    assert response.status_code == 200
    assert lab.admission.active == 0
    assert "clamped" in Q.QUERY_DEADLINE_SEMANTICS


# ---------------------------------------------------------- the answer shape
def test_a_search_answers_ranked_rows_and_names_the_method_that_ran(lab):
    with api() as client:
        hybrid = post(client, lab, "hybrid").json()
        vector = post(client, lab, "vector").json()
        bm25 = post(client, lab, "bm25").json()

    assert hybrid["method"] == "hybrid"
    assert [c["id"] for c in hybrid["items"]] == ["c0", "c1", "c2"]
    assert hybrid["knowledge_base_id"] == lab.kb_id
    assert len(vector["items"]) == 3 and vector["method"] == "vector"
    assert bm25["items"][0]["retrieval_method"] == "bm25"


def test_a_request_that_never_reaches_retrieval_is_still_refused_by_name(lab):
    """Validation refusals keep their status and their message, and take no
    slot: an empty question is answered before admission."""
    with api() as client:
        empty = client.post(f"{V1}/searches",
                            json={"query": "", "knowledge_base_id": lab.kb_id})
        assert empty.status_code == 400
        assert "Query is required" in empty.json()["error"]["message"]

        unknown = client.post(f"{V1}/searches", json={
            "query": QUESTION, "knowledge_base_id": lab.kb_id, "method": "telepathy"})
        assert unknown.status_code == 400
        assert unknown.json()["error"]["details"]["supported"] == ["hybrid", "bm25", "vector"]


def test_a_search_failure_is_still_a_500_and_releases_its_slot(lab, monkeypatch):
    lab.watched["watch"] = lambda: (_ for _ in ()).throw(RuntimeError("store broke"))
    with api() as client:
        response = post(client, lab, "bm25")
    assert response.status_code == 500
    assert lab.admission.active == 0
    recent = lab.registry.snapshot()["queries"]["recent"][0]
    assert recent["status"] == "failed" and recent["mode"] == SEARCH_MODE


def test_no_answer_model_is_ever_reached_from_a_search(lab, monkeypatch):
    """The justification for giving this endpoint no answer budget."""
    model = GatedAnswerModel()
    model.release()
    monkeypatch.setattr(entrypoint.services.pipeline_cache, "_build", lambda session, kb: SimpleNamespace(
        settings=SimpleNamespace(embedding_model_name="test/embedding"),
        hybrid_retriever=Retriever(), vector_db=Store(),
        embedding_model=Embedding(lab.transport, lab.embeddings),
        llm_model=model, answer_model=model,
    ))
    with api() as client:
        for method in METHODS:
            assert post(client, lab, method).status_code == 200
    assert model.calls == 0
    assert lab.answers.snapshot()["acquired_total"] == 0
