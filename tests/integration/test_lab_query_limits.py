"""The Lab's retrieval endpoints are not a way around the query limits.

Four endpoints do the front half of a query on the request thread, exactly
as ``/api/query`` does -- embed the question, search the store, build the
knowledge base's lexical index if this pipeline has not built it yet -- and
they ran under no limit at all. A burst of them could hold every request
thread, each waiting an unbounded time for an embedding slot, which is the
starvation ``QUERY_MAX_ACTIVE`` exists to prevent: the bound was on one
endpoint rather than on the work.

They make no answer-model call, so they are given no answer budget. What
they are given is the three things the query path already had, and these
tests pin each: the same admission counter (so the bound is on request
threads doing retrieval, whichever endpoint asked), the query deadline
(which is what bounds the embedding-slot wait), and the measurement, under
their own ``mode``.

The pipeline is a stub built *through* the cache, so the lease is real. No
provider is reached.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import app as flask_app
from components.ingest import limits as L
from components.knowledgebase.manager import KnowledgeBaseManager
from components.observability import telemetry as T
from components.query import limits as Q
from core.models import DocumentChunk, RetrievalResult

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
    monkeypatch.setattr(T, "_registry", fresh)
    return fresh


@pytest.fixture
def lab(tmp_path, monkeypatch, registry):
    monkeypatch.chdir(tmp_path)
    manager = KnowledgeBaseManager(str(tmp_path / "kbs.json"))
    monkeypatch.setattr(flask_app, "kb_manager", manager)
    flask_app.app.config.update(TESTING=True)

    admission = Q.QueryAdmission(2)
    monkeypatch.setattr(flask_app, "query_admission", admission)
    answers = L.ProviderBudget(2)
    monkeypatch.setattr(Q, "_answer_budget", answers)
    embeddings = L.ProviderBudget(2)
    transport = FakeEmbeddingTransport()
    watched = {}

    # Built through the cache, not around it: the lease the endpoint takes
    # is then a real lease on a real entry.
    flask_app.pipeline_cache.clear()
    monkeypatch.setattr(
        flask_app.pipeline_cache, "_build",
        lambda session, kb: LabPipeline(transport, embeddings, watch=watched.get("watch")),
    )
    kb = manager.create("lab-kb", chunker={"type": "structure_first"})
    return SimpleNamespace(kb_id=kb["kb_id"], admission=admission, answers=answers,
                           embeddings=embeddings, transport=transport, registry=registry,
                           watched=watched)


#: Every endpoint that reaches retrieval, with a request that gets there.
ENDPOINTS = {
    "lab.search_vector": ("/api/chunks/search-vector",
                          lambda kb: {"query": QUESTION, "kb_id": kb, "offset": 0, "limit": 5}),
    "lab.search_bm25": ("/api/chunks/search-bm25",
                        lambda kb: {"query": QUESTION, "kb_id": kb, "offset": 0, "limit": 5}),
    "lab.experiment_search": ("/api/experiment/search_chunks",
                              lambda kb: {"query": QUESTION, "kb_id": kb, "method": "bm25", "top_k": 5}),
    "lab.experiment_rank": ("/api/experiment/rank_chunks",
                            lambda kb: {"query": QUESTION, "kb_id": kb, "method": "bm25",
                                        "top_k": 5, "chunk_ids": ["c0"]}),
}


def post(client, lab, mode):
    path, body = ENDPOINTS[mode]
    return client.post(path, json=body(lab.kb_id))


# ------------------------------------------------------------- admitted
@pytest.mark.parametrize("mode", sorted(ENDPOINTS))
def test_a_lab_search_is_admitted_measured_and_takes_no_answer_slot(lab, mode):
    with flask_app.app.test_client() as client:
        response = post(client, lab, mode)
    assert response.status_code == 200, response.get_json()

    assert lab.admission.snapshot()["accepted_total"] == 1
    assert lab.admission.active == 0, "released on the way out"
    # No answer-model call is made here, which is why these endpoints are
    # given no answer budget -- only admission, the deadline and the trace.
    assert lab.answers.snapshot()["acquired_total"] == 0

    queries = lab.registry.snapshot()["queries"]
    assert queries["measured"] == 1
    assert queries["recent"][0]["mode"] == mode, "Lab traffic is distinguishable from chat"
    assert queries["recent"][0]["status"] == "succeeded"
    assert queries["outcomes"] == {"succeeded": 1}


def test_the_pipeline_is_leased_for_the_length_of_a_lab_search(lab):
    """A burst of other sessions must not evict this pipeline and close its
    store while the search is reading it."""
    seen = []
    lab.watched["watch"] = lambda: seen.append(flask_app.pipeline_cache.snapshot()["leased"])
    with flask_app.app.test_client() as client:
        assert post(client, lab, "lab.search_bm25").status_code == 200
    assert seen == [1], "leased while the retriever was running"
    assert flask_app.pipeline_cache.snapshot()["leased"] == 0


# ------------------------------------------------------------- refused
@pytest.mark.parametrize("mode", sorted(ENDPOINTS))
def test_a_lab_search_is_refused_when_every_query_slot_is_in_use(lab, mode):
    """The same 503 a question gets, for the same reason, with the same
    Retry-After: one shape, whichever endpoint was asked."""
    assert lab.admission.try_enter() and lab.admission.try_enter()
    try:
        with flask_app.app.test_client() as client:
            response = post(client, lab, mode)
    finally:
        lab.admission.leave()
        lab.admission.leave()

    assert response.status_code == 503
    body = response.get_json()
    assert body["overloaded"] is True and body["reason"] == "admission"
    assert body["retry_after_seconds"] >= Q.RETRY_AFTER_MIN
    assert response.headers["Retry-After"] == str(int(body["retry_after_seconds"]))
    assert lab.registry.snapshot()["counters"]["query.rejected"] == 1


def test_lab_searches_and_questions_share_one_bound(lab, monkeypatch):
    """The bound is on request threads doing retrieval, not on one route.
    With the only slot held by a Lab search, a question is refused too --
    and the reverse -- which is what makes the starvation guarantee hold
    across the whole retrieval surface rather than on /api/query alone."""
    single = Q.QueryAdmission(1)
    monkeypatch.setattr(flask_app, "query_admission", single)

    assert single.try_enter(), "stand in for a Lab search in flight"
    with flask_app.app.test_client() as client:
        question = client.post("/api/query", json={"question": "soru", "kb_id": lab.kb_id})
        assert question.status_code == 503
        assert question.get_json()["reason"] == "admission"
        # ... and health still answers, and says why.
        health = client.get("/api/health").get_json()
        assert health["state"] == "overloaded" and health["ready"] is True
        assert "every query slot is in use" in health["reasons"][0]
        assert health["query"]["active"] == 1
    single.leave()

    # The reverse direction: a question in flight refuses a Lab search.
    assert single.try_enter()
    with flask_app.app.test_client() as client:
        refused = post(client, lab, "lab.experiment_search")
    single.leave()
    assert refused.status_code == 503 and refused.get_json()["reason"] == "admission"


# ------------------------------------------------------------- deadline
def test_the_query_deadline_reaches_a_lab_search(lab, monkeypatch):
    """The endpoint's embedding call goes through the same wrapper a
    question's does, so the same guard stops it: 504, nothing embedded."""
    monkeypatch.setattr(Q.QueryGuard, "for_timeout",
                        classmethod(lambda cls, seconds, clock=None: cls(deadline=-1.0)))
    with flask_app.app.test_client() as client:
        response = post(client, lab, "lab.search_vector")

    assert response.status_code == 504
    body = response.get_json()
    assert body["timed_out"] is True and body["success"] is False
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
    with flask_app.app.test_client() as client:
        response = post(client, lab, "lab.search_bm25")
    assert response.status_code == 200
    assert lab.admission.active == 0
    assert "clamped" in Q.QUERY_DEADLINE_SEMANTICS


# -------------------------------------------------------- unchanged shape
def test_the_answers_are_what_the_lab_screen_already_expected(lab):
    with flask_app.app.test_client() as client:
        search = post(client, lab, "lab.experiment_search").get_json()
        ranked = post(client, lab, "lab.experiment_rank").get_json()
        vector = post(client, lab, "lab.search_vector").get_json()
        bm25 = post(client, lab, "lab.search_bm25").get_json()

    assert search["success"] and search["retrieval_method"] == "bm25"
    assert [c["chunk_id"] for c in search["chunks"]] == ["c0", "c1", "c2"]
    assert ranked["success"] and ranked["results"][0] == {
        "chunk_id": "c0", "found": True, "rank": 0, "score": 1.0,
        "retrieval_method": "bm25", "search_term": QUESTION,
    }
    assert vector["success"] and len(vector["chunks"]) == 3
    assert vector["search_metadata"]["search_method"] == "vector"
    assert bm25["success"] and bm25["chunks"][0]["retrieval_method"] == "bm25"


def test_a_request_that_never_reaches_retrieval_still_answers_as_before(lab):
    """Validation refusals keep their status and their message."""
    with flask_app.app.test_client() as client:
        assert client.post("/api/chunks/search-bm25",
                           json={"query": "", "kb_id": lab.kb_id}).status_code == 400
        missing_kb = client.post("/api/chunks/search-vector", json={"query": QUESTION})
        assert missing_kb.status_code == 400
        assert "Knowledge base" in missing_kb.get_json()["error"]


def test_an_endpoint_failure_is_still_a_500_and_releases_its_slot(lab, monkeypatch):
    lab.watched["watch"] = lambda: (_ for _ in ()).throw(RuntimeError("store broke"))
    with flask_app.app.test_client() as client:
        response = post(client, lab, "lab.search_bm25")
    assert response.status_code == 500
    assert lab.admission.active == 0
    recent = lab.registry.snapshot()["queries"]["recent"][0]
    assert recent["status"] == "failed" and recent["mode"] == "lab.search_bm25"


def test_no_answer_model_is_ever_reached_from_the_lab(lab, monkeypatch):
    """The justification for giving these endpoints no answer budget."""
    model = GatedAnswerModel()
    model.release()
    monkeypatch.setattr(flask_app.pipeline_cache, "_build", lambda session, kb: SimpleNamespace(
        settings=SimpleNamespace(embedding_model_name="test/embedding"),
        hybrid_retriever=Retriever(), vector_db=Store(),
        embedding_model=Embedding(lab.transport, lab.embeddings),
        llm_model=model, answer_model=model,
    ))
    with flask_app.app.test_client() as client:
        for mode in sorted(ENDPOINTS):
            assert post(client, lab, mode).status_code == 200
    assert model.calls == 0
    assert lab.answers.snapshot()["acquired_total"] == 0
