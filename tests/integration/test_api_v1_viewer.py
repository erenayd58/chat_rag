"""The two routes the Viewer needs, over the real stack.

Step 12 moved the Viewer into the console's own front end. It had been a
second server on `:8765` reading `/api/demo/*`; it is a screen now, and what a
screen can read is `/api/v1`. Two things it needs had no answer there, and
this module drives both against the real packager, the real chunkers and the
real retrieval engine -- only the answer model and the embedding model are
doubles, for the reason `api_v1_doubles` gives.

**The payload.** One document's whole analysis as a reader's view of it. What
makes it a resource of its own rather than a convenience over the per-method
rows is the unit-offset mapping: where a chunk starts and ends *inside* a
canonical unit. That is what lets three methods be drawn down one column of
text, it is written by the packager, and no client can compute it.

**The analysis query.** One question, one document, through several chunking
methods at once. Every other query on this contract searches a knowledge base,
which has exactly one chunker; this is the only one that can compare them, and
it is what the Viewer's *Sorgu* runs on.

The engine behind the second is process-wide and holds its indexes, so the
fixture drops it on the way in and out. A test that answered out of another
test's arms would pass for the wrong reason exactly once.
"""

from __future__ import annotations

import io
import time

import pytest
from api_v1_doubles import (
    DOCUMENT, OTHER_DOCUMENT, PATIENCE_SECONDS, CitingLLM, DeterministicEmbedding,
)
from fastapi.testclient import TestClient

from application import analysis_query
from application.services import build_services
from components.viewer import analysis
from components.viewer import methods as M
from interfaces.http.v1 import PREFIX, create_app

V1 = PREFIX


@pytest.fixture
def api(tmp_path, monkeypatch):
    from pipeline.rag_pipeline import RAGPipeline

    monkeypatch.setenv("RETRIEVAL_PROFILE", "hybrid_rrf")
    llm = CitingLLM()
    monkeypatch.setattr(RAGPipeline, "_create_llm", lambda self: llm)
    monkeypatch.setattr(RAGPipeline, "_create_embedding",
                        lambda self: DeterministicEmbedding())

    analysis._queue.join()
    with analysis._lock:
        analysis._inflight.clear()
    monkeypatch.setattr(analysis, "root", lambda: tmp_path / "viewer-live")
    analysis_query.reset()

    services = build_services()
    with TestClient(create_app(services)) as client:
        client.services = services
        client.llm = llm
        yield client
    analysis_query.reset()
    services.ingest_jobs.close(timeout=PATIENCE_SECONDS)
    services.pipeline_cache.clear()
    analysis._queue.join()
    with analysis._lock:
        analysis._inflight.clear()


# ------------------------------------------------------------------ helpers
def _ok(response):
    assert response.status_code == 200, response.text
    return response.json()


def _knowledge_base(client, name):
    response = client.post(f"{V1}/knowledge-bases",
                           json={"name": name, "chunker": {"type": "structure_first"}})
    assert response.status_code == 201, response.text
    return response.json()["id"]


def _ingested(client, kb_id, *, text=DOCUMENT, name="rapor.md",
              methods=(M.STANDARD, M.MARKDOWN)):
    """One upload, followed to a settled job and then to a ready analysis."""
    response = client.post(
        f"{V1}/documents",
        files={"file": (name, io.BytesIO(text.encode("utf-8")), "text/markdown")},
        data={"knowledge_base_id": kb_id, "methods": list(methods)},
    )
    assert response.status_code == 202, response.text
    job_id = response.json()["id"]

    deadline = time.monotonic() + PATIENCE_SECONDS
    while time.monotonic() < deadline:
        job = _ok(client.get(f"{V1}/ingest-jobs/{job_id}"))
        if job["status"] not in ("queued", "running"):
            break
        time.sleep(0.05)
    assert job["status"] == "succeeded", job
    document_id = job["result"]["document_id"]

    while time.monotonic() < deadline:
        state = _ok(client.get(f"{V1}/documents/{document_id}/analysis"))
        if state["status"] in ("ready", "failed"):
            break
        time.sleep(0.05)
    assert state["status"] == "ready", state
    return document_id, state


# ------------------------------------------------------------- the payload
def test_the_payload_carries_the_units_the_arms_and_where_they_cut(api):
    kb_id = _knowledge_base(api, "Yillik raporlar")
    document_id, state = _ingested(api, kb_id)

    found = _ok(api.get(f"{V1}/documents/{document_id}/analysis/payload"))
    assert found["document_id"] == document_id
    assert found["content_id"] == state["content_id"]
    assert set(found["ready_methods"]) == {M.STANDARD, M.MARKDOWN}

    payload = found["payload"]
    assert payload["units"], "the canonical reading is the page the arms print onto"
    assert payload["pages"]
    assert set(payload["arms"]) == set(found["ready_methods"]), (
        "the arms in the payload are exactly the methods it says are ready"
    )

    for method in found["ready_methods"]:
        arm = payload["arms"][method]
        assert arm["chunks"], method
        # The reason this resource exists: per unit, which chunk covers which
        # offsets of it. Without this the methods cannot be aligned on the text.
        assert arm["seg"], f"{method} carries no unit-offset mapping"
        unit_ids = {unit["i"] for unit in payload["units"]}
        assert set(arm["seg"]) <= unit_ids
        for rows in arm["seg"].values():
            for chunk_index, start, end, _how in rows:
                assert 0 <= chunk_index < len(arm["chunks"])
                assert 0 <= start <= end


def test_a_document_with_no_analysis_is_not_ready_rather_than_absent(api):
    kb_id = _knowledge_base(api, "Yillik raporlar")
    document_id, _state = _ingested(api, kb_id)
    analysis.discard(document_id)

    refused = api.get(f"{V1}/documents/{document_id}/analysis/payload")
    assert refused.status_code == 409, refused.text
    body = refused.json()["error"]
    assert body["type"] == "not_ready"
    # The state comes with it, so a screen polls instead of giving up.
    assert body["details"]["state"]["status"]


def test_an_unknown_document_has_no_payload(api):
    refused = api.get(f"{V1}/documents/kimse-yok/analysis/payload")
    assert refused.status_code == 409
    assert refused.json()["error"]["type"] == "not_ready"


# ------------------------------------------------------- the analysis query
def test_one_question_is_answered_through_every_ready_method(api):
    kb_id = _knowledge_base(api, "Yillik raporlar")
    document_id, _state = _ingested(api, kb_id)

    found = _ok(api.post(f"{V1}/analysis-queries", json={
        "document_id": document_id, "question": "takipteki alacaklar ne oldu?"}))

    assert set(found["methods"]) == {M.STANDARD, M.MARKDOWN}
    assert [arm["method"] for arm in found["arms"]] == found["methods"]
    assert found["answer_model"] == "citing-test-model"

    for arm in found["arms"]:
        assert arm["sources"], f"{arm['method']} retrieved nothing"
        assert all(source["text"] for source in arm["sources"])
        assert arm["answer"]["text"], arm
        assert arm["status"] in ("ok", "insufficient"), arm
        # Two arms ran, so each can say how much of its context the other
        # also found. That number is what the comparison is for.
        assert arm["unit_overlap"] is not None


def test_naming_one_method_answers_in_the_same_shape_as_naming_four(api):
    kb_id = _knowledge_base(api, "Yillik raporlar")
    document_id, _state = _ingested(api, kb_id)

    found = _ok(api.post(f"{V1}/analysis-queries", json={
        "document_id": document_id, "question": "sermaye yeterliligi",
        "methods": [M.STANDARD]}))

    assert found["methods"] == [M.STANDARD]
    assert len(found["arms"]) == 1
    # One arm has nothing to overlap with, and says so rather than claiming 0.
    assert found["arms"][0]["unit_overlap"] is None


def test_retrieval_only_skips_the_answer_model(api):
    kb_id = _knowledge_base(api, "Yillik raporlar")
    document_id, _state = _ingested(api, kb_id)
    before = api.llm.calls

    found = _ok(api.post(f"{V1}/analysis-queries", json={
        "document_id": document_id, "question": "takipteki alacaklar",
        "methods": [M.STANDARD], "answer": False}))

    assert api.llm.calls == before, "answer: false must not reach the answer model"
    assert found["arms"][0]["sources"]
    assert found["arms"][0]["answer"] is None
    assert found["answer_model"] is None
    # Still reported, because retrieval still used one: the arm's dense leg is
    # the deployment's configured model, not a second one the Viewer chose.
    assert found["embedding_model"] == api.services.settings.embedding_model_name


def test_the_three_refusals_stay_three_different_answers(api):
    kb_id = _knowledge_base(api, "Yillik raporlar")
    document_id, _state = _ingested(api, kb_id, methods=(M.STANDARD,))

    empty = api.post(f"{V1}/analysis-queries", json={
        "document_id": document_id, "question": "   "})
    assert empty.status_code == 400
    assert empty.json()["error"]["type"] == "invalid_request"

    unknown = api.post(f"{V1}/analysis-queries", json={
        "document_id": document_id, "question": "soru", "methods": ["kimse-yok"]})
    assert unknown.status_code == 400
    assert unknown.json()["error"]["type"] == "invalid_request"
    assert M.STANDARD in unknown.json()["error"]["details"]["supported"]

    # A real method this upload did not select is a different answer: the
    # content may have it, it is still not this document's to serve.
    unselected = api.post(f"{V1}/analysis-queries", json={
        "document_id": document_id, "question": "soru", "methods": [M.MARKDOWN]})
    assert unselected.status_code == 404
    assert unselected.json()["error"]["type"] == "not_found"


def test_a_re_analysed_document_is_not_answered_out_of_its_old_arms(api):
    """The engine keys its indexes on the analysis, not on the document id.

    A document that gained a method has different rows under the same id, and
    answering out of the ones already indexed would silently hide the new arm.
    """
    kb_id = _knowledge_base(api, "Yillik raporlar")
    document_id, _state = _ingested(api, kb_id, methods=(M.STANDARD,))

    first = _ok(api.post(f"{V1}/analysis-queries", json={
        "document_id": document_id, "question": "takipteki alacaklar"}))
    assert first["methods"] == [M.STANDARD]

    added = api.post(f"{V1}/documents/{document_id}/analysis/methods",
                     json={"methods": [M.MARKDOWN]})
    assert added.status_code == 202, added.text
    deadline = time.monotonic() + PATIENCE_SECONDS
    while time.monotonic() < deadline:
        state = _ok(api.get(f"{V1}/documents/{document_id}/analysis"))
        if M.MARKDOWN in state["ready_methods"]:
            break
        time.sleep(0.05)
    assert M.MARKDOWN in state["ready_methods"], state

    again = _ok(api.post(f"{V1}/analysis-queries", json={
        "document_id": document_id, "question": "takipteki alacaklar"}))
    assert set(again["methods"]) == {M.STANDARD, M.MARKDOWN}


def test_two_documents_are_kept_apart(api):
    kb_id = _knowledge_base(api, "Yillik raporlar")
    yearly, _s1 = _ingested(api, kb_id, methods=(M.STANDARD,))
    green, _s2 = _ingested(api, kb_id, name="surdurulebilirlik.md",
                           text=OTHER_DOCUMENT, methods=(M.STANDARD,))

    found = _ok(api.post(f"{V1}/analysis-queries", json={
        "document_id": green, "question": "karbon ayak izi"}))
    assert found["document_id"] == green
    sources = found["arms"][0]["sources"]
    assert sources
    assert all("takipteki" not in source["text"].casefold() for source in sources), (
        "a question of one document must not be answered out of another's arms"
    )
    assert yearly != green
