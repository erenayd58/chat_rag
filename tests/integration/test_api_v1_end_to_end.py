"""`/api/v1`, end to end, over the real persistence.

``tests/migration/test_api_v1_contract.py`` drives this surface as a contract:
its stores are doubles, so what it proves is that the *shapes* and the
*refusals* are what the contract says. This module proves the other half --
that the flows actually work over what the product now runs on:

    FastAPI  ->  application/  ->  PostgreSQL + pgvector
                                   the real chunker, the real retriever,
                                   the real ingest workers, the real packager

Only the two things that would leave the machine are replaced: the answer
model and the embedding model. Both are deterministic, so an assertion here is
about the product and never about a provider being reachable. Everything else
is what a deployment runs -- the tables the migrations built, the vectors
pgvector stores, the jobs the manager runs on its own threads, and the
analysis the packager builds on its own worker.

Driven through Starlette's transport rather than through the Flask bridge, for
the same reason Step 11 will: this is the surface a client speaks, and a
failure here should name the API rather than the console it is mounted in.
"""

from __future__ import annotations

import io
import time

import pytest
from api_v1_doubles import (
    DOCUMENT, OTHER_DOCUMENT, PATIENCE_SECONDS, CitingLLM, DeterministicEmbedding,
)
from fastapi.testclient import TestClient

from chat_rag.application.services import build_services
from chat_rag.components.viewer import analysis
from chat_rag.components.viewer import methods as M
from interfaces.http.v1 import PREFIX, create_app

V1 = PREFIX


# -------------------------------------------------------------- the fixture
@pytest.fixture
def api(tmp_path, monkeypatch):
    """One application over the test database, served at `/api/v1`.

    The container is the real one -- ``build_services`` composes exactly what
    a process composes -- with the two providers replaced at the pipeline's
    own factories, so every pipeline the cache builds for every knowledge base
    gets them and nothing else about a pipeline is a double.
    """
    from chat_rag.pipeline.rag_pipeline import RAGPipeline

    # The profile with both legs: a dense index to rebuild and a lexical one
    # to go stale. bm25_only, the process default, has neither.
    monkeypatch.setenv("RETRIEVAL_PROFILE", "hybrid_rrf")

    llm = CitingLLM()
    monkeypatch.setattr(RAGPipeline, "_create_llm", lambda self: llm)
    monkeypatch.setattr(RAGPipeline, "_create_embedding",
                        lambda self: DeterministicEmbedding())

    # The packager writes into the test's own directory; its worker is drained
    # on the way out, below, so nothing it queued outlives the test.
    monkeypatch.setattr(analysis, "root", lambda: tmp_path / "viewer-live")

    services = build_services()
    with TestClient(create_app(services)) as client:
        client.services = services
        client.llm = llm
        yield client
    services.ingest_jobs.close(timeout=PATIENCE_SECONDS)
    # This container's own packager, drained before its pipelines go. Outside
    # the activation ``analysis.state()`` is the process default's, which is
    # never the one a container built here owns -- draining that one waited
    # on an empty queue while this one went on building into the next test.
    with services.activate():
        analysis.state().queue.join()
        with analysis.state().lock:
            analysis.state().inflight.clear()
    services.pipeline_cache.clear()


def _created(response):
    assert response.status_code == 201, response.text
    return response.json()


def _ok(response):
    assert response.status_code == 200, response.text
    return response.json()


def _knowledge_base(client, name, **extra):
    return _created(client.post(f"{V1}/knowledge-bases", json={
        "name": name, "chunker": {"type": "structure_first"}, **extra}))


def _upload(client, kb_id, *, name="rapor.md", text=DOCUMENT, methods=(M.STANDARD,)):
    """One upload, followed to its settled job."""
    response = client.post(
        f"{V1}/documents",
        files={"file": (name, io.BytesIO(text.encode("utf-8")), "text/markdown")},
        data={"knowledge_base_id": kb_id, "methods": list(methods)},
    )
    assert response.status_code == 202, response.text
    job = response.json()
    assert response.headers["location"] == f"{V1}/ingest-jobs/{job['id']}"
    return _settled(client, job["id"])


def _settled(client, job_id):
    """Poll one job until it is neither queued nor running."""
    deadline = time.monotonic() + PATIENCE_SECONDS
    while time.monotonic() < deadline:
        job = _ok(client.get(f"{V1}/ingest-jobs/{job_id}"))
        if job["status"] not in ("queued", "running"):
            return job
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} never settled")


def _ready_analysis(client, document_id):
    """Poll one document's analysis until the packager has finished with it."""
    deadline = time.monotonic() + PATIENCE_SECONDS
    while time.monotonic() < deadline:
        state = _ok(client.get(f"{V1}/documents/{document_id}/analysis"))
        if state["status"] in ("ready", "failed"):
            return state
        time.sleep(0.05)
    raise AssertionError(f"the analysis of {document_id} never settled")


def _document_of(client, job):
    assert job["status"] == "succeeded", job
    assert job["result"]["document_id"], job
    return job["result"]["document_id"]


# ====================================================== the knowledge base alone
def test_a_knowledge_base_is_created_read_renamed_listed_and_deleted(api):
    created = _knowledge_base(api, "Yillik raporlar")
    kb_id = created["id"]

    assert _ok(api.get(f"{V1}/knowledge-bases/{kb_id}"))["name"] == "Yillik raporlar"

    renamed = _ok(api.patch(f"{V1}/knowledge-bases/{kb_id}",
                            json={"name": "Yillik raporlar 2024"}))
    assert renamed["name"] == "Yillik raporlar 2024"
    assert renamed["chunker"] == created["chunker"], "creation-time facts are not editable"

    listed = _ok(api.get(f"{V1}/knowledge-bases"))
    assert [item["id"] for item in listed["items"]] == [kb_id]
    assert listed["page"] == {"offset": 0, "limit": 50, "total": 1}

    assert api.delete(f"{V1}/knowledge-bases/{kb_id}").status_code == 204
    assert api.get(f"{V1}/knowledge-bases/{kb_id}").status_code == 404


def test_a_second_knowledge_base_of_the_same_name_is_refused_and_creates_nothing(api):
    _knowledge_base(api, "Yillik raporlar")
    refused = api.post(f"{V1}/knowledge-bases", json={
        "name": "Yillik raporlar", "chunker": {"type": "structure_first"}})
    assert refused.status_code == 400
    assert refused.json()["error"]["type"] == "invalid_request"
    assert _ok(api.get(f"{V1}/knowledge-bases"))["page"]["total"] == 1


# ========================================= upload -> job -> document -> chunks
def test_an_upload_becomes_a_job_a_document_and_stored_chunks(api):
    kb_id = _knowledge_base(api, "Yillik raporlar")["id"]

    job = _upload(api, kb_id)
    document_id = _document_of(api, job)
    assert job["knowledge_base_id"] == kb_id
    assert job["content_id"], "a settled job names the bytes it ingested"
    assert job["result"]["chunk_count"] > 0
    assert job["restart_settled"] is False

    document = _ok(api.get(f"{V1}/documents/{document_id}"))
    assert document["knowledge_base_id"] == kb_id
    assert document["name"] == "rapor.md"
    assert document["content_id"] == job["content_id"]
    assert document["chunk_count"] == job["result"]["chunk_count"]
    assert document["ingest_job_id"] == job["id"]
    assert document["size_bytes"] > 0

    chunks = _ok(api.get(f"{V1}/documents/{document_id}/chunks"))
    assert chunks["page"]["total"] == document["chunk_count"]
    assert all(row["document_id"] == document_id for row in chunks["items"])
    assert any("alacaklar" in row["content"].casefold() for row in chunks["items"])

    # Nothing above named a knowledge base: a document belongs to exactly one
    # and its ledger row is what says which, so a client is not asked to.

    # A parser with no canonical reading of this format is the documented
    # 404 -- not an empty page, which a client would read as "no content".
    units = api.get(f"{V1}/documents/{document_id}/units")
    assert units.status_code == 404, units.text
    assert units.json()["error"]["type"] == "not_found"


def test_the_job_list_carries_the_queue_and_narrows_to_one_knowledge_base(api):
    first = _knowledge_base(api, "Birinci")["id"]
    second = _knowledge_base(api, "Ikinci")["id"]
    _upload(api, first)
    _upload(api, second, name="surdurulebilirlik.md", text=OTHER_DOCUMENT)

    every = _ok(api.get(f"{V1}/ingest-jobs"))
    assert every["page"]["total"] == 2
    assert set(every["capacity"]) == {"running", "queued", "queue_capacity", "workers"}
    assert every["capacity"]["workers"] >= 1

    only_first = _ok(api.get(f"{V1}/ingest-jobs", params={"knowledge_base_id": first}))
    assert [job["knowledge_base_id"] for job in only_first["items"]] == [first]
    assert _ok(api.get(f"{V1}/ingest-jobs", params={"active": "true"}))["page"]["total"] == 0


# ================================================ the analysis and its methods
def test_an_uploads_analysis_is_built_and_one_methods_rows_are_served(api):
    kb_id = _knowledge_base(api, "Yillik raporlar")["id"]
    document_id = _document_of(api, _upload(api, kb_id))

    state = _ready_analysis(api, document_id)
    assert state["status"] == "ready", state
    assert state["selected_methods"] == [M.STANDARD]
    assert state["ready_methods"] == [M.STANDARD]
    assert state["failed_methods"] == []
    assert state["content_id"]
    assert state["unit_count"] > 0
    assert state["content"]["shared_with_document_ids"] == [document_id]

    rows = _ok(api.get(
        f"{V1}/documents/{document_id}/analysis/methods/{M.STANDARD}/chunks"))
    assert rows["page"]["total"] > 0
    assert rows["method"] == M.STANDARD
    # The rows are the packager's own JSONL, passed through -- the same
    # representation a frozen arm is read over.
    assert all(item["text"] for item in rows["items"])
    assert all(item["chunk_id"] for item in rows["items"])


def test_a_variant_is_added_to_a_document_that_is_already_here(api):
    kb_id = _knowledge_base(api, "Yillik raporlar")["id"]
    document_id = _document_of(api, _upload(api, kb_id))
    _ready_analysis(api, document_id)

    added = api.post(f"{V1}/documents/{document_id}/analysis/methods",
                     json={"methods": [M.MARKDOWN]})
    assert added.status_code == 202, added.text
    assert M.MARKDOWN in added.json()["selected_methods"]

    state = _ready_analysis(api, document_id)
    assert set(state["ready_methods"]) == {M.STANDARD, M.MARKDOWN}
    variant = _ok(api.get(
        f"{V1}/documents/{document_id}/analysis/methods/{M.MARKDOWN}/chunks"))
    assert variant["page"]["total"] > 0
    assert variant["method"] == M.MARKDOWN


def test_a_method_this_upload_never_selected_is_not_its_to_serve(api):
    kb_id = _knowledge_base(api, "Yillik raporlar")["id"]
    document_id = _document_of(api, _upload(api, kb_id))
    _ready_analysis(api, document_id)

    unknown = api.get(f"{V1}/documents/{document_id}/analysis/methods/kimse-yok/chunks")
    assert unknown.status_code == 400
    assert unknown.json()["error"]["type"] == "invalid_request"

    unselected = api.get(
        f"{V1}/documents/{document_id}/analysis/methods/{M.MARKDOWN}/chunks")
    assert unselected.status_code == 404
    assert unselected.json()["error"]["type"] == "not_found"


# ========================================================== search and query
@pytest.mark.parametrize("method", ["bm25", "vector", "hybrid"])
def test_a_search_finds_the_ingested_document_by_every_method(api, method):
    kb_id = _knowledge_base(api, "Yillik raporlar")["id"]
    document_id = _document_of(api, _upload(api, kb_id))

    found = _ok(api.post(f"{V1}/searches", json={
        "query": "takipteki alacaklar", "knowledge_base_id": kb_id,
        "method": method, "limit": 5}))
    assert found["items"], f"{method} found nothing in a corpus that has it"
    assert found["knowledge_base_id"] == kb_id
    assert all(item["document_id"] == document_id for item in found["items"])
    assert found["items"][0]["score"] is not None
    assert found["items"][0]["retrieval_method"]


def test_a_search_never_reaches_another_knowledge_bases_corpus(api):
    first = _knowledge_base(api, "Birinci")["id"]
    second = _knowledge_base(api, "Ikinci")["id"]
    _upload(api, first)
    other_document = _document_of(
        api, _upload(api, second, name="surdurulebilirlik.md", text=OTHER_DOCUMENT))

    found = _ok(api.post(f"{V1}/searches", json={
        "query": "karbon ayak izi", "knowledge_base_id": second, "method": "hybrid"}))
    assert found["items"]
    assert {item["document_id"] for item in found["items"]} == {other_document}

    elsewhere = _ok(api.post(f"{V1}/searches", json={
        "query": "karbon ayak izi", "knowledge_base_id": first, "method": "bm25"}))
    assert all(item["document_id"] != other_document for item in elsewhere["items"])


def test_a_question_is_answered_with_citations_that_quote_the_corpus(api):
    kb_id = _knowledge_base(api, "Yillik raporlar")["id"]
    document_id = _document_of(api, _upload(api, kb_id))

    answer = _ok(api.post(f"{V1}/queries", json={
        "question": "Takipteki alacaklar nasil degisti?",
        "knowledge_base_id": kb_id, "top_k": 3}))

    assert answer["answer"], "an answered question with no answer"
    assert answer["knowledge_base_id"] == kb_id
    assert answer["citations"], "an answer with no sources cannot be checked"
    assert answer["grounded"] is True
    assert any(citation["used"] for citation in answer["citations"])
    assert all(citation["content"] for citation in answer["citations"])
    assert {citation["document_id"] for citation in answer["citations"]} == {document_id}
    assert answer["timing"]["query_id"]
    assert answer["timing"]["total_seconds"] is not None
    assert api.llm.calls == 1


def test_an_unservable_retrieval_method_is_refused_with_the_supported_list(api):
    kb_id = _knowledge_base(api, "Yillik raporlar")["id"]
    refused = api.post(f"{V1}/searches", json={
        "query": "alacaklar", "knowledge_base_id": kb_id, "method": "telepathy"})
    assert refused.status_code == 400
    error = refused.json()["error"]
    assert error["type"] == "invalid_request"
    assert set(error["details"]["supported"]) == {"hybrid", "bm25", "vector"}


# ======================================================== the embedding index
def test_the_embedding_index_is_reported_and_rebuilt_in_place(api):
    kb_id = _knowledge_base(api, "Yillik raporlar")["id"]
    document_id = _document_of(api, _upload(api, kb_id))

    before = _ok(api.get(f"{V1}/knowledge-bases/{kb_id}/embedding-index"))
    assert before["state"] == "compatible"
    assert before["dense_available"] is True

    rebuilt = _ok(api.post(f"{V1}/knowledge-bases/{kb_id}/embedding-index/rebuild"))
    assert rebuilt["result"]["chunks"] > 0
    assert rebuilt["result"]["dimension"] == DeterministicEmbedding.DIMENSION
    assert rebuilt["index"]["state"] == "compatible"

    # A re-index rewrites vectors and nothing else: same chunks, still found.
    after = _ok(api.get(f"{V1}/documents/{document_id}/chunks"))
    assert after["page"]["total"] == before["stored"]["chunk_count"]
    assert _ok(api.post(f"{V1}/searches", json={
        "query": "sermaye yeterliligi", "knowledge_base_id": kb_id,
        "method": "vector"}))["items"]


# ========================================================= two of everything
def test_two_knowledge_bases_hold_two_corpora_and_two_ledgers(api):
    first = _knowledge_base(api, "Birinci")["id"]
    second = _knowledge_base(api, "Ikinci")["id"]
    first_document = _document_of(api, _upload(api, first))
    second_document = _document_of(
        api, _upload(api, second, name="surdurulebilirlik.md", text=OTHER_DOCUMENT))

    every = _ok(api.get(f"{V1}/documents"))
    assert {row["id"] for row in every["items"]} == {first_document, second_document}

    narrowed = _ok(api.get(f"{V1}/documents", params={"knowledge_base_id": second}))
    assert [row["id"] for row in narrowed["items"]] == [second_document]

    assert _ok(api.get(f"{V1}/knowledge-bases/{first}/chunks"))["page"]["total"] > 0
    searched = _ok(api.get(f"{V1}/knowledge-bases/{second}/chunks",
                           params={"search": "karbon"}))
    assert searched["page"]["total"] > 0
    assert all("karbon" in row["content"].casefold() for row in searched["items"])


def test_two_documents_in_one_knowledge_base_are_both_retrievable(api):
    kb_id = _knowledge_base(api, "Karisik")["id"]
    first = _document_of(api, _upload(api, kb_id))
    second = _document_of(
        api, _upload(api, kb_id, name="surdurulebilirlik.md", text=OTHER_DOCUMENT))

    assert _ok(api.get(f"{V1}/documents",
                       params={"knowledge_base_id": kb_id}))["page"]["total"] == 2

    alacaklar = _ok(api.post(f"{V1}/searches", json={
        "query": "takipteki alacaklar karsilik orani", "knowledge_base_id": kb_id,
        "method": "bm25"}))
    karbon = _ok(api.post(f"{V1}/searches", json={
        "query": "karbon ayak izi yenilenebilir enerji", "knowledge_base_id": kb_id,
        "method": "bm25"}))
    assert alacaklar["items"][0]["document_id"] == first
    assert karbon["items"][0]["document_id"] == second


# =================================================================== deletion
def test_a_deleted_document_leaves_no_chunk_a_search_can_still_find(api):
    """The lexical index is built once per pipeline and never re-read against
    the store, so a delete that does not say so leaves every cached pipeline
    answering with chunks that are gone -- including the one that deleted."""
    kb_id = _knowledge_base(api, "Yillik raporlar")["id"]
    keep = _document_of(api, _upload(api, kb_id))
    drop = _document_of(
        api, _upload(api, kb_id, name="surdurulebilirlik.md", text=OTHER_DOCUMENT))

    # Both pipelines must have an index built before the delete, or the
    # staleness this guards against cannot happen.
    assert _ok(api.post(f"{V1}/searches", json={
        "query": "karbon ayak izi", "knowledge_base_id": kb_id,
        "method": "bm25"}))["items"]
    onlooker = api.services.get_pipeline("another-session", kb_id)
    assert onlooker.hybrid_retriever.hybrid_search("karbon ayak izi", top_k=5)

    assert api.delete(f"{V1}/documents/{drop}").status_code == 204

    assert api.get(f"{V1}/documents/{drop}").status_code == 404
    assert _ok(api.get(f"{V1}/documents/{keep}"))["id"] == keep

    for method in ("bm25", "vector", "hybrid"):
        found = _ok(api.post(f"{V1}/searches", json={
            "query": "karbon ayak izi", "knowledge_base_id": kb_id, "method": method}))
        assert all(item["document_id"] != drop for item in found["items"]), method

    assert all(result.chunk.doc_id != drop for result
               in onlooker.hybrid_retriever.hybrid_search("karbon ayak izi", top_k=5))


def test_deleting_a_knowledge_base_takes_its_corpus_and_leaves_the_ledger_row(api):
    kb_id = _knowledge_base(api, "Yillik raporlar")["id"]
    document_id = _document_of(api, _upload(api, kb_id))

    assert api.delete(f"{V1}/knowledge-bases/{kb_id}").status_code == 204

    # The record that a file was ever ingested survives its knowledge base.
    surviving = _ok(api.get(f"{V1}/documents/{document_id}"))
    assert surviving["knowledge_base_id"] == kb_id
    assert _ok(api.get(f"{V1}/documents"))["page"]["total"] == 1

    # Its vectors do not: the collection went with the knowledge base.
    reborn = _knowledge_base(api, "Yeni")["id"]
    assert _ok(api.get(f"{V1}/knowledge-bases/{reborn}/chunks"))["page"]["total"] == 0
