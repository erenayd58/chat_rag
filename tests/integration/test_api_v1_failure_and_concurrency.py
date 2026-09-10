"""What `/api/v1` does when two things happen at once, and when one goes wrong.

``tests/unit/test_fastapi_adapter.py`` drives each refusal through a seam:
that proves the exception table routes it, which is a statement about the
adapter. This module provokes the same answers -- and the ones no seam can
fake -- out of the running product, where the queue really is full, the two
uploads really are racing and the delete really does overtake the search.

The cases, and why each is here rather than in a unit test:

* **overload** is a bound the job manager keeps under a lock. Faked, it says
  nothing about whether the bound holds.
* **the same bytes twice** is decided by the content hash and the in-flight
  registry, both of which need two real submissions to be exercised at all.
* **upload against delete** and **rebuild against search** are the two orders
  a client can produce that nothing in the code chooses. What is asserted is
  the invariant that has to hold in *both* orders -- never a winner.
* **a shared content** is the deletion rule with the most ways to be wrong:
  one document's chunks must go while another's analysis of the same bytes
  stays, and only the last upload takes the shared analysis down.
"""

from __future__ import annotations

import io
import time
from concurrent.futures import ThreadPoolExecutor

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


@pytest.fixture
def api(tmp_path, monkeypatch):
    """The same deployment the end-to-end suite uses, on one ingest worker.

    One worker and a queue of one is what makes a second upload wait and a
    third refused -- the smallest arrangement in which the bound is
    observable, and the same code path a busy deployment takes.
    """
    from chat_rag.pipeline.rag_pipeline import RAGPipeline

    monkeypatch.setenv("RETRIEVAL_PROFILE", "hybrid_rrf")
    monkeypatch.setenv("INGEST_WORKERS", "1")
    monkeypatch.setenv("INGEST_QUEUE_CAPACITY", "1")
    monkeypatch.setattr(RAGPipeline, "_create_llm", lambda self: CitingLLM())
    monkeypatch.setattr(RAGPipeline, "_create_embedding",
                        lambda self: DeterministicEmbedding())

    analysis.state().queue.join()
    with analysis.state().lock:
        analysis.state().inflight.clear()
    monkeypatch.setattr(analysis, "root", lambda: tmp_path / "viewer-live")

    services = build_services()
    with TestClient(create_app(services)) as client:
        client.services = services
        yield client
    services.ingest_jobs.close(timeout=PATIENCE_SECONDS)
    services.pipeline_cache.clear()
    analysis.state().queue.join()
    with analysis.state().lock:
        analysis.state().inflight.clear()


def _ok(response):
    assert response.status_code == 200, response.text
    return response.json()


def _knowledge_base(client, name="Yillik raporlar"):
    created = client.post(f"{V1}/knowledge-bases", json={
        "name": name, "chunker": {"type": "structure_first"}})
    assert created.status_code == 201, created.text
    return created.json()["id"]


def _submit(client, kb_id, *, name="rapor.md", text=DOCUMENT):
    """One upload, not waited for. The response, whatever it is."""
    return client.post(
        f"{V1}/documents",
        files={"file": (name, io.BytesIO(text.encode("utf-8")), "text/markdown")},
        data={"knowledge_base_id": kb_id, "methods": [M.STANDARD]})


def _settled(client, job_id):
    deadline = time.monotonic() + PATIENCE_SECONDS
    while time.monotonic() < deadline:
        job = _ok(client.get(f"{V1}/ingest-jobs/{job_id}"))
        if job["status"] not in ("queued", "running"):
            return job
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} never settled")


def _ingest(client, kb_id, **kwargs):
    accepted = _submit(client, kb_id, **kwargs)
    assert accepted.status_code == 202, accepted.text
    job = _settled(client, accepted.json()["id"])
    assert job["status"] == "succeeded", job
    return job


def _ready_analysis(client, document_id):
    deadline = time.monotonic() + PATIENCE_SECONDS
    while time.monotonic() < deadline:
        state = _ok(client.get(f"{V1}/documents/{document_id}/analysis"))
        if state["status"] in ("ready", "failed"):
            return state
        time.sleep(0.05)
    raise AssertionError(f"the analysis of {document_id} never settled")


def _refusal(response, status, kind):
    assert response.status_code == status, response.text
    error = response.json()["error"]
    assert error["type"] == kind, error
    assert error["message"], "a refusal with no message explains nothing"
    return error


# ============================================================ what is not here
@pytest.mark.parametrize("path", [
    "/knowledge-bases/yok", "/documents/yok", "/ingest-jobs/yok",
])
def test_a_resource_that_is_not_here_is_one_answer_and_not_a_fault(api, path):
    _refusal(api.get(f"{V1}{path}"), 404, "not_found")


def test_an_upload_into_a_knowledge_base_that_is_not_here_writes_nothing(api):
    _refusal(_submit(api, "yok"), 404, "not_found")
    assert _ok(api.get(f"{V1}/documents"))["page"]["total"] == 0
    assert _ok(api.get(f"{V1}/ingest-jobs"))["page"]["total"] == 0


def test_a_payload_this_surface_cannot_read_is_a_400_and_names_the_field(api):
    unreadable = api.post(f"{V1}/knowledge-bases", content=b"{not json",
                          headers={"content-type": "application/json"})
    error = _refusal(unreadable, 400, "invalid_request")
    assert error["details"]["fields"], "a rejected payload that names nothing"

    wrong_type = api.post(f"{V1}/queries", json={"question": "soru", "top_k": "cok"})
    assert _refusal(wrong_type, 400, "invalid_request")["details"]["fields"]


def test_a_knowledge_base_of_a_name_already_taken_is_refused_twice_over(api):
    first = _knowledge_base(api, "Yillik raporlar")
    _refusal(api.post(f"{V1}/knowledge-bases", json={
        "name": "Yillik raporlar", "chunker": {"type": "structure_first"}}),
        400, "invalid_request")

    # And by rename, which is the same rule reached from the other side.
    second = _knowledge_base(api, "Baska")
    _refusal(api.patch(f"{V1}/knowledge-bases/{second}",
                       json={"name": "Yillik raporlar"}), 400, "invalid_request")
    assert _ok(api.get(f"{V1}/knowledge-bases/{first}"))["name"] == "Yillik raporlar"


def test_a_variant_that_is_selected_and_not_built_is_not_ready_and_says_so(api):
    """The distinction a polling client depends on: 409 with the state means
    "keep asking", 404 means "never ask again"."""
    kb_id = _knowledge_base(api)
    document_id = _ingest(api, kb_id)["result"]["document_id"]
    _ready_analysis(api, document_id)

    queued = api.post(f"{V1}/documents/{document_id}/analysis/methods",
                      json={"methods": [M.MARKDOWN]})
    assert queued.status_code == 202, queued.text

    answered = api.get(
        f"{V1}/documents/{document_id}/analysis/methods/{M.MARKDOWN}/chunks")
    if answered.status_code == 409:
        assert _refusal(answered, 409, "not_ready")["details"]["state"]
    else:
        # The packager can finish between the two calls; that is the other
        # legal answer and it is the one this test must not forbid.
        assert answered.status_code == 200, answered.text


# ================================================================== overload
def test_a_full_ingest_queue_refuses_rather_than_queues_and_says_when_to_retry(api):
    """Nothing is queued behind a refusal and nothing is kept: the document
    the third upload carried is not ingested later and not half-ingested now."""
    kb_id = _knowledge_base(api)
    accepted, refused = [], []
    for index in range(12):
        response = _submit(api, kb_id, name=f"rapor-{index}.md",
                           text=DOCUMENT + f"\n\nEk {index}.\n")
        (accepted if response.status_code == 202 else refused).append(response)

    assert refused, "one worker and a queue of one refused nothing out of twelve"
    error = _refusal(refused[0], 503, "overloaded")
    assert refused[0].headers["retry-after"]
    assert error["details"]["retry_after_seconds"] > 0

    for response in accepted:
        _settled(api, response.json()["id"])
    assert _ok(api.get(f"{V1}/documents"))["page"]["total"] == len(accepted)


# ======================================================= two of the same bytes
def test_the_same_bytes_submitted_twice_at_once_are_one_parse_and_one_document(api):
    """The second submission attaches to the first job rather than parsing the
    file again, and ``attached_uploads`` is how a client is told so."""
    kb_id = _knowledge_base(api)
    with ThreadPoolExecutor(max_workers=2) as pool:
        both = [future.result() for future in
                [pool.submit(_submit, api, kb_id), pool.submit(_submit, api, kb_id)]]

    assert [response.status_code for response in both] == [202, 202]
    jobs = [response.json() for response in both]
    assert jobs[0]["id"] == jobs[1]["id"], "the same bytes were parsed twice"

    settled = _settled(api, jobs[0]["id"])
    assert settled["status"] == "succeeded"
    assert settled["attached_uploads"] >= 1
    assert _ok(api.get(f"{V1}/documents"))["page"]["total"] == 1


# ================================================== an upload against a delete
def test_deleting_while_another_upload_runs_leaves_the_ledger_and_the_store_agreed(api):
    """Whichever order the two land in, what must not exist afterwards is a
    document the ledger knows and the store does not, or the other way round."""
    kb_id = _knowledge_base(api)
    doomed = _ingest(api, kb_id)["result"]["document_id"]

    second = _submit(api, kb_id, name="cevre.md", text=OTHER_DOCUMENT)
    assert second.status_code == 202, second.text
    with ThreadPoolExecutor(max_workers=2) as pool:
        deletion = pool.submit(api.delete, f"{V1}/documents/{doomed}")
        arrival = pool.submit(_settled, api, second.json()["id"])
        assert deletion.result().status_code == 204
        survivor = arrival.result()

    assert survivor["status"] == "succeeded", survivor
    listed = _ok(api.get(f"{V1}/documents"))
    assert [row["id"] for row in listed["items"]] == [survivor["result"]["document_id"]]

    for row in listed["items"]:
        stored = _ok(api.get(f"{V1}/documents/{row['id']}/chunks"))
        assert stored["page"]["total"] == row["chunk_count"], "ledger and store disagree"

    corpus = _ok(api.get(f"{V1}/knowledge-bases/{kb_id}/chunks"))
    assert all(row["document_id"] != doomed for row in corpus["items"])


# ==================================================== a rebuild against a search
def test_searching_while_the_index_is_rebuilt_answers_from_one_side_or_the_other(api):
    """A re-index rewrites every vector in place. A search running across it
    must come back with an answer -- the old vectors or the new ones -- and
    never with a fault or with half a corpus."""
    kb_id = _knowledge_base(api)
    document_id = _ingest(api, kb_id)["result"]["document_id"]

    def search():
        return [api.post(f"{V1}/searches", json={
            "query": "takipteki alacaklar", "knowledge_base_id": kb_id,
            "method": "hybrid"}) for _ in range(8)]

    with ThreadPoolExecutor(max_workers=2) as pool:
        searching = pool.submit(search)
        rebuilding = pool.submit(
            api.post, f"{V1}/knowledge-bases/{kb_id}/embedding-index/rebuild")
        answers = searching.result()
        assert rebuilding.result().status_code == 200, rebuilding.result().text

    for answer in answers:
        assert answer.status_code == 200, answer.text
        found = answer.json()["items"]
        assert all(item["document_id"] == document_id for item in found)

    assert _ok(api.get(f"{V1}/knowledge-bases/{kb_id}/embedding-index"))[
        "state"] == "compatible"
    assert _ok(api.post(f"{V1}/searches", json={
        "query": "takipteki alacaklar", "knowledge_base_id": kb_id,
        "method": "vector"}))["items"]


# ================================================== a content two uploads share
def test_only_the_last_upload_of_a_content_takes_its_shared_analysis(api):
    """One file, ingested into two knowledge bases, is two uploads and one
    content. Deleting one upload takes its chunks and its ledger row; the
    other keeps the analysis they share, because it is an analysis of the
    bytes and not of either upload."""
    first = _knowledge_base(api, "Birinci")
    second = _knowledge_base(api, "Ikinci")
    one = _ingest(api, first)["result"]["document_id"]
    two = _ingest(api, second)["result"]["document_id"]

    shared = _ready_analysis(api, one)
    assert _ready_analysis(api, two)["content_id"] == shared["content_id"]
    assert set(shared["content"]["shared_with_document_ids"]) == {one, two}

    assert api.delete(f"{V1}/documents/{one}").status_code == 204

    # The other upload is untouched: its rows, its analysis, its variants.
    kept = _ok(api.get(f"{V1}/documents/{two}/analysis"))
    assert kept["status"] == "ready"
    assert kept["ready_methods"] == shared["ready_methods"]
    assert _ok(api.get(
        f"{V1}/documents/{two}/analysis/methods/{M.STANDARD}/chunks"))["page"]["total"] > 0
    assert _ok(api.get(f"{V1}/documents/{two}/chunks"))["page"]["total"] > 0

    # And the deleted one took its own corpus with it.
    assert _ok(api.get(f"{V1}/knowledge-bases/{first}/chunks"))["page"]["total"] == 0
    assert _ok(api.get(f"{V1}/knowledge-bases/{second}/chunks"))["page"]["total"] > 0

    assert api.delete(f"{V1}/documents/{two}").status_code == 204
    assert _ok(api.get(f"{V1}/documents/{two}/analysis"))["status"] == "missing"
