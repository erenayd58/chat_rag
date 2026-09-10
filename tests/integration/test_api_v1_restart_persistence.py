"""What survives a restart, read back through `/api/v1`.

Steps 8 and 9 moved every record and every vector into PostgreSQL, which is
the reason a restart is now supposed to be uneventful. "Supposed to" is not a
test, and the pieces are held by four different owners:

===========================  ==================================================
the knowledge bases           PostgreSQL, ``knowledge_bases``
the ingest ledger             PostgreSQL, ``documents``
the chunks and their vectors  PostgreSQL + pgvector, ``chunk_vectors``
the embedding manifest        PostgreSQL, ``vector_collections``
the analysis state            PostgreSQL, ``contents``; its artifacts on disk
the ingest jobs               the journal, settled at start-up against the
                              ledger
===========================  ==================================================

A restart here is the real thing minus the operating system: the workers are
stopped, the pipeline cache is cleared and closed, the container is thrown
away, and a **new** container is composed against the same database and the
same directories -- through ``runtime.bootstrap.resume_background_work``,
which is what an entrypoint runs. Nothing is carried over in memory, so
anything that reads back did so out of the database.

The one thing deliberately *not* asserted to survive is a lexical index: it is
process-local by design, rebuilt from the store on first use, and the test
below checks it is rebuilt rather than that it persisted.
"""

from __future__ import annotations

import io
import os
import time
from datetime import datetime

import pytest
from api_v1_doubles import (
    DOCUMENT, OTHER_DOCUMENT, PATIENCE_SECONDS, CitingLLM, DeterministicEmbedding,
)
from fastapi.testclient import TestClient

from chat_rag.application import documents as document_use_case
from chat_rag.application.services import build_services
from chat_rag.components.ingest.journal import JobJournal
from chat_rag.components.viewer import analysis
from chat_rag.components.viewer import methods as M
from chat_rag.config import paths
from interfaces.http.v1 import PREFIX, create_app
from runtime import bootstrap

V1 = PREFIX


class Deployment:
    """One process's worth of application, startable and stoppable.

    Two of these in one test are two runs of the same deployment over one
    database -- which is what a restart is, and the only part of it a test can
    honestly leave out is the interpreter exiting.
    """

    def __init__(self, tmp_path):
        self.tmp_path = tmp_path
        self.services = None
        self.client = None
        self.llm = None

    def start(self, *, resume: bool = True) -> TestClient:
        self.llm = CitingLLM()
        self.services = build_services()
        if resume:
            # What both entrypoints run before serving: pick up the packager's
            # unfinished work, settle the previous process's jobs against the
            # ledger, sweep what it staged.
            bootstrap.resume_background_work(self.services)
        self.client = TestClient(create_app(self.services))
        self.client.__enter__()
        return self.client

    def stop(self) -> None:
        if self.client is not None:
            self.client.__exit__(None, None, None)
        if self.services is not None:
            self.services.ingest_jobs.close(timeout=PATIENCE_SECONDS)
            self.services.pipeline_cache.clear()
        self.client = self.services = None

    def restart(self) -> TestClient:
        self.stop()
        return self.start()


@pytest.fixture
def deployment(tmp_path, monkeypatch):
    """A deployment on the test database, with its two providers replaced."""
    from chat_rag.pipeline.rag_pipeline import RAGPipeline

    monkeypatch.setenv("RETRIEVAL_PROFILE", "hybrid_rrf")
    monkeypatch.setattr(RAGPipeline, "_create_llm", lambda self: CitingLLM())
    monkeypatch.setattr(RAGPipeline, "_create_embedding",
                        lambda self: DeterministicEmbedding())

    analysis._queue.join()
    with analysis._lock:
        analysis._inflight.clear()
    # The packager's directory outlives the "process" the way a real one does.
    monkeypatch.setattr(analysis, "root", lambda: tmp_path / "viewer-live")

    running = Deployment(tmp_path)
    running.start(resume=False)
    yield running
    running.stop()
    analysis._queue.join()
    with analysis._lock:
        analysis._inflight.clear()


def _ok(response):
    assert response.status_code == 200, response.text
    return response.json()


def _knowledge_base(client, name):
    created = client.post(f"{V1}/knowledge-bases", json={
        "name": name, "chunker": {"type": "structure_first"}})
    assert created.status_code == 201, created.text
    return created.json()["id"]


def _ingest(client, kb_id, *, name="rapor.md", text=DOCUMENT):
    response = client.post(
        f"{V1}/documents",
        files={"file": (name, io.BytesIO(text.encode("utf-8")), "text/markdown")},
        data={"knowledge_base_id": kb_id, "methods": [M.STANDARD]})
    assert response.status_code == 202, response.text
    job_id = response.json()["id"]
    deadline = time.monotonic() + PATIENCE_SECONDS
    while time.monotonic() < deadline:
        job = _ok(client.get(f"{V1}/ingest-jobs/{job_id}"))
        if job["status"] not in ("queued", "running"):
            assert job["status"] == "succeeded", job
            return job
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} never settled")


def _ready_analysis(client, document_id):
    deadline = time.monotonic() + PATIENCE_SECONDS
    while time.monotonic() < deadline:
        state = _ok(client.get(f"{V1}/documents/{document_id}/analysis"))
        if state["status"] in ("ready", "failed"):
            return state
        time.sleep(0.05)
    raise AssertionError(f"the analysis of {document_id} never settled")


# ================================================================= the records
def test_knowledge_bases_and_their_documents_read_back_after_a_restart(deployment):
    client = deployment.client
    first = _knowledge_base(client, "Yillik raporlar")
    second = _knowledge_base(client, "Surdurulebilirlik")
    first_job = _ingest(client, first)
    second_job = _ingest(client, second, name="cevre.md", text=OTHER_DOCUMENT)
    # A document carries its analysis block, and the packager is still working
    # on it when the job settles: compare two settled pictures, not a settled
    # one against a moving one.
    for job in (first_job, second_job):
        _ready_analysis(client, job["result"]["document_id"])

    before = _ok(client.get(f"{V1}/knowledge-bases"))
    documents_before = _ok(client.get(f"{V1}/documents"))

    client = deployment.restart()

    assert _ok(client.get(f"{V1}/knowledge-bases")) == before
    assert _ok(client.get(f"{V1}/documents")) == documents_before

    for job in (first_job, second_job):
        document = _ok(client.get(f"{V1}/documents/{job['result']['document_id']}"))
        assert document["content_id"] == job["content_id"]
        assert document["ingest_job_id"] == job["id"]


def test_the_stored_chunks_and_their_vectors_survive_a_restart(deployment):
    client = deployment.client
    kb_id = _knowledge_base(client, "Yillik raporlar")
    document_id = _ingest(client, kb_id)["result"]["document_id"]

    chunks_before = _ok(client.get(f"{V1}/documents/{document_id}/chunks"))
    index_before = _ok(client.get(f"{V1}/knowledge-bases/{kb_id}/embedding-index"))
    assert index_before["state"] == "compatible"

    client = deployment.restart()

    assert _ok(client.get(f"{V1}/documents/{document_id}/chunks")) == chunks_before

    # The manifest is a stored fact, not a process's memory of one: the same
    # embedding still matches the vectors that were written last time.
    index_after = _ok(client.get(f"{V1}/knowledge-bases/{kb_id}/embedding-index"))
    assert index_after["state"] == "compatible"
    assert index_after["stored"]["fingerprint"] == index_before["stored"]["fingerprint"]
    assert index_after["stored"]["chunk_count"] == index_before["stored"]["chunk_count"]


def test_retrieval_works_on_a_process_that_indexed_nothing_itself(deployment):
    """The lexical index is process-local by design. What has to survive is
    not the index -- it is the store it is rebuilt from, on first use, by a
    process that has ingested nothing."""
    client = deployment.client
    kb_id = _knowledge_base(client, "Yillik raporlar")
    document_id = _ingest(client, kb_id)["result"]["document_id"]

    client = deployment.restart()

    for method in ("bm25", "vector", "hybrid"):
        found = _ok(client.post(f"{V1}/searches", json={
            "query": "takipteki alacaklar", "knowledge_base_id": kb_id,
            "method": method}))
        assert found["items"], f"{method} found nothing after a restart"
        assert all(item["document_id"] == document_id for item in found["items"])

    answer = _ok(client.post(f"{V1}/queries", json={
        "question": "Takipteki alacaklar nasil degisti?", "knowledge_base_id": kb_id}))
    assert answer["citations"] and answer["grounded"] is True


# ================================================================ the analysis
def test_a_finished_analysis_and_its_rows_survive_a_restart(deployment):
    client = deployment.client
    kb_id = _knowledge_base(client, "Yillik raporlar")
    document_id = _ingest(client, kb_id)["result"]["document_id"]
    before = _ready_analysis(client, document_id)
    rows_before = _ok(client.get(
        f"{V1}/documents/{document_id}/analysis/methods/{M.STANDARD}/chunks"))

    client = deployment.restart()

    after = _ok(client.get(f"{V1}/documents/{document_id}/analysis"))
    assert after["status"] == "ready"
    assert after["content_id"] == before["content_id"]
    assert after["selected_methods"] == before["selected_methods"]
    assert after["ready_methods"] == before["ready_methods"]
    assert _ok(client.get(
        f"{V1}/documents/{document_id}/analysis/methods/{M.STANDARD}/chunks")
    ) == rows_before


def test_a_variant_can_be_added_to_a_document_ingested_by_a_previous_process(deployment):
    """The canonical is on disk and the content is in the database, so the
    file is neither re-uploaded nor re-parsed to gain a method."""
    client = deployment.client
    kb_id = _knowledge_base(client, "Yillik raporlar")
    document_id = _ingest(client, kb_id)["result"]["document_id"]
    _ready_analysis(client, document_id)

    client = deployment.restart()

    added = client.post(f"{V1}/documents/{document_id}/analysis/methods",
                        json={"methods": [M.MARKDOWN]})
    assert added.status_code == 202, added.text
    state = _ready_analysis(client, document_id)
    assert set(state["ready_methods"]) == {M.STANDARD, M.MARKDOWN}


# ==================================================================== the jobs
def test_a_job_that_finished_before_the_restart_still_answers_truthfully(deployment):
    """A job id outlives the process that minted it: the journal records every
    transition and start-up settles it against the ledger."""
    client = deployment.client
    kb_id = _knowledge_base(client, "Yillik raporlar")
    job = _ingest(client, kb_id)

    client = deployment.restart()

    after = _ok(client.get(f"{V1}/ingest-jobs/{job['id']}"))
    assert after["status"] == "succeeded"
    assert after["document_id"] == job["result"]["document_id"]
    assert after["knowledge_base_id"] == kb_id

    # And in the list, not only by id: a client polling the collection is the
    # one most likely to be asking about a job a restart settled.
    listed = _ok(client.get(f"{V1}/ingest-jobs"))
    assert [row["id"] for row in listed["items"]] == [job["id"]]
    assert _ok(client.get(f"{V1}/ingest-jobs",
                          params={"active": "true"}))["page"]["total"] == 0


def test_a_job_the_restart_interrupted_is_settled_and_not_reported_as_running(deployment):
    """The half-written case: a journal that still says ``running`` when the
    process comes back. The ledger decides -- no document carrying the job's
    id means nothing was committed, and the client is told so rather than
    left polling a job no worker will ever pick up."""
    client = deployment.client
    kb_id = _knowledge_base(client, "Yillik raporlar")

    # What the dead process left behind: a record that still says "running"
    # and no ledger row to go with it. Written through the journal itself,
    # because the journal is the whole of what a restart has to read.
    JobJournal(paths.ingest_journal()).record({
        "job_id": "yarimkalan01", "status": "running", "kb_id": kb_id,
        "filename": "yarim.md", "chunking_mode": "standard",
        "methods": [M.STANDARD], "content_sha256": "c0ffee" * 8,
        "submitted_at": datetime.now().isoformat(timespec="seconds"),
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "attached_uploads": 0, "journalled_at": time.time(),
    })

    client = deployment.restart()

    settled = _ok(client.get(f"{V1}/ingest-jobs/yarimkalan01"))
    assert settled["status"] not in ("queued", "running"), settled
    assert settled["restart_settled"] is True
    assert settled["document_id"] is None
    assert _ok(client.get(f"{V1}/documents"))["page"]["total"] == 0


def test_nothing_a_previous_process_staged_is_left_behind(deployment):
    """An upload staged by a process that died belongs to no job, and the
    sweep at start-up is what stops it accumulating on disk for ever."""
    staging = paths.upload_staging()
    orphan = os.path.join(staging, "upload_deadbeef.md")
    os.makedirs(staging, exist_ok=True)
    with open(orphan, "w", encoding="utf-8") as handle:
        handle.write(DOCUMENT)

    deployment.restart()

    assert not os.path.exists(orphan)


# ============================================================ nothing is lost
def test_a_restart_settles_the_journal_without_re_running_any_work(deployment):
    """Recovery reports; it does not re-ingest. A second restart finds the
    same answers and produces no second document."""
    client = deployment.client
    kb_id = _knowledge_base(client, "Yillik raporlar")
    job = _ingest(client, kb_id)

    for _ in range(2):
        client = deployment.restart()
        assert _ok(client.get(f"{V1}/documents"))["page"]["total"] == 1
        assert _ok(client.get(f"{V1}/knowledge-bases/{kb_id}/chunks"))["page"]["total"] == \
            job["result"]["chunk_count"]

    assert document_use_case.of_ingest_job(
        deployment.services, job["id"])["doc_id"] == job["result"]["document_id"]
