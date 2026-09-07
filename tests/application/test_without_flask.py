"""The product's behaviour, driven with no web framework anywhere near it.

This is the evidence for the boundary rather than a second copy of the API
tests. Those prove the Flask adapter still maps correctly; these prove there
is something underneath it worth adapting -- that a FastAPI router, a queue
consumer or a script could call the same use cases and get the same answers.

The rule every test here follows: **no Flask**. No test client, no request
context, no ``flask.request``, no application object. A container is composed
with :func:`application.services.build_services`-shaped doubles, a use case is
called with ordinary arguments, and what comes back is an ordinary dict or an
:mod:`application.errors` exception. ``test_the_boundary_holds`` states that
rule as an assertion, so a use case that starts importing Flask fails here
before it fails a review.

Five flows, chosen because they are the ones a migration would break:

    knowledge base lifecycle   create, rename, delete, and the store handles
    upload and ingest          submission's refusals, and a job end to end
    query                      the answer, and every refusal it can give
    the Viewer read model      selected vs ready, over a real packager
    cross-store deletion       what a deleted document takes with it
"""

from __future__ import annotations

import ast
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from application import (
    chunks, documents, goldsets, ingest, knowledge_bases, ops, query, workspace,
)
from application.errors import InvalidRequest, NotFound, NotReady, Unavailable
from application.services import Services
from components.goldset import GoldSetManager
from components.ingest import IngestManager, PipelineCache
from components.knowledgebase.manager import KnowledgeBaseManager
from components.query import QueryAdmission
from components.viewer import analysis
from components.viewer import methods as M
from config import Settings
from config.ingest import IngestLimits
from core.exceptions import IngestOverloaded, LLMException, QueryOverloaded
from core.models import DocumentChunk, RetrievalResult
from utils.document_tracker import DocumentTracker

REPO = Path(__file__).resolve().parents[2]


# --------------------------------------------------------------- the doubles
class Store:
    """A vector store that records what it was asked to do."""

    def __init__(self):
        self.deleted: list[str] = []
        self.rows: list[dict] = []

    def delete_by_doc_id(self, doc_id):
        self.deleted.append(doc_id)

    def get_all_chunks(self):
        return []

    def get_chunks_paginated(self, offset=0, limit=20, filter_dict=None):
        return {'chunks': self.rows, 'total': len(self.rows),
                'offset': offset, 'limit': limit}

    def get_name(self):
        return "StubStore"

    def close(self):
        pass


class Retriever:
    requires_document_embeddings = False

    def keyword_search(self, text, top_k=10, **kwargs):
        return [RetrievalResult(
            chunk=DocumentChunk(chunk_id="c1", content="parca", doc_id="d1",
                                doc_title="Rapor", chunk_index=0, total_chunks=1,
                                metadata={"doc_id": "d1"}),
            score=0.9, retrieval_method="bm25", rank=0,
        )]

    def hybrid_search(self, text, top_k=5, *a, **k):
        return self.keyword_search(text, top_k)


class Chunker:
    last_canonical_units = None
    last_deep_result = None

    def get_name(self):
        return "StubChunker"


class Pipeline:
    """Enough of a RAGPipeline for the flows below, and nothing more."""

    def __init__(self, *, answer=None, store=None, ingest_error=None):
        self.settings = SimpleNamespace(embedding_model_name="test/embedding")
        self.vector_db = store or Store()
        self.hybrid_retriever = Retriever()
        self.chunker = Chunker()
        self.last_parse_seconds = 0.5
        self.last_deep_analysis_report = None
        self._answer = answer
        self._ingest_error = ingest_error

    def query(self, question, top_k=5, temperature=0.3, max_tokens=500):
        if isinstance(self._answer, Exception):
            raise self._answer
        return {'answer': self._answer or "cevap",
                'sources': [{'label': 'S1', 'doc_id': 'd1'}],
                'metadata': {'retrieval_method': 'bm25'}}

    def ingest_document_from_file(self, file_path, doc_title, deep_analysis=False):
        if self._ingest_error is not None:
            raise self._ingest_error
        return [DocumentChunk(chunk_id="c1", content="parca", doc_id="doc-live",
                              doc_title=doc_title, chunk_index=0, total_chunks=1,
                              metadata={"doc_id": "doc-live"})]


def build(tmp_path, *, pipeline=None, workers=1, queue=4) -> Services:
    """A container composed the way ``build_services`` composes one, with this
    test's own stores. No Flask, no environment, no network."""
    settings = Settings()
    settings.query_timeout = 5.0
    made = pipeline or Pipeline()

    services = Services(
        settings=settings,
        kb_manager=KnowledgeBaseManager(str(tmp_path / "kbs.json")),
        gold_manager=GoldSetManager(str(tmp_path / "gold.json")),
        pipeline_cache=PipelineCache(build=lambda session, kb: made, max_entries=4),
        query_admission=QueryAdmission(2),
        documents=lambda: DocumentTracker(str(tmp_path / "ledger.json")),
        default_pipeline=made,
        sync_waiters=threading.BoundedSemaphore(1),
    )
    services.ingest_jobs = IngestManager(
        IngestLimits(workers=workers, queue_capacity=queue, job_timeout_seconds=60),
        execute=lambda job: ingest.execute_job(services, job),
    )
    return services


@pytest.fixture
def container(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(analysis, "root", lambda: tmp_path / "viewer-live")
    monkeypatch.setattr(analysis, "enqueue", lambda key: key)
    services = build(tmp_path)
    yield services
    services.ingest_jobs.close(timeout=20)


# ------------------------------------------------------- the boundary itself
def test_the_boundary_holds():
    """No use case may import a web framework.

    Stated as a test because it is the whole claim: everything else here would
    still pass if ``application`` quietly grew a ``from flask import request``
    inside one function.
    """
    offenders = []
    for path in sorted((REPO / "application").rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            if any(n.split(".")[0] in {"flask", "flask_cors", "werkzeug"} for n in names):
                offenders.append(f"{path.name}:{node.lineno}")
    assert offenders == [], f"application code imports a web framework: {offenders}"


def test_the_cli_composes_the_same_application_without_loading_flask(tmp_path):
    """The boundary, proved by a second caller rather than by a rule.

    ``python -m cli`` used to reach its knowledge bases and pipelines through
    ``import app``, so an evaluation run loaded a web framework to read a
    ledger. It now calls ``default_services()``; the same container, the same
    pipeline cache, the same seams -- and, checked here in a real interpreter,
    no Flask in ``sys.modules`` at all.
    """
    import os
    import subprocess
    import sys

    # Its own data root, and the repository only on the path: composing the
    # container writes a store, and it must not be the developer's.
    environment = dict(os.environ, PYTHONPATH=str(REPO),
                       CHAT_RAG_DATA_DIR=str(tmp_path / "state"))
    finished = subprocess.run(
        [sys.executable, "-c",
         "import cli.runtime, sys;"
         "print('flask' in sys.modules or 'flask_cors' in sys.modules);"
         "print(cli.runtime.services.kb_manager is not None)"],
        cwd=tmp_path, env=environment, capture_output=True, text=True, timeout=300,
    )
    assert finished.returncode == 0, finished.stderr[-2000:]
    loaded, composed = finished.stdout.split()[-2:]
    assert loaded == "False", "the CLI loaded a web framework"
    assert composed == "True", "the CLI did not compose the application"


def test_the_adapter_is_the_only_place_that_knows_a_status_code():
    """The other half: no use case names an HTTP status."""
    import application.errors as errors

    assert not hasattr(errors.InvalidRequest, "status_code")
    source = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted((REPO / "application").rglob("*.py"))
    )
    for code in ("jsonify", "make_response", "Response("):
        assert code not in source, f"{code} reached the application layer"


# --------------------------------------------------- knowledge base lifecycle
def test_a_knowledge_base_can_be_created_renamed_and_deleted_without_flask(container):
    created = knowledge_bases.create(container, {"name": "Yillik raporlar",
                                                 "chunker": {"type": "structure_first"}})
    kb_id = created["kb_id"]

    assert [kb["kb_id"] for kb in knowledge_bases.list_all(container)] == [kb_id]
    assert knowledge_bases.get(container, kb_id)["name"] == "Yillik raporlar"

    renamed = knowledge_bases.update(container, kb_id, {"name": "Raporlar"})
    assert renamed["name"] == "Raporlar"

    assert knowledge_bases.delete(container, kb_id)["deleted"] is True
    # A real deletion frees the name again.
    assert knowledge_bases.create(container, {"name": "Raporlar"})["kb_id"] != kb_id


def test_the_refusals_a_knowledge_base_makes_are_application_refusals(container):
    with pytest.raises(NotFound):
        knowledge_bases.get(container, "no-such-kb")
    with pytest.raises(NotFound):
        knowledge_bases.delete(container, "no-such-kb")

    first = knowledge_bases.create(container, {"name": "A"})["kb_id"]
    knowledge_bases.create(container, {"name": "B"})
    with pytest.raises(InvalidRequest, match="already exists"):
        knowledge_bases.update(container, first, {"name": "B"})
    with pytest.raises(InvalidRequest, match="Nothing to update"):
        knowledge_bases.update(container, first, {})
    with pytest.raises(InvalidRequest):
        knowledge_bases.create(container, {"name": "C", "chunker": {"type": "invented"}})


def test_deleting_a_knowledge_base_closes_its_store_handles_first(container):
    """The order is the reason this is a use case: a live Chroma client keeps
    the store's sqlite file open, and on Windows that is what makes the
    directory undeletable."""
    kb_id = knowledge_bases.create(container, {"name": "A"})["kb_id"]
    container.get_pipeline("someone", kb_id)
    assert container.pipeline_cache.snapshot()["size"] == 1

    knowledge_bases.delete(container, kb_id)
    assert container.pipeline_cache.snapshot()["size"] == 0


# ------------------------------------------------------------ upload + ingest
def _upload(tmp_path, name="rapor.pdf", body=b"%PDF-1.4 rapor"):
    source = tmp_path / ("src-" + name)
    source.write_bytes(body)

    def save(target):
        Path(target).write_bytes(source.read_bytes())

    return ingest.Upload(filename=name, save=save)


def test_an_upload_is_validated_before_anything_is_written(container, tmp_path):
    with pytest.raises(InvalidRequest, match="Knowledge base selection"):
        ingest.submit(container, upload=_upload(tmp_path), kb_id="", session_id="s")
    with pytest.raises(NotFound, match="not found"):
        ingest.submit(container, upload=_upload(tmp_path), kb_id="ghost", session_id="s")


def test_deep_analysis_is_refused_when_the_chunker_has_no_path_for_it(container, tmp_path):
    """Refused explicitly, and carrying the flag the console branches on --
    never a silent fall back to Standard."""
    kb_id = knowledge_bases.create(container, {"name": "A"})["kb_id"]
    with pytest.raises(InvalidRequest) as refusal:
        ingest.submit(container, upload=_upload(tmp_path), kb_id=kb_id, session_id="s",
                      methods=[M.STANDARD, M.DEEP])
    assert refusal.value.details == {"deep_analysis_unavailable": True}


def test_one_upload_runs_a_whole_ingest_job_and_registers_the_document(container, tmp_path):
    """The flow a FastAPI router would call: submit, wait, read the outcome."""
    kb_id = knowledge_bases.create(container, {"name": "A"})["kb_id"]

    accepted = ingest.submit(container, upload=_upload(tmp_path), kb_id=kb_id,
                             session_id="s")
    ingest.await_settlement(container, accepted)
    settled = ingest.outcome(container, accepted)

    assert settled.kind == ingest.SUCCEEDED, settled.error
    assert settled.result["chunks_created"] == 1
    assert settled.result["doc_id"] == "doc-live"

    # The ledger write is the job's last act, so a registered document is a
    # committed one -- which is what makes restart settlement truthful.
    rows = documents.list_all(container, kb_id)
    assert [row["doc_id"] for row in rows] == ["doc-live"]
    assert documents.of_ingest_job(container, accepted.job.job_id)["doc_id"] == "doc-live"


def test_a_full_queue_refuses_the_upload_rather_than_queueing_it(tmp_path, monkeypatch):
    """Backpressure is application behaviour, not a Flask concern: the refusal
    carries its own retry estimate and nothing is kept."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(analysis, "root", lambda: tmp_path / "viewer-live")
    monkeypatch.setattr(analysis, "enqueue", lambda key: key)

    blocked = threading.Event()

    class Slow(Pipeline):
        def ingest_document_from_file(self, file_path, doc_title, deep_analysis=False):
            blocked.wait(5)
            return super().ingest_document_from_file(file_path, doc_title, deep_analysis)

    container = build(tmp_path, pipeline=Slow(), workers=1, queue=1)
    try:
        kb_id = knowledge_bases.create(container, {"name": "A"})["kb_id"]
        # Distinct bytes: the same content submitted twice is deliberately
        # *attached* to the job already carrying it rather than queued again,
        # so identical uploads would never fill the queue.
        for index in range(2):
            ingest.submit(container, upload=_upload(tmp_path, f"r{index}.pdf",
                                                   b"%PDF rapor " + bytes([index])),
                          kb_id=kb_id, session_id="s")
        with pytest.raises(IngestOverloaded) as full:
            ingest.submit(container, upload=_upload(tmp_path, "r9.pdf", b"%PDF rapor 9"),
                          kb_id=kb_id, session_id="s")
        assert full.value.retry_after_seconds > 0
    finally:
        blocked.set()
        container.ingest_jobs.close(timeout=20)


def test_a_failed_job_says_which_kind_of_failure_it_was(tmp_path, monkeypatch):
    """A store that holds another embedding model's vectors is a *conflict*,
    not a fault: the outcome names it so the adapter can answer 409 and the
    console can offer a re-index."""
    from core.exceptions import IndexIncompatibleException

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(analysis, "root", lambda: tmp_path / "viewer-live")
    monkeypatch.setattr(analysis, "enqueue", lambda key: key)
    broken = Pipeline(ingest_error=IndexIncompatibleException("another model"))
    container = build(tmp_path, pipeline=broken)
    try:
        kb_id = knowledge_bases.create(container, {"name": "A"})["kb_id"]
        accepted = ingest.submit(container, upload=_upload(tmp_path), kb_id=kb_id,
                                 session_id="s")
        ingest.await_settlement(container, accepted)
        assert ingest.outcome(container, accepted).kind == ingest.REINDEX_REQUIRED
    finally:
        container.ingest_jobs.close(timeout=20)


# -------------------------------------------------------------------- query
def test_a_question_is_answered_with_its_sources_and_its_timings(container):
    kb_id = knowledge_bases.create(container, {"name": "A"})["kb_id"]
    answered = query.answer(container, question="ornitorenk tarifesi",
                            session_id="s", kb_id=kb_id)

    assert answered["answer"] == "cevap"
    assert answered["sources"] == [{'label': 'S1', 'doc_id': 'd1'}]
    # The measurement the console shows, produced by the scope rather than by
    # any route: existing metadata is untouched and ``query`` is added.
    assert answered["metadata"]["retrieval_method"] == "bm25"
    assert set(answered["metadata"]["query"]["stages"]) >= set()
    assert answered["metadata"]["query"]["query_id"]
    assert container.query_admission.active == 0, "the slot came back"


def test_an_empty_question_is_refused_before_it_takes_a_slot(container):
    with pytest.raises(InvalidRequest, match="Question is required"):
        query.answer(container, question="   ", session_id="s")
    assert container.query_admission.snapshot()["accepted_total"] == 0


def test_a_full_admission_counter_refuses_the_question_at_once(container):
    assert container.query_admission.try_enter()
    assert container.query_admission.try_enter()
    try:
        with pytest.raises(QueryOverloaded) as full:
            query.answer(container, question="soru", session_id="s")
    finally:
        container.query_admission.leave()
        container.query_admission.leave()
    assert full.value.reason == "admission"


def test_a_missing_answer_model_is_an_unavailable_capability_not_a_fault(tmp_path, monkeypatch):
    """Retrieval and ingestion need no language model, so a missing one is a
    feature that is unavailable now -- and the flag says which feature."""
    monkeypatch.chdir(tmp_path)
    container = build(tmp_path, pipeline=Pipeline(answer=LLMException("no provider")))
    try:
        with pytest.raises(Unavailable) as missing:
            query.answer(container, question="soru", session_id="s")
        assert missing.value.details == {"generation_unavailable": True}
        assert container.query_admission.active == 0
    finally:
        container.ingest_jobs.close(timeout=20)


def test_a_lab_search_runs_under_the_same_bound_a_question_does(container):
    """The bound is on the *work*, not on one route: with every slot held, a
    read-only search is refused exactly as a question is."""
    kb_id = knowledge_bases.create(container, {"name": "A"})["kb_id"]
    found = chunks.search_bm25(container, query="parca", kb_id=kb_id, session_id="s")
    assert [row["chunk_id"] for row in found["chunks"]] == ["c1"]
    assert found["search_metadata"]["search_method"] == "bm25"

    assert container.query_admission.try_enter() and container.query_admission.try_enter()
    try:
        with pytest.raises(QueryOverloaded):
            chunks.search_bm25(container, query="parca", kb_id=kb_id, session_id="s")
    finally:
        container.query_admission.leave()
        container.query_admission.leave()


def test_a_search_refusal_is_not_counted_as_a_failed_query(container):
    """An invalid search leaves the bound cleanly. Counting it as a failure
    would make the metrics lie in the direction that matters."""
    from components.observability import telemetry as T

    kb_id = knowledge_bases.create(container, {"name": "A"})["kb_id"]
    with pytest.raises(InvalidRequest, match="Query is required"):
        chunks.search_bm25(container, query="  ", kb_id=kb_id, session_id="s")
    recent = T.metrics().snapshot(recent=1)["queries"]["recent"]
    assert recent and recent[0]["status"] == "succeeded"


# ------------------------------------------------------- the Viewer read model
def _units(count=3):
    return [{"document_id": "shared", "unit_id": f"u{i}", "order": i,
             "text": f"Paragraf {i}.", "type": "paragraph",
             "section_path": [], "source": {"page": 1, "block": i}}
            for i in range(count)]


def test_the_workspace_read_model_is_built_without_a_request(container, tmp_path):
    """The Viewer's panel is a read model over three stores, and building it
    needs no host header, no session and no request -- ``console_url`` is the
    one thing HTTP knows, and it is a parameter."""
    kb_id = knowledge_bases.create(container, {"name": "Yillik"})["kb_id"]
    path = tmp_path / "rapor.pdf"
    path.write_text("pdf", encoding="utf-8")
    container.documents().mark_as_ingested(file_path=str(path), doc_id="doc-1",
                                           chunk_count=3, kb_id=kb_id)
    workspace.stage_analysis("doc-1", label="rapor.pdf", units=_units(),
                             methods=[M.STANDARD, M.MARKDOWN], kb_id=kb_id,
                             content_sha="a" * 48)

    snapshot = workspace.snapshot(container, console_url="http://console:5005")
    assert snapshot["console_url"] == "http://console:5005"
    assert snapshot["totals"]["documents"] == 1
    base = snapshot["knowledge_bases"][0]
    assert base["kb_id"] == kb_id and base["document_count"] == 1

    viewer = base["documents"][0]["viewer"]
    # visible = selected ∩ ready. Nothing is built, so the upload may be shown
    # nothing -- while its selection is still recorded in full.
    assert sorted(viewer["requested"]) == sorted([M.STANDARD, M.MARKDOWN])
    assert viewer["ready_methods"] == []


def test_an_upload_is_only_ever_offered_the_methods_it_chose(container, tmp_path):
    """Two uploads of one PDF share a content and keep separate selections."""
    kb_id = knowledge_bases.create(container, {"name": "A"})["kb_id"]
    sha = "c0ffee" * 8
    workspace.stage_analysis("doc-1", label="one.pdf", units=_units(),
                             methods=[M.STANDARD], kb_id=kb_id, content_sha=sha)
    workspace.stage_analysis("doc-2", label="two.pdf", units=_units(),
                             methods=[M.MARKDOWN], kb_id=kb_id, content_sha=sha)

    first = workspace.analysis_state("doc-1")
    second = workspace.analysis_state("doc-2")
    assert first["key"] == second["key"], "one content"
    assert first["selected_methods"] == [M.STANDARD]
    assert second["selected_methods"] == [M.MARKDOWN]

    # Nothing is packaged, so asking for rows is *not ready* rather than
    # *not here*, and the answer carries the state it got to.
    with pytest.raises(NotReady) as pending:
        workspace.chunk_rows("doc-1")
    assert pending.value.state["status"]

    with pytest.raises(InvalidRequest, match="unknown chunking method"):
        workspace.chunk_rows("doc-1", "not-a-method")


# ------------------------------------------------------- cross-store deletion
def test_deleting_a_document_reaches_all_three_stores(container, tmp_path):
    """Vectors, ledger row and analysis. The rule a schema has to reproduce,
    stated where the schema migration will read it."""
    kb_id = knowledge_bases.create(container, {"name": "A"})["kb_id"]
    path = tmp_path / "one.pdf"
    path.write_text("pdf", encoding="utf-8")
    container.documents().mark_as_ingested(file_path=str(path), doc_id="doc-1",
                                           chunk_count=3, kb_id=kb_id)
    workspace.stage_analysis("doc-1", label="one.pdf", units=_units(),
                             methods=[M.STANDARD], kb_id=kb_id, content_sha="d" * 48)

    documents.delete(container, "doc-1")

    assert container.default_pipeline.vector_db.deleted == ["doc-1"]
    assert documents.list_all(container) == []
    assert workspace.analysis_state("doc-1")["status"] == analysis.STATUS_MISSING


# --------------------------------------------------------------- operational
def test_the_service_state_is_readable_without_a_request(container):
    state, ready, reasons = ops.service_state(container)
    assert state == "ok" and ready is True and reasons == []

    # Saturating the query counter is an *overloaded* service, not a broken
    # one: it still answers, and says why.
    container.query_admission.try_enter()
    container.query_admission.try_enter()
    try:
        state, ready, reasons = ops.service_state(container)
    finally:
        container.query_admission.leave()
        container.query_admission.leave()
    assert state == "overloaded" and ready is True
    assert "every query slot is in use" in reasons[0]

    health = ops.health(container)
    assert health["status"] == "healthy" and "ingest" in health and "query" in health


def test_a_gold_set_entry_takes_the_documents_recorded_hash(container, tmp_path):
    """The application carries the sha the ingest already recorded, rather
    than hashing the file again -- an entry that knows which bytes it was
    confirmed against can warn when the corpus is replaced."""
    kb_id = knowledge_bases.create(container, {"name": "A"})["kb_id"]
    path = tmp_path / "one.pdf"
    path.write_text("pdf", encoding="utf-8")
    container.documents().mark_as_ingested(file_path=str(path), doc_id="doc-1",
                                           chunk_count=1, kb_id=kb_id)
    recorded = container.documents().get_document_by_doc_id("doc-1")["file_hash"]

    entry = goldsets.upsert(container, {"kb_id": kb_id, "question": "soru?",
                                        "document_id": "doc-1",
                                        "correct_chunk_id": "c1"})
    assert entry["document_sha256"] == recorded
    assert len(goldsets.list_all(container, kb_id)) == 1

    goldsets.delete(container, entry["entry_id"])
    with pytest.raises(NotFound):
        goldsets.delete(container, entry["entry_id"])
