"""`/api/v1`, driven as the contract it is.

This file lives in the migration suite on purpose. What it asserts is what the
FastAPI implementation will have to satisfy unchanged: resource shapes, field
names, identities, status codes and the refusal taxonomy. Nothing here reaches
into Flask beyond the test client that stands in for a transport, and nothing
here asserts a message, a float, an ordering the product does not promise, or
any part of the two fields the contract publishes as pass-through
(``metadata`` on a chunk, ``diagnostics`` on an answer).

The doubles are the same shape the application suite uses: a stub pipeline, a
real knowledge-base manager, a real ingest ledger and the real analysis
packager over a temporary directory. No provider is reached and no model is
loaded -- which is also the point of ``/api/v1/meta/*``: discovery has to work
on a machine that can run nothing.
"""

from __future__ import annotations

import io
import threading
from types import SimpleNamespace

import pytest

import app as flask_app
from application import ingest as app_ingest
from application import workspace as app_workspace
from components.ingest import IngestManager
from components.knowledgebase.manager import KnowledgeBaseManager
from components.viewer import analysis
from components.viewer import methods as M
from config.ingest import IngestLimits
from core.models import DocumentChunk, RetrievalResult
from utils.document_tracker import DocumentTracker

V1 = "/api/v1"


# ------------------------------------------------------------- the doubles
class Store:
    def __init__(self):
        self.rows: list[dict] = []
        self.deleted: list[str] = []

    def add(self, doc_id, count=3):
        for index in range(count):
            self.rows.append({
                "chunk_id": f"{doc_id}-c{index}",
                "content": f"{doc_id} parca {index}",
                "metadata": {"doc_id": doc_id, "chunk_index": index,
                             "total_chunks": count, "section_title": f"Bolum {index}",
                             "chunking_mode": "standard"},
            })

    def get_chunks_paginated(self, offset=0, limit=20, filter_dict=None):
        rows = [r for r in self.rows
                if not filter_dict or r["metadata"].get("doc_id") == filter_dict.get("doc_id")]
        return {"chunks": rows[offset:offset + limit], "total": len(rows),
                "offset": offset, "limit": limit}

    def search_chunks_by_text(self, search_text, offset=0, limit=20):
        rows = [r for r in self.rows if search_text.casefold() in r["content"].casefold()]
        return {"chunks": rows[offset:offset + limit], "total": len(rows),
                "offset": offset, "limit": limit}

    def delete_by_doc_id(self, doc_id):
        self.deleted.append(doc_id)
        self.rows = [r for r in self.rows if r["metadata"].get("doc_id") != doc_id]

    def get_all_chunks(self):
        return []

    def get_name(self):
        return "StubStore"

    def close(self):
        pass


class Retriever:
    requires_document_embeddings = False

    def _hits(self):
        return [RetrievalResult(
            chunk=DocumentChunk(chunk_id="c0", content="parca", doc_id="doc-live",
                                doc_title="Rapor.pdf", chunk_index=0, total_chunks=1,
                                metadata={"doc_id": "doc-live", "chunking_mode": "standard"}),
            score=0.9, retrieval_method="bm25", rank=0,
        )]

    def keyword_search(self, text, top_k=10, **kwargs):
        return self._hits()

    def hybrid_search(self, text, top_k=5, *a, **k):
        return self._hits()


class Chunker:
    last_canonical_units = None
    last_deep_result = None

    def get_name(self):
        return "StubChunker"


class Pipeline:
    def __init__(self, store):
        self.settings = SimpleNamespace(embedding_model_name="test/embedding")
        self.vector_db = store
        self.hybrid_retriever = Retriever()
        self.chunker = Chunker()
        self.last_parse_seconds = 0.4
        self.last_deep_analysis_report = None

    def query(self, question, top_k=5, temperature=0.3, max_tokens=500):
        return {
            "answer": "Takipteki alacaklar azaldi [S1].",
            "sources": [{
                "label": "S1", "chunk_id": "c0", "doc_id": "doc-live",
                "document": "Rapor.pdf", "heading": "Bolum 1", "pages": [3],
                "chunking_mode": "standard", "score": 0.9, "used": True,
                "content": "Takipteki alacaklar 2024 yilinda azaldi.",
            }],
            "metadata": {"retrieval_method": "bm25", "answer": {"grounded": True}},
        }

    def ingest_document_from_file(self, file_path, doc_title, deep_analysis=False):
        return [DocumentChunk(chunk_id="doc-live-c0", content="parca", doc_id="doc-live",
                              doc_title=doc_title, chunk_index=0, total_chunks=1,
                              metadata={"doc_id": "doc-live"})]

    def embedding_index_status(self):
        return {"compatible": True, "model": "test/embedding", "vectors": 3}


@pytest.fixture
def api(tmp_path, monkeypatch):
    """A console whose every store is this test's own, served over `/api/v1`."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(analysis, "root", lambda: tmp_path / "viewer-live")
    # No build: these tests are about the contract, not about producing a
    # variant. The packager has its own suite.
    monkeypatch.setattr(analysis, "enqueue", lambda key: key)

    store = Store()
    pipeline = Pipeline(store)
    ledger_file = str(tmp_path / "ledger.json")
    container = flask_app.services

    monkeypatch.setattr(container, "kb_manager", KnowledgeBaseManager(str(tmp_path / "kbs.json")))
    monkeypatch.setattr(container, "documents", lambda: DocumentTracker(ledger_file))
    monkeypatch.setattr(container, "get_pipeline", lambda *a, **k: pipeline)
    monkeypatch.setattr(container, "default_pipeline", pipeline)
    monkeypatch.setattr(container, "sync_waiters", threading.BoundedSemaphore(1))
    jobs = IngestManager(
        IngestLimits(workers=1, queue_capacity=4, job_timeout_seconds=60),
        execute=lambda job: app_ingest.execute_job(container, job),
    )
    monkeypatch.setattr(container, "ingest_jobs", jobs)

    flask_app.app.config.update(TESTING=True)
    with flask_app.app.test_client() as client:
        yield SimpleNamespace(client=client, store=store, container=container,
                              ledger=lambda: DocumentTracker(ledger_file), root=tmp_path)
    jobs.close(timeout=20)


def _json(response):
    body = response.get_json()
    assert isinstance(body, dict), response.data[:300]
    return body


def _error(response, kind: str):
    body = _json(response)
    assert "error" in body, body
    assert body["error"]["type"] == kind, body["error"]
    assert body["error"]["message"], "a refusal with no message explains nothing"
    return body["error"]


def _kb(api, name="Yillik raporlar", **extra):
    created = api.client.post(f"{V1}/knowledge-bases",
                              json={"name": name, "chunker": {"type": "structure_first"},
                                    **extra})
    assert created.status_code == 201, created.get_json()
    return _json(created)


def _units(count=3):
    return [{"document_id": "shared", "unit_id": f"u{i}", "order": i,
             "text": f"Paragraf {i}.", "type": "paragraph",
             "section_path": [], "source": {"page": 1, "block": i}}
            for i in range(count)]


def _ingested(api, doc_id, *, kb_id, name, sha="c0ffee" * 8, methods=(M.STANDARD,)):
    """One upload, as an ingest leaves it: a ledger row, a store and a record."""
    path = api.root / name
    path.write_text("pdf", encoding="utf-8")
    api.ledger().mark_as_ingested(file_path=str(path), doc_id=doc_id, chunk_count=3,
                                  kb_id=kb_id, metadata={"original_filename": name})
    api.store.add(doc_id)
    app_workspace.stage_analysis(doc_id, label=name, units=_units(), methods=list(methods),
                                 kb_id=kb_id, content_sha=sha)
    return doc_id


# =========================================================== the conventions
def test_a_collection_always_carries_its_page(api):
    """Both keys, on every list, whether or not everything fits. A client that
    has to branch on which kind of list it got will branch wrongly."""
    _kb(api)
    body = _json(api.client.get(f"{V1}/knowledge-bases"))
    assert set(body) >= {"items", "page"}
    assert set(body["page"]) == {"offset", "limit", "total"}
    assert body["page"]["total"] == 1 and len(body["items"]) == 1


def test_a_resource_is_the_object_itself_with_no_envelope(api):
    created = _kb(api)
    body = _json(api.client.get(f"{V1}/knowledge-bases/{created['id']}"))
    assert body["id"] == created["id"]
    assert "success" not in body and "data" not in body


def test_pagination_walks_a_collection_and_clamps_what_it_is_asked_for(api):
    from interfaces.http.v1.envelope import MAX_LIMIT

    for index in range(5):
        _kb(api, name=f"KB {index}")

    first = _json(api.client.get(f"{V1}/knowledge-bases?offset=0&limit=2"))
    second = _json(api.client.get(f"{V1}/knowledge-bases?offset=2&limit=2"))
    assert [k["id"] for k in first["items"]] != [k["id"] for k in second["items"]]
    assert first["page"]["total"] == second["page"]["total"] == 5

    # A ceiling, not a suggestion: without one, one request makes any list
    # endpoint as expensive as the caller likes.
    capped = _json(api.client.get(f"{V1}/knowledge-bases?limit=100000"))
    assert capped["page"]["limit"] == MAX_LIMIT
    # And nonsense is the default rather than a refusal: a page size is a
    # presentation decision, and a pasted link should not 400.
    assert _json(api.client.get(f"{V1}/knowledge-bases?limit=abc"))["page"]["limit"] > 0


def test_every_refusal_says_which_kind_it_is(api):
    """The taxonomy a client branches on. ``type`` is the contract; the
    message is not, and neither is the status if HTTP ever changes its mind."""
    assert api.client.get(f"{V1}/knowledge-bases/nope").status_code == 404
    _error(api.client.get(f"{V1}/knowledge-bases/nope"), "not_found")

    empty = api.client.post(f"{V1}/queries", json={"question": "   "})
    assert empty.status_code == 400
    _error(empty, "invalid_request")

    unknown = api.client.post(f"{V1}/searches", json={"query": "x", "method": "telepathy"})
    assert unknown.status_code == 400
    assert _error(unknown, "invalid_request")["details"]["supported"]

    missing_job = api.client.get(f"{V1}/ingest-jobs/never-existed")
    assert missing_job.status_code == 404
    _error(missing_job, "not_found")


# ================================================================= discovery
def test_chunking_methods_come_from_the_library_registry(api):
    """Discovery has to work on a machine that can run nothing, which is why
    an unavailable method is listed with its reason rather than hidden."""
    from amsc.chunking import registry

    body = _json(api.client.get(f"{V1}/meta/chunking-methods"))
    keys = [item["key"] for item in body["items"]]
    assert keys, "no chunking method was offered at all"
    assert set(keys) == set(registry.order()), (
        "the API and the library registry disagree about which methods exist"
    )
    for item in body["items"]:
        assert set(item) == {"key", "label", "summary", "engine", "available",
                             "unavailable_reason", "uses_model", "default",
                             "orchestration", "baseline"}
        assert isinstance(item["available"], bool)
        if not item["available"]:
            assert item["unavailable_reason"], "an unavailable method must say why"
    assert sum(1 for item in body["items"] if item["default"]) >= 1


def test_a_newly_registered_method_appears_with_no_edit_to_this_api(api):
    """The architectural claim, checked rather than asserted in prose.

    The library's own example method is registered for the length of this
    test. Nothing in ``interfaces/http/v1`` names it, and no catalogue in this
    repository is edited -- it appears because the registry is the only list.
    """
    from amsc.chunking import registry
    from amsc.chunking.example import FIXED_WINDOW

    before = [i["key"] for i in _json(api.client.get(f"{V1}/meta/chunking-methods"))["items"]]
    assert FIXED_WINDOW.key not in before

    registry.register(FIXED_WINDOW)
    try:
        body = _json(api.client.get(f"{V1}/meta/chunking-methods"))
        found = {item["key"]: item for item in body["items"]}
        assert FIXED_WINDOW.key in found, "a registered method did not reach the API"
        entry = found[FIXED_WINDOW.key]
        assert entry["label"] == FIXED_WINDOW.label
        assert entry["summary"] == FIXED_WINDOW.summary
        assert entry["engine"] == FIXED_WINDOW.kind

        # ...and an upload can ask for it by that key, which is the half that
        # makes discovery useful rather than decorative.
        kb = _kb(api, name="fifth")
        accepted = api.client.post(
            f"{V1}/documents",
            data={"knowledge_base_id": kb["id"], "file": (io.BytesIO(b"%PDF x"), "f.pdf"),
                  "methods": [M.STANDARD, FIXED_WINDOW.key]},
            content_type="multipart/form-data",
        )
        assert accepted.status_code == 202, accepted.get_json()
        assert FIXED_WINDOW.key in _json(accepted)["methods"]
    finally:
        registry.unregister(FIXED_WINDOW.key)

    after = [i["key"] for i in _json(api.client.get(f"{V1}/meta/chunking-methods"))["items"]]
    assert FIXED_WINDOW.key not in after, "unregistering did not take it away again"


def test_no_module_behind_this_api_keeps_a_method_catalogue_of_its_own():
    """The other half of the discovery claim: not "the registry is one source"
    but "there is no second one".

    A duplicated catalogue looks like a list of method keys written down
    together, so that is what is looked for -- two or more of them as literals
    in one file. A single key is not a catalogue and is sometimes legitimate
    (``"hybrid"`` is also the name of a *retrieval* method, which is a
    different thing that happens to share a word).

    The console's own form and its labels are held to the same rule from the
    other side by ``tests/unit/test_chunker_extension.py``.
    """
    import re
    from pathlib import Path

    from amsc.chunking import registry

    root = Path(__file__).resolve().parents[2]
    patterns = {key: re.compile(r'["\']' + re.escape(key) + r'["\']')
                for key in registry.order()}

    offenders = {}
    for tree in ("interfaces", "application"):
        for path in sorted((root / tree).rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            text = path.read_text(encoding="utf-8")
            named = sorted(key for key, pattern in patterns.items() if pattern.search(text))
            if len(named) > 1:
                offenders[path.relative_to(root).as_posix()] = named

    assert offenders == {}, (
        "these enumerate chunking methods instead of reading the registry: "
        + repr(offenders)
    )


def test_retrieval_methods_report_what_this_profile_can_actually_serve(api):
    kb = _kb(api)
    body = _json(api.client.get(f"{V1}/meta/retrieval-methods?knowledge_base_id={kb['id']}"))
    names = {item["name"] for item in body["items"]}
    assert {"vector", "bm25", "hybrid"} <= names
    for item in body["items"]:
        if not item["available"]:
            assert item["unavailable_reason"], item


def test_health_answers_without_a_model_a_store_or_a_provider(api):
    body = _json(api.client.get(f"{V1}/health"))
    assert body["state"] in {"ok", "overloaded", "degraded"}
    assert body["ready"] is True
    assert set(body["capacity"]) == {"ingest", "query"}


# ========================================================== knowledge bases
def test_a_knowledge_base_is_created_read_renamed_and_deleted(api):
    created = _kb(api)
    kb_id = created["id"]
    assert created["name"] == "Yillik raporlar"
    assert created["chunker"]["type"] == "structure_first"

    renamed = _json(api.client.patch(f"{V1}/knowledge-bases/{kb_id}", json={"name": "Raporlar"}))
    assert renamed["name"] == "Raporlar" and renamed["id"] == kb_id

    assert api.client.delete(f"{V1}/knowledge-bases/{kb_id}").status_code == 204
    assert api.client.get(f"{V1}/knowledge-bases/{kb_id}").status_code == 404


def test_a_knowledge_base_exposes_no_storage_location(api):
    """The pgvector migration replaces where the vectors live. A client that
    was told about it would have to be changed too."""
    created = _kb(api)
    body = _json(api.client.get(f"{V1}/knowledge-bases/{created['id']}"))
    assert set(body) == {"id", "name", "chunker", "retrieval_method",
                         "embedding_model", "extra"}
    serialised = repr(body)
    for leak in ("vector_db_path", "vector_db_provider", "chroma", str(api.root)):
        assert leak not in serialised, f"{leak!r} reached the wire"


def test_a_rejected_knowledge_base_creates_nothing(api):
    _kb(api, name="Taken")
    duplicate = api.client.post(f"{V1}/knowledge-bases", json={"name": "Taken"})
    assert duplicate.status_code == 400
    _error(duplicate, "invalid_request")
    assert _json(api.client.get(f"{V1}/knowledge-bases"))["page"]["total"] == 1


# ================================================================ documents
def test_a_document_names_its_upload_and_its_content_separately(api):
    """The two identities. One PDF ingested twice is two documents and one
    content, and a schema that conflates them either loses an upload or shares
    a choice that is not shared."""
    kb = _kb(api)["id"]
    _ingested(api, "doc-1", kb_id=kb, name="one.pdf")
    _ingested(api, "doc-2", kb_id=kb, name="two.pdf", methods=(M.MARKDOWN,))

    items = {d["id"]: d for d in _json(api.client.get(f"{V1}/documents"))["items"]}
    assert set(items) == {"doc-1", "doc-2"}
    assert items["doc-1"]["knowledge_base_id"] == kb
    assert items["doc-1"]["name"] == "one.pdf"
    # Same bytes staged for both, so one shared analysis...
    assert items["doc-1"]["analysis"]["content_id"] == items["doc-2"]["analysis"]["content_id"]
    # ...and two separate upload-level selections.
    assert items["doc-1"]["analysis"]["selected_methods"] == [M.STANDARD]
    assert items["doc-2"]["analysis"]["selected_methods"] == [M.MARKDOWN]


def test_a_document_exposes_no_file_path(api):
    """The ledger is keyed by an absolute path on the machine that ran the
    upload. It is the field PostgreSQL replaces with a row id, and no client
    has ever needed it."""
    kb = _kb(api)["id"]
    _ingested(api, "doc-1", kb_id=kb, name="one.pdf")
    body = _json(api.client.get(f"{V1}/documents/doc-1"))
    assert set(body) == {"id", "knowledge_base_id", "name", "content_id", "size_bytes",
                         "chunk_count", "chunking_mode", "status", "ingested_at",
                         "ingest_job_id", "analysis"}
    assert str(api.root) not in repr(body)
    assert "file_path" not in repr(body)


def test_an_uploads_analysis_shows_selected_and_ready_apart(api):
    """``visible = selected ∩ ready``, on the wire.

    Neither half alone is right: the selection alone promises a variant that
    does not exist, and what is ready alone shows another upload's variants.
    """
    kb = _kb(api)["id"]
    _ingested(api, "doc-1", kb_id=kb, name="one.pdf", methods=(M.STANDARD, M.MARKDOWN))

    body = _json(api.client.get(f"{V1}/documents/doc-1/analysis"))
    assert set(body["selected_methods"]) == {M.STANDARD, M.MARKDOWN}
    # Nothing was built, so this upload may be asked about nothing -- while its
    # selection is still recorded in full.
    assert body["ready_methods"] == []
    assert set(body["content"]) == {"requested_methods", "ready_methods",
                                    "shared_with_document_ids"}
    assert "doc-1" in body["content"]["shared_with_document_ids"]


def test_an_analysis_that_does_not_exist_is_a_state_not_a_404(api):
    """A document with no analysis is a fact about the document, and a client
    polling for a build should not have to read a 404 as progress."""
    kb = _kb(api)["id"]
    path = api.root / "bare.pdf"
    path.write_text("pdf", encoding="utf-8")
    api.ledger().mark_as_ingested(file_path=str(path), doc_id="doc-bare",
                                  chunk_count=0, kb_id=kb)

    response = api.client.get(f"{V1}/documents/doc-bare/analysis")
    assert response.status_code == 200
    assert _json(response)["status"] == "missing"


def test_the_three_refusals_a_variant_request_can_get_are_three(api):
    """A client does different things with each, so they are different answers.

    Legacy answered 404 to all three, which left "wait, it is building" and
    "stop asking, it is not yours" indistinguishable.
    """
    kb = _kb(api)["id"]
    _ingested(api, "doc-1", kb_id=kb, name="one.pdf", methods=(M.STANDARD,))
    _ingested(api, "doc-2", kb_id=kb, name="two.pdf", methods=(M.MARKDOWN,))

    unknown = api.client.get(f"{V1}/documents/doc-1/analysis/methods/telepathy/chunks")
    assert unknown.status_code == 400
    _error(unknown, "invalid_request")

    # The content has markdown -- doc-2 asked for it -- and it is still not
    # doc-1's to serve.
    other = api.client.get(f"{V1}/documents/doc-1/analysis/methods/{M.MARKDOWN}/chunks")
    assert other.status_code == 404
    _error(other, "not_found")

    pending = api.client.get(f"{V1}/documents/doc-1/analysis/methods/{M.STANDARD}/chunks")
    assert pending.status_code == 409
    assert _error(pending, "not_ready")["details"]["state"]["status"]


def test_deleting_a_document_leaves_the_shared_content_for_the_other_upload(api):
    kb = _kb(api)["id"]
    _ingested(api, "doc-1", kb_id=kb, name="one.pdf", methods=(M.STANDARD,))
    _ingested(api, "doc-2", kb_id=kb, name="two.pdf", methods=(M.MARKDOWN,))

    assert api.client.delete(f"{V1}/documents/doc-1").status_code == 204
    assert api.store.deleted == ["doc-1"]
    assert api.client.get(f"{V1}/documents/doc-1").status_code == 404

    survivor = _json(api.client.get(f"{V1}/documents/doc-2"))["analysis"]
    assert survivor["status"] != "missing"
    assert survivor["selected_methods"] == [M.MARKDOWN]
    assert survivor["content"]["shared_with_document_ids"] == ["doc-2"]


def test_a_documents_chunks_are_paged_and_scoped_to_it(api):
    kb = _kb(api)["id"]
    _ingested(api, "doc-1", kb_id=kb, name="one.pdf")
    _ingested(api, "doc-2", kb_id=kb, name="two.pdf", sha="d" * 48)

    body = _json(api.client.get(f"{V1}/documents/doc-1/chunks?limit=2"))
    assert body["page"]["total"] == 3 and len(body["items"]) == 2
    assert {row["document_id"] for row in body["items"]} == {"doc-1"}
    row = body["items"][0]
    assert {"id", "document_id", "content", "chunk_index", "total_chunks",
            "section", "chunking_mode", "metadata"} == set(row)


# =============================================================== the upload
def test_an_upload_is_always_a_job(api):
    """202 and the job, never a document: the parse, the model calls and the
    store writes happen on a worker, and a contract written now does not
    inherit the legacy surface's synchronous wait."""
    kb = _kb(api)["id"]
    response = api.client.post(
        f"{V1}/documents",
        data={"knowledge_base_id": kb, "file": (io.BytesIO(b"%PDF rapor"), "rapor.pdf")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 202
    job = _json(response)
    assert response.headers["Location"] == f"{V1}/ingest-jobs/{job['id']}"
    assert job["status"] in {"queued", "running", "succeeded"}
    assert job["knowledge_base_id"] == kb
    assert job["name"] == "rapor.pdf"
    assert job["methods"], "an upload always resolves to at least one method"

    # The same job, polled where the Location header said to poll it.
    polled = _json(api.client.get(f"{V1}/ingest-jobs/{job['id']}"))
    assert polled["id"] == job["id"]
    assert set(polled) >= {"id", "status", "knowledge_base_id", "document_id",
                           "content_id", "name", "methods", "error", "result",
                           "restart_settled", "attached_uploads"}


def test_an_upload_without_a_knowledge_base_is_refused_and_writes_nothing(api):
    response = api.client.post(
        f"{V1}/documents",
        data={"file": (io.BytesIO(b"%PDF rapor"), "rapor.pdf")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 400
    _error(response, "invalid_request")
    assert _json(api.client.get(f"{V1}/ingest-jobs"))["page"]["total"] == 0


def test_an_upload_into_an_unknown_knowledge_base_is_a_404(api):
    response = api.client.post(
        f"{V1}/documents",
        data={"knowledge_base_id": "ghost", "file": (io.BytesIO(b"%PDF x"), "r.pdf")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 404
    _error(response, "not_found")


def test_an_unknown_job_is_a_404_that_means_only_one_thing(api):
    """A job id survives a restart -- the journal records it and start-up
    settles it against the ledger -- so 404 means "older than the retention
    window", never "we lost it"."""
    response = api.client.get(f"{V1}/ingest-jobs/job-from-last-year")
    assert response.status_code == 404
    _error(response, "not_found")


# ============================================================ asking, looking
def test_a_question_answers_with_its_citations(api):
    kb = _kb(api)["id"]
    body = _json(api.client.post(f"{V1}/queries",
                                 json={"knowledge_base_id": kb, "question": "alacaklar?"}))
    assert body["answer"]
    assert body["knowledge_base_id"] == kb
    assert body["grounded"] is True
    assert body["retrieval_method"] == "bm25"

    citation = body["citations"][0]
    assert set(citation) == {"label", "chunk_id", "document_id", "document", "section",
                             "pages", "chunking_mode", "used", "score", "content"}
    assert citation["label"] == "S1" and citation["used"] is True
    # The chunk verbatim, because that is what a citation quotes.
    assert citation["content"] == "Takipteki alacaklar 2024 yilinda azaldi."
    assert "diagnostics" in body, "the non-contractual half is still carried"


def test_a_search_returns_ranked_chunks_and_no_answer(api):
    kb = _kb(api)["id"]
    body = _json(api.client.post(f"{V1}/searches",
                                 json={"knowledge_base_id": kb, "query": "alacak",
                                       "method": "bm25", "limit": 5}))
    assert body["method"] == "bm25" and body["knowledge_base_id"] == kb
    assert "answer" not in body
    row = body["items"][0]
    assert row["id"] == "c0" and row["retrieval_method"] == "bm25"
    assert isinstance(row["score"], (int, float))


def test_an_empty_question_is_refused_before_it_takes_a_query_slot(api):
    kb = _kb(api)["id"]
    before = api.container.query_admission.snapshot()["accepted_total"]
    assert api.client.post(f"{V1}/queries",
                           json={"knowledge_base_id": kb, "question": " "}).status_code == 400
    assert api.container.query_admission.snapshot()["accepted_total"] == before


def test_both_surfaces_answer_from_the_same_application(api):
    """The parity claim. ``/api/v1`` is a second adapter, not a second
    implementation: a knowledge base created through one is the same record the
    other lists, with no synchronisation between them."""
    created = _kb(api, name="Shared")
    legacy = api.client.get("/api/kb").get_json()
    assert [kb["kb_id"] for kb in legacy["knowledge_bases"]] == [created["id"]]

    api.client.delete(f"{V1}/knowledge-bases/{created['id']}")
    assert api.client.get("/api/kb").get_json()["knowledge_bases"] == []
