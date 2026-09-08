"""The FastAPI adapter itself: the things the contract suite cannot see.

``tests/migration/test_api_v1_contract.py`` drives `/api/v1` as a contract and
is deliberately blind to what serves it -- which is exactly what made the port
possible. This file is the other half: the questions that only have an answer
because the adapter is FastAPI, and the places where FastAPI's defaults had to
be bent to keep the contract rather than the contract bent to keep the
defaults.

Four groups:

* **the generated document.** Every served route is in it, every body and
  every answer is typed, and nothing the persistence layer keys its rows by
  appears anywhere in it.
* **validation at the boundary.** FastAPI answers a bad payload with a 422 and
  a ``detail`` list; this surface has always answered 400 ``invalid_request``,
  and a client branches on that. So does an unreadable page size, which is
  answered with the default rather than refused at all.
* **the exception table.** Each application refusal, provoked through a seam
  rather than asserted about a dictionary, because a table nothing routes
  through is a table that can be wrong.
* **the process's own application.** What ``interfaces.http.create_app``
  composes: the contract, one operator route beside it that is deliberately
  off the contract, and one container underneath both.

Until Step 13 there was a fifth group here -- coexistence -- for the bridge
that mounted this application inside the Flask console so one process could
serve both. That console is gone and this application is the process
(``asgi.py``), so what those tests were guarding is now the group above.
"""

from __future__ import annotations

import io
import json
import threading
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import asgi as entrypoint
from application import ingest as app_ingest
from application.errors import Conflict, InvalidRequest, NotFound, NotReady, Unavailable
from components.ingest import IngestManager
from components.knowledgebase.manager import KnowledgeBaseManager
from components.viewer import analysis
from components.viewer import methods as M
from config.ingest import IngestLimits
from core.exceptions import IngestOverloaded, QueryOverloaded, QueryTimeout
from core.models import DocumentChunk, RetrievalResult
import interfaces.http as http
from interfaces.http import v1
from interfaces.http.v1 import envelope
from utils.document_tracker import DocumentTracker

V1 = v1.PREFIX

#: Names the current persistence keys its rows by. None of them is a product
#: fact and none may appear in the published schema -- they are what the
#: PostgreSQL and pgvector migrations replace.
STORAGE_DETAIL = (
    "file_path", "vector_db_path", "vector_db_provider", "storage_path",
    "chroma", "collection_name", "embedding_provider",
)


# ------------------------------------------------------------- the doubles
class Store:
    """The same shape the contract suite's store has: enough to answer a read."""

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

    def model_chain(self):
        return {"embedding": {"model": "test/embedding"}, "answer": {"model": "test/answer"}}


@pytest.fixture
def wired(tmp_path, monkeypatch):
    """The process container, with every store replaced by this test's own.

    The same seams the application suite replaces, on the same object both
    surfaces are given -- which is what makes the parity test below mean
    something.
    """
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(analysis, "root", lambda: tmp_path / "viewer-live")
    monkeypatch.setattr(analysis, "enqueue", lambda key: key)

    store = Store()
    pipeline = Pipeline(store)
    ledger_file = str(tmp_path / "ledger.json")
    container = entrypoint.services

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

    yield SimpleNamespace(container=container, store=store, root=tmp_path,
                          ledger=lambda: DocumentTracker(ledger_file))
    jobs.close(timeout=20)


@pytest.fixture
def api(wired):
    """The FastAPI application, driven directly -- no Flask in the way.

    ``raise_server_exceptions=False`` so the client behaves like a server: an
    unhandled exception is the 500 a caller would receive, which is the thing
    the exception table is for.
    """
    with TestClient(v1.create_app(wired.container),
                    raise_server_exceptions=False) as client:
        yield SimpleNamespace(client=client, **vars(wired))


@pytest.fixture
def schema(api):
    return api.client.get(v1.OPENAPI_PATH).json()


def _kb(api, name="Yillik raporlar"):
    made = api.client.post(f"{V1}/knowledge-bases",
                           json={"name": name, "chunker": {"type": "structure_first"}})
    assert made.status_code == 201, made.text
    return made.json()


# ========================================================= the generated document
def test_the_openapi_document_is_served_and_names_this_contract(schema):
    """Served from the contract's own prefix, so a client that knows the base
    URL knows where the document is."""
    assert schema["openapi"].startswith("3.")
    assert schema["info"]["title"]
    assert schema["info"]["version"]
    assert v1.OPENAPI_PATH.startswith(f"{V1}/")


def test_every_served_route_appears_in_the_openapi_document(api):
    """Generated, so it cannot drift -- checked anyway, because "generated"
    is a claim about the code and this is the claim being made."""
    served = {
        (method, path)
        for path, methods, _ in http.served_routes(api.client.app.routes)
        for method in methods
        if method not in {"HEAD", "OPTIONS"} and not path.endswith("openapi.json")
    }
    documented = {
        (method.upper(), path)
        for path, operations in api.client.get(v1.OPENAPI_PATH).json()["paths"].items()
        for method in operations
    }
    assert served == documented, sorted(served ^ documented)


def test_every_request_body_and_every_answer_is_typed(schema):
    """A generated client is only worth the types in it. An operation with an
    untyped body, or a 2xx with no schema, is one a client has to guess at."""
    untyped = []
    for path, operations in schema["paths"].items():
        for verb, operation in operations.items():
            body = operation.get("requestBody")
            if body and not any("schema" in media
                                for media in body["content"].values()):
                untyped.append(f"{verb.upper()} {path} (body)")
            ok = [code for code in operation["responses"] if code.startswith("2")]
            assert ok, f"{verb.upper()} {path} documents no success"
            for code in ok:
                answer = operation["responses"][code]
                if code == "204":
                    continue  # nothing to return, so nothing to type
                if not any("schema" in media
                           for media in (answer.get("content") or {}).values()):
                    untyped.append(f"{verb.upper()} {path} -> {code}")
    assert untyped == [], untyped


def test_every_operation_publishes_the_refusal_taxonomy(schema):
    """A client that only knows the happy path handles a 503 as a bug."""
    for path, operations in schema["paths"].items():
        for verb, operation in operations.items():
            answers = set(operation["responses"])
            assert {"400", "404", "500", "503", "504"} <= answers, (
                f"{verb.upper()} {path} publishes only {sorted(answers)}")


def test_the_document_advertises_no_status_this_surface_never_sends(schema):
    """FastAPI documents a 422 wherever it validates something. This surface
    answers a payload it cannot read with 400 ``invalid_request`` -- so a 422
    in the document would be a promise no server here keeps, and it would send
    a generated client looking for a body shape that never arrives."""
    assert "422" not in json.dumps(schema)
    assert "ValidationError" not in json.dumps(schema.get("components", {}))


def test_the_bodies_a_client_sends_are_models_not_free_dictionaries(schema):
    """The four writes that take a payload each name a schema, so a rejected
    field is a documented field rather than a surprise."""
    written = {
        (verb.upper(), path)
        for path, operations in schema["paths"].items()
        for verb, operation in operations.items()
        if operation.get("requestBody")
    }
    assert ("POST", f"{V1}/knowledge-bases") in written
    assert ("PATCH", f"{V1}/knowledge-bases/{{kb_id}}") in written
    assert ("POST", f"{V1}/queries") in written
    assert ("POST", f"{V1}/searches") in written
    assert ("POST", f"{V1}/documents") in written


def test_no_storage_implementation_detail_reaches_the_published_schema(schema):
    """The pgvector and PostgreSQL migrations replace all of these. A client
    that was told about one would have to be changed with them."""
    serialised = json.dumps(schema).casefold()
    leaked = [name for name in STORAGE_DETAIL if f'"{name}"' in serialised]
    assert leaked == [], f"the schema publishes storage internals: {leaked}"


def test_the_document_names_no_chunking_method(schema):
    """Discovery is a runtime answer from the library's registry. A method
    baked into the published schema would be a second catalogue, and it would
    be wrong on the machine that cannot run it."""
    from amsc.chunking import registry

    serialised = json.dumps(schema)
    named = sorted(key for key in registry.order() if f'"{key}"' in serialised)
    assert named == [], f"the schema enumerates chunking methods: {named}"


def test_every_refusal_is_typed_and_not_only_described(schema):
    """A generated client types the happy path from the response models; the
    refusal body is produced by the exception handlers and never returned by a
    route, so without this it is the one shape a client has to hand-write --
    and ``type``, which the whole taxonomy is branched on, is exactly what is
    in it."""
    for path, operations in schema["paths"].items():
        for verb, operation in operations.items():
            for code in ("400", "404", "409", "500", "503", "504"):
                answer = operation["responses"][code]
                media = (answer.get("content") or {}).get("application/json") or {}
                assert media.get("schema"), f"{verb.upper()} {path} -> {code} is untyped"

    error = schema["components"]["schemas"]["ApiError"]
    assert set(error["required"]) == {"type", "message"}
    assert set(error["properties"]) == {"type", "message", "details"}


def test_the_two_headers_this_surface_sends_are_declared_where_it_sends_them(schema):
    """``Location`` and ``Retry-After`` are answers, not decoration: one says
    where to poll, the other how long to wait. A client generated from a
    document that does not mention them has to be told separately."""
    created = schema["paths"][f"{V1}/knowledge-bases"]["post"]["responses"]["201"]
    accepted = schema["paths"][f"{V1}/documents"]["post"]["responses"]["202"]
    assert "Location" in created["headers"]
    assert "Location" in accepted["headers"]

    for path, operations in schema["paths"].items():
        for verb, operation in operations.items():
            headers = operation["responses"]["503"].get("headers") or {}
            assert "Retry-After" in headers, f"{verb.upper()} {path}"


def test_the_document_publishes_exactly_the_operations_the_contract_lists(schema):
    """``docs/api-v1.md`` is the published endpoint list and this is the
    machine-readable one. They are generated from different things -- prose
    and the routing table -- so a route added to one and not the other is a
    client reading two different contracts."""
    import re
    from pathlib import Path

    doc = Path(__file__).resolve().parents[2] / "docs" / "api-v1.md"
    published = {
        (verb, path)
        for verbs, path in re.findall(r"`([A-Z|]+)\s+(/api/v1/[^`\s]+)`",
                                      doc.read_text(encoding="utf-8"))
        for verb in verbs.split("|")
    }
    generated = {
        (verb.upper(), path.replace("{", "<").replace("}", ">"))
        for path, operations in schema["paths"].items()
        for verb in operations
    }
    # The document serves itself; prose lists it, the routing table cannot.
    published.discard(("GET", f"{V1}/openapi.json"))
    assert published == generated, sorted(published ^ generated)


# ================================================ validation at the boundary
def test_an_unreadable_body_is_the_products_400_and_not_fastapis_422(api):
    """FastAPI's default is a 422 with a ``detail`` list, which is a second
    vocabulary for "your request was wrong". This surface has one."""
    refused = api.client.post(f"{V1}/queries", content=b"{not json",
                              headers={"content-type": "application/json"})
    assert refused.status_code == 400
    body = refused.json()
    assert body["error"]["type"] == "invalid_request"
    assert "detail" not in body


def test_a_wrongly_typed_field_is_refused_with_the_field_named(api):
    refused = api.client.post(f"{V1}/queries", json={"question": "x", "top_k": "many"})
    assert refused.status_code == 400
    error = refused.json()["error"]
    assert error["type"] == "invalid_request"
    assert any("top_k" in field["field"] for field in error["details"]["fields"])


def test_an_unknown_field_is_refused_rather_than_ignored(api):
    """``extra="forbid"`` on the schemas, from the client's side: a payload
    with a misspelt field is a payload that would silently not do what its
    author meant."""
    refused = api.client.post(f"{V1}/knowledge-bases",
                              json={"name": "x", "chunkers": {"type": "y"}})
    assert refused.status_code == 400
    assert refused.json()["error"]["type"] == "invalid_request"


def test_a_missing_body_is_read_as_an_empty_one(api):
    """FastAPI refuses a body-less POST as a missing field. On this surface a
    payload is optional and the *use case* decides what a missing field means,
    so a request with no body does exactly what one with ``{}`` does -- which
    is what the Flask adapter did and what a client sending neither expects.

    Proved by the pair: both take the same default name, so the second is
    refused as a duplicate, by the product, in the product's own words.
    """
    none = api.client.post(f"{V1}/knowledge-bases")
    assert none.status_code == 201, none.text

    empty = api.client.post(f"{V1}/knowledge-bases", json={})
    assert empty.status_code == 400
    assert empty.json()["error"]["type"] == "invalid_request"


def test_an_unreadable_page_size_is_the_default_not_a_refusal(api):
    """The one place FastAPI's validation is deliberately not used. A page
    size is a presentation decision; refusing ``?limit=abc`` breaks a pasted
    link and teaches nobody anything."""
    _kb(api)
    for query in ("limit=abc", "offset=-4", "limit=0", "limit=100000",
                  "offset=nope&limit=nonsense"):
        page = api.client.get(f"{V1}/knowledge-bases?{query}").json()["page"]
        assert 1 <= page["limit"] <= envelope.MAX_LIMIT, query
        assert page["offset"] >= 0, query


def test_a_wrong_method_is_refused_in_this_surfaces_vocabulary(api):
    """Starlette's own 405 carries a ``detail`` string. A client on this
    contract branches on ``error.type``."""
    refused = api.client.put(f"{V1}/health")
    assert refused.status_code == 405
    assert refused.json()["error"]["type"] == "invalid_request"


def test_an_unroutable_path_under_the_prefix_is_a_not_found_refusal(api):
    refused = api.client.get(f"{V1}/there-is-no-such-thing")
    assert refused.status_code == 404
    assert refused.json()["error"]["type"] == "not_found"


# ===================================================== the exception table
@pytest.mark.parametrize("error, status, kind", [
    (InvalidRequest("no"), 400, "invalid_request"),
    (NotFound("gone"), 404, "not_found"),
    (NotReady("building", state={"status": "running"}), 409, "not_ready"),
    (Conflict("in use"), 409, "conflict"),
    (Unavailable("no model"), 503, "unavailable"),
    (QueryOverloaded("full", reason="admission", retry_after_seconds=7), 503, "overloaded"),
    (IngestOverloaded("queue full", retry_after_seconds=11), 503, "overloaded"),
    (QueryTimeout(), 504, "timeout"),
    (RuntimeError("nobody decided about this"), 500, "internal"),
])
def test_every_refusal_is_translated_centrally(api, monkeypatch, error, status, kind):
    """Provoked through a seam rather than asserted about a table, because a
    mapping nothing routes through is a mapping that can be wrong.

    ``/api/v1/health`` is the vehicle: it takes no payload, so what comes back
    is the exception table and nothing else.
    """
    from application import ops

    def explode(_services):
        raise error

    monkeypatch.setattr(ops, "health", explode)
    refused = api.client.get(f"{V1}/health")
    assert refused.status_code == status
    assert refused.json()["error"]["type"] == kind


def test_an_overloaded_refusal_carries_retry_after_and_which_limit_refused(api, monkeypatch):
    """'raise QUERY_MAX_ACTIVE' and 'raise ANSWER_MAX_INFLIGHT' are different
    decisions, and the caller's retry should not have to guess which."""
    from application import ops

    def explode(_services):
        raise QueryOverloaded("no slot", reason="answer_capacity",
                              retry_after_seconds=9)

    monkeypatch.setattr(ops, "health", explode)
    refused = api.client.get(f"{V1}/health")
    assert refused.status_code == 503
    assert refused.headers["Retry-After"] == "9"
    details = refused.json()["error"]["details"]
    assert details["reason"] == "answer_capacity"
    assert details["retry_after_seconds"] == 9


def test_a_not_ready_refusal_carries_where_the_work_got_to(api, monkeypatch):
    from application import ops

    def explode(_services):
        raise NotReady("still building", state={"status": "running", "unit_count": 4})

    monkeypatch.setattr(ops, "health", explode)
    refused = api.client.get(f"{V1}/health")
    assert refused.status_code == 409
    assert refused.json()["error"]["details"]["state"]["status"] == "running"


def test_a_timeout_says_how_long_it_waited(api, monkeypatch):
    from application import ops

    monkeypatch.setattr(ops, "health", lambda _s: (_ for _ in ()).throw(QueryTimeout()))
    refused = api.client.get(f"{V1}/health")
    assert refused.status_code == 504
    assert refused.json()["error"]["details"]["timeout_seconds"] > 0


def test_no_route_handler_catches_an_application_error_of_its_own():
    """The architectural half of the table: it is one place because no router
    is allowed to be a second one."""
    import ast
    from pathlib import Path

    routers = Path(v1.__file__).parent / "routers"
    offenders = []
    for path in sorted(routers.glob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, (ast.Try, ast.ExceptHandler)):
                offenders.append(path.name)
    assert offenders == [], (
        "these translate exceptions instead of letting the table do it: "
        + repr(sorted(set(offenders))))


# =================================================== the answers themselves
def test_a_deletion_answers_204_with_no_body(api):
    created = _kb(api)
    gone = api.client.delete(f"{V1}/knowledge-bases/{created['id']}")
    assert gone.status_code == 204
    assert gone.content == b""


def test_an_upload_answers_202_and_where_to_poll_it(api):
    kb = _kb(api)["id"]
    accepted = api.client.post(
        f"{V1}/documents",
        data={"knowledge_base_id": kb},
        files={"file": ("rapor.pdf", io.BytesIO(b"%PDF rapor"), "application/pdf")},
    )
    assert accepted.status_code == 202
    job = accepted.json()
    assert accepted.headers["Location"] == f"{V1}/ingest-jobs/{job['id']}"
    assert api.client.get(accepted.headers["Location"]).json()["id"] == job["id"]


def test_a_created_knowledge_base_answers_201_and_where_to_read_it(api):
    created = api.client.post(f"{V1}/knowledge-bases", json={"name": "Yeni"})
    assert created.status_code == 201
    location = created.headers["Location"]
    assert location == f"{V1}/knowledge-bases/{created.json()['id']}"
    assert api.client.get(location).status_code == 200


def test_an_upload_without_a_file_is_the_products_own_refusal(api):
    """Not FastAPI's "field required": the message is the one the legacy
    surface and this one have always given."""
    refused = api.client.post(f"{V1}/documents", data={"knowledge_base_id": "x"})
    assert refused.status_code == 400
    assert refused.json()["error"]["type"] == "invalid_request"


def test_a_document_carries_no_field_the_ledger_keys_it_by(api):
    kb = _kb(api)["id"]
    path = api.root / "one.pdf"
    path.write_text("pdf", encoding="utf-8")
    api.ledger().mark_as_ingested(file_path=str(path), doc_id="doc-1", chunk_count=3,
                                  kb_id=kb, metadata={"original_filename": "one.pdf"})

    body = api.client.get(f"{V1}/documents/doc-1").text.casefold()
    for name in STORAGE_DETAIL:
        assert f'"{name}"' not in body, name
    assert str(api.root).casefold().replace("\\", "\\\\") not in body


def test_the_embedding_index_report_reaches_the_client_unchanged(api):
    """A pass-through answer, and the one place a declared model could quietly
    reshape one: the store's manifest report is what an operator reads to
    decide whether to re-index, and this adapter neither pads it with nulls
    nor drops a field it has not heard of."""
    kb = _kb(api)["id"]
    body = api.client.get(f"{V1}/knowledge-bases/{kb}/embedding-index").json()
    assert body == {"compatible": True, "model": "test/embedding", "vectors": 3}


def test_a_registered_chunking_method_reaches_the_api_with_no_edit_here(api):
    """The discovery claim, on the FastAPI side of the port: the registry is
    read at request time, so a method registered now is offered now."""
    from amsc.chunking import registry
    from amsc.chunking.example import FIXED_WINDOW

    def keys():
        return [item["key"]
                for item in api.client.get(f"{V1}/meta/chunking-methods").json()["items"]]

    assert FIXED_WINDOW.key not in keys()
    registry.register(FIXED_WINDOW)
    try:
        assert FIXED_WINDOW.key in keys()
    finally:
        registry.unregister(FIXED_WINDOW.key)
    assert FIXED_WINDOW.key not in keys()


def test_pagination_walks_a_collection_and_clamps_what_it_is_asked_for(api):
    for index in range(5):
        api.client.post(f"{V1}/knowledge-bases", json={"name": f"KB {index}"})

    first = api.client.get(f"{V1}/knowledge-bases?offset=0&limit=2").json()
    second = api.client.get(f"{V1}/knowledge-bases?offset=2&limit=2").json()
    assert [k["id"] for k in first["items"]] != [k["id"] for k in second["items"]]
    assert first["page"] == {"offset": 0, "limit": 2, "total": 5}
    assert second["page"] == {"offset": 2, "limit": 2, "total": 5}

    capped = api.client.get(f"{V1}/knowledge-bases?limit=100000").json()
    assert capped["page"]["limit"] == envelope.MAX_LIMIT

    beyond = api.client.get(f"{V1}/knowledge-bases?offset=99").json()
    assert beyond["items"] == [] and beyond["page"]["total"] == 5


def test_a_knowledge_base_and_a_document_live_and_die_through_this_api(api):
    """One representative lifecycle, end to end, over the real managers: the
    ledger, the analysis packager and the ingest jobs are the product's own."""
    from application import workspace

    kb = _kb(api, name="Lifecycle")["id"]

    path = api.root / "rapor.pdf"
    path.write_text("pdf", encoding="utf-8")
    api.ledger().mark_as_ingested(file_path=str(path), doc_id="doc-1", chunk_count=3,
                                  kb_id=kb, metadata={"original_filename": "rapor.pdf"})
    api.store.add("doc-1")
    workspace.stage_analysis(
        "doc-1", label="rapor.pdf", kb_id=kb, content_sha="a" * 48,
        methods=[M.STANDARD],
        units=[{"document_id": "shared", "unit_id": "u0", "order": 0, "text": "P.",
                "type": "paragraph", "section_path": [], "source": {"page": 1}}])

    listed = api.client.get(f"{V1}/documents?knowledge_base_id={kb}").json()
    assert [row["id"] for row in listed["items"]] == ["doc-1"]
    assert listed["items"][0]["analysis"]["selected_methods"] == [M.STANDARD]

    chunks = api.client.get(f"{V1}/documents/doc-1/chunks?limit=2").json()
    assert chunks["page"]["total"] == 3 and len(chunks["items"]) == 2

    answered = api.client.post(f"{V1}/queries",
                               json={"knowledge_base_id": kb, "question": "alacaklar?"})
    assert answered.status_code == 200
    assert answered.json()["grounded"] is True

    searched = api.client.post(f"{V1}/searches",
                               json={"knowledge_base_id": kb, "query": "alacak",
                                     "method": "bm25"})
    assert searched.status_code == 200 and searched.json()["items"]

    assert api.client.delete(f"{V1}/documents/doc-1").status_code == 204
    assert api.client.get(f"{V1}/documents/doc-1").status_code == 404
    assert api.client.delete(f"{V1}/knowledge-bases/{kb}").status_code == 204
    assert api.client.get(f"{V1}/knowledge-bases").json()["page"]["total"] == 0


# =============================================== the process's own application
def test_the_process_serves_the_contract_and_exactly_one_route_beside_it(wired):
    """``interfaces.http.create_app`` is the whole server. The operator route
    is the only thing on it that ``/api/v1`` does not publish, and it is off
    the contract deliberately -- its body reports internals that are free to
    change."""
    whole = http.surface(http.create_app(wired.container))
    contract = http.surface(v1.create_app(wired.container))
    assert whole - contract == {("GET", "/api/ops/metrics")}
    assert contract, "the contract serves nothing"


def test_the_operator_route_answers_from_the_same_container(wired):
    """One container under both, handed in rather than imported: the metrics
    an operator reads are this process's own."""
    with TestClient(http.create_app(wired.container),
                    raise_server_exceptions=False) as client:
        assert client.post(f"{V1}/knowledge-bases", json={"name": "Ops"}).status_code == 201
        body = client.get("/api/ops/metrics").json()
    assert body["success"] is True
    assert body["ingest"]["workers"] == wired.container.ingest_jobs.snapshot()["workers"]


def test_a_caller_with_no_session_of_its_own_shares_one_cache_entry(wired):
    """The session id selects a cached pipeline and nothing else. There is no
    cookie on this surface, so a plain client gets the shared entry rather
    than a pipeline of its own per request."""
    seen: list[str] = []
    wired.container.get_pipeline = lambda session_id, kb_id=None: (
        seen.append(session_id) or Pipeline(wired.store))

    with TestClient(v1.create_app(wired.container)) as client:
        answered = client.get(f"{V1}/meta/models")
        assert answered.status_code == 200
        assert "set-cookie" not in answered.headers, "this surface starts no session"

    assert seen == ["global"], seen
