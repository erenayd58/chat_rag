"""The JSON shapes the Viewer screen actually consumes from this console.

The Viewer reads a handful of keys from three routes: the payload's ``arms`` /
``units`` / ``pages`` / ``meta`` / ``label`` and its ``live`` block, one
method's chunk rows, which it indexes for "Dokümana sor", and the analysis
state it polls while a build runs. Both sides had tests only against
hand-written mocks of the other; nothing pinned the producing side.

These tests drive the real packager over a small canonical in a temporary root
and read the routes over `/api/v1`. They assert the keys the screen reads, not
the whole payload -- that is ``amsc.viewer.corpus.load_corpus`` output,
published as pass-through and pinned in the chunk repository.

Until Step 13 the caller was the Viewer's own server, relaying ``/api/demo/*``
from a second process. The Viewer is a screen of this console's front end now,
and the keys are the same ones, read from the contract.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import asgi as entrypoint
import interfaces.http as http
from components.viewer import analysis
from components.viewer import methods as M

V1 = http.v1.PREFIX

#: What the Viewer's live-document code reads off ``payload["live"]``.
LIVE_KEYS = {"docId", "docIds", "key", "kbId", "kbName", "requested", "methods", "deepSource", "preparedAt"}
#: The shape one chunking method's rows come back in.
ARM_KEYS = {"items", "page", "method", "engine", "content_id"}
#: What the index and the page read off a chunk row (the structural row schema).
ROW_KEYS = {"chunk_id", "text", "unit_ids", "token_count"}


def _unit(order, unit_id, unit_type, text, page, *, level=None, path=()):
    row = {
        "document_id": "shape-doc", "unit_id": unit_id, "order": order, "text": text,
        "type": unit_type, "section_path": list(path), "source": {"page": page, "block": order},
    }
    if level is not None:
        row.update(heading_level=level, semantic_role="section", opens_section=True)
    return row


def _corpus(sections=2, paragraphs=5):
    units, order = [], 0
    for section in range(1, sections + 1):
        order += 1
        title = f"{section}. BOLUM"
        units.append(_unit(order, f"h-{order:04d}", "heading", title, section, level=1, path=[title]))
        for para in range(paragraphs):
            order += 1
            body = (f"Bu {section}. bolumun {para + 1}. paragrafidir. " * 12).strip()
            units.append(_unit(order, f"p-{order:04d}", "paragraph", body, section, path=[title]))
    return units


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    """A packager root of this test's own, drained before and after."""
    analysis._queue.join()
    with analysis._lock:
        analysis._inflight.clear()
    monkeypatch.setattr(analysis, "root", lambda: tmp_path / "viewer-live")
    yield tmp_path / "viewer-live"
    analysis._queue.join()
    with analysis._lock:
        analysis._inflight.clear()


@pytest.fixture
def client(workspace):
    with TestClient(http.create_app(entrypoint.services),
                    raise_server_exceptions=False) as test_client:
        yield test_client


@pytest.fixture
def document(workspace):
    """One live document with two ready variants, built by the real packager."""
    analysis.stage(doc_id="shape-doc", label="Sekil.pdf", units=_corpus(),
                   methods=["markdown", "structure-only"], kb_id="kb1", kb_name="sekil-kb",
                   chunking_mode="standard", content_sha="shape-sha")
    analysis._queue.join()
    state = analysis.read_state("shape-doc", "shape-sha")
    assert state["status"] == analysis.STATUS_READY, state
    return "shape-doc"


# --------------------------------------------------------------- /payload


def test_the_payload_route_returns_the_corpus_shape_plus_the_live_block(client, document):
    response = client.get(f"{V1}/documents/{document}/analysis/payload")
    body = response.json()
    assert response.status_code == 200, response.text
    assert set(body) == {"document_id", "content_id", "label", "ready_methods", "payload"}
    assert body["document_id"] == document and body["label"] == "Sekil.pdf"
    assert body["ready_methods"] == ["markdown", "structure-only"]

    payload = body["payload"]
    for key in ("units", "arms", "pages", "meta", "label", "live"):
        assert key in payload, key
    assert payload["label"] == "Sekil.pdf"
    assert sorted(payload["arms"]) == ["markdown", "structure-only"], "exactly the ready arms"
    for arm in payload["arms"].values():
        assert {"chunks", "seg", "m"} <= set(arm), "the page's row builder reads chunks/seg/m"
        assert arm["chunks"], "an offered arm has chunks"
    assert payload["meta"]["deep"] is None, "no Deep panel without a Deep variant"


def test_the_live_block_names_the_document_and_every_method_status(client, document):
    live = client.get(f"{V1}/documents/{document}/analysis/payload").json()["payload"]["live"]
    assert set(live) == LIVE_KEYS
    assert live["docId"] == document and live["docIds"] == [document]
    assert live["key"] == analysis.key_for(document, "shape-sha")
    assert live["kbId"] == "kb1" and live["kbName"] == "sekil-kb"
    assert live["requested"] == ["markdown", "structure-only"]
    # The page filters its method picker on these statuses: one entry per
    # registry method, ready ones offered, the rest hidden. (The JSON is
    # written with sorted keys, so order comes from the page, not from here.)
    assert set(live["methods"]) == set(M.ORDER)
    assert live["methods"]["markdown"]["status"] == analysis.STATUS_READY
    assert live["methods"]["structure-only"]["status"] == analysis.STATUS_READY
    assert live["methods"]["agentic"]["status"] == analysis.STATUS_MISSING
    assert live["methods"]["hybrid"]["status"] == analysis.STATUS_MISSING
    assert live["deepSource"] is None


def test_a_document_with_no_payload_is_not_ready_and_carries_its_state(client, workspace):
    """409 rather than 404: nothing has been built *yet*, and a client polling
    for a build has to be able to tell that from "no such document"."""
    response = client.get(f"{V1}/documents/nobody/analysis/payload")
    assert response.status_code == 409
    error = response.json()["error"]
    assert error["type"] == "not_ready"
    assert error["details"]["state"]["status"] == analysis.STATUS_MISSING
    assert error["details"]["state"]["doc_id"] == "nobody"
    assert "nobody" in error["message"]


# ---------------------------------------------------------------- /chunks


def test_the_chunks_route_serves_each_ready_arm_in_the_shape_the_viewer_indexes(client, document):
    """One method per request on this contract. The relay used to hand over
    every arm at once because it was building indexes for all of them in
    another process; a screen asks for the one a reader opened."""
    for method in ("markdown", "structure-only"):
        response = client.get(
            f"{V1}/documents/{document}/analysis/methods/{method}/chunks?limit=500")
        body = response.json()
        assert response.status_code == 200, response.text
        assert set(body) == ARM_KEYS, method
        assert body["method"] == method
        assert body["engine"] == M.METHODS[method].engine
        assert body["content_id"] == analysis.key_for(document, "shape-sha")
        assert body["page"]["total"] == len(body["items"]) > 0
        assert ROW_KEYS <= set(body["items"][0]), "rows are the chunker's own row schema"
        assert body["items"] == analysis.chunk_rows(document, method, "shape-sha"), (
            "the rows served are exactly the packaged chunks.jsonl"
        )


def test_the_chunk_rows_are_paged_like_every_other_collection(client, document):
    whole = client.get(
        f"{V1}/documents/{document}/analysis/methods/markdown/chunks?limit=500").json()
    page = client.get(
        f"{V1}/documents/{document}/analysis/methods/markdown/chunks?offset=1&limit=2").json()
    assert page["page"] == {"offset": 1, "limit": 2, "total": whole["page"]["total"]}
    assert page["items"] == whole["items"][1:3]


def test_the_chunks_route_keeps_its_three_refusals_apart(client, document):
    """Unknown to this deployment, known but not this upload's, and not built
    yet are three different things to a client."""
    unknown = client.get(f"{V1}/documents/{document}/analysis/methods/turbo/chunks")
    assert unknown.status_code == 400
    assert "turbo" in unknown.json()["error"]["message"]

    other = client.get(f"{V1}/documents/{document}/analysis/methods/agentic/chunks")
    assert other.status_code == 404, "a known method that was never packaged is not invented"
    assert other.json()["error"]["details"]["state"]["status"] == analysis.STATUS_READY

    nobody = client.get(f"{V1}/documents/nobody/analysis/methods/markdown/chunks")
    assert nobody.status_code == 409
    assert nobody.json()["error"]["details"]["state"]["status"] == analysis.STATUS_MISSING


# ------------------------------------------------------- state and methods


def test_the_state_route_reports_what_the_screen_polls(client, document):
    state = client.get(f"{V1}/documents/{document}/analysis").json()
    for key in ("status", "content_id", "selected_methods", "ready_methods",
                "failed_methods", "unit_count", "deep_source", "updated_at", "content"):
        assert key in state, key
    assert state["status"] == analysis.STATUS_READY
    assert state["selected_methods"] == ["markdown", "structure-only"]
    assert state["ready_methods"] == ["markdown", "structure-only"]
    assert state["failed_methods"] == []
    assert state["content_id"] == analysis.key_for(document, "shape-sha")
    # The shared analysis, kept apart from this upload's own selection.
    assert state["content"]["ready_methods"] == ["markdown", "structure-only"]
    assert state["content"]["shared_with_document_ids"] == [document]


def test_post_methods_adds_a_variant_and_the_new_arm_becomes_fetchable(client, document):
    response = client.post(f"{V1}/documents/{document}/analysis/methods",
                           json={"methods": ["agentic"]})
    state = response.json()
    assert response.status_code == 202, response.text
    assert state["selected_methods"] == ["markdown", "structure-only", "agentic"]
    assert state["status"] in {analysis.STATUS_PENDING, analysis.STATUS_RUNNING,
                               analysis.STATUS_READY}

    analysis._queue.join()
    state = client.get(f"{V1}/documents/{document}/analysis").json()
    assert state["status"] == analysis.STATUS_READY
    assert state["ready_methods"] == ["markdown", "structure-only", "agentic"]
    arm = client.get(
        f"{V1}/documents/{document}/analysis/methods/agentic/chunks?limit=500").json()
    assert arm["engine"] == "deep_analysis" and arm["items"]
    payload = client.get(f"{V1}/documents/{document}/analysis/payload").json()["payload"]
    assert payload["live"]["methods"]["agentic"]["status"] == analysis.STATUS_READY
    assert payload["live"]["deepSource"] == analysis.SOURCE_DETERMINISTIC, "no model was called for it"


def test_post_methods_on_an_unknown_document_is_a_404(client, workspace):
    response = client.post(f"{V1}/documents/nobody/analysis/methods",
                           json={"methods": ["markdown"]})
    assert response.status_code == 404
    assert response.json()["error"]["type"] == "not_found"


def test_post_methods_normalises_like_an_upload(client, document):
    state = client.post(f"{V1}/documents/{document}/analysis/methods",
                        json={"methods": ["turbo", "markdown"]}).json()
    assert state["selected_methods"] == ["markdown", "structure-only"], "unknown names are dropped"
    analysis._queue.join()
