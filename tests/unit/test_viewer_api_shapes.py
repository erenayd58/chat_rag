"""The JSON shapes Viewer v3 actually consumes from this console.

The Viewer's own server relays three console routes and reads a handful of
keys from each: the payload's ``arms`` / ``units`` / ``pages`` / ``meta`` /
``label`` and its ``live`` block, the chunk rows' ``arms[method].rows`` (plus
``kind`` and ``label``) which it indexes for "Dokümana sor", and the state
record it polls while a build runs. Both sides had tests only against
hand-written mocks of the other; nothing pinned the producing side.

These tests drive the real packager over a small canonical in a temporary
root and read the routes through the Flask test client. They assert the keys
the page and the relay read, not the full payload -- the payload is
``amsc.viewer_corpus.load_corpus`` output and is pinned in the chunk repository.
"""

from __future__ import annotations

import pytest

import app as flask_app
from components.viewer import analysis
from components.viewer import methods as M

#: What the Viewer's live-document code reads off ``payload["live"]``.
LIVE_KEYS = {"docId", "docIds", "key", "kbId", "kbName", "requested", "methods", "deepSource", "preparedAt"}
#: What ``rag_chat.ChatEngine.register_live`` reads off each relayed arm.
ARM_KEYS = {"kind", "label", "chunk_count", "rows"}
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
    flask_app.app.config.update(TESTING=True)
    with flask_app.app.test_client() as test_client:
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
    response = client.get(f"/api/demo/viewer-analysis/{document}/payload")
    body = response.get_json()
    assert response.status_code == 200
    assert set(body) == {"success", "doc_id", "payload"}
    assert body["success"] is True and body["doc_id"] == document

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
    live = client.get(f"/api/demo/viewer-analysis/{document}/payload").get_json()["payload"]["live"]
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


def test_a_document_with_no_payload_is_a_404_carrying_its_state(client, workspace):
    response = client.get("/api/demo/viewer-analysis/nobody/payload")
    body = response.get_json()
    assert response.status_code == 404
    assert body["success"] is False
    assert body["state"]["status"] == analysis.STATUS_MISSING
    assert body["state"]["doc_id"] == "nobody"
    assert "nobody" in body["error"]


# ---------------------------------------------------------------- /chunks


def test_the_chunks_route_serves_every_ready_arm_in_the_shape_the_viewer_indexes(client, document):
    response = client.get(f"/api/demo/viewer-analysis/{document}/chunks")
    body = response.get_json()
    assert response.status_code == 200
    assert set(body) == {"success", "doc_id", "label", "key", "arms"}
    assert body["doc_id"] == document and body["label"] == "Sekil.pdf"
    assert body["key"] == analysis.key_for(document, "shape-sha")
    assert sorted(body["arms"]) == ["markdown", "structure-only"]
    for method, arm in body["arms"].items():
        assert set(arm) == ARM_KEYS, method
        assert arm["kind"] == M.METHODS[method].engine
        assert arm["label"] == M.METHODS[method].label
        assert arm["chunk_count"] == len(arm["rows"]) > 0
        assert ROW_KEYS <= set(arm["rows"][0]), "rows are the chunker's own row schema"
        assert arm["rows"] == analysis.chunk_rows(document, method, "shape-sha"), (
            "the rows served are exactly the packaged chunks.jsonl"
        )


def test_the_chunks_route_can_select_one_method(client, document):
    body = client.get(f"/api/demo/viewer-analysis/{document}/chunks?method=markdown").get_json()
    assert list(body["arms"]) == ["markdown"]


def test_the_chunks_route_refuses_an_unknown_method_and_reports_a_missing_one(client, document):
    response = client.get(f"/api/demo/viewer-analysis/{document}/chunks?method=turbo")
    assert response.status_code == 400
    assert "turbo" in response.get_json()["error"]

    response = client.get(f"/api/demo/viewer-analysis/{document}/chunks?method=agentic")
    body = response.get_json()
    assert response.status_code == 404, "a known method that was never packaged is not invented"
    assert body["success"] is False and body["state"]["status"] == analysis.STATUS_READY

    response = client.get("/api/demo/viewer-analysis/nobody/chunks")
    assert response.status_code == 404
    assert response.get_json()["state"]["status"] == analysis.STATUS_MISSING


# ------------------------------------------------------- state and methods


def test_the_state_route_reports_what_the_workspace_panel_polls(client, document):
    body = client.get(f"/api/demo/viewer-analysis/{document}").get_json()
    assert body["success"] is True
    state = body["state"]
    for key in ("key", "status", "doc_id", "doc_ids", "requested", "methods",
                "ready_methods", "failed_methods", "label", "kb_id", "kb_name", "updated_at"):
        assert key in state, key
    assert state["status"] == analysis.STATUS_READY
    assert state["ready_methods"] == ["markdown", "structure-only"]
    assert state["failed_methods"] == []
    assert set(state["methods"]) == {"markdown", "structure-only"}


def test_post_methods_adds_a_variant_and_the_new_arm_becomes_fetchable(client, document):
    response = client.post(f"/api/demo/viewer-analysis/{document}/methods", json={"methods": ["agentic"]})
    body = response.get_json()
    assert response.status_code == 200 and body["success"] is True
    assert body["state"]["requested"] == ["markdown", "structure-only", "agentic"]
    assert body["state"]["status"] in {analysis.STATUS_PENDING, analysis.STATUS_RUNNING, analysis.STATUS_READY}

    analysis._queue.join()
    state = client.get(f"/api/demo/viewer-analysis/{document}").get_json()["state"]
    assert state["status"] == analysis.STATUS_READY
    assert state["ready_methods"] == ["markdown", "structure-only", "agentic"]
    arms = client.get(f"/api/demo/viewer-analysis/{document}/chunks").get_json()["arms"]
    assert arms["agentic"]["kind"] == "deep_analysis" and arms["agentic"]["rows"]
    payload = client.get(f"/api/demo/viewer-analysis/{document}/payload").get_json()["payload"]
    assert payload["live"]["methods"]["agentic"]["status"] == analysis.STATUS_READY
    assert payload["live"]["deepSource"] == analysis.SOURCE_DETERMINISTIC, "no model was called for it"


def test_post_methods_on_an_unknown_document_is_a_404(client, workspace):
    response = client.post("/api/demo/viewer-analysis/nobody/methods", json={"methods": ["markdown"]})
    assert response.status_code == 404
    assert response.get_json()["success"] is False


def test_post_methods_normalises_like_an_upload(client, document):
    body = client.post(f"/api/demo/viewer-analysis/{document}/methods",
                       json={"methods": ["turbo", "markdown"]}).get_json()
    assert body["state"]["requested"] == ["markdown", "structure-only"], "unknown names are dropped"
    analysis._queue.join()
