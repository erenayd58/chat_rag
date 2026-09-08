"""Adding a chunking method: the extension path, driven end to end.

The method is the library's shipped example (``amsc.chunking.example``),
registered in ``amsc.chunking.registry`` for the length of a test and nothing else.
What is proved is that this one registration is enough for the product:
the console's catalogue and its API list it, an upload can ask for it, the
packager runs it over a real canonical without loading any model, the
Viewer routes serve it under its own kind and label, the Viewer v3 shell
build lists it -- and that unregistering it makes every one of those
surfaces forget it again. No route, template, script or registry file in
this repository is edited to get there.

The last tests are the drift guards: the few places that still spell a
method name (the frontend's mode labels, the indexing chunker's options)
are held to the registries they mirror.

Nothing here writes down how many methods there are. Every list is derived
from the registry and the fifth method is located by key, so this file does
not itself become the fixed method list it exists to make unnecessary -- a
real fifth method added to ``_BUILTIN`` leaves it green.
"""

from __future__ import annotations

import io
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

import app as flask_app
from application import workspace as app_workspace
from amsc.chunking import registry
from amsc.chunking.example import FIXED_WINDOW
from components.chunker import registry as indexing
from components.knowledgebase.manager import KnowledgeBaseManager, normalize_chunker_config
from components.viewer import analysis
from components.viewer import methods as M

ROOT = Path(__file__).resolve().parents[2]


# ------------------------------------------------------------- fixtures
@pytest.fixture
def fifth():
    registry.register(FIXED_WINDOW)
    try:
        yield FIXED_WINDOW
    finally:
        registry.unregister(FIXED_WINDOW.key)


def _unit(order, unit_id, unit_type, text, page, *, level=None, path=()):
    row = {
        "document_id": "fifth-doc", "unit_id": unit_id, "order": order, "text": text,
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


# ------------------------------------------------- the console knows it
def test_one_registration_is_a_console_method(fifth):
    assert set(M.ORDER) == set(registry.order()), "the console's universe is the registry's"
    assert list(M.ORDER)[:len(M.PRODUCT_ORDER)] == list(M.PRODUCT_ORDER), "product order first"
    assert M.ORDER[-1] == "fixed-window", "a method the product order does not name is listed last"
    assert "fixed-window" in M.METHODS
    method = M.METHODS["fixed-window"]
    assert (method.label, method.engine, method.uses_model, method.needs_embedder, method.deep) == (
        "Sabit Pencere", "fixed_window", False, False, False,
    )
    assert M.resolve("fixed-window").available is True
    assert M.normalise(["fixed-window", "markdown"]) == ["markdown", "fixed-window"], "product order first"
    row = next(r for r in M.catalogue() if r["key"] == "fixed-window")
    assert row["available"] is True and row["default"] is False and row["engine"] == "fixed_window"
    assert next(r for r in M.catalogue() if r["key"] == "structure-only")["default"] is True


def test_the_api_offers_it(client, fifth, monkeypatch):
    """Both surfaces, because both read the same registry and neither keeps a
    list. The product contract's own discovery is driven end to end by
    ``tests/migration/test_api_v1_contract.py``; what is checked here is that
    a chunker author gets it on the compatibility surface too, with no edit."""
    monkeypatch.setattr(M, "embedder_available", lambda: (True, ""))
    body = client.get("/api/demo/methods").get_json()
    assert [row["key"] for row in body["methods"]] == list(M.ORDER)
    fifth_row = body["methods"][-1]
    assert fifth_row["key"] == "fixed-window" and fifth_row["label"] == "Sabit Pencere"
    assert fifth_row["available"] and not fifth_row["default"]

    versioned = client.get("/api/v1/meta/chunking-methods").get_json()
    assert [row["key"] for row in versioned["items"]] == list(M.ORDER)


class _Chunker:
    last_canonical_units = None
    last_deep_result = None

    def get_name(self):
        return "StructuralChunker"

    def chunk_text_deep(self, *args, **kwargs):  # pragma: no cover
        raise AssertionError


class _Pipeline:
    def __init__(self):
        self.settings = SimpleNamespace(deep_analysis_api_key_env="FAKE_DEEP_KEY")
        self.chunker = _Chunker()
        self.last_deep_analysis_report = None

    def ingest_document_from_file(self, file_path, doc_title, deep_analysis=False):
        return [SimpleNamespace(doc_id="doc-fifth")]


def test_an_upload_can_ask_for_it(tmp_path, monkeypatch, client, fifth):
    from werkzeug.datastructures import MultiDict

    monkeypatch.chdir(tmp_path)
    manager = KnowledgeBaseManager(str(tmp_path / "kbs.json"))
    monkeypatch.setattr(flask_app.services, "kb_manager", manager)
    kb = manager.create("fifth-kb", chunker={"type": "structure_first"})
    monkeypatch.setattr(flask_app.services, "get_pipeline", lambda *a, **k: _Pipeline())
    staged = []
    monkeypatch.setattr(app_workspace, "stage_analysis",
                        lambda doc_id, **kw: staged.append(dict(kw, doc_id=doc_id)) or {"status": "pending"})
    data = MultiDict([("file", (io.BytesIO(b"kucuk bir belge"), "belge.txt")), ("kb_id", kb["kb_id"]),
                      ("methods", "fixed-window")])
    response = client.post("/api/documents/upload", data=data, content_type="multipart/form-data")
    assert response.status_code == 200, response.get_json()
    assert staged[-1]["methods"] == ["fixed-window"]
    assert response.get_json()["chunking_mode"] == "standard", "an analysis method never changes the indexing"


# --------------------------------------- the packager runs it, the Viewer reads it
def test_the_packager_runs_it_and_the_viewer_routes_serve_it(client, fifth, tmp_path, monkeypatch):
    def no_model():
        raise AssertionError("no boundary model may be loaded for a method that needs none")

    monkeypatch.setattr(analysis, "_boundary_embedder", no_model)
    analysis.stage(doc_id="fifth-doc", label="Besinci.pdf", units=_corpus(),
                   methods=["structure-only", "fixed-window"], kb_id="kb1", kb_name="fifth-kb",
                   chunking_mode="standard", content_sha="fifth-sha")
    analysis._queue.join()

    state = analysis.read_state("fifth-doc", "fifth-sha")
    assert state["status"] == analysis.STATUS_READY, state
    assert state["ready_methods"] == ["structure-only", "fixed-window"]
    assert state["methods"]["fixed-window"]["status"] == analysis.STATUS_READY
    assert state["methods"]["fixed-window"]["chunk_count"] >= 2

    payload = analysis.payload("fifth-doc", "fifth-sha")
    assert payload["arms"]["fixed-window"]["kind"] == "fixed_window"
    assert payload["live"]["methods"]["fixed-window"]["status"] == "ready"

    chunks = client.get("/api/demo/viewer-analysis/fifth-doc/chunks?method=fixed-window").get_json()
    assert chunks["success"] is True
    arm = chunks["arms"]["fixed-window"]
    assert arm["kind"] == "fixed_window" and arm["label"] == "Sabit Pencere"
    assert arm["chunk_count"] == len(arm["rows"]) >= 2
    assert all(":fw-chunk-" in row["chunk_id"] for row in arm["rows"])
    assert all({"chunk_id", "text", "unit_ids", "token_count"} <= set(row) for row in arm["rows"])

    # The Viewer v3 shell a fresh clone builds lists it too.
    from amsc.viewer.build import build_viewer

    output = tmp_path / "v3" / "index.html"
    build_viewer({}, output, root=tmp_path)
    html_text = output.read_text(encoding="utf-8")
    data = re.search(r'<script id="viewer-data" type="application/json">(.*?)</script>', html_text, re.S).group(1)
    assert '"fixed-window"' in data and "Sabit Pencere" in data


def test_a_method_can_be_added_to_an_existing_document_later(client, fifth):
    analysis.stage(doc_id="fifth-doc", label="Besinci.pdf", units=_corpus(),
                   methods=["structure-only"], content_sha="fifth-sha")
    analysis._queue.join()
    response = client.post("/api/demo/viewer-analysis/fifth-doc/methods", json={"methods": ["fixed-window"]})
    assert response.status_code == 200, response.get_json()
    analysis._queue.join()
    assert analysis.read_state("fifth-doc", "fifth-sha")["ready_methods"] == ["structure-only", "fixed-window"]


# --------------------------------------------- the Viewer needs no rebuild
def test_the_viewer_is_told_about_it_without_a_page_rebuild(fifth, tmp_path):
    """The exposure rule: a registered method cannot be invisible in the
    Viewer because somebody forgot ``python -m amsc.viewer.build``.

    The page embeds the registry at build time, which is all a file opened
    from disk can carry. Served, it reads the registry from its own server at
    boot -- so a page built before this method existed still lists it. Both
    halves are checked here: the stale build, and the live route.
    """
    from amsc.viewer import server as viewer_server
    from amsc.viewer.build import build_viewer

    registry.unregister(FIXED_WINDOW.key)
    output = tmp_path / "v3" / "index.html"
    build_viewer({}, output, root=tmp_path)
    registry.register(FIXED_WINDOW)

    page = output.read_text(encoding="utf-8")
    embedded = re.search(r'<script id="viewer-data" type="application/json">(.*?)</script>',
                         page, re.S).group(1)
    assert "fixed-window" not in embedded, "the page really was built without it"

    served = viewer_server.method_registry_payload()
    assert served["order"][-1] == "fixed-window"
    assert served["labels"]["fixed-window"] == M.METHODS["fixed-window"].label
    assert served["meta"]["fixed-window"]["kind"] == M.METHODS["fixed-window"].engine
    assert "/api/methods" in page and "refreshMethods" in page


# ------------------------------------------------------- and gone again
def test_once_unregistered_the_console_forgets_it(client):
    registry.register(FIXED_WINDOW)
    registry.unregister(FIXED_WINDOW.key)

    assert set(M.ORDER) == set(registry.order())
    assert "fixed-window" not in M.ORDER
    assert "fixed-window" not in M.METHODS
    assert M.normalise(["fixed-window"]) == ["structure-only"], "an unknown name falls back to Standard"
    assert [row["key"] for row in M.catalogue()] == list(M.ORDER)
    analysis.stage(doc_id="fifth-doc", label="Besinci.pdf", units=_corpus(),
                   methods=["structure-only"], content_sha="fifth-sha")
    analysis._queue.join()
    refused = client.get("/api/demo/viewer-analysis/fifth-doc/chunks?method=fixed-window")
    assert refused.status_code == 400 and "unknown chunking method" in refused.get_json()["error"]


# ------------------------------------------------------ drift guards
def test_the_frontend_mode_labels_are_the_registry_labels():
    """``chunkingModeLabel`` maps an ingest mode to a product name; the names
    are the registry's, held here so a rename there cannot leave the badge
    behind."""
    script = (ROOT / "static" / "js" / "api.js").read_text(encoding="utf-8")
    assert f"chunking_mode === 'deep_analysis') return '{M.label(M.DEEP)}'" in script
    assert f"chunking_mode === 'standard') return '{M.label(M.STANDARD)}'" in script


def test_the_upload_form_hard_codes_no_method_key():
    script = (ROOT / "static" / "js" / "kb_detail.js").read_text(encoding="utf-8")
    # ``'hybrid'`` also names a *retrieval* method on this page, so the guard
    # is on how a chunking method is picked, not on the bare word.
    assert "m.key ===" not in script and "m.key ==" not in script
    for key in ("structure-only", "agentic", "markdown"):
        assert f"'{key}'" not in script and f'"{key}"' not in script, key
    assert "m.default" in script, "the default comes from the catalogue"


def test_the_viewer_builders_and_the_console_share_one_identity():
    from amsc.viewer import corpus as viewer_corpus
    from amsc.viewer import build as viewer_build

    assert dict(viewer_build.METHOD_LABELS) == {key: M.METHODS[key].label for key in registry.order()}
    assert dict(viewer_build.METHOD_SUMMARIES) == {key: M.METHODS[key].summary for key in registry.order()}
    assert dict(viewer_corpus.ARM_KINDS) == {key: M.METHODS[key].engine for key in registry.order()}
    assert set(M.ORDER) == set(viewer_build.METHOD_ORDER), "same universe, the console's own order"


def test_the_indexing_chunkers_are_one_table(client):
    from components.chunker import create_chunker
    from core.exceptions import ConfigurationException

    assert list(indexing.ids()) == ["v4", "structure_first"]
    for chunker in indexing.INDEXING_CHUNKERS:
        for alias in chunker.aliases:
            assert normalize_chunker_config({"type": alias.upper()})["type"] == chunker.id
    with pytest.raises(ValueError, match=re.escape(indexing.expected())):
        normalize_chunker_config({"type": "turbo"})
    settings = SimpleNamespace(chunker_type="turbo")
    with pytest.raises(ConfigurationException, match=re.escape(indexing.expected())):
        create_chunker(settings)
    with pytest.raises(ValueError, match="Structure-first accepts no runtime chunker params"):
        normalize_chunker_config({"type": "structure_first", "params": {"x": 1}})
    with pytest.raises(ValueError, match="Frozen V4 accepts no runtime chunker params"):
        normalize_chunker_config({"type": "v4", "params": {"x": 1}})
    assert normalize_chunker_config(None) == {"type": "structure_first", "params": {}}


def test_an_analysis_method_is_still_never_an_indexing_chunker(fifth):
    """The two registries stay apart: registering an analysis method does not
    make it a knowledge base's chunker."""
    with pytest.raises(ValueError):
        normalize_chunker_config({"type": "fixed-window"})
    with pytest.raises(ValueError):
        normalize_chunker_config({"type": "fixed_window"})
