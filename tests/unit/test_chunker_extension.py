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

from fastapi.testclient import TestClient

import asgi as entrypoint
import interfaces.http as http

V1 = http.v1.PREFIX
from chat_rag.application import workspace as app_workspace
from amsc.chunking import registry
from amsc.chunking.example import FIXED_WINDOW
from chat_rag.components.chunker import registry as indexing
from chat_rag.components.knowledgebase.manager import KnowledgeBaseManager, normalize_chunker_config
from chat_rag.components.viewer import analysis
from chat_rag.components.viewer import methods as M

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
    analysis.state().queue.join()
    with analysis.state().lock:
        analysis.state().inflight.clear()
    monkeypatch.setattr(analysis, "root", lambda: tmp_path / "viewer-live")
    yield tmp_path / "viewer-live"
    analysis.state().queue.join()
    with analysis.state().lock:
        analysis.state().inflight.clear()


@pytest.fixture
def client(workspace):
    with TestClient(http.create_app(entrypoint.services),
                    raise_server_exceptions=False) as test_client:
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
    """One surface, reading the registry and keeping no list of its own. The
    contract's discovery is driven end to end by
    ``tests/migration/test_api_v1_contract.py``; what is checked here is that
    a chunker author gets it with no edit to this repository at all."""
    monkeypatch.setattr(M, "embedder_available", lambda: (True, ""))
    body = client.get(f"{V1}/meta/chunking-methods").json()
    assert [row["key"] for row in body["items"]] == list(M.ORDER)
    fifth_row = body["items"][-1]
    assert fifth_row["key"] == "fixed-window" and fifth_row["label"] == "Sabit Pencere"
    assert fifth_row["available"] and not fifth_row["default"]


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
    monkeypatch.chdir(tmp_path)
    manager = KnowledgeBaseManager(str(tmp_path / "kbs.json"))
    monkeypatch.setattr(entrypoint.services, "kb_manager", manager)
    kb = manager.create("fifth-kb", chunker={"type": "structure_first"})
    monkeypatch.setattr(entrypoint.services, "get_pipeline", lambda *a, **k: _Pipeline())
    staged = []
    monkeypatch.setattr(app_workspace, "stage_analysis",
                        lambda doc_id, **kw: staged.append(dict(kw, doc_id=doc_id)) or {"status": "pending"})
    response = client.post(
        f"{V1}/documents",
        files={"file": ("belge.txt", io.BytesIO(b"kucuk bir belge"), "text/plain")},
        data={"knowledge_base_id": kb["kb_id"], "methods": ["fixed-window"]},
    )
    assert response.status_code == 202, response.text
    job = entrypoint.services.ingest_jobs.get(response.json()["id"])
    assert entrypoint.services.ingest_jobs.wait(job, 30)
    assert staged[-1]["methods"] == ["fixed-window"]
    assert job.snapshot()["result"]["chunking_mode"] == "standard", (
        "an analysis method never changes the indexing")


# --------------------------------------- the packager runs it, the Viewer reads it
def test_the_packager_runs_it_and_the_viewer_routes_serve_it(client, fifth, tmp_path, monkeypatch):
    def no_model():
        raise AssertionError("no boundary model may be loaded for a method that needs none")

    monkeypatch.setattr(analysis, "_boundary_embedder", no_model)
    analysis.stage(doc_id="fifth-doc", label="Besinci.pdf", units=_corpus(),
                   methods=["structure-only", "fixed-window"], kb_id="kb1", kb_name="fifth-kb",
                   chunking_mode="standard", content_sha="fifth-sha")
    analysis.state().queue.join()

    state = analysis.read_state("fifth-doc", "fifth-sha")
    assert state["status"] == analysis.STATUS_READY, state
    assert state["ready_methods"] == ["structure-only", "fixed-window"]
    assert state["methods"]["fixed-window"]["status"] == analysis.STATUS_READY
    assert state["methods"]["fixed-window"]["chunk_count"] >= 2

    payload = analysis.payload("fifth-doc", "fifth-sha")
    assert payload["arms"]["fixed-window"]["kind"] == "fixed_window"
    assert payload["live"]["methods"]["fixed-window"]["status"] == "ready"

    arm = client.get(
        f"{V1}/documents/fifth-doc/analysis/methods/fixed-window/chunks?limit=500").json()
    assert arm["method"] == "fixed-window" and arm["engine"] == "fixed_window"
    assert arm["page"]["total"] == len(arm["items"]) >= 2
    assert all(":fw-chunk-" in row["chunk_id"] for row in arm["items"])
    assert all({"chunk_id", "text", "unit_ids", "token_count"} <= set(row) for row in arm["items"])


def test_a_method_can_be_added_to_an_existing_document_later(client, fifth):
    analysis.stage(doc_id="fifth-doc", label="Besinci.pdf", units=_corpus(),
                   methods=["structure-only"], content_sha="fifth-sha")
    analysis.state().queue.join()
    response = client.post(f"{V1}/documents/fifth-doc/analysis/methods",
                           json={"methods": ["fixed-window"]})
    assert response.status_code == 202, response.text
    analysis.state().queue.join()
    assert analysis.read_state("fifth-doc", "fifth-sha")["ready_methods"] == ["structure-only", "fixed-window"]


# --------------------------------------------- the Viewer needs no rebuild
def test_the_viewer_is_told_about_it_without_a_rebuild_of_anything(client, fifth):
    """The exposure rule: a registered method cannot be invisible in the
    Viewer because somebody forgot to rebuild something.

    It used to be a real risk. The Viewer was a built HTML page with the
    registry embedded in it at build time, and it only stayed current because
    the server it was served from re-read the registry at boot. There is no
    such page and no such server: the Viewer is a screen of the Next.js
    console, and its method chips come from this route at run time. Registering
    a method is therefore the whole of exposing it -- which is what is checked
    here, over the process's own application.

    The front end's half of the same rule -- that it writes no method key down
    anywhere -- is ``frontend/tests/catalogue.test.tsx``.
    """
    listed = client.get(f"{V1}/meta/chunking-methods").json()["items"]
    row = listed[-1]
    assert row["key"] == "fixed-window"
    assert row["label"] == M.METHODS["fixed-window"].label
    assert row["engine"] == M.METHODS["fixed-window"].engine


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
    analysis.state().queue.join()
    refused = client.get(f"{V1}/documents/fifth-doc/analysis/methods/fixed-window/chunks")
    assert refused.status_code == 400
    assert "unknown chunking method" in refused.json()["error"]["message"]


# ------------------------------------------------------ drift guards
#: The front end's own drift guards moved with the front end. A method key
#: written down in the console's source, and a picker that knows a real one,
#: are ``frontend/tests/catalogue.test.tsx``; a Flask-era path anywhere in it
#: is ``frontend/tests/surface.test.ts``. Both read that source, which is where
#: the risk is, and neither can be written here.


def test_the_viewer_builders_and_the_console_share_one_identity():
    from amsc.viewer import corpus as viewer_corpus
    from amsc.viewer import build as viewer_build

    assert dict(viewer_build.METHOD_LABELS) == {key: M.METHODS[key].label for key in registry.order()}
    assert dict(viewer_build.METHOD_SUMMARIES) == {key: M.METHODS[key].summary for key in registry.order()}
    assert dict(viewer_corpus.ARM_KINDS) == {key: M.METHODS[key].engine for key in registry.order()}
    assert set(M.ORDER) == set(viewer_build.METHOD_ORDER), "same universe, the console's own order"


def test_the_indexing_chunkers_are_one_table(client):
    from chat_rag.components.chunker import create_chunker
    from chat_rag.core.exceptions import ConfigurationException

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
