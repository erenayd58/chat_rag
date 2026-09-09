"""The chunking-method wire contract, pinned where five layers meet.

One set of strings -- ``markdown``, ``structure-only``, ``agentic``,
``hybrid`` -- travels from the registry in ``components/viewer/methods.py``
through ``GET /api/v1/meta/chunking-methods``, the upload's repeated
``methods`` field, the packaged variant directories and the Viewer screen's
method chips. Nothing tested it directly; renaming a key or reordering
``ORDER`` would keep every existing test green and break the product.

These tests pin *observable* behaviour: the keys, labels and engines the API
reports, how a request's selection is normalised, what an upload does with
it, and the one invariant that keeps two registries apart -- an analysis
method is never an indexing chunker. They do not pin summaries or the shape
of the dataclass, which a registry refactor is free to change.

They also do not pin *how many* methods there are. The four keys below are a
contract because renaming one breaks the API, the form and a packaged variant
directory at once; the registry holding exactly those four is not, because
adding a fifth is the supported extension path
(``chunk/docs/adding-a-chunker.md``). So every list here is either a
statement about these four specifically, or derived from ``M.ORDER`` -- and a
valid new method needs no edit to this file. ``test_chunker_extension.py``
proves that by registering one.
"""

from __future__ import annotations

import io
from types import SimpleNamespace

import pytest

from fastapi.testclient import TestClient

import asgi as entrypoint
import interfaces.http as http

V1 = http.v1.PREFIX
from application import workspace as app_workspace
from components.chunker import create_chunker
from components.knowledgebase.manager import KnowledgeBaseManager, normalize_chunker_config
from components.viewer import methods as M
from core.exceptions import ConfigurationException

WIRE_KEYS = ("markdown", "structure-only", "agentic", "hybrid")
WIRE_ENGINES = {
    "markdown": "markdown_recursive",
    "structure-only": "structure_first",
    "agentic": "deep_analysis",
    "hybrid": "hybrid_h1",
}
WIRE_LABELS = {
    "markdown": "Markdown",
    "structure-only": "Standard",
    "agentic": "Deep Analysis",
    "hybrid": "Hybrid",
}


@pytest.fixture
def hybrid_available(monkeypatch):
    monkeypatch.setattr(M, "embedder_available", lambda: (True, ""))


@pytest.fixture
def hybrid_unavailable(monkeypatch):
    monkeypatch.setattr(M, "embedder_available", lambda: (False, "model not on this machine"))


# ------------------------------------------------------------ the registry


def test_the_wire_keys_their_order_labels_and_engines_are_pinned():
    """The four shipped keys, their engines, their labels and the order the
    console lists them in. Not an equality against the registry: a fifth
    method registered beside them is valid and is listed after them."""
    assert set(WIRE_KEYS) <= set(M.METHODS)
    assert [key for key in M.ORDER if key in WIRE_KEYS] == list(WIRE_KEYS)
    assert {key: M.METHODS[key].engine for key in WIRE_KEYS} == WIRE_ENGINES
    assert {key: M.METHODS[key].label for key in WIRE_KEYS} == WIRE_LABELS
    assert M.DEFAULT_SELECTION == ("structure-only",)
    assert (M.STANDARD, M.DEEP, M.MARKDOWN, M.HYBRID) == (
        "structure-only", "agentic", "markdown", "hybrid",
    )


def test_only_deep_uses_a_model_and_only_hybrid_needs_an_embedder():
    """Of the shipped four -- and whatever else is registered, the console
    reports the library's own answer rather than one of its own."""
    from amsc.chunking import registry

    assert [key for key in WIRE_KEYS if M.METHODS[key].uses_model] == ["agentic"]
    assert [key for key in WIRE_KEYS if M.METHODS[key].needs_embedder] == ["hybrid"]
    for key in M.ORDER:
        entry = registry.get(key)
        assert M.METHODS[key].uses_model == entry.uses_model, key
        assert M.METHODS[key].needs_embedder == entry.needs_embedder, key


def test_normalise_accepts_a_list_a_comma_string_or_nothing(hybrid_available):
    assert M.normalise(None) == ["structure-only"]
    assert M.normalise([]) == ["structure-only"]
    assert M.normalise("") == ["structure-only"]
    assert M.normalise(["markdown"]) == ["markdown"]
    assert M.normalise("markdown, agentic") == ["markdown", "agentic"]
    assert M.normalise("hybrid;markdown") == ["markdown", "hybrid"]


def test_normalise_splits_a_separator_wherever_it_appears(hybrid_available):
    """The shape the FastAPI route actually produces.

    ``methods: list[str]`` means one comma-separated form field arrives as a
    *one-element list containing a comma*, not as a string -- so a version that
    only split bare strings never split anything, and the whole selection was
    silently replaced by the fallback. Splitting is applied per item now, which
    is why the list forms below and the string forms above agree.
    """
    assert M.normalise(["markdown,agentic"]) == ["markdown", "agentic"]
    assert M.normalise(["markdown;agentic"]) == ["markdown", "agentic"]
    assert M.normalise([" markdown , agentic "]) == ["markdown", "agentic"]
    # Mixed: one repeated field that itself lists two. The answer is in display
    # order, as it is for every other input -- derived from ORDER rather than
    # written out, so adding a method cannot make this a false statement.
    mixed = ["structure-only", "markdown", "agentic"]
    assert M.normalise(["structure-only", "markdown,agentic"]) == [
        key for key in M.ORDER if key in mixed]


def test_normalise_leaves_a_plain_repeated_field_alone(hybrid_available):
    """The spelling the browser uses must be untouched by the split above: a
    name with no separator in it splits to itself."""
    assert M.normalise(["markdown"]) == ["markdown"]
    assert M.normalise(["markdown", "agentic"]) == ["markdown", "agentic"]
    assert M.normalise(["hybrid", "agentic", "markdown", "structure-only"]) == list(WIRE_KEYS)


def test_normalise_ignores_the_empty_pieces_a_separator_leaves(hybrid_available):
    """A trailing comma, a doubled one, or a field that is only separators.
    None of them is a method name, and none of them may become one."""
    assert M.normalise(["markdown,"]) == ["markdown"]
    assert M.normalise(["markdown,,agentic"]) == ["markdown", "agentic"]
    assert M.normalise([","]) == ["structure-only"], "nothing usable falls back to Standard"
    assert M.normalise([" "]) == ["structure-only"]


def test_normalise_reorders_to_display_order_and_drops_unknown_names(hybrid_available):
    assert M.normalise(["hybrid", "agentic", "markdown", "structure-only"]) == list(WIRE_KEYS)
    assert M.normalise(["turbo", "markdown"]) == ["markdown"]
    assert M.normalise(["turbo"]) == ["structure-only"], "nothing usable falls back to Standard"
    assert M.normalise(["Markdown"]) == ["structure-only"], "keys are case-sensitive wire ids"


def test_an_unavailable_method_is_dropped_rather_than_half_honoured(hybrid_unavailable):
    assert M.normalise(["hybrid", "markdown"]) == ["markdown"]
    assert M.normalise(["hybrid"]) == ["structure-only"]
    assert M.resolve("hybrid").available is False
    assert M.resolve("hybrid").reason == "model not on this machine"
    assert [m.key for m in M.offered()] == [k for k in M.ORDER if k != "hybrid"]


def test_the_catalogue_reports_every_method_offered_or_not(hybrid_unavailable):
    rows = M.catalogue()
    assert [row["key"] for row in rows] == list(M.ORDER)
    assert set(WIRE_KEYS) <= {row["key"] for row in rows}
    for row in rows:
        assert set(row) == {"key", "label", "summary", "engine", "available", "reason", "uses_model", "default"}
        assert row["available"] or row["reason"], "unavailable carries its reason"
    assert next(row for row in rows if row["key"] == "hybrid")["available"] is False


# ------------------------------------------------------------- the API


@pytest.fixture
def client():
    with TestClient(http.create_app(entrypoint.services),
                    raise_server_exceptions=False) as test_client:
        yield test_client


def test_the_meta_route_is_the_catalogue_verbatim(client, hybrid_available):
    body = client.get(f"{V1}/meta/chunking-methods").json()
    assert [row["key"] for row in body["items"]] == list(M.ORDER)
    reported = {row["key"]: row for row in body["items"]}
    assert {key: reported[key]["engine"] for key in WIRE_KEYS} == WIRE_ENGINES
    assert {key: reported[key]["label"] for key in WIRE_KEYS} == WIRE_LABELS


# ------------------------------------------------------- the upload route


class _Chunker:
    """Deep-capable, like the structure-first chunker the product runs."""

    last_canonical_units = None
    last_deep_result = None

    def get_name(self):
        return "StructuralChunker"

    def chunk_text_deep(self, *args, **kwargs):  # pragma: no cover - never reached
        raise AssertionError("the route must not chunk directly")


class _Pipeline:
    def __init__(self):
        self.settings = SimpleNamespace(deep_analysis_api_key_env="FAKE_DEEP_KEY")
        self.chunker = _Chunker()
        self.last_deep_analysis_report = None
        self.seen_deep_analysis = None

    def ingest_document_from_file(self, file_path, doc_title, deep_analysis=False):
        self.seen_deep_analysis = deep_analysis
        return [SimpleNamespace(doc_id="doc-under-test")]


@pytest.fixture
def upload(tmp_path, monkeypatch, client):
    """A client, a knowledge base of its own, a stub pipeline and a recorder
    for what the route hands the Viewer packager."""
    monkeypatch.chdir(tmp_path)
    manager = KnowledgeBaseManager(str(tmp_path / "kbs.json"))
    monkeypatch.setattr(entrypoint.services, "kb_manager", manager)
    kb = manager.create("wire-kb", chunker={"type": "structure_first"})
    pipeline = _Pipeline()
    monkeypatch.setattr(entrypoint.services, "get_pipeline", lambda *a, **k: pipeline)
    staged = []
    monkeypatch.setattr(
        app_workspace, "stage_analysis",
        lambda doc_id, **kw: staged.append(dict(kw, doc_id=doc_id)) or {"status": "pending"},
    )

    def post(*fields):
        """One upload, and what it left behind.

        The form is repeated ``methods`` fields, which is what a browser sends
        and what the contract declares; the answer is the job, so what a test
        reads is the settled job rather than a synchronous body.
        """
        data: dict = {"knowledge_base_id": kb["kb_id"]}
        for name, value in fields:
            data.setdefault(name, []).append(value)
        response = client.post(
            f"{V1}/documents",
            files={"file": ("belge.txt", io.BytesIO(b"kucuk bir test belgesi"), "text/plain")},
            data=data,
        )
        assert response.status_code == 202, response.text
        manager = entrypoint.services.ingest_jobs
        job = manager.get(response.json()["id"])
        assert manager.wait(job, 30), "the job did not settle"
        settled = job.snapshot()
        assert settled["status"] == "succeeded", settled
        return settled["result"], staged[-1], pipeline

    return SimpleNamespace(post=post, kb_id=kb["kb_id"], manager=manager)


def test_the_repeated_methods_field_selects_the_analysis_arms(upload, hybrid_available):
    body, staged, pipeline = upload.post(("methods", "hybrid"), ("methods", "markdown"))
    assert staged["methods"] == ["markdown", "hybrid"], "normalised to display order"
    assert body["chunking_mode"] == "standard"
    assert pipeline.seen_deep_analysis is False


def test_selecting_deep_analysis_flips_the_whole_ingest_to_deep(upload):
    body, staged, pipeline = upload.post(("methods", "markdown"), ("methods", "agentic"))
    assert staged["methods"] == ["markdown", "agentic"]
    assert body["chunking_mode"] == "deep_analysis"
    assert pipeline.seen_deep_analysis is True
    assert staged["chunking_mode"] == "deep_analysis"


def test_no_methods_means_standard(upload):
    body, staged, pipeline = upload.post()
    assert staged["methods"] == ["structure-only"]
    assert body["chunking_mode"] == "standard"
    assert pipeline.seen_deep_analysis is False


def test_the_wire_format_is_repeated_fields(upload):
    """What the browser sends, and the spelling everything else is measured
    against: one ``methods`` field per method."""
    body, staged, pipeline = upload.post(("methods", "agentic"), ("methods", "markdown"))
    assert staged["methods"] == ["markdown", "agentic"]
    assert body["chunking_mode"] == "deep_analysis" and pipeline.seen_deep_analysis is True


def test_one_comma_joined_field_is_the_other_documented_spelling(upload):
    """``docs/api-v1.md``: "repeated, or one comma-separated field".

    This test used to assert the opposite -- that the comma form was *not*
    part of the contract -- on the reasoning that the route read the form with
    ``getlist``. That was the Flask route. The FastAPI one declares
    ``methods: list[str]``, so a single comma-joined field arrives as the
    one-element list ``["agentic,markdown"]``: a sequence, so the branch in
    ``normalise`` that split strings never ran, neither name matched, and the
    selection fell back to Standard.

    The failure was silent, which is what makes it worth a test rather than a
    docs edit. A client using the documented spelling got a successful 202 for
    an upload analysed with a method it did not ask for, and nothing anywhere
    said so.
    """
    body, staged, pipeline = upload.post(("methods", "agentic,markdown"))
    assert staged["methods"] == ["markdown", "agentic"], (
        "the comma-separated spelling the contract documents selects both"
    )
    assert body["chunking_mode"] == "deep_analysis" and pipeline.seen_deep_analysis is True


def test_the_two_spellings_agree(upload):
    """Neither is a dialect of the other: the same selection, written either
    way, has to reach the pipeline as the same list."""
    _, repeated, _ = upload.post(("methods", "markdown"), ("methods", "agentic"))
    _, joined, _ = upload.post(("methods", "markdown,agentic"))
    assert repeated["methods"] == joined["methods"]


def test_a_comma_joined_field_drops_unknown_names_like_any_other(upload):
    """Splitting is not admitting: what comes out of it goes through the same
    registry check, so a name nobody registered is dropped and does not take
    the valid names beside it with it."""
    _, staged, _ = upload.post(("methods", "markdown, turbo ,agentic"))
    assert staged["methods"] == ["markdown", "agentic"], (
        "surrounding spaces are trimmed and the unknown name is dropped alone"
    )


def test_an_unavailable_method_is_dropped_from_the_upload(upload, hybrid_unavailable):
    _, staged, _ = upload.post(("methods", "hybrid"), ("methods", "markdown"))
    assert staged["methods"] == ["markdown"]


def test_deep_analysis_is_a_method_key_and_not_a_flag_beside_the_list(upload):
    """There used to be a second spelling -- a ``deep_analysis`` boolean that
    replaced the selection -- for the Flask-era upload form. That form is gone
    and so is the flag: Deep Analysis is asked for by its registry key, like
    every other method, and a leftover boolean selects nothing."""
    body, staged, pipeline = upload.post(("deep_analysis", "true"))
    assert staged["methods"] == ["structure-only"], "an unknown field selects nothing"
    assert body["chunking_mode"] == "standard" and pipeline.seen_deep_analysis is False

    body, staged, pipeline = upload.post(("methods", "agentic"))
    assert staged["methods"] == ["agentic"]
    assert body["chunking_mode"] == "deep_analysis" and pipeline.seen_deep_analysis is True


def test_the_response_and_ledger_speak_in_modes_not_method_keys(upload):
    body, _, _ = upload.post(("methods", "markdown"), ("methods", "agentic"))
    assert body["chunking_mode"] in {"standard", "deep_analysis"}
    from utils import DocumentTracker

    (record,) = DocumentTracker().ingested_docs.values()
    assert record["chunking_mode"] in {"standard", "deep_analysis"}
    assert record["chunking_mode"] not in M.METHODS


# ------------------------------- analysis methods never index anything


def test_analysis_methods_never_change_the_knowledge_bases_chunker(upload, hybrid_available):
    before = dict(upload.manager.get(upload.kb_id)["chunker"])
    chunker_before = None
    for selection in ([("methods", "markdown")], [("methods", "agentic")],
                      [("methods", "hybrid"), ("methods", "markdown"), ("methods", "agentic")]):
        _, _, pipeline = upload.post(*selection)
        chunker_before = chunker_before or pipeline.chunker
        assert pipeline.chunker is chunker_before, "the pipeline's indexing chunker is untouched"
    assert upload.manager.get(upload.kb_id)["chunker"] == before
    assert before == {"type": "structure_first", "params": {}}


def _analysis_ids() -> list[str]:
    """Every registered analysis method's key and engine kind -- read from the
    registry, so a method added later is checked too."""
    from amsc.chunking import registry

    return sorted({name for method in registry.methods()
                   for name in (method.key, method.kind)})


@pytest.mark.parametrize("name", _analysis_ids())
def test_an_analysis_method_key_or_engine_is_not_an_indexing_chunker(name):
    """The indexing registry (v4 / structure_first) must refuse every
    analysis-method id and every engine name, so a future shared registry
    cannot quietly turn a Viewer selection into a retrieval change."""
    if name == "structure_first":
        pytest.skip("structure_first is the one legitimate indexing name the engine label reuses")
    settings = SimpleNamespace(chunker_type=name)
    with pytest.raises(ConfigurationException):
        create_chunker(settings)
    with pytest.raises(ValueError):
        normalize_chunker_config({"type": name})


def test_the_indexing_registry_still_resolves_every_alias_it_documents():
    for alias in ("v4", "frozenv4chunker", "frozen_v4_chunker"):
        assert normalize_chunker_config({"type": alias})["type"] == "v4"
    for alias in ("structure_first", "structurefirst", "structural", "structuralchunker", "structural_chunker"):
        assert normalize_chunker_config({"type": alias})["type"] == "structure_first"


# ------------------------------------------------- the chunk-side copies


def test_the_viewer_reader_and_page_agree_with_the_console_registry():
    """The library's Viewer reader and page builder and this console all read
    one registry (``amsc.chunking.registry``) now, so this can no longer drift by
    accident; it is kept as the guard that says so. Read from
    ``viewer_corpus``, the shared reader, rather than from the legacy v2
    page that merely re-exports it."""
    from amsc.viewer import corpus as viewer_corpus
    from amsc.viewer import build as viewer_v3

    for key in M.ORDER:
        assert viewer_corpus.ARM_KINDS[key] == M.METHODS[key].engine, key
    assert set(viewer_v3.METHOD_ORDER) == set(M.ORDER)
    assert {key: viewer_v3.METHOD_LABELS[key] for key in WIRE_KEYS} == WIRE_LABELS
    assert set(viewer_v3.METHOD_SUMMARIES) == set(M.ORDER)
