"""Packaging an ingested document for the Viewer.

The contract these tests hold to is the expensive one: the Viewer gets a real
analysis of a real document, and getting it costs no second parse and no
second model call. A Deep Analysis upload is reused exactly as it ran; a
Standard upload gets the deterministic quality contract, which never reaches a
provider at all.
"""

from __future__ import annotations

import json

import pytest

from chat_rag.components.viewer import analysis


# --- a small canonical corpus, built here so no fixture file can drift ------


def _unit(order, unit_id, unit_type, text, page, *, level=None, path=(),
          document_id="probe-doc"):
    row = {
        "document_id": document_id, "unit_id": unit_id, "order": order, "text": text,
        "type": unit_type, "section_path": list(path), "source": {"page": page, "block": order},
    }
    if level is not None:
        row.update(heading_level=level, semantic_role="section", opens_section=True)
    return row


def _corpus(sections=3, paragraphs=6, document_id="probe-doc"):
    units, order = [], 0
    for section in range(1, sections + 1):
        order += 1
        title = f"{section}. BOLUM BASLIGI"
        units.append(_unit(order, f"h-{order:04d}", "heading", title, section, level=1,
                           path=[title], document_id=document_id))
        for para in range(paragraphs):
            order += 1
            body = (f"Bu {section}. bolumun {para + 1}. paragrafidir. " * 14).strip()
            units.append(_unit(order, f"p-{order:04d}", "paragraph", body, section,
                               path=[title], document_id=document_id))
    return units


@pytest.fixture(autouse=True)
def workspace(tmp_path, monkeypatch):
    """An analysis root of this test's own, and a worker that starts and ends
    each test idle.

    The packaging worker is a module-level singleton every test shares. It
    must be drained *before* this fixture's redirect is undone -- a job still
    running when ``root`` goes back to its real value would write into the
    repository -- which is why the drain lives here and not in a fixture that
    tears down later.
    """
    analysis.state().queue.join()
    with analysis.state().lock:
        analysis.state().inflight.clear()
    monkeypatch.setattr(analysis, "root", lambda: tmp_path / "viewer-live")
    yield tmp_path / "viewer-live"
    analysis.state().queue.join()
    with analysis.state().lock:
        analysis.state().inflight.clear()


def _stage_and_build(**overrides):
    fields = dict(doc_id="probe-doc", label="Probe belgesi", units=_corpus(),
                  kb_id="kb1", kb_name="probe-kb", chunking_mode="standard")
    fields.update(overrides)
    analysis.stage(**fields)
    # stage() queues the build; waiting on the queue is what a caller does,
    # and it keeps the worker and the test off the same document at once.
    analysis.state().queue.join()
    return analysis.read_state("probe-doc")


def _deep_run(units):
    """A DeepAnalysisResult of the kind an ingest hands over, made here."""
    from amsc.deep.pipeline import MODE_DEEP, DeepAnalysisSettings, chunk_document
    from amsc.document.models import RawDocumentUnit

    models = [RawDocumentUnit.model_validate(u) for u in units]
    result = chunk_document(models, mode=MODE_DEEP,
                            settings=DeepAnalysisSettings(use_llm=False, verify=False))
    assert result.deep is not None
    return result.deep


# --- one document, several variants ----------------------------------------


def test_one_upload_runs_every_selected_method_over_one_canonical(workspace):
    """Three methods, one parse. The canonical is written once and reused."""
    analysis.stage(doc_id="multi", label="Cok yontemli", units=_corpus(),
                   methods=["markdown", "structure-only", "agentic"],
                   kb_id="kb1", kb_name="probe-kb", content_sha="c0ffee")
    analysis.state().queue.join()
    state = analysis.read_state("multi", "c0ffee")
    assert state["status"] == analysis.STATUS_READY
    assert state["ready_methods"] == ["markdown", "structure-only", "agentic"]

    payload = analysis.payload("multi", "c0ffee")
    assert sorted(payload["arms"]) == ["agentic", "markdown", "structure-only"]
    # Each arm is a real chunking of the same units, not a copy of another.
    counts = {arm: len(payload["arms"][arm]["chunks"]) for arm in payload["arms"]}
    assert all(count > 0 for count in counts.values()), counts
    assert payload["arms"]["markdown"]["chunks"] != payload["arms"]["structure-only"]["chunks"]
    # One parse: exactly one canonical file for the whole document.
    key = analysis.key_for("multi", "c0ffee")
    assert list(analysis.document_dir(key).glob("*.jsonl")) == [analysis.units_path(key)]


def test_only_the_selected_methods_are_produced(workspace):
    """A method that was not asked for is absent -- never invented."""
    analysis.stage(doc_id="one", label="Tek yontem", units=_corpus(),
                   methods=["structure-only"], content_sha="beef")
    analysis.state().queue.join()
    payload = analysis.payload("one", "beef")
    assert list(payload["arms"]) == ["structure-only"]
    assert payload["meta"]["deep"] is None, "no Deep panel for a document with no Deep variant"
    assert payload["live"]["methods"]["agentic"]["status"] == analysis.STATUS_MISSING


def test_the_same_bytes_are_one_document_however_often_they_are_uploaded(workspace):
    """A second upload of the same PDF enriches the document it already is.

    One directory, one canonical, one variant per method -- and each upload
    keeps its own choice of which of them it is asking about.
    """
    analysis.stage(doc_id="first", label="Ayni belge.pdf", units=_corpus(),
                   methods=["structure-only"], content_sha="same-bytes")
    analysis.state().queue.join()
    analysis.stage(doc_id="second", label="Ayni belge.pdf", units=_corpus(),
                   methods=["markdown"], content_sha="same-bytes")
    analysis.state().queue.join()

    key = analysis.key_for("first", "same-bytes")
    assert key == analysis.key_for("second", "same-bytes")
    directories = [d.name for d in workspace.iterdir() if d.is_dir()]
    assert len(directories) == 1, f"the same PDF made {len(directories)} documents: {directories}"

    # Content level: the second upload's method joined the first's, and both
    # variants are on disk, built once.
    state = analysis._read_state_file(key)
    assert state["ready_methods"] == ["markdown", "structure-only"]
    assert analysis.chunks_path(key, "markdown") is not None
    assert analysis.chunks_path(key, "structure-only") is not None

    # Upload level: each console record answers for what it asked for.
    assert list(analysis.payload("second", "same-bytes")["arms"]) == ["markdown"]
    assert list(analysis.payload("first", "same-bytes")["arms"]) == ["structure-only"]

    # Both console records point at the one analysis.
    assert sorted(analysis.states()) == ["first", "second"]
    assert analysis.states()["first"]["key"] == analysis.states()["second"]["key"]


def test_two_files_with_one_name_stay_two_documents(workspace):
    """Identity is the content, so a different PDF is a different document."""
    analysis.stage(doc_id="a", label="rapor.pdf", units=_corpus(sections=2),
                   methods=["structure-only"], content_sha="aaa")
    analysis.stage(doc_id="b", label="rapor.pdf", units=_corpus(sections=3),
                   methods=["structure-only"], content_sha="bbb")
    analysis.state().queue.join()
    assert analysis.key_for("a", "aaa") != analysis.key_for("b", "bbb")
    assert len(analysis.payload("a", "aaa")["pages"]) == 2
    assert len(analysis.payload("b", "bbb")["pages"]) == 3


def test_a_variant_can_be_added_later_without_reparsing(workspace):
    """Scenario 3: ask for another method; nothing is parsed or rebuilt."""
    analysis.stage(doc_id="grow", label="Buyuyen belge", units=_corpus(),
                   methods=["structure-only"], content_sha="grow1")
    analysis.state().queue.join()
    key = analysis.key_for("grow", "grow1")
    before = analysis.units_path(key).stat().st_mtime_ns
    markdown_before = analysis.variant_dir(key, "markdown").exists()

    analysis.add_methods("grow", ["markdown"], "grow1")
    analysis.state().queue.join()

    assert not markdown_before
    assert analysis.units_path(key).stat().st_mtime_ns == before, "the canonical was rewritten"
    payload = analysis.payload("grow", "grow1")
    assert sorted(payload["arms"]) == ["markdown", "structure-only"]
    assert analysis.read_state("grow", "grow1")["ready_methods"] == ["markdown", "structure-only"]


def test_deleting_one_upload_keeps_the_others_analysis(workspace):
    analysis.stage(doc_id="first", label="Ayni.pdf", units=_corpus(),
                   methods=["structure-only"], content_sha="shared")
    analysis.stage(doc_id="second", label="Ayni.pdf", units=_corpus(),
                   methods=["structure-only"], content_sha="shared")
    analysis.state().queue.join()
    analysis.discard("first", "shared")
    assert analysis.payload("second", "shared") is not None
    analysis.discard("second", "shared")
    assert analysis.payload("second", "shared") is None


def test_a_variant_that_cannot_run_is_recorded_not_faked(workspace, monkeypatch):
    """A chunker that fails leaves a failed variant and no arm at all."""
    real = analysis._chunk_rows

    def explode(method, units):
        if method == "markdown":
            raise RuntimeError("no markdown for you")
        return real(method, units)

    monkeypatch.setattr(analysis, "_chunk_rows", explode)
    analysis.stage(doc_id="partial", label="Kismi", units=_corpus(),
                   methods=["markdown", "structure-only"], content_sha="partial")
    analysis.state().queue.join()
    state = analysis.read_state("partial", "partial")
    assert state["status"] == analysis.STATUS_READY
    assert state["ready_methods"] == ["structure-only"]
    assert state["failed_methods"] == ["markdown"]
    assert list(analysis.payload("partial", "partial")["arms"]) == ["structure-only"]


# --- what the Viewer gets --------------------------------------------------


def test_a_standard_upload_becomes_a_viewable_document(workspace):
    state = _stage_and_build()
    assert state["status"] == analysis.STATUS_READY

    payload = analysis.payload("probe-doc")
    assert sorted(payload["arms"]) == ["agentic", "structure-only"], (
        "the Viewer's Standard-vs-Deep comparison needs both arms"
    )
    assert payload["units"] and payload["pages"] == [1, 2, 3]
    assert payload["story"]["sections"], "Debug reads the per-section decision story"
    assert payload["arms"]["agentic"]["sq"]["chunk_count"] > 0, "structural quality is measured"
    assert payload["live"]["docId"] == "probe-doc"
    assert payload["live"]["kbName"] == "probe-kb"


def test_a_standard_upload_costs_no_model_call(workspace):
    """The Deep side of the comparison is the free deterministic contract."""
    _stage_and_build()
    deep = analysis.payload("probe-doc")["meta"]["deep"]
    assert deep["calls"] == {"proposer": 0, "verifier": 0, "total": 0}
    assert deep["estCostUsd"] == 0
    assert deep["status"] == "deterministic"
    assert analysis.read_state("probe-doc")["deep_source"] == analysis.SOURCE_DETERMINISTIC


def test_a_deep_upload_is_reused_and_never_run_again(workspace, monkeypatch):
    """The one guarantee worth a test of its own: staging a run means the
    packager must not chunk the document a second time."""
    units = _corpus()
    run = _deep_run(units)

    import amsc.deep.pipeline as deep_pipeline

    def refuse(*args, **kwargs):  # pragma: no cover - the failure is the point
        raise AssertionError("Deep Analysis must not run a second time for the Viewer")

    monkeypatch.setattr(deep_pipeline, "chunk_document", refuse)
    analysis.stage(doc_id="probe-doc", label="Probe belgesi", units=units, deep_result=run,
                   kb_id="kb1", kb_name="probe-kb", chunking_mode="deep_analysis")
    analysis.state().queue.join()
    state = analysis.read_state("probe-doc")

    assert state["status"] == analysis.STATUS_READY
    assert state["deep_source"] == analysis.SOURCE_INGEST
    payload = analysis.payload("probe-doc")
    assert payload["arms"]["agentic"]["chunks"], "the ingest's own deep rows reached the Viewer"


def test_the_canonical_is_the_one_the_ingest_chunked(workspace):
    """A tree pinned to a different canonical is what the Viewer refuses; the
    units written here are the ingest's own, so the pin holds."""
    _stage_and_build()
    rows = [json.loads(line) for line in
            analysis.units_path("probe-doc").read_text(encoding="utf-8").splitlines() if line.strip()]
    assert [r["unit_id"] for r in rows] == [u["unit_id"] for u in _corpus()]
    manifest = json.loads((analysis.run_dir("probe-doc") / "arm" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["units_file"] == "probe-doc/units.jsonl"
    assert manifest["arm_kind"] == "deep_analysis"


# --- lifecycle -------------------------------------------------------------


def test_deleting_a_document_drops_only_its_own_analysis(workspace):
    _stage_and_build()
    (workspace / "keepsake.json").write_text("{}", encoding="utf-8")
    assert analysis.discard("probe-doc") is True
    assert analysis.read_state("probe-doc")["status"] == analysis.STATUS_MISSING
    assert (workspace / "keepsake.json").is_file(), "nothing outside the document's own directory"
    assert analysis.discard("probe-doc") is False


def test_a_document_id_cannot_escape_the_analysis_root(workspace):
    assert ".." not in analysis._safe_id("../../etc/passwd")
    assert analysis.document_dir("../../etc/passwd").parent == workspace


def test_an_interrupted_build_is_resumed_from_disk(workspace):
    """A restart mid-build leaves a record and every input it needs; the next
    look queues it again rather than losing the document."""
    _stage_and_build()
    analysis.payload_path("probe-doc").unlink()

    assert analysis.read_state("probe-doc")["status"] == analysis.STATUS_PENDING
    assert analysis.resume_incomplete() == ["probe-doc"]
    analysis.state().queue.join()
    assert analysis.read_state("probe-doc")["status"] == analysis.STATUS_READY
    assert analysis.payload("probe-doc") is not None


def test_a_build_failure_is_a_state_not_a_crash(workspace, monkeypatch):
    monkeypatch.setattr(analysis, "_build", lambda doc_id: (_ for _ in ()).throw(RuntimeError("boom")))
    analysis.stage(doc_id="probe-doc", label="Probe", units=_corpus())
    analysis.state().queue.join()
    state = analysis.read_state("probe-doc")
    assert state["status"] == analysis.STATUS_FAILED
    assert "boom" in state["error"]


def test_a_document_ingested_earlier_is_recovered_on_the_worker(workspace, monkeypatch):
    """Catch-up path: the request records what to build, and the canonical is
    recovered where it belongs -- off the request, on the worker."""
    import threading

    seen = []
    main_thread = threading.current_thread().ident

    def resolver(doc_id, kb_id):
        seen.append((doc_id, kb_id, threading.current_thread().ident))
        return _corpus()

    monkeypatch.setattr(analysis.state(), "unit_resolver", resolver)
    state = analysis.request_build(doc_id="probe-doc", label="Eski belge", kb_id="kb1")
    assert state["status"] == analysis.STATUS_PENDING
    assert "unit_count" not in state, "the request writes no canonical of its own"

    analysis.state().queue.join()
    assert [(d, k) for d, k, _ in seen] == [("probe-doc", "kb1")]
    assert seen[0][2] != main_thread, "the recovery must not run on the request's thread"
    assert analysis.read_state("probe-doc")["status"] == analysis.STATUS_READY
    assert analysis.payload("probe-doc")["arms"]["structure-only"]["chunks"]


def test_a_document_with_no_recoverable_canonical_fails_clearly(workspace, monkeypatch):
    monkeypatch.setattr(analysis.state(), "unit_resolver", lambda doc_id, kb_id: None)
    analysis.request_build(doc_id="probe-doc", label="Eski belge")
    analysis.state().queue.join()
    state = analysis.read_state("probe-doc")
    assert state["status"] == analysis.STATUS_FAILED
    assert "structured parser" in state["error"]


# --- one document, several uploads of it -----------------------------------
#
# The analysis directory is the content, but a Deep run is produced by one
# upload's ingest and carries that upload's id. The two identities must not be
# allowed to disagree, and one arm that cannot be built must not take the
# document's other arms with it.


def test_the_same_pdf_uploaded_again_still_packages_every_method(workspace):
    """The same bytes under a new upload id are still one working document.

    The second upload lands on the first upload's canonical while bringing a
    Deep run stamped with its own id. That disagreement used to abort the
    build at the Deep step -- and take Markdown, which has nothing to do with
    Deep, down with it -- leaving a document that advertised arms the Viewer
    could not then fetch.
    """
    wanted = ["markdown", "structure-only", "agentic"]
    first = _corpus(document_id="upload-1")
    analysis.stage(doc_id="upload-1", label="Ayni.pdf", units=first, methods=wanted,
                   deep_result=_deep_run(first), content_sha="same-bytes")
    analysis.state().queue.join()

    second = _corpus(document_id="upload-2")
    analysis.stage(doc_id="upload-2", label="Ayni.pdf", units=second, methods=wanted,
                   deep_result=_deep_run(second), content_sha="same-bytes")
    analysis.state().queue.join()

    key = analysis.key_for("upload-2", "same-bytes")
    state = analysis.read_state("upload-2", "same-bytes")
    assert state["status"] == analysis.STATUS_READY, state.get("error")
    assert state["failed_methods"] == []
    assert [d.name for d in workspace.iterdir() if d.is_dir()] == [key]

    # Deep packaged: the run was adopted under the canonical's identity rather
    # than refused for carrying the second upload's name.
    assert state["methods"]["agentic"]["status"] == analysis.STATUS_READY
    run_summary = json.loads((analysis.run_dir(key) / "summary.json").read_text(encoding="utf-8"))
    assert run_summary["document_id"] == "upload-1", "the run answers to the canonical"

    # Every method that was asked for is ready, and ready means fetchable.
    assert state["ready_methods"] == wanted
    for method in state["ready_methods"]:
        assert analysis.chunks_path(key, method) is not None, f"{method} says ready with no chunks"
        assert analysis.chunk_rows("upload-2", method, "same-bytes"), f"{method} serves no rows"


def test_a_deep_failure_leaves_the_other_methods_standing(workspace, monkeypatch):
    """One arm failing is a state, not the end of the document.

    Markdown does not depend on Deep and must survive it. Standard does --
    Deep's packaging is what writes it -- so with Deep gone it is chunked on
    its own instead of being advertised and missing.
    """
    import amsc.deep.arm as deep_arm

    def refuse(*args, **kwargs):
        raise ValueError("the deep tree was built for someone else")

    monkeypatch.setattr(deep_arm, "package", refuse)
    units = _corpus()
    analysis.stage(doc_id="probe-doc", label="Probe belgesi", units=units,
                   methods=["markdown", "structure-only", "agentic"],
                   deep_result=_deep_run(units))
    analysis.state().queue.join()

    key = analysis.key_for("probe-doc")
    state = analysis.read_state("probe-doc")
    assert state["status"] == analysis.STATUS_READY, "a failed arm is not a failed document"
    assert state["methods"]["agentic"]["status"] == analysis.STATUS_FAILED
    assert state["failed_methods"] == ["agentic"]

    assert state["ready_methods"] == ["markdown", "structure-only"]
    assert analysis.chunks_path(key, "agentic") is None, "nothing packaged, nothing advertised"
    for method in state["ready_methods"]:
        assert analysis.chunk_rows("probe-doc", method), f"{method} serves no rows"
