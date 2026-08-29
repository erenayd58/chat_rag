"""Packaging an ingested document for the Viewer v2.

The contract these tests hold to is the expensive one: the Viewer gets a real
analysis of a real document, and getting it costs no second parse and no
second model call. A Deep Analysis upload is reused exactly as it ran; a
Standard upload gets the deterministic quality contract, which never reaches a
provider at all.
"""

from __future__ import annotations

import json

import pytest

from components.viewer import analysis


# --- a small canonical corpus, built here so no fixture file can drift ------


def _unit(order, unit_id, unit_type, text, page, *, level=None, path=()):
    row = {
        "document_id": "probe-doc", "unit_id": unit_id, "order": order, "text": text,
        "type": unit_type, "section_path": list(path), "source": {"page": page, "block": order},
    }
    if level is not None:
        row.update(heading_level=level, semantic_role="section", opens_section=True)
    return row


def _corpus(sections=3, paragraphs=6):
    units, order = [], 0
    for section in range(1, sections + 1):
        order += 1
        title = f"{section}. BOLUM BASLIGI"
        units.append(_unit(order, f"h-{order:04d}", "heading", title, section, level=1, path=[title]))
        for para in range(paragraphs):
            order += 1
            body = (f"Bu {section}. bolumun {para + 1}. paragrafidir. " * 14).strip()
            units.append(_unit(order, f"p-{order:04d}", "paragraph", body, section, path=[title]))
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
    analysis._queue.join()
    with analysis._lock:
        analysis._inflight.clear()
    monkeypatch.setattr(analysis, "root", lambda: tmp_path / "viewer-live")
    yield tmp_path / "viewer-live"
    analysis._queue.join()
    with analysis._lock:
        analysis._inflight.clear()


def _stage_and_build(**overrides):
    fields = dict(doc_id="probe-doc", label="Probe belgesi", units=_corpus(),
                  kb_id="kb1", kb_name="probe-kb", chunking_mode="standard")
    fields.update(overrides)
    analysis.stage(**fields)
    # stage() queues the build; waiting on the queue is what a caller does,
    # and it keeps the worker and the test off the same document at once.
    analysis._queue.join()
    return analysis.read_state("probe-doc")


def _deep_run(units):
    """A DeepAnalysisResult of the kind an ingest hands over, made here."""
    from amsc.deep_pipeline import MODE_DEEP, DeepAnalysisSettings, chunk_document
    from amsc.models import RawDocumentUnit

    models = [RawDocumentUnit.model_validate(u) for u in units]
    result = chunk_document(models, mode=MODE_DEEP,
                            settings=DeepAnalysisSettings(use_llm=False, verify=False))
    assert result.deep is not None
    return result.deep


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

    import amsc.deep_pipeline as deep_pipeline

    def refuse(*args, **kwargs):  # pragma: no cover - the failure is the point
        raise AssertionError("Deep Analysis must not run a second time for the Viewer")

    monkeypatch.setattr(deep_pipeline, "chunk_document", refuse)
    analysis.stage(doc_id="probe-doc", label="Probe belgesi", units=units, deep_result=run,
                   kb_id="kb1", kb_name="probe-kb", chunking_mode="deep_analysis")
    analysis._queue.join()
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
    analysis._queue.join()
    assert analysis.read_state("probe-doc")["status"] == analysis.STATUS_READY
    assert analysis.payload("probe-doc") is not None


def test_a_build_failure_is_a_state_not_a_crash(workspace, monkeypatch):
    monkeypatch.setattr(analysis, "_build", lambda doc_id: (_ for _ in ()).throw(RuntimeError("boom")))
    analysis.stage(doc_id="probe-doc", label="Probe", units=_corpus())
    analysis._queue.join()
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

    monkeypatch.setattr(analysis, "_unit_resolver", resolver)
    state = analysis.request_build(doc_id="probe-doc", label="Eski belge", kb_id="kb1")
    assert state["status"] == analysis.STATUS_PENDING
    assert "unit_count" not in state, "the request writes no canonical of its own"

    analysis._queue.join()
    assert [(d, k) for d, k, _ in seen] == [("probe-doc", "kb1")]
    assert seen[0][2] != main_thread, "the recovery must not run on the request's thread"
    assert analysis.read_state("probe-doc")["status"] == analysis.STATUS_READY
    assert analysis.payload("probe-doc")["arms"]["structure-only"]["chunks"]


def test_a_document_with_no_recoverable_canonical_fails_clearly(workspace, monkeypatch):
    monkeypatch.setattr(analysis, "_unit_resolver", lambda doc_id, kb_id: None)
    analysis.request_build(doc_id="probe-doc", label="Eski belge")
    analysis._queue.join()
    state = analysis.read_state("probe-doc")
    assert state["status"] == analysis.STATUS_FAILED
    assert "structured parser" in state["error"]
