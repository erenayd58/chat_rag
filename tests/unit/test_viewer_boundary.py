"""The Viewer service boundary: who owns what, and what survives a failure.

Everything the browser sees about a live document comes down one path:

    ingest -> components.viewer.analysis (stage, queue, build)
           -> amsc.viewer.corpus.load_corpus  (the shared payload reader)
           -> viewer-payload.json             (the published state)
           -> /api/demo/viewer-analysis/<id>/payload
           -> amsc.viewer.server relay -> Viewer v3

These tests hold that path to the parts of it that are easy to break and
expensive to notice:

* the packaging worker reads the *shared reader*, never a Viewer page module,
  so the console cannot come to depend on either page's build;
* a build that fails leaves the last published payload exactly where it was,
  because a document that was viewable must not stop being viewable because a
  later variant could not be produced;
* deleting a document while its analysis is being built removes it, rather
  than losing the race and leaving a rebuilt directory behind;
* restart recovery and repeated staging are deterministic;
* the scratch files a killed process leaves are swept, so the one unbounded
  thing in the workspace stays bounded.

Race-sensitive tests synchronise on events, never on sleeps: the packaging
worker is a real thread and a timing assumption here would be a flake later.
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from components.viewer import analysis
from components.viewer import methods as M


def _unit(order, unit_id, unit_type, text, page, *, level=None, path=()):
    row = {
        "document_id": "edge-doc", "unit_id": unit_id, "order": order, "text": text,
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


@pytest.fixture(autouse=True)
def workspace(tmp_path, monkeypatch):
    """A packaging root of this test's own, with the worker idle either side."""
    analysis._queue.join()
    with analysis._lock:
        analysis._inflight.clear()
        analysis._revoked.clear()
    monkeypatch.setattr(analysis, "root", lambda: tmp_path / "viewer-live")
    yield tmp_path / "viewer-live"
    analysis._queue.join()
    with analysis._lock:
        analysis._inflight.clear()
        analysis._revoked.clear()


def _build(doc_id="edge-doc", **overrides):
    fields = dict(doc_id=doc_id, label="Kenar.pdf", units=_corpus(),
                  methods=[M.STANDARD], kb_id="kb1", kb_name="kenar-kb",
                  chunking_mode="standard")
    fields.update(overrides)
    analysis.stage(**fields)
    analysis._queue.join()
    return analysis.read_state(fields["doc_id"], fields.get("content_sha"))


# ---------------------------------------------------------------- ownership


def test_the_packager_reads_the_shared_reader_not_a_viewer_page(workspace):
    """The console's dependency is the payload reader, not either page.

    ``viewer_corpus`` is the contract both Viewer pages and this console are
    written against. Importing a *page* module here would make the console
    depend on a template it never renders -- and on Viewer v2, which is only
    kept for the research build.
    """
    for page in ("amsc.viewer_v2", "amsc.viewer.build", "amsc.viewer_v2_template",
                 "amsc.viewer.template"):
        sys.modules.pop(page, None)

    state = _build()

    assert state["status"] == analysis.STATUS_READY, state
    assert "amsc.viewer.corpus" in sys.modules, "the shared reader is what a build uses"
    leaked = [page for page in ("amsc.viewer_v2", "amsc.viewer.build",
                                "amsc.viewer_v2_template", "amsc.viewer.template")
              if page in sys.modules]
    assert leaked == [], f"packaging pulled in a Viewer page module: {leaked}"


def test_the_console_states_no_method_identity_of_its_own(workspace):
    """Phase 5's registry is the authority; this side only adds availability.

    A method registered in the library reaches every console surface with no
    edit here, which is what keeps a new chunker from needing a Viewer-specific
    list.
    """
    from amsc.chunking import registry

    assert set(M.ORDER) == set(registry.order())
    for key in registry.order():
        entry = registry.get(key)
        assert M.METHODS[key].label == entry.label
        assert M.METHODS[key].summary == entry.summary
        assert M.METHODS[key].engine == entry.kind


# ------------------------------------------------- a failure keeps the state


def test_a_failed_rebuild_keeps_the_last_published_payload(workspace, monkeypatch):
    """A document that was viewable stays viewable.

    Publication is the payload file. A rebuild that dies after the state is
    marked ``running`` must not take the published payload with it: the
    browser reading the document mid-rebuild, and after a failed one, gets the
    last good analysis rather than a 404.
    """
    assert _build()["status"] == analysis.STATUS_READY
    published = analysis.payload("edge-doc")
    assert published is not None and published["arms"]
    key = analysis.key_for("edge-doc")
    on_disk = analysis.payload_path(key).read_bytes()

    import amsc.viewer.corpus as corpus

    def refuse(*args, **kwargs):
        raise RuntimeError("the reader failed on this rebuild")

    monkeypatch.setattr(corpus, "load_corpus", refuse)
    analysis.add_methods("edge-doc", [M.MARKDOWN])
    analysis._queue.join()

    state = analysis.read_state("edge-doc")
    assert state["status"] == analysis.STATUS_FAILED
    assert "RuntimeError" in (state.get("error") or "")
    # The published artifact is untouched, byte for byte.
    assert analysis.payload_path(key).read_bytes() == on_disk
    # And so is the analysis it serves: the same arms, over the same units.
    served = analysis.payload("edge-doc")
    for part in ("arms", "units", "pages", "meta", "label"):
        assert served[part] == published[part], part
    # The one thing that did change is the record of what was asked for --
    # the method that failed is named as requested and reported absent, not
    # quietly dropped.
    assert M.MARKDOWN in served["live"]["requested"]
    assert served["live"]["methods"][M.MARKDOWN]["status"] != analysis.STATUS_READY
    # And the arms that were on disk before are still fetchable.
    assert analysis.chunk_rows("edge-doc", M.STANDARD)


def test_a_repaired_rebuild_republishes_over_the_stale_payload(workspace, monkeypatch):
    """Keeping the old payload is not the same as being stuck with it."""
    assert _build()["status"] == analysis.STATUS_READY
    published = analysis.payload("edge-doc")

    import amsc.viewer.corpus as corpus

    real = corpus.load_corpus
    monkeypatch.setattr(corpus, "load_corpus",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("nope")))
    analysis.add_methods("edge-doc", [M.MARKDOWN])
    analysis._queue.join()
    assert analysis.read_state("edge-doc")["status"] == analysis.STATUS_FAILED

    monkeypatch.setattr(corpus, "load_corpus", real)
    analysis.add_methods("edge-doc", [M.MARKDOWN])
    analysis._queue.join()

    state = analysis.read_state("edge-doc")
    assert state["status"] == analysis.STATUS_READY
    fresh = analysis.payload("edge-doc")
    assert fresh != published
    assert M.MARKDOWN in fresh["arms"] and M.STANDARD in fresh["arms"]


# ------------------------------------------------ deletion during packaging


def test_deleting_a_document_mid_build_leaves_nothing_behind(workspace, monkeypatch):
    """The delete wins, whichever of the two finishes last.

    A build writes its directory back into existence on every record it saves,
    so a delete that lands in the middle of one used to be undone by the build
    that was already running: the console forgot the document and the
    workspace went on listing a half-built analysis of it.
    """
    reached, release = threading.Event(), threading.Event()
    real_rows = analysis._chunk_rows

    def blocking(method, units):
        reached.set()
        assert release.wait(60), "the test never released the build"
        return real_rows(method, units)

    monkeypatch.setattr(analysis, "_chunk_rows", blocking)
    analysis.stage(doc_id="edge-doc", label="Kenar.pdf", units=_corpus(),
                   methods=[M.STANDARD], kb_id="kb1", chunking_mode="standard")
    key = analysis.key_for("edge-doc")
    assert reached.wait(60), "the build never started"

    assert analysis.discard("edge-doc") is True
    release.set()
    analysis._queue.join()

    assert not analysis.document_dir(key).exists(), "the build rebuilt a deleted document"
    assert analysis.states() == {}
    assert analysis.read_state("edge-doc")["status"] == analysis.STATUS_MISSING


def test_a_build_that_fails_after_a_delete_records_no_state_for_it(workspace, monkeypatch):
    """The failure path recreates nothing either."""
    reached, release = threading.Event(), threading.Event()

    def blow_up(method, units):
        reached.set()
        assert release.wait(60)
        raise RuntimeError("the chunker died")

    monkeypatch.setattr(analysis, "_chunk_rows", blow_up)
    analysis.stage(doc_id="edge-doc", label="Kenar.pdf", units=_corpus(),
                   methods=[M.STANDARD], kb_id="kb1", chunking_mode="standard")
    key = analysis.key_for("edge-doc")
    assert reached.wait(60)

    assert analysis.discard("edge-doc") is True
    release.set()
    analysis._queue.join()

    assert not analysis.document_dir(key).exists()
    assert analysis.read_state("edge-doc")["status"] == analysis.STATUS_MISSING


def test_asking_for_the_document_again_withdraws_the_delete(workspace, monkeypatch):
    """A delete followed by a re-upload is a request, not a tombstone."""
    reached, release = threading.Event(), threading.Event()
    real_rows = analysis._chunk_rows

    def blocking(method, units):
        reached.set()
        assert release.wait(60)
        return real_rows(method, units)

    monkeypatch.setattr(analysis, "_chunk_rows", blocking)
    analysis.stage(doc_id="edge-doc", label="Kenar.pdf", units=_corpus(),
                   methods=[M.STANDARD], kb_id="kb1", chunking_mode="standard")
    assert reached.wait(60)
    analysis.discard("edge-doc")
    # The same bytes come back while the doomed build is still running.
    monkeypatch.setattr(analysis, "_chunk_rows", real_rows)
    analysis.stage(doc_id="edge-doc", label="Kenar.pdf", units=_corpus(),
                   methods=[M.STANDARD], kb_id="kb1", chunking_mode="standard")
    release.set()
    analysis._queue.join()

    state = analysis.read_state("edge-doc")
    assert state["status"] == analysis.STATUS_READY, state
    assert analysis.payload("edge-doc") is not None


# ------------------------------------------------------ repeats and restarts


def test_staging_the_same_document_twice_builds_one_analysis(workspace):
    """Duplicate staging is idempotent: one key, one directory, one payload."""
    first = _build()
    payload_first = analysis.payload("edge-doc")
    second = _build()

    assert first["key"] == second["key"]
    assert [child.name for child in (workspace).iterdir()] == [first["key"]]
    assert second["status"] == analysis.STATUS_READY
    # Nothing was chunked again: the same arms, from the same canonical.
    assert set(analysis.payload("edge-doc")["arms"]) == set(payload_first["arms"])
    assert second["unit_count"] == first["unit_count"]


def test_restart_recovery_queues_the_unfinished_and_only_those(workspace):
    """What a restart picks up is decided by disk, and it is the same set twice.

    A finished document is not rebuilt on every start (that would re-chunk the
    whole corpus), and an interrupted one always is.
    """
    _build(doc_id="done-doc")
    _build(doc_id="stuck-doc")
    stuck = analysis.key_for("stuck-doc")
    # Exactly what a kill mid-build leaves: a running record, no payload.
    analysis.payload_path(stuck).unlink()
    analysis._set_state(stuck, status=analysis.STATUS_RUNNING)

    first = analysis.resume_incomplete()
    analysis._queue.join()
    assert first == [stuck]

    # Deterministic: with the same disk, the same answer -- and now that the
    # rebuild finished, nothing is outstanding.
    assert analysis.resume_incomplete() == []
    analysis._queue.join()
    assert analysis.read_state("stuck-doc")["status"] == analysis.STATUS_READY
    assert analysis.read_state("done-doc")["status"] == analysis.STATUS_READY


def test_a_ready_document_whose_payload_vanished_is_rebuilt(workspace):
    """The published artifact is the truth about ``ready``."""
    _build()
    key = analysis.key_for("edge-doc")
    analysis.payload_path(key).unlink()

    assert analysis.read_state("edge-doc")["status"] == analysis.STATUS_PENDING
    assert analysis.resume_incomplete() == [key]
    analysis._queue.join()
    assert analysis.read_state("edge-doc")["status"] == analysis.STATUS_READY


# --------------------------------------------------------- scratch discipline


def test_the_scratch_files_of_a_killed_process_are_swept(workspace):
    """The one thing here that could grow without a bound.

    Records are written to ``<name>.<pid>.<tid>.tmp`` and renamed into place,
    and the thread ids never repeat -- so a process killed between the two
    steps leaves a file nothing will ever overwrite.
    """
    _build()
    key = analysis.key_for("edge-doc")
    directory = analysis.document_dir(key)
    orphan = directory / "state.json.999.888.tmp"
    orphan.write_text("{}", encoding="utf-8")
    (directory / "units.jsonl.999.777.tmp").write_text("", encoding="utf-8")

    assert analysis.sweep_scratch() == 2
    assert not orphan.exists()
    # The real records are untouched.
    assert analysis.payload("edge-doc") is not None
    assert json.loads((directory / "state.json").read_text(encoding="utf-8"))["key"] == key
    # And a second sweep is a no-op.
    assert analysis.sweep_scratch() == 0


def test_a_restart_sweeps_before_it_resumes(workspace):
    """Recovery cleans up as well as picking up."""
    _build()
    key = analysis.key_for("edge-doc")
    (analysis.document_dir(key) / "state.json.111.222.tmp").write_text("{}", encoding="utf-8")

    analysis.resume_incomplete()
    analysis._queue.join()

    assert list(analysis.document_dir(key).rglob("*.tmp")) == []


# ------------------------------------------------- nothing generated is versioned


def _versioned(repo: Path) -> list[str]:
    """Every path a clone of ``repo`` would get: tracked plus not-ignored."""
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        cwd=repo, capture_output=True, text=True,
    )
    if result.returncode != 0:  # pragma: no cover - not a checkout
        pytest.skip("not a git checkout")
    return result.stdout.splitlines()


def test_no_viewer_runtime_state_is_in_version_control():
    """A fresh clone starts with no analyses, and cannot inherit anyone's.

    Everything under the packaging root is regenerable from an ingest: a
    canonical, packaged arms, a state record and a published payload. Checking
    any of it in would ship one developer's documents to everybody and make
    the console's answer depend on a file nobody rebuilds.
    """
    repo = Path(__file__).resolve().parents[2]
    offenders = [
        path for path in _versioned(repo)
        if path.startswith("artifacts/viewer-live/")
        or Path(path).name in ("viewer-payload.json", "units.jsonl")
    ]
    assert offenders == [], f"generated Viewer state is versioned: {offenders}"


def test_the_console_needs_no_built_viewer_page_to_run():
    """The page is the Viewer's build artifact; the console never reads it.

    The console serves JSON and knows nothing about the HTML, which is what
    lets ``start-demo.ps1`` build the page on a fresh clone without the
    console having an opinion about it.
    """
    repo = Path(__file__).resolve().parents[2]
    for module in ("app.py", "components/viewer/analysis.py", "components/viewer/methods.py"):
        source = (repo / module).read_text(encoding="utf-8")
        assert "viewer-v3" not in source and "viewer-v2" not in source, module
        assert "index.html" not in source, module
