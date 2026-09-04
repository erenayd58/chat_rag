"""One document's Viewer records, written by the worker while a poll reads them.

The packaging worker writes ``state.json`` at every stage of a build and
rewrites ``viewer-payload.json`` at the end of it. Meanwhile the console reads
both from request threads: the workspace snapshot walks every state record,
the browser polls one document's state while its build runs, and the Viewer
fetches the payload.

Those two sides used to collide, and on Windows the collision is not subtle.
``os.replace`` cannot replace a file that another handle has open, so a reader
merely *having* ``state.json`` open made the worker's write raise
``PermissionError``; and the reader that lost the same race got a
``PermissionError`` of its own, which ``_read_state_file`` catches and reports
as an unreadable -- that is, failed -- record. The worse half was silent: a
write merges onto whatever it just read, so a write that read the placeholder
persisted a record with no ``doc_ids``, and the document dropped out of the
workspace it belonged to.

Everything under one analysis directory is written by this module, in this
process, so one lock per key covering reads as well as writes is the whole
fix; there is no second process to coordinate with.
"""

from __future__ import annotations

import json
import threading

import pytest

from components.viewer import analysis

#: ``payload()`` derives the directory from the content hash, so the two have
#: to agree or the test would read a key nothing was ever written to.
CONTENT_SHA = "concurrency-probe"
KEY = "doc-concurrency-probe"


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    """An analysis root of this test's own, with the worker left idle."""
    analysis._queue.join()
    with analysis._lock:
        analysis._inflight.clear()
    monkeypatch.setattr(analysis, "root", lambda: tmp_path / "viewer-live")
    yield tmp_path / "viewer-live"
    analysis._queue.join()
    with analysis._lock:
        analysis._inflight.clear()


def _seed(**fields):
    assert analysis.content_key(CONTENT_SHA) == KEY
    analysis._set_state(
        KEY, status=analysis.STATUS_PENDING, doc_ids=["upload-one", "upload-two"],
        requested=["structure-only"], content_sha=CONTENT_SHA, label="Probe.pdf", **fields
    )


# ------------------------------------------------------- state records


def test_polling_a_document_never_breaks_the_build_writing_its_state(workspace):
    """The reported failure, as the product produces it: a status poll running
    against a build that is recording its stages."""
    _seed()
    reads: list[dict] = []
    errors: list[BaseException] = []
    stop = threading.Event()

    def poll() -> None:
        while not stop.is_set():
            try:
                reads.append(analysis._read_state_file(KEY))
            except BaseException as error:  # noqa: BLE001 - the point of the test
                errors.append(error)
                return

    reader = threading.Thread(target=poll, daemon=True)
    reader.start()
    try:
        for index in range(200):
            analysis._set_state(KEY, status=analysis.STATUS_RUNNING, unit_count=index)
    finally:
        stop.set()
        reader.join(timeout=10)

    assert not errors, errors
    assert reads, "the reader never ran"
    assert analysis._read_state_file(KEY)["unit_count"] == 199


def test_a_poll_never_sees_a_document_as_failed_while_it_is_only_being_written(workspace):
    """The reader's half. An unreadable record is indistinguishable from a
    genuinely failed one to every caller above this module, so a transient
    read must not produce it."""
    _seed()
    seen: list[dict] = []
    stop = threading.Event()

    def poll() -> None:
        while not stop.is_set():
            seen.append(analysis._read_state_file(KEY))

    reader = threading.Thread(target=poll, daemon=True)
    reader.start()
    try:
        for index in range(200):
            analysis._set_state(KEY, status=analysis.STATUS_RUNNING, unit_count=index)
    finally:
        stop.set()
        reader.join(timeout=10)

    unreadable = [state for state in seen if state.get("error") == "unreadable state record"]
    assert not unreadable, f"{len(unreadable)} of {len(seen)} reads reported a healthy document as failed"
    # And no observation ever lost the identity that ties this analysis to its
    # console records -- the loss that made the document vanish from the list.
    assert all(state.get("doc_ids") == ["upload-one", "upload-two"] for state in seen)


def test_concurrent_writers_do_not_drop_each_others_fields(workspace):
    """Two threads recording different things about one document: staging from
    a request and the worker recording its stage."""
    _seed()
    errors: list[BaseException] = []

    def record(field: str, count: int) -> None:
        try:
            for index in range(count):
                analysis._set_state(KEY, **{field: index})
        except BaseException as error:  # noqa: BLE001
            errors.append(error)

    threads = [threading.Thread(target=record, args=("unit_count", 150)),
               threading.Thread(target=record, args=("payload_bytes", 150))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    assert not errors, errors
    final = analysis._read_state_file(KEY)
    assert final["unit_count"] == 149 and final["payload_bytes"] == 149
    assert final["doc_ids"] == ["upload-one", "upload-two"], "identity survived both writers"
    assert final["requested"] == ["structure-only"]


# ------------------------------------------------------- the payload


def test_reading_the_payload_never_breaks_the_build_rewriting_it(workspace):
    """``viewer-payload.json`` is rewritten at the end of every build and read
    by the Viewer for the same document."""
    _seed()
    path = analysis.payload_path(KEY)
    analysis._write_json(path, {"arms": {}, "units": []})
    misses: list[None] = []
    errors: list[BaseException] = []
    stop = threading.Event()

    def poll() -> None:
        while not stop.is_set():
            try:
                if analysis.payload("upload-one", CONTENT_SHA) is None:
                    misses.append(None)
            except BaseException as error:  # noqa: BLE001
                errors.append(error)
                return

    reader = threading.Thread(target=poll, daemon=True)
    reader.start()
    try:
        for index in range(200):
            with analysis._state_lock(KEY):
                analysis._write_json(path, {"arms": {}, "units": [], "n": index})
    finally:
        stop.set()
        reader.join(timeout=10)

    assert not errors, errors
    assert not misses, "a payload that is on disk read back as absent"


# ------------------------------- an unreadable record is not overwritten


def test_a_state_record_that_cannot_be_read_is_not_merged_over(workspace):
    """With the lock in place this only happens to a genuinely damaged file,
    and then the right move is to leave it: merging onto a blank record would
    replace the document's identity with the absence of one."""
    _seed()
    path = analysis.document_dir(KEY) / "state.json"
    path.write_text('{"key": "doc-concurrency-probe", "doc_ids": ["upload-one"', encoding="utf-8")

    with pytest.raises(ValueError):
        analysis._set_state(KEY, status=analysis.STATUS_READY)

    # Untouched, so the records it held can still be recovered by hand.
    assert path.read_text(encoding="utf-8").startswith('{"key": "doc-concurrency-probe"')
    # And a caller that only reads is told the record is unusable, not given
    # an empty one that looks like a document with no analysis.
    state = analysis._read_state_file(KEY)
    assert state["status"] == analysis.STATUS_FAILED
    assert state["error"] == "unreadable state record"


def test_a_document_with_no_state_record_at_all_still_starts_cleanly(workspace):
    """The absent case must stay distinct from the unreadable one: a first
    stage has no record to merge onto and that is not an error."""
    assert analysis._read_state_file("doc-brand-new")["status"] == analysis.STATUS_MISSING
    state = analysis._set_state("doc-brand-new", status=analysis.STATUS_PENDING, doc_ids=["x"])
    assert state["status"] == analysis.STATUS_PENDING and state["doc_ids"] == ["x"]
    assert json.loads((analysis.document_dir("doc-brand-new") / "state.json")
                      .read_text(encoding="utf-8"))["doc_ids"] == ["x"]
