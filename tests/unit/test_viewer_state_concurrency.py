"""One document's Viewer records, written by the worker while a poll reads them.

The packaging worker records every stage of a build and rewrites
``viewer-payload.json`` at the end of it. Meanwhile the console reads both from
request threads: the workspace snapshot reads every analysis record, the
browser polls one document's state while its build runs, and the Viewer fetches
the payload.

Those two sides used to collide, and on Windows the collision was not subtle.
The record was a ``state.json`` beside the artifacts, and ``os.replace`` cannot
replace a file another handle has open -- so a reader merely *having* it open
made the worker's write raise ``PermissionError``, and the reader that lost the
same race reported a healthy document as an unreadable, that is failed, record.
The worse half was silent: a write merged onto whatever it had just read, so a
write that read the placeholder persisted a record with no ``doc_ids``, and the
document dropped out of the workspace it belonged to.

Step 8 made the record a row, and the two halves are now answered differently:

* **the record** -- a write reads and updates it inside one transaction, with
  the row locked, so a concurrent write cannot be merged over and a concurrent
  read cannot see a half-applied one. Unlike the lock this replaced, that holds
  for two processes as well as two threads.
* **the payload** -- still a file, still rewritten at the end of every build
  and read by the Viewer at the same moment, so it still needs the per-key lock
  that covers reads as well as writes (``analysis._artifact_lock``).

Every test below is the same question asked of whichever of the two now
answers it.
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
            with analysis._artifact_lock(KEY):
                analysis._write_json(path, {"arms": {}, "units": [], "n": index})
    finally:
        stop.set()
        reader.join(timeout=10)

    assert not errors, errors
    assert not misses, "a payload that is on disk read back as absent"


# ------------------------------------ a write that fails changes nothing


def test_a_write_that_fails_leaves_the_record_whole(workspace, monkeypatch):
    """What replaced "an unreadable record is not merged over".

    The damaged-file case is gone with the file. Its consequence is not: a
    write that fails part way must leave the record exactly as it was, because
    a record merged onto a blank one replaces the document's identity with the
    absence of one -- and that is what made a document vanish from the
    workspace it belonged to. The failure is injected *after* the membership
    has been written inside the transaction, so only a rollback satisfies this.
    """
    from storage.repositories import ContentRepository

    _seed()
    before = analysis._read_state_file(KEY)

    real = ContentRepository._apply

    def apply_then_fail(self, row, changes):
        real(self, row, changes)
        raise RuntimeError("the connection went away")

    monkeypatch.setattr(ContentRepository, "_apply", apply_then_fail)
    with pytest.raises(RuntimeError):
        analysis._set_state(KEY, status=analysis.STATUS_READY, doc_ids=[])

    after = analysis._read_state_file(KEY)
    assert after["status"] == before["status"]
    assert after["doc_ids"] == ["upload-one", "upload-two"], "identity survived"
    assert after["requested"] == ["structure-only"]


def test_a_document_with_no_record_at_all_still_starts_cleanly(workspace):
    """A first stage has no record to merge onto, and that is not an error."""
    assert analysis._read_state_file("doc-brand-new")["status"] == analysis.STATUS_MISSING
    state = analysis._set_state("doc-brand-new", status=analysis.STATUS_PENDING, doc_ids=["x"])
    assert state["status"] == analysis.STATUS_PENDING and state["doc_ids"] == ["x"]
    # And it is really persisted: read back from the store, not from the
    # dictionary the write returned.
    assert analysis._read_state_file("doc-brand-new")["doc_ids"] == ["x"]
