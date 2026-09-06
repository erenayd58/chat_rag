"""The ingest ledger under two writers, and under a crash mid-write.

``DocumentTracker`` used to load the whole ledger into memory on construction
and write the whole ledger back from that memory on every change, with a plain
``open(path, "w")`` and no lock. The upload route constructs a fresh tracker
per request, so two requests that each built one before either saved wrote a
ledger that lacked the other's record.

Phase 1A reproduced both failures deterministically here as strict xfails.
Phase 1B made them pass: every mutation now re-reads the ledger inside a lock
on that file, applies its own change and replaces the file in one step, so a
concurrent writer's record survives and an interrupted write leaves the
previous ledger intact. These tests are what hold that.

Nothing here touches the developer's ``.ingested_documents.json`` -- every
tracker is pointed at a file under ``tmp_path``.

Phase 8 removed the last way back to the old behaviour: ``_save_tracking_data``
survived Phase 1B as an unused private method that wrote this instance's whole
in-memory view over the file, documented as able to drop a record written
elsewhere. Nothing called it, so it was one wrong ``self._save...()`` away from
undoing all of the above. The test at the end of this file holds that gone.
"""

from __future__ import annotations

import json
import os
import pathlib
import threading

import pytest

from utils import DocumentTracker
from utils import document_tracker as tracker_module


@pytest.fixture
def ledger(tmp_path):
    return str(tmp_path / "state" / "ingested_documents.json")


@pytest.fixture
def documents(tmp_path):
    def make(name: str) -> str:
        path = tmp_path / f"{name}.pdf"
        path.write_bytes(b"%PDF-1.7 " + name.encode())
        return str(path)
    return make


def _records(ledger: str) -> dict:
    with open(ledger, encoding="utf-8") as handle:
        return json.load(handle)


# --------------------------------------------------------------- two writers


def test_two_overlapping_uploads_both_keep_their_records(ledger, documents):
    """Two requests, each with its own tracker, both constructed before either
    saved -- exactly the upload route's per-request ``DocumentTracker()``."""
    first = DocumentTracker(ledger)
    second = DocumentTracker(ledger)

    first.mark_as_ingested(file_path=documents("a"), doc_id="doc-a", chunk_count=1)
    second.mark_as_ingested(file_path=documents("b"), doc_id="doc-b", chunk_count=1)

    assert {row["doc_id"] for row in _records(ledger).values()} == {"doc-a", "doc-b"}


def test_n_simultaneous_uploads_leave_n_records(ledger, documents):
    """The same collision from threads, at the width the dev server allows.

    The barrier guarantees every writer has loaded before any writer saves,
    which is precisely the state that used to leave one record behind.
    """
    writers = 8
    paths = [documents(f"t{index}") for index in range(writers)]
    trackers = [DocumentTracker(ledger) for _ in range(writers)]
    gate = threading.Barrier(writers)
    errors = []

    def upload(index: int) -> None:
        try:
            gate.wait(timeout=10)
            trackers[index].mark_as_ingested(file_path=paths[index], doc_id=f"doc-{index}", chunk_count=1)
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=upload, args=(index,)) for index in range(writers)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    assert not errors
    assert len(_records(ledger)) == writers


def test_a_stale_tracker_merges_instead_of_replacing(ledger, documents):
    """The mechanism the two tests above rest on.

    A save re-reads the ledger and applies only its own change to it, so a
    record written since this tracker loaded is still there afterwards -- and
    the instance's own view is refreshed to the ledger rather than left
    claiming a document it just dropped.
    """
    DocumentTracker(ledger).mark_as_ingested(file_path=documents("a"), doc_id="doc-a", chunk_count=1)
    stale = DocumentTracker(ledger)  # loaded holding doc-a only
    DocumentTracker(ledger).mark_as_ingested(file_path=documents("b"), doc_id="doc-b", chunk_count=1)
    assert {row["doc_id"] for row in _records(ledger).values()} == {"doc-a", "doc-b"}

    stale.mark_as_ingested(file_path=documents("c"), doc_id="doc-c", chunk_count=1)

    assert {row["doc_id"] for row in _records(ledger).values()} == {"doc-a", "doc-b", "doc-c"}
    assert {row["doc_id"] for row in stale.ingested_docs.values()} == {"doc-a", "doc-b", "doc-c"}


def test_removing_a_document_leaves_every_other_record_alone(ledger, documents):
    """Removal is a change to the ledger, not a rewrite of one view of it."""
    first = documents("a")
    DocumentTracker(ledger).mark_as_ingested(file_path=first, doc_id="doc-a", chunk_count=1)
    stale = DocumentTracker(ledger)
    DocumentTracker(ledger).mark_as_ingested(file_path=documents("b"), doc_id="doc-b", chunk_count=1)

    stale.remove_document(first)

    assert {row["doc_id"] for row in _records(ledger).values()} == {"doc-b"}


# ----------------------------------------------------------- torn write


def test_a_crash_mid_write_does_not_destroy_the_ledger(ledger, documents, monkeypatch):
    DocumentTracker(ledger).mark_as_ingested(file_path=documents("a"), doc_id="doc-a", chunk_count=1)
    assert {row["doc_id"] for row in _records(ledger).values()} == {"doc-a"}

    real_dump = json.dump

    def crash_half_way(payload, handle, **kwargs):
        text = json.dumps(payload, **kwargs)
        handle.write(text[: len(text) // 2])
        raise OSError("disk full")

    monkeypatch.setattr(tracker_module.json, "dump", crash_half_way)
    DocumentTracker(ledger).mark_as_ingested(file_path=documents("b"), doc_id="doc-b", chunk_count=1)
    monkeypatch.setattr(tracker_module.json, "dump", real_dump)

    # The record that was there before the failed write must still be there:
    # either the old ledger survived, or the new one was written completely.
    survivors = {row["doc_id"] for row in DocumentTracker(ledger).ingested_docs.values()}
    assert "doc-a" in survivors
    # And the half-written bytes must not be sitting anywhere a reader looks.
    assert _records(ledger), "the ledger itself was left unreadable"
    assert not list(pathlib.Path(ledger).parent.glob("*.tmp")), "a scratch file was left behind"


def test_the_loader_resets_an_unreadable_ledger_to_empty_silently(ledger, tmp_path, capsys):
    """Constructing a tracker never raises, whatever the file holds.

    Writes are atomic now, so this state no longer follows from an interrupted
    write; it is what a ledger damaged from outside looks like. Loading still
    reports nothing rather than failing, and the write that follows keeps the
    file instead of replacing it -- see the quarantine test below.
    """
    tmp_path.joinpath("state").mkdir()
    with open(ledger, "w", encoding="utf-8") as handle:
        handle.write('{"only": {"doc_id": "half-wr')
    tracker = DocumentTracker(ledger)
    assert tracker.ingested_docs == {}
    assert "Could not load tracking file" in capsys.readouterr().out


def test_a_ledger_that_cannot_be_read_is_never_written_over(ledger, documents, monkeypatch):
    """An OS-level read failure is transient -- on Windows another process
    renaming the file over it is enough. Treating that as an empty ledger and
    writing would replace every record that is still there, so the change is
    refused instead."""
    DocumentTracker(ledger).mark_as_ingested(file_path=documents("a"), doc_id="doc-a", chunk_count=1)
    before = _records(ledger)

    tracker = DocumentTracker(ledger)

    def refuse(self):
        raise PermissionError("the ledger is open in another process")

    monkeypatch.setattr(DocumentTracker, "_read_records", refuse)
    tracker.mark_as_ingested(file_path=documents("b"), doc_id="doc-b", chunk_count=1)

    assert _records(ledger) == before, "an unreadable ledger was overwritten"


def test_an_unparseable_ledger_is_kept_rather_than_destroyed(ledger, documents, tmp_path):
    """A ledger whose contents cannot be parsed is not something the next
    write should land on top of. It is moved aside under a name that says what
    it is, so whatever it held stays recoverable, and a new one is started."""
    os.makedirs(os.path.dirname(ledger), exist_ok=True)
    with open(ledger, "w", encoding="utf-8") as handle:
        handle.write('{"C:/rapor.pdf": {"doc_id": "doc-from-before", "chunk_co')

    DocumentTracker(ledger).mark_as_ingested(file_path=documents("new"), doc_id="doc-new", chunk_count=1)

    assert {row["doc_id"] for row in _records(ledger).values()} == {"doc-new"}
    kept = sorted(tmp_path.joinpath("state").glob("ingested_documents.json.corrupt-*"))
    assert len(kept) == 1, f"the unreadable ledger was not kept: {kept}"
    assert "doc-from-before" in kept[0].read_text(encoding="utf-8")


def test_the_ledger_has_no_write_path_that_skips_the_re_read(ledger, documents):
    """Every change goes through ``_mutate``, which re-reads inside the lock.

    ``_save_tracking_data`` was the exception -- it replaced the file with
    this instance's in-memory snapshot, which is exactly the lost update the
    rest of this file exists to prevent. It had no caller and was removed;
    this is what stops it, or another like it, from coming back unnoticed.
    """
    tracker = DocumentTracker(str(ledger))

    assert not hasattr(tracker, "_save_tracking_data")

    public = [
        name for name in dir(tracker)
        if not name.startswith("__") and callable(getattr(tracker, name))
    ]
    import inspect

    for name in public:
        source = inspect.getsource(getattr(tracker, name))
        if "_write_records(" not in source:
            continue
        assert name in ("_mutate", "_write_records"), (
            f"{name} writes the ledger without going through _mutate, so a "
            "concurrent writer's record can be lost"
        )
