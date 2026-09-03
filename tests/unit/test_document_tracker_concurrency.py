"""The ingest ledger under two writers, and under a crash mid-write.

``DocumentTracker`` loads the whole ledger into memory on construction and
writes the whole ledger back from that memory on every change, with a plain
``open(path, "w")`` and no lock. The upload route constructs a fresh tracker
per request. Two requests that both constructed their tracker before either
saved therefore each write a ledger that lacks the other's record.

These tests state the contract the ledger *should* honour and are marked
``xfail(strict=True)``: today they fail deterministically, which is the
evidence; once the ledger is fixed they will pass, and ``strict`` turns that
into a failure until the marker is removed. Nothing here touches the
developer's ``.ingested_documents.json`` -- every tracker is pointed at a
file under ``tmp_path``.
"""

from __future__ import annotations

import json
import threading

import pytest

from utils import DocumentTracker
from utils import document_tracker as tracker_module

LEDGER_RACE = "Phase 1B: the ledger is an unlocked whole-file read-modify-write; a concurrent writer's record is lost"
LEDGER_TORN_WRITE = "Phase 1B: the ledger is rewritten in place; a crash mid-write leaves an unparseable file the loader resets to empty"


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


@pytest.mark.xfail(strict=True, reason=LEDGER_RACE)
def test_two_overlapping_uploads_both_keep_their_records(ledger, documents):
    """Two requests, each with its own tracker, both constructed before either
    saved -- exactly the upload route's per-request ``DocumentTracker()``."""
    first = DocumentTracker(ledger)
    second = DocumentTracker(ledger)

    first.mark_as_ingested(file_path=documents("a"), doc_id="doc-a", chunk_count=1)
    second.mark_as_ingested(file_path=documents("b"), doc_id="doc-b", chunk_count=1)

    assert {row["doc_id"] for row in _records(ledger).values()} == {"doc-a", "doc-b"}


@pytest.mark.xfail(strict=True, reason=LEDGER_RACE)
def test_n_simultaneous_uploads_leave_n_records(ledger, documents):
    """The same lost update from threads: every writer loads before any
    writer saves (the barrier guarantees it), so each saves a one-record
    ledger and the last one to finish is the only record left."""
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


def test_the_per_request_tracker_saves_only_what_it_loaded(ledger, documents):
    """The mechanism behind the two failures above, stated plainly so a fix
    that changes it is a visible decision: a tracker writes its own memory,
    it does not merge with what is on disk at save time."""
    DocumentTracker(ledger).mark_as_ingested(file_path=documents("a"), doc_id="doc-a", chunk_count=1)
    stale = DocumentTracker(ledger)  # loaded with doc-a
    DocumentTracker(ledger).mark_as_ingested(file_path=documents("b"), doc_id="doc-b", chunk_count=1)
    assert {row["doc_id"] for row in _records(ledger).values()} == {"doc-a", "doc-b"}

    stale.mark_as_ingested(file_path=documents("c"), doc_id="doc-c", chunk_count=1)
    on_disk = {row["doc_id"] for row in _records(ledger).values()}
    assert "doc-c" in on_disk and "doc-a" in on_disk
    # Whether doc-b survives is the whole question; today it does not.
    assert on_disk == set(stale.ingested_docs[path]["doc_id"] for path in stale.ingested_docs)


# ----------------------------------------------------------- torn write


@pytest.mark.xfail(strict=True, reason=LEDGER_TORN_WRITE)
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


def test_the_loader_resets_an_unreadable_ledger_to_empty_silently(ledger, tmp_path, capsys):
    """What happens after such a crash today, so the consequence is on
    record: the loader does not raise, it starts over with nothing."""
    tmp_path.joinpath("state").mkdir()
    with open(ledger, "w", encoding="utf-8") as handle:
        handle.write('{"only": {"doc_id": "half-wr')
    tracker = DocumentTracker(ledger)
    assert tracker.ingested_docs == {}
    assert "Could not load tracking file" in capsys.readouterr().out
