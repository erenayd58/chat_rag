"""The ingest ledger under two writers, and under a failure mid-write.

``DocumentTracker`` used to be one JSON file. It loaded the whole ledger into
memory on construction and wrote the whole ledger back from that memory on
every change, and the upload route builds a fresh tracker per request -- so two
requests that each built one before either saved wrote a ledger that lacked the
other's record. Phase 1A reproduced that here as strict xfails; Phase 1B made
it pass with a re-read inside a per-file lock and an atomic rename.

Step 8 made the ledger a table, and this file is the same set of questions
asked of it. The questions did not change, because they were never about
files: does a concurrent writer's record survive, does a stale reader see what
someone else wrote, does removing one document leave the rest alone, and does a
failed write leave the ledger as it was.

Three of the answers are now given by the database rather than by this code,
and that is the point of asking them here:

* a lost update is impossible because two uploads write two rows, not two
  copies of one document;
* a write is atomic because it is a transaction, not a rename;
* a *failed* write leaves the previous state because it rolls back, and there
  is no half-written store to quarantine, recover or refuse to overwrite.

What went with the files is the machinery that defended them: the
``.corrupt-<stamp>`` quarantine, the "never write over a ledger that cannot be
read" rule, and the scratch-file sweep. Those described failure modes of a
whole-document rewrite. The failure mode of a row is a rolled-back
transaction, and it is exercised below.

Nothing here touches the developer's database: every test runs against the
suite's own, truncated before each one (``tests/conftest.py``).
"""

from __future__ import annotations

import threading

import pytest

from storage import DocumentRepository, session_scope
from utils import DocumentTracker


@pytest.fixture
def ledger(tmp_path):
    """The path a tracker is pointed at.

    Still passed, still ignored: the records are rows. Kept in the fixture
    because every caller in the product still constructs a tracker this way,
    and a test should exercise the constructor the product uses.
    """
    return str(tmp_path / "state" / "ingested_documents.json")


@pytest.fixture
def documents(tmp_path):
    def make(name: str) -> str:
        path = tmp_path / f"{name}.pdf"
        path.write_bytes(b"%PDF-1.7 " + name.encode())
        return str(path)
    return make


def _records() -> dict:
    """The ledger as it is, read straight from the repository rather than
    through a tracker -- so a tracker's own view can be compared with it."""
    with session_scope() as session:
        return {row["doc_id"]: row for row in DocumentRepository(session).list()}


# --------------------------------------------------------------- two writers


def test_two_overlapping_uploads_both_keep_their_records(ledger, documents):
    """Two requests, each with its own tracker, both constructed before either
    saved -- exactly the upload route's per-request ``DocumentTracker()``."""
    first = DocumentTracker(ledger)
    second = DocumentTracker(ledger)

    first.mark_as_ingested(file_path=documents("a"), doc_id="doc-a", chunk_count=1)
    second.mark_as_ingested(file_path=documents("b"), doc_id="doc-b", chunk_count=1)

    assert set(_records()) == {"doc-a", "doc-b"}


def test_n_simultaneous_uploads_leave_n_records(ledger, documents):
    """The same collision from threads, at the width the dev server allows.

    The barrier guarantees every writer has been constructed before any writer
    saves, which is precisely the state that used to leave one record behind.
    """
    writers = 8
    paths = [documents(f"t{index}") for index in range(writers)]
    trackers = [DocumentTracker(ledger) for _ in range(writers)]
    gate = threading.Barrier(writers)
    errors = []

    def upload(index: int) -> None:
        try:
            gate.wait(timeout=10)
            trackers[index].mark_as_ingested(
                file_path=paths[index], doc_id=f"doc-{index}", chunk_count=1
            )
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=upload, args=(index,)) for index in range(writers)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    assert not errors
    assert len(_records()) == writers


def test_a_stale_tracker_sees_what_another_writer_recorded(ledger, documents):
    """The mechanism the two tests above rest on.

    A tracker holds no copy of the ledger, so a record written since it was
    constructed is simply there. The instance's own view used to be a snapshot
    taken at construction, and writing it back is what dropped other people's
    documents.
    """
    DocumentTracker(ledger).mark_as_ingested(file_path=documents("a"), doc_id="doc-a", chunk_count=1)
    stale = DocumentTracker(ledger)  # constructed while only doc-a existed
    DocumentTracker(ledger).mark_as_ingested(file_path=documents("b"), doc_id="doc-b", chunk_count=1)

    stale.mark_as_ingested(file_path=documents("c"), doc_id="doc-c", chunk_count=1)

    assert set(_records()) == {"doc-a", "doc-b", "doc-c"}
    assert {row["doc_id"] for row in stale.ingested_docs.values()} == {
        "doc-a", "doc-b", "doc-c"
    }


def test_removing_a_document_leaves_every_other_record_alone(ledger, documents):
    """Removal is a change to the ledger, not a rewrite of one view of it."""
    first = documents("a")
    DocumentTracker(ledger).mark_as_ingested(file_path=first, doc_id="doc-a", chunk_count=1)
    stale = DocumentTracker(ledger)
    DocumentTracker(ledger).mark_as_ingested(file_path=documents("b"), doc_id="doc-b", chunk_count=1)

    stale.remove_by_doc_id("doc-a")

    assert set(_records()) == {"doc-b"}


def test_the_same_upload_recorded_twice_is_one_row(ledger, documents):
    """``doc_id`` is unique, and the write is an upsert.

    A job that is retried, or two workers that somehow finish the same
    document, leave one record carrying the later of the two -- not two records
    the console would show as two documents.
    """
    path = documents("a")
    DocumentTracker(ledger).mark_as_ingested(file_path=path, doc_id="doc-a", chunk_count=1)
    DocumentTracker(ledger).mark_as_ingested(file_path=path, doc_id="doc-a", chunk_count=7)

    records = _records()
    assert list(records) == ["doc-a"]
    assert records["doc-a"]["chunk_count"] == 7


def test_racing_writers_of_one_document_leave_one_row(ledger, documents):
    """The same upsert, contended. Two threads writing one ``doc_id`` at once
    must leave one row and no error: the unique constraint decides it, and the
    conflict clause is what turns "the other one won" into "the row is there"
    rather than an IntegrityError surfacing as a failed ingest."""
    path = documents("a")
    errors = []
    gate = threading.Barrier(6)

    def write(count: int) -> None:
        try:
            gate.wait(timeout=10)
            DocumentTracker(ledger).mark_as_ingested(
                file_path=path, doc_id="doc-a", chunk_count=count
            )
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=write, args=(n,)) for n in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    assert not errors, errors
    assert list(_records()) == ["doc-a"]


# ----------------------------------------------------------- a failed write


def test_a_failed_write_leaves_the_ledger_as_it_was(ledger, documents, monkeypatch):
    """What replaced the torn-write defences.

    The record that was there before a failed write must still be there, and
    the failed one must not be half-there. That used to need a scratch file and
    a rename; it is now the transaction, and the tracker reports the failure by
    returning ``False`` -- which is what makes the ingest roll its chunks back
    out of the vector store instead of registering a document nobody can find.
    """
    DocumentTracker(ledger).mark_as_ingested(file_path=documents("a"), doc_id="doc-a", chunk_count=1)
    before = _records()

    real = DocumentRepository.upsert

    def insert_then_fail(self, record):
        real(self, record)
        raise OSError("connection reset")

    monkeypatch.setattr(DocumentRepository, "upsert", insert_then_fail)
    recorded = DocumentTracker(ledger).mark_as_ingested(
        file_path=documents("b"), doc_id="doc-b", chunk_count=1
    )

    assert recorded is False, "a write that failed must not report success"
    assert _records() == before, "a failed write changed the ledger"


def test_a_ledger_that_cannot_be_read_is_never_written_over(ledger, documents, monkeypatch):
    """A read failure is transient -- a dropped connection is enough. Treating
    it as an empty ledger and writing anyway would replace every record that is
    still there, so the change must not be applied over them."""
    DocumentTracker(ledger).mark_as_ingested(file_path=documents("a"), doc_id="doc-a", chunk_count=1)
    before = _records()

    def refuse(self, *args, **kwargs):
        raise ConnectionError("the connection was closed")

    monkeypatch.setattr(DocumentRepository, "upsert", refuse)
    assert DocumentTracker(ledger).mark_as_ingested(
        file_path=documents("b"), doc_id="doc-b", chunk_count=1
    ) is False

    assert _records() == before, "an unreadable ledger was overwritten"
