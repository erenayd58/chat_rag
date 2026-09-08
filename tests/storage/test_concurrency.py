"""The invariants that only break when two writers arrive at once.

Everything this file drives had a process-local lock before Step 8 -- a
``threading.RLock`` per ledger file, per analysis key, per store. Those were
correct for one process and silently wrong for two, and "two" is not
hypothetical: the deployment runs one server process today, and the whole
point of moving to a database is that it need not.

So each test below takes a rule the product depends on, has several threads
attack it through the ordinary code path at the same moment, and asserts the
rule held -- with the enforcement in the database, where a second process is
subject to it too.

The threads are real and so are the sessions: every one opens its own unit of
work from the pool, exactly as a request thread does. A barrier makes them
collide rather than queue, which is the difference between a test that
exercises contention and one that happens not to.
"""

from __future__ import annotations

import threading

import pytest

from components.knowledgebase.manager import KnowledgeBaseManager
from components.viewer import analysis
from storage import ContentRepository, DocumentRepository, session_scope
from storage.models import Content, ContentDocument
from utils import DocumentTracker

#: Enough threads to lose a race reliably, few enough to stay inside the
#: connection pool without queueing for the length of the test.
WRITERS = 6


def _race(work, count=WRITERS, timeout=30):
    """Run ``work(index)`` on ``count`` threads that all start together.

    Returns ``(results, errors)``. Nothing is asserted here: which of the
    writers wins is not the property under test, only what is left behind.
    """
    gate = threading.Barrier(count)
    results: list = []
    errors: list[BaseException] = []
    lock = threading.Lock()

    def run(index: int) -> None:
        try:
            gate.wait(timeout=timeout)
            outcome = work(index)
            with lock:
                results.append(outcome)
        except BaseException as error:  # noqa: BLE001 - collected, then asserted on
            with lock:
                errors.append(error)

    threads = [threading.Thread(target=run, args=(index,)) for index in range(count)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=timeout)
    assert not any(thread.is_alive() for thread in threads), "a writer never finished"
    return results, errors


# ==========================================================================
# duplicate content creation
# ==========================================================================
def test_several_uploads_of_one_document_create_one_content():
    """The dedup invariant, contended.

    Six uploads of the same PDF land at once. Each reads "no content for these
    bytes yet" and each tries to create one. Exactly one row must exist
    afterwards, and every upload must be attached to it -- the alternative is
    six analyses of one document, five of which nothing will ever look at
    again, each holding a full packaged corpus on disk.
    """
    key = "doc-contended-bytes"

    def upload(index: int):
        with session_scope() as session:
            return ContentRepository(session).upsert_state(
                key,
                merge=lambda state: {
                    "doc_ids": sorted(set(state["doc_ids"]) | {f"upload-{index}"})
                },
                fields={"status": analysis.STATUS_PENDING},
                content_sha="contended-bytes",
            )

    _, errors = _race(upload)

    assert not errors, errors
    with session_scope() as session:
        assert session.query(Content).count() == 1
        state = ContentRepository(session).get(key)
    assert state["doc_ids"] == sorted(f"upload-{i}" for i in range(WRITERS)), (
        "an upload was dropped by another upload's write"
    )


def test_concurrent_writers_do_not_drop_each_others_selections():
    """The same collision one level down: each upload records the methods *it*
    asked for, and none of them may lose another's.

    This is the failure the file store was fixed for once already, at a level
    where the fix could only ever hold within one process.
    """
    key = "doc-contended-bytes"
    with session_scope() as session:
        ContentRepository(session).upsert_state(key, fields={"status": "pending"})

    def choose(index: int):
        method = ["structure-only", "markdown"][index % 2]

        def merge(state):
            selections = dict(state["selections"])
            selections[f"upload-{index}"] = [method]
            return {
                "doc_ids": sorted(set(state["doc_ids"]) | {f"upload-{index}"}),
                "selections": selections,
                "requested": sorted(set(state["requested"]) | {method}),
            }

        with session_scope() as session:
            return ContentRepository(session).upsert_state(key, merge=merge)

    _, errors = _race(choose)

    assert not errors, errors
    with session_scope() as session:
        state = ContentRepository(session).get(key)
    assert len(state["selections"]) == WRITERS
    for index in range(WRITERS):
        assert state["selections"][f"upload-{index}"] == [
            ["structure-only", "markdown"][index % 2]
        ]
    assert sorted(state["requested"]) == ["markdown", "structure-only"]


# ==========================================================================
# duplicate membership
# ==========================================================================
def test_attaching_one_upload_repeatedly_leaves_one_membership():
    """Idempotent by constraint, not by a check-then-insert that two threads
    can both pass."""
    key = "doc-contended-bytes"

    def attach(_index: int):
        with session_scope() as session:
            return ContentRepository(session).upsert_state(
                key, merge=lambda state: {"doc_ids": sorted(set(state["doc_ids"]) | {"upload-1"})}
            )

    _, errors = _race(attach)

    assert not errors, errors
    with session_scope() as session:
        assert session.query(ContentDocument).count() == 1


# ==========================================================================
# unique knowledge-base names
# ==========================================================================
def test_only_one_of_several_simultaneous_creations_of_a_name_succeeds():
    """The rule the console shows as "a knowledge base named X already exists".

    It used to be a scan of the in-memory records followed by a file write, and
    two requests could both pass the scan -- which is how eight knowledge bases
    with one name came to share a single empty store. The unique index decides
    it now, and the loser gets the ``ValueError`` the route already turns into
    a 400.
    """
    manager = KnowledgeBaseManager()

    def create(_index: int):
        try:
            return manager.create(name="Yillik raporlar")
        except ValueError as error:
            return error

    results, errors = _race(create)

    assert not errors, errors
    created = [r for r in results if isinstance(r, dict)]
    refused = [r for r in results if isinstance(r, ValueError)]
    assert len(created) == 1, f"{len(created)} knowledge bases were created"
    assert len(refused) == WRITERS - 1
    assert all("already exists" in str(error) for error in refused)
    assert len(manager.list()) == 1


# ==========================================================================
# competing ingest state
# ==========================================================================
def test_several_workers_finishing_one_document_leave_one_ledger_row():
    """A retried job, or two workers that somehow both finish the same
    document. The unique ``doc_id`` and the upsert make that one row carrying
    one of the writers' values -- not two rows the console lists as two
    documents, and not an IntegrityError surfacing as a failed ingest."""
    tracker = DocumentTracker()

    def record(index: int):
        return tracker.mark_as_ingested(
            file_path=f"/staging/upload-{index}.pdf", doc_id="doc-1",
            chunk_count=index, kb_id="kb-1",
        )

    results, errors = _race(record)

    assert not errors, errors
    assert all(results), "a writer reported a failure it did not have"
    with session_scope() as session:
        rows = DocumentRepository(session).list()
    assert len(rows) == 1 and rows[0]["doc_id"] == "doc-1"


def test_competing_updates_of_one_analysis_all_land():
    """Two things recording different facts about one document at once -- the
    request staging it and the worker recording its stage.

    Each write reads and updates under the row's lock, so neither can be merged
    over: both fields are there at the end, and so is the identity that ties
    the analysis to its console records.
    """
    key = "doc-contended-bytes"
    with session_scope() as session:
        ContentRepository(session).upsert_state(key, fields={
            "status": "pending", "doc_ids": ["upload-1"], "requested": ["structure-only"]})

    def update(index: int):
        field = "unit_count" if index % 2 else "payload_bytes"
        for value in range(20):
            with session_scope() as session:
                ContentRepository(session).upsert_state(
                    key, fields={field: index * 100 + value})
        return field

    _, errors = _race(update, count=4)

    assert not errors, errors
    with session_scope() as session:
        state = ContentRepository(session).get(key)
    assert state["unit_count"] is not None and state["payload_bytes"] is not None
    assert state["doc_ids"] == ["upload-1"], "identity survived both writers"
    assert state["requested"] == ["structure-only"]


# ==========================================================================
# destructive operations
# ==========================================================================
def test_two_deletions_of_the_last_upload_do_not_both_claim_it():
    """The reference-count rule, contended.

    Two requests delete two uploads of one content at the same moment. Exactly
    one of them must see "nothing points at this any more" -- if both did, both
    would remove the directory, and if neither did the analysis would be left
    owned by nobody and never cleaned up.
    """
    key = "doc-contended-bytes"
    with session_scope() as session:
        ContentRepository(session).upsert_state(
            key, fields={"doc_ids": ["upload-0", "upload-1"], "status": "ready"})

    def detach(index: int):
        with session_scope() as session:
            return ContentRepository(session).detach_document(key, f"upload-{index}")

    results, errors = _race(detach, count=2)

    assert not errors, errors
    assert sorted(results) == [0, 1], (
        "exactly one deletion must find itself last; both saw " + repr(results)
    )
    with session_scope() as session:
        assert session.query(ContentDocument).count() == 0


def test_deleting_a_knowledge_base_from_two_requests_deletes_it_once():
    manager = KnowledgeBaseManager()
    created = manager.create(name="Yillik raporlar")

    def delete(_index: int):
        return manager.delete_with_storage(created["kb_id"])

    results, errors = _race(delete)

    assert not errors, errors
    assert [r["deleted"] for r in results].count(True) == 1
    assert manager.list() == []
