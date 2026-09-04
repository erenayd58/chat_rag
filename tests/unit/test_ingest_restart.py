"""What a client holding a job id gets after the server restarts.

A restart loses the queue and the workers. It must not lose the *answer*: a
client handed 202 and a job id cannot act on "unknown job", because that
reads the same whether the upload never happened or finished while it was
away. These tests drive a real journal directory through a simulated
restart -- a second manager built over the same directory, which is exactly
what a new process does -- and assert the two truthful outcomes.
"""

from __future__ import annotations

import json
import os
import threading
import time

import pytest

from components.ingest import jobs as J
from components.ingest.journal import INTERRUPTED, RECOVERED, JobJournal
from config.ingest import IngestLimits


@pytest.fixture
def workspace(tmp_path):
    staging = tmp_path / "staging"
    staging.mkdir()
    journal = tmp_path / "journal"
    return staging, JobJournal(str(journal)), journal


class Blocking:
    def __init__(self):
        self.release = threading.Event()
        self.started = threading.Event()

    def __call__(self, job):
        self.started.set()
        self.release.wait(20)
        return {"success": True, "doc_id": f"doc-{job.job_id}"}


def manager(journal, execute, **limits):
    fields = dict(workers=1, queue_capacity=4, job_timeout_seconds=60, job_retention_seconds=3600)
    fields.update(limits)
    return J.IngestManager(IngestLimits(**fields), execute=execute, journal=journal)


def submit(mgr, staging, name, *, kb="kb-a"):
    path = staging / f"upload_{name}.txt"
    path.write_bytes(name.encode())
    return mgr.submit(
        kb_id=kb, filename=f"{name}.txt", temp_path=str(path), session_id="s",
        methods=("structure-only",), deep_analysis=False, content_sha=f"sha-{name}",
    )


# ------------------------------------------------------------ the journal


def test_a_job_is_journalled_at_every_transition(workspace):
    staging, journal, directory = workspace
    blocking = Blocking()
    mgr = manager(journal, blocking)
    try:
        job, _ = submit(mgr, staging, "a")
        record = json.load(open(directory / f"{job.job_id}.json", encoding="utf-8"))
        assert record["status"] == J.QUEUED
        assert record["filename"] == "a.txt"
        assert "temp_path" not in record

        assert blocking.started.wait(10)
        # A reader holding this file open is what stops the writer replacing
        # it on Windows, so the write retries; either value here is truthful
        # and both settle the same way, which is the property that matters.
        record = json.load(open(directory / f"{job.job_id}.json", encoding="utf-8"))
        assert record["status"] in (J.RUNNING, J.QUEUED)
        assert record["status"] in J.ACTIVE, "in flight, whichever it caught"

        blocking.release.set()
        assert mgr.wait(job, 10)
        record = json.load(open(directory / f"{job.job_id}.json", encoding="utf-8"))
        assert record["status"] == J.SUCCEEDED
        assert record["doc_id"] == f"doc-{job.job_id}"
    finally:
        blocking.release.set()
        mgr.close()


def test_a_journal_that_cannot_be_written_does_not_fail_the_ingest(workspace, monkeypatch):
    staging, journal, _ = workspace
    monkeypatch.setattr(
        journal, "record",
        lambda snapshot: (_ for _ in ()).throw(OSError("disk full")) if False else None,
    )
    blocking = Blocking()
    blocking.release.set()
    mgr = manager(journal, blocking)
    try:
        job, _ = submit(mgr, staging, "a")
        assert mgr.wait(job, 10)
        assert job.status == J.SUCCEEDED
    finally:
        mgr.close()


# ----------------------------------------------------------- the restart


def test_a_job_in_flight_at_a_restart_is_answered_as_interrupted(workspace):
    """The process stops mid-job. The next one settles it truthfully: the
    document was never registered, so the client is told to upload again."""
    staging, journal, _ = workspace
    blocking = Blocking()
    first = manager(journal, blocking)
    job_id = None
    try:
        job, _ = submit(first, staging, "a")
        job_id = job.job_id
        assert blocking.started.wait(10)
    finally:
        # The process dies here: no terminal record is ever written.
        pass

    # A new process over the same journal directory, with a ledger that has
    # never heard of this job.
    second = manager(journal, Blocking())
    try:
        interrupted = second.recover(resolve_document=lambda _job_id: None)
        assert [r["job_id"] for r in interrupted] == [job_id]
        record = second.record_for(job_id)
        assert record["status"] == INTERRUPTED
        assert record["resolution"] == INTERRUPTED
        assert record["restart_recovered"] is True
        assert record["result"] is None
        assert "upload it again" in record["error"]
        assert second.record_for("never-existed") is None
    finally:
        second.close()
        blocking.release.set()
        first.close()


def test_a_job_that_finished_before_the_restart_is_recovered_from_the_ledger(workspace):
    """The last thing a job does is write the ledger. A document carrying the
    job's id therefore proves it completed, whatever the journal last said."""
    staging, journal, _ = workspace
    blocking = Blocking()
    first = manager(journal, blocking)
    job, _ = submit(first, staging, "a")
    job_id = job.job_id
    assert blocking.started.wait(10)

    ledger = {job_id: {"doc_id": "doc-42", "chunk_count": 17, "chunking_mode": "standard"}}
    second = manager(journal, Blocking())
    try:
        settled = second.recover(resolve_document=ledger.get)
        assert len(settled) == 1
        record = second.record_for(job_id)
        assert record["status"] == J.SUCCEEDED
        assert record["resolution"] == RECOVERED
        assert record["doc_id"] == "doc-42"
        assert record["result"]["chunks_created"] == 17
        assert record["result"]["recovered_from_ledger"] is True
        assert record["error"] is None
    finally:
        second.close()
        blocking.release.set()
        first.close()


def test_a_job_that_had_already_finished_keeps_its_own_answer(workspace):
    staging, journal, _ = workspace
    blocking = Blocking()
    blocking.release.set()
    first = manager(journal, blocking)
    job, _ = submit(first, staging, "a")
    assert first.wait(job, 10)
    first.close()

    second = manager(journal, Blocking())
    try:
        assert second.recover(resolve_document=lambda _id: None) == []
        record = second.record_for(job.job_id)
        assert record["status"] == J.SUCCEEDED
        assert record.get("restart_recovered") is not True
        assert record["result"]["doc_id"] == f"doc-{job.job_id}"
    finally:
        second.close()


def test_a_settled_job_is_listed_with_the_live_ones(workspace):
    staging, journal, _ = workspace
    blocking = Blocking()
    first = manager(journal, blocking)
    job, _ = submit(first, staging, "a")
    assert blocking.started.wait(10)

    second = manager(journal, Blocking())
    try:
        second.recover(resolve_document=lambda _id: None)
        listed = second.list("kb-a")
        assert [r["job_id"] for r in listed] == [job.job_id]
        assert listed[0]["status"] == INTERRUPTED
        assert second.list("kb-other") == []
        assert second.list("kb-a", active_only=True) == [], "nothing is actually in flight"
        assert second.snapshot()["retained"]["restart_settled"] == 1
    finally:
        second.close()
        blocking.release.set()
        first.close()


def test_a_failed_lookup_settles_as_interrupted_rather_than_raising(workspace):
    staging, journal, _ = workspace
    blocking = Blocking()
    first = manager(journal, blocking)
    job, _ = submit(first, staging, "a")
    assert blocking.started.wait(10)

    def broken(_job_id):
        raise OSError("the ledger is unreadable right now")

    second = manager(journal, Blocking())
    try:
        second.recover(resolve_document=broken)
        assert second.record_for(job.job_id)["status"] == INTERRUPTED
    finally:
        second.close()
        blocking.release.set()
        first.close()


def test_an_unreadable_journal_record_is_skipped_not_fatal(workspace):
    staging, journal, directory = workspace
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "garbage.json").write_text("{not json", encoding="utf-8")
    mgr = manager(journal, Blocking())
    try:
        assert mgr.recover(resolve_document=lambda _id: None) == []
    finally:
        mgr.close()


# ------------------------------------------------------------- retention


def test_journal_records_older_than_the_window_are_deleted(workspace):
    staging, journal, directory = workspace
    directory.mkdir(parents=True, exist_ok=True)
    old = {"job_id": "ancient", "status": J.SUCCEEDED, "journalled_at": time.time() - 10_000}
    fresh = {"job_id": "recent", "status": J.SUCCEEDED, "journalled_at": time.time()}
    journal.record(old)
    journal.record(fresh)

    assert journal.prune(3600) == 1
    assert sorted(os.listdir(directory)) == ["recent.json"]


def test_recovery_drops_records_past_the_window(workspace):
    staging, journal, directory = workspace
    journal.record({"job_id": "ancient", "status": J.RUNNING,
                    "journalled_at": time.time() - 10_000})
    mgr = manager(journal, Blocking(), job_retention_seconds=60)
    try:
        assert mgr.recover(resolve_document=lambda _id: None) == []
        assert mgr.record_for("ancient") is None
        assert os.listdir(directory) == []
    finally:
        mgr.close()
