"""The job registry is a cache, not a store: how a finished job leaves memory.

Three things could have made it grow without bound, and each has a test
here: a finished job never dropped because no new upload arrived to trigger
pruning; a registry that only obeys a time window and so grows without limit
under load; and a failed job holding its traceback, which holds every local
of every frame the ingest passed through -- the chunk list, the canonical
units, the pipeline.
"""

from __future__ import annotations

import gc
import threading
import time

import pytest

from components.ingest import jobs as J
from components.ingest.journal import JobJournal
from config.ingest import IngestLimits


class Immediate:
    def __init__(self, fail: set[str] | None = None, payload=None):
        self.fail = fail or set()
        self.payload = payload

    def __call__(self, job):
        if job.filename in self.fail:
            raise RuntimeError(f"{job.filename} exploded")
        return {"success": True, "doc_id": f"doc-{job.job_id}", "payload": self.payload}


@pytest.fixture
def staging(tmp_path):
    directory = tmp_path / "staging"
    directory.mkdir()
    return directory


def manager(execute, journal=None, **limits):
    fields = dict(workers=1, queue_capacity=20, job_timeout_seconds=60, job_retention_seconds=3600)
    fields.update(limits)
    return J.IngestManager(IngestLimits(**fields), execute=execute, journal=journal)


def submit(mgr, staging, name, *, kb="kb-a"):
    path = staging / f"upload_{name}.txt"
    path.write_bytes(name.encode())
    return mgr.submit(
        kb_id=kb, filename=f"{name}.txt", temp_path=str(path), session_id="s",
        methods=("structure-only",), deep_analysis=False, content_sha=f"sha-{name}",
    )


def test_a_finished_job_leaves_on_a_read_not_only_on_the_next_upload(staging, monkeypatch):
    """The failure mode this closes: uploads stop, the last jobs are past
    their window, and nothing ever comes along to prune them."""
    clock = {"now": 1000.0}
    monkeypatch.setattr(J.time, "monotonic", lambda: clock["now"])
    mgr = manager(Immediate(), job_retention_seconds=60)
    try:
        job, _ = submit(mgr, staging, "a")
        assert mgr.wait(job, 10)
        assert mgr.get(job.job_id) is job

        clock["now"] += 61  # past the window, and no new submission arrives
        assert mgr.snapshot()["retained"]["finished"] == 0, "a status read prunes"
        assert mgr.get(job.job_id) is None
        assert mgr.record_for(job.job_id) is None
        assert mgr.list() == []
    finally:
        mgr.close()


def test_the_registry_is_capped_however_long_the_window_is(staging, monkeypatch):
    monkeypatch.setattr(J, "MAX_FINISHED", 3)
    mgr = manager(Immediate(), job_retention_seconds=100_000)
    try:
        jobs = []
        for index in range(8):
            job, _ = submit(mgr, staging, f"n{index}")
            assert mgr.wait(job, 10)
            jobs.append(job)
        assert mgr.snapshot()["retained"]["finished"] == 3
        assert [mgr.get(job.job_id) is not None for job in jobs] == [False] * 5 + [True] * 3
    finally:
        mgr.close()


def test_the_journal_record_goes_when_the_job_does(staging, tmp_path, monkeypatch):
    import os

    monkeypatch.setattr(J, "MAX_FINISHED", 1)
    directory = tmp_path / "journal"
    mgr = manager(Immediate(), journal=JobJournal(str(directory)), job_retention_seconds=100_000)
    try:
        first, _ = submit(mgr, staging, "first")
        assert mgr.wait(first, 10)
        second, _ = submit(mgr, staging, "second")
        assert mgr.wait(second, 10)
        assert sorted(os.listdir(directory)) == [f"{second.job_id}.json"]
    finally:
        mgr.close()


def test_a_settled_restart_record_is_pruned_like_any_other(staging, tmp_path):
    directory = tmp_path / "journal"
    journal = JobJournal(str(directory))
    journal.record({"job_id": "old", "status": J.SUCCEEDED, "kb_id": "kb-a",
                    "journalled_at": time.time() - 30})
    mgr = manager(Immediate(), journal=journal, job_retention_seconds=10)
    try:
        mgr.recover(resolve_document=lambda _id: None)
        # recover() already refuses anything past the window.
        assert mgr.record_for("old") is None
        assert mgr.snapshot()["retained"]["restart_settled"] == 0
    finally:
        mgr.close()


def test_dropping_a_traceback_frees_what_its_frames_held():
    """The mechanism, measured where nothing else can hold a reference.

    ``logger.error(exc_info=True)`` is called before the strip, and a test
    runner's log capture keeps those records for the length of the test, so
    the frames are asserted on here rather than through a job.
    """
    import weakref

    class Bulky:
        """Stands in for the chunk list: big, and a local of the failing frame."""

    def explode():
        bulky = Bulky()
        assert bulky is not None
        raise RuntimeError("ingest failed with a big stack")

    try:
        explode()
    except RuntimeError as error:
        caught = error
    # The frame's local is reachable only through the traceback right now.
    frames = caught.__traceback__.tb_next.tb_frame
    watch = weakref.ref(frames.f_locals["bulky"])
    del frames
    assert watch() is not None, "the traceback is what keeps it alive"

    J._drop_traceback(caught)
    del caught
    gc.collect()
    assert watch() is None, "dropping the traceback released the frame's locals"


def test_a_failed_job_keeps_its_exception_but_not_its_frames(staging):
    """The route reads the exception's type to choose a status code, so the
    object stays; the frames it arrived with do not."""
    def explode(job):
        raise RuntimeError("ingest failed with a big stack")

    mgr = manager(explode)
    try:
        job, _ = submit(mgr, staging, "bad")
        assert mgr.wait(job, 10)
        assert job.status == J.FAILED
        assert isinstance(job.exception, RuntimeError), "the type the route branches on"
        assert job.exception.__traceback__ is None
        assert job.exception.__context__ is None and job.exception.__cause__ is None
        assert "big stack" in job.error
    finally:
        mgr.close()


def test_a_chained_exception_is_unlinked_all_the_way_down(staging):
    def explode(job):
        try:
            raise ValueError("the root cause")
        except ValueError as inner:
            raise RuntimeError("the wrapper") from inner

    mgr = manager(explode)
    try:
        job, _ = submit(mgr, staging, "bad")
        assert mgr.wait(job, 10)
        assert job.exception.__cause__ is None
        assert job.exception.__traceback__ is None
    finally:
        mgr.close()


def test_the_snapshot_reports_the_retention_policy(staging):
    mgr = manager(Immediate(), job_retention_seconds=42)
    try:
        retained = mgr.snapshot()["retained"]
        assert retained == {"finished": 0, "restart_settled": 0,
                            "max": J.MAX_FINISHED, "retention_seconds": 42}
    finally:
        mgr.close()
