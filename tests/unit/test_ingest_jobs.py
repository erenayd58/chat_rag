"""The job manager: admission, the queue bound, the lifecycle, the file.

Every test drives a manager of its own with a fake ``execute`` that blocks on
events the test holds, so what is asserted is an ordering the test forced,
never one it waited for. The doubles never parse, embed or call anything.
"""

from __future__ import annotations

import os
import threading

import pytest

from chat_rag.components.ingest import jobs as J
from chat_rag.components.ingest.limits import JobGuard, current_guard
from chat_rag.config.ingest import IngestLimits
from chat_rag.core.exceptions import IngestOverloaded


class Execution:
    """A fake ingest: each job blocks until released, and the double keeps
    count of how many are inside at once."""

    def __init__(self, *, fail: set[str] | None = None):
        self.lock = threading.Lock()
        self.inside = 0
        self.peak = 0
        self.started: dict[str, threading.Event] = {}
        self.release = threading.Event()
        self.fail = fail or set()
        self.seen_files: dict[str, bool] = {}

    def __call__(self, job: J.IngestJob) -> dict:
        with self.lock:
            self.inside += 1
            self.peak = max(self.peak, self.inside)
            self.seen_files[job.job_id] = os.path.isfile(job.temp_path)
            self.started.setdefault(job.job_id, threading.Event()).set()
        try:
            self.release.wait(20)
            current_guard().check()
            if job.filename in self.fail:
                raise RuntimeError(f"{job.filename} exploded")
            return {"success": True, "doc_id": f"doc-{job.job_id}", "filename": job.filename}
        finally:
            with self.lock:
                self.inside -= 1

    def wait_started(self, job: J.IngestJob, timeout: float = 10.0) -> bool:
        with self.lock:
            event = self.started.setdefault(job.job_id, threading.Event())
        return event.wait(timeout)


@pytest.fixture
def staging(tmp_path):
    directory = tmp_path / "staging"
    directory.mkdir()
    return directory


def make_manager(execution, **limits):
    fields = dict(workers=1, queue_capacity=2, job_timeout_seconds=60, job_retention_seconds=3600)
    fields.update(limits)
    return J.IngestManager(IngestLimits(**fields), execute=execution, name="test")


def submit(manager, staging, name, *, kb="kb-a", methods=("structure-only",), deep=False):
    path = staging / f"upload_{name}.txt"
    path.write_bytes(name.encode())
    return manager.submit(
        kb_id=kb, filename=f"{name}.txt", temp_path=str(path), session_id="s",
        methods=methods, deep_analysis=deep, content_sha=f"sha-{name}",
    )


def files(staging) -> list[str]:
    return sorted(p.name for p in staging.iterdir() if p.is_file())


# --------------------------------------------------------------- admission


def test_the_queue_never_exceeds_its_bound_and_overload_is_deterministic(staging):
    execution = Execution()
    manager = make_manager(execution, workers=1, queue_capacity=2)
    try:
        first, _ = submit(manager, staging, "a")
        assert execution.wait_started(first)
        second, _ = submit(manager, staging, "b")
        third, _ = submit(manager, staging, "c")
        assert manager.snapshot()["queued"] == 2
        assert manager.position(second) == 1 and manager.position(third) == 2

        with pytest.raises(IngestOverloaded) as refused:
            submit(manager, staging, "d")
        assert refused.value.retry_after_seconds >= 5
        assert manager.snapshot()["queued"] == 2, "the refused job was never queued"
        assert manager.stats["rejected"] == 1
        assert "upload_d.txt" not in files(staging), "the refused file is gone at once"

        # The same refusal again: overload is a function of state, not luck.
        with pytest.raises(IngestOverloaded):
            submit(manager, staging, "e")
        assert manager.stats["rejected"] == 2

        execution.release.set()
        assert manager.drain(10)
        assert manager.stats["peak_queued"] == 2
        assert execution.peak == 1
        # Room again: the next submission is accepted.
        job, _ = submit(manager, staging, "f")
        assert manager.wait(job, 10) and job.status == J.SUCCEEDED
    finally:
        execution.release.set()
        manager.close()
    assert files(staging) == []


def test_admission_counts_running_and_waiting_together(staging):
    execution = Execution()
    manager = make_manager(execution, workers=2, queue_capacity=0)
    try:
        a, _ = submit(manager, staging, "a", kb="kb-a")
        b, _ = submit(manager, staging, "b", kb="kb-b")
        assert execution.wait_started(a) and execution.wait_started(b)
        with pytest.raises(IngestOverloaded):
            submit(manager, staging, "c", kb="kb-c")
    finally:
        execution.release.set()
        manager.close()


def test_more_submitters_than_workers_do_not_mean_more_execution(staging):
    """Twelve threads submit at once; two workers exist; two jobs run."""
    execution = Execution()
    manager = make_manager(execution, workers=2, queue_capacity=10)
    barrier = threading.Barrier(12)
    outcomes: list = []
    lock = threading.Lock()

    def submitter(index):
        barrier.wait(10)
        try:
            job, _ = submit(manager, staging, f"n{index}", kb=f"kb-{index}")
            with lock:
                outcomes.append(job)
        except IngestOverloaded as refused:
            with lock:
                outcomes.append(refused)

    threads = [threading.Thread(target=submitter, args=(i,)) for i in range(12)]
    try:
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(10)
        accepted = [o for o in outcomes if isinstance(o, J.IngestJob)]
        assert len(accepted) == 12, "workers plus queue is exactly twelve"
        running = [job for job in accepted if execution.wait_started(job, timeout=0.5)]
        assert len(running) == 2
        assert execution.peak == 2
        assert manager.snapshot()["running"] == 2 and manager.snapshot()["queued"] == 10
        execution.release.set()
        assert manager.drain(20)
        assert execution.peak == 2
        assert manager.stats["peak_active"] == 2
        assert all(job.status == J.SUCCEEDED for job in accepted)
    finally:
        execution.release.set()
        manager.close()
    assert files(staging) == []


# ---------------------------------------------------------------- identity


def test_the_same_document_twice_in_flight_is_one_job(staging):
    execution = Execution()
    manager = make_manager(execution)
    try:
        first, attached_first = submit(manager, staging, "same")
        (staging / "upload_same-2.txt").write_bytes(b"same")
        second, attached_second = manager.submit(
            kb_id="kb-a", filename="same.txt", temp_path=str(staging / "upload_same-2.txt"),
            session_id="s", methods=("structure-only",), deep_analysis=False, content_sha="sha-same",
        )
        assert attached_first is False and attached_second is True
        assert second is first
        assert first.attached_count == 1
        assert manager.stats == {**manager.stats, "submitted": 2, "accepted": 1, "attached": 1}
        assert files(staging) == ["upload_same.txt"], "the twin's file was discarded"
        execution.release.set()
        assert manager.wait(first, 10)
        assert first.status == J.SUCCEEDED
        # Once finished, the same content is a new job again (a re-upload
        # is allowed, as it always was).
        again, attached = submit(manager, staging, "same")
        assert attached is False and again is not first
    finally:
        execution.release.set()
        manager.close()


def test_different_methods_for_one_file_are_different_jobs(staging):
    execution = Execution()
    manager = make_manager(execution, queue_capacity=4)
    try:
        standard, _ = submit(manager, staging, "same", methods=("structure-only",))
        (staging / "upload_same-deep.txt").write_bytes(b"same")
        deep, attached = manager.submit(
            kb_id="kb-a", filename="same.txt", temp_path=str(staging / "upload_same-deep.txt"),
            session_id="s", methods=("structure-only", "agentic"), deep_analysis=True, content_sha="sha-same",
        )
        assert attached is False and deep is not standard
    finally:
        execution.release.set()
        manager.close()


# ------------------------------------------------- one writer per knowledge base


def test_one_ingest_runs_per_knowledge_base_while_others_proceed(staging):
    execution = Execution()
    manager = make_manager(execution, workers=2, queue_capacity=4)
    try:
        a, _ = submit(manager, staging, "a", kb="kb-1")
        assert execution.wait_started(a)
        b, _ = submit(manager, staging, "b", kb="kb-1")
        c, _ = submit(manager, staging, "c", kb="kb-2")
        assert execution.wait_started(c), "a different knowledge base is not held up"
        assert not execution.wait_started(b, timeout=0.2), "the same knowledge base waits"
        assert b.status == J.QUEUED and manager.position(b) == 1
        assert manager.snapshot()["busy_knowledge_bases"] == ["kb-1", "kb-2"]
        execution.release.set()
        assert manager.drain(10)
        assert [j.status for j in (a, b, c)] == [J.SUCCEEDED] * 3
    finally:
        execution.release.set()
        manager.close()


# ------------------------------------------------------------ terminal states


def test_a_failed_job_releases_its_worker_and_its_file(staging):
    execution = Execution(fail={"bad.txt"})
    manager = make_manager(execution)
    try:
        bad, _ = submit(manager, staging, "bad")
        execution.release.set()
        assert manager.wait(bad, 10)
        assert bad.status == J.FAILED
        assert "exploded" in bad.error
        assert isinstance(bad.exception, RuntimeError)
        assert manager.snapshot()["running"] == 0
        assert files(staging) == []
        good, _ = submit(manager, staging, "good")
        assert manager.wait(good, 10) and good.status == J.SUCCEEDED
        assert manager.stats["failed"] == 1 and manager.stats["succeeded"] == 1
    finally:
        execution.release.set()
        manager.close()


def test_a_job_past_its_deadline_ends_timed_out_with_nothing_returned(staging, monkeypatch):
    execution = Execution()
    manager = make_manager(execution)
    # A guard that is already over: the job runs, reaches its first seam
    # and stops there. No clock is waited on.
    monkeypatch.setattr(J.JobGuard, "for_timeout", classmethod(lambda cls, s: JobGuard(deadline=-1.0)))
    try:
        late, _ = submit(manager, staging, "late")
        execution.release.set()
        assert manager.wait(late, 10)
        assert late.status == J.TIMED_OUT
        assert late.result is None
        assert "deadline" in late.error
        assert files(staging) == []
        assert manager.stats["timed_out"] == 1
    finally:
        execution.release.set()
        manager.close()


def test_cancelling_a_queued_job_removes_it_at_once(staging):
    execution = Execution()
    manager = make_manager(execution)
    try:
        running, _ = submit(manager, staging, "running")
        assert execution.wait_started(running)
        waiting, _ = submit(manager, staging, "waiting")
        cancelled = manager.cancel(waiting.job_id)
        assert cancelled is waiting and waiting.status == J.CANCELLED
        assert waiting.done.is_set()
        assert manager.snapshot()["queued"] == 0
        assert files(staging) == ["upload_running.txt"]
        # Its identity is free again.
        again, attached = submit(manager, staging, "waiting")
        assert attached is False and again is not waiting
    finally:
        execution.release.set()
        manager.close()


def test_cancelling_a_running_job_stops_it_at_its_next_seam(staging):
    execution = Execution()
    manager = make_manager(execution)
    try:
        running, _ = submit(manager, staging, "running")
        assert execution.wait_started(running)
        assert manager.cancel(running.job_id, "operator") is running
        assert running.status == J.RUNNING, "not interrupted mid-stage"
        assert running.guard.cancelled
        execution.release.set()  # the fake reaches its seam and checks the guard
        assert manager.wait(running, 10)
        assert running.status == J.CANCELLED
        assert running.result is None
        assert files(staging) == []
    finally:
        execution.release.set()
        manager.close()


def test_cancelling_an_unknown_or_finished_job_changes_nothing(staging):
    execution = Execution()
    manager = make_manager(execution)
    try:
        assert manager.cancel("nope") is None
        job, _ = submit(manager, staging, "done")
        execution.release.set()
        assert manager.wait(job, 10)
        assert manager.cancel(job.job_id) is job
        assert job.status == J.SUCCEEDED
    finally:
        manager.close()


def test_a_finished_job_is_forgotten_after_retention(staging, monkeypatch):
    execution = Execution()
    execution.release.set()
    manager = make_manager(execution)
    monkeypatch.setattr(J, "MAX_FINISHED", 1)
    try:
        first, _ = submit(manager, staging, "first")
        assert manager.wait(first, 10)
        second, _ = submit(manager, staging, "second")
        assert manager.wait(second, 10)
        assert manager.get(second.job_id) is second
        assert manager.get(first.job_id) is None
        assert manager.get("unknown") is None
    finally:
        manager.close()


def test_the_snapshot_shows_what_the_api_needs_and_nothing_private(staging):
    execution = Execution()
    manager = make_manager(execution)
    try:
        job, _ = submit(manager, staging, "shown")
        described = manager.describe(job)
        assert described["status"] in (J.QUEUED, J.RUNNING)
        assert described["filename"] == "shown.txt"
        assert "temp_path" not in described and "exception" not in described
        listing = manager.list("kb-a", active_only=True)
        assert [entry["job_id"] for entry in listing] == [job.job_id]
        assert manager.list("kb-other") == []
    finally:
        execution.release.set()
        manager.close()


def test_the_file_is_still_there_while_the_job_runs_and_gone_after(staging):
    execution = Execution()
    manager = make_manager(execution)
    try:
        job, _ = submit(manager, staging, "kept")
        assert execution.wait_started(job)
        assert execution.seen_files[job.job_id] is True
        execution.release.set()
        assert manager.wait(job, 10)
    finally:
        manager.close()
    assert files(staging) == []


def test_sweeping_the_staging_directory_removes_orphans(staging):
    (staging / "upload_orphan.pdf").write_bytes(b"%PDF")
    (staging / "nested").mkdir()
    removed = J.sweep_staging(str(staging))
    assert [os.path.basename(p) for p in removed] == ["upload_orphan.pdf"]
    assert files(staging) == []
    assert J.sweep_staging(str(staging / "missing")) == []
