"""Ingest jobs: bounded admission, a small worker pool, an explicit lifecycle.

An upload used to be one request thread doing everything -- parse, chunk,
call the model, embed, write the store, write the ledger -- for as long as
that took, and eight of them could do it at once because eight was the
server's thread count. This module puts the same work behind a job:

    submitted --> queued --> running --> succeeded
                   |            |------> failed
                   |            |------> timed_out
                   |            '------> cancelled
                   '--> cancelled
    (refused at the door: overloaded -- never a job, never retained)

* **Admission** is decided under one lock. A job is accepted while fewer
  than ``INGEST_WORKERS + INGEST_QUEUE_CAPACITY`` jobs are running or
  waiting, and refused with :class:`IngestOverloaded` otherwise; so the
  process never runs more than ``INGEST_WORKERS`` at once and, with every
  worker busy, never holds more than ``INGEST_QUEUE_CAPACITY`` waiting. A
  refusal is a response, not a state: retaining refused submissions would
  be the unbounded growth this exists to prevent.
* **Identity** is the knowledge base, the file's content hash and the
  requested methods. A second upload with the same identity while the first
  is still queued or running is *attached* to it: the caller gets the same
  job back and the second file is discarded. Two different method sets for
  one file are two jobs, as they always were.
* **One ingest per knowledge base at a time.** Two writers to one store
  raced on the BM25 rebuild and on the chunker's per-ingest state, so a
  worker skips a queued job whose knowledge base is busy and takes the next
  one; different knowledge bases run in parallel up to the worker count.
* **The file** belongs to the job from submission to its terminal state and
  is deleted in every one of them, including refusal and attachment. A job
  exists only in memory, so after a restart a file still in the staging
  directory belongs to nobody; :func:`sweep_staging` removes it at start-up.
* **Restart** loses queued and running jobs -- nothing is resumed and
  nothing half-done survives, because nothing is committed before the store
  write and the ledger write at the very end of a job. What a restart must
  *not* lose is the answer owed to a client holding a ``job_id``, so every
  job journals three records (accepted, started, terminal) and the next
  start-up settles whatever was in flight: ``succeeded`` when the ledger
  holds a document that job wrote, ``interrupted`` when it does not. See
  ``journal.py``; the ledger stays the authority for what was ingested.

This is not a workflow engine: one queue, one kind of job, one function that
runs it. The function is injected so the machinery is testable without a
parser, a store or a model.
"""

from __future__ import annotations

import logging
import os
import statistics
import threading
import time
import uuid
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Optional

from chat_rag.config.ingest import IngestLimits
from chat_rag.core.exceptions import IngestInterrupted, IngestOverloaded

from chat_rag.components.observability import events
from chat_rag.components.observability import telemetry as T

from .journal import INTERRUPTED, JobJournal
from .limits import JobGuard, budgets, use_guard

logger = logging.getLogger("chat_rag.ingest")

QUEUED = "queued"
RUNNING = "running"
SUCCEEDED = "succeeded"
FAILED = "failed"
TIMED_OUT = "timed_out"
CANCELLED = "cancelled"

ACTIVE = frozenset({QUEUED, RUNNING})
TERMINAL = frozenset({SUCCEEDED, FAILED, TIMED_OUT, CANCELLED, INTERRUPTED})

#: Finished jobs kept for polling, whatever the retention time says. The
#: registry is bounded by this *and* by the retention window: whichever bites
#: first. A job record is a few hundred bytes plus the result summary, so five
#: hundred of them is well under a megabyte and cannot grow past it.
MAX_FINISHED = 500


def _discard_file(path: Optional[str]) -> None:
    """Best effort: a file that is already gone is not a failure."""
    if not path:
        return
    try:
        os.remove(path)
    except FileNotFoundError:
        pass
    except OSError as error:  # noqa: BLE001 - a leftover file is not a failed job
        logger.warning("could not remove the staged upload %s: %s", path, error)


def _drop_traceback(error: Optional[BaseException]) -> None:
    """Keep a failed job's exception, drop the frames hanging off it.

    A traceback holds every local of every frame it passed through -- for an
    ingest that is the chunk list, the canonical units and the pipeline. The
    exception itself is a few hundred bytes and the route still needs its type
    to choose a status code, so the object stays and the frames go.
    """
    while error is not None:
        error.__traceback__ = None
        following = error.__cause__ or error.__context__
        error.__cause__ = None
        error.__context__ = None
        error = following


def sweep_staging(directory: str) -> list[str]:
    """Remove every file in the staging directory; return what was removed.

    Called once at start-up, before any job can be submitted. Jobs live in
    memory, so nothing in the directory can belong to one.
    """
    removed: list[str] = []
    try:
        names = os.listdir(directory)
    except FileNotFoundError:
        return removed
    for name in names:
        path = os.path.join(directory, name)
        if os.path.isfile(path):
            _discard_file(path)
            removed.append(path)
    return removed


@dataclass
class IngestJob:
    job_id: str
    kb_id: str
    filename: str
    temp_path: str
    session_id: str
    methods: tuple[str, ...]
    deep_analysis: bool
    chunking_mode: str
    content_sha: str
    identity: tuple
    status: str = QUEUED
    submitted_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    error: Optional[str] = None
    exception: Optional[BaseException] = None
    result: Optional[dict] = None
    doc_id: Optional[str] = None
    attached_count: int = 0
    #: Set on a record settled by a restart rather than observed running.
    restart_recovered: bool = False
    resolution: Optional[str] = None
    error_category: str = "none"
    guard: JobGuard = field(default_factory=JobGuard)
    done: threading.Event = field(default_factory=threading.Event)
    #: Where this job's time went. Written by the stages the pipeline opens
    #: (components/observability/telemetry.py) and read by /api/ops/metrics.
    trace: Optional[T.JobTrace] = None
    _submitted_mono: float = field(default_factory=time.monotonic)
    _started_mono: Optional[float] = None
    _finished_mono: Optional[float] = None

    @property
    def active(self) -> bool:
        return self.status in ACTIVE

    def snapshot(self, position: Optional[int] = None) -> dict[str, Any]:
        """What the API shows. Never the temp path, never an exception object."""
        seconds = None
        if self._started_mono is not None:
            end = self._finished_mono if self._finished_mono is not None else time.monotonic()
            seconds = round(end - self._started_mono, 2)
        return {
            "job_id": self.job_id,
            "status": self.status,
            "kb_id": self.kb_id,
            "filename": self.filename,
            "chunking_mode": self.chunking_mode,
            "methods": list(self.methods),
            "content_sha256": self.content_sha,
            "submitted_at": self.submitted_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "position": position,
            "run_seconds": seconds,
            "attached_uploads": self.attached_count,
            "doc_id": self.doc_id,
            "error": self.error,
            "result": self.result,
            "restart_recovered": self.restart_recovered,
            "resolution": self.resolution,
            "error_category": self.error_category,
            # Where the time went, for this one job. Counts and seconds only.
            "timing": self.trace.as_dict() if self.trace is not None else None,
            # Wall-clock, for the journal's retention window. The rest of the
            # timing is monotonic and meaningless across a restart.
            "journalled_at": time.time(),
        }


class IngestManager:
    """The queue, the workers and the lifecycle, behind one condition."""

    def __init__(
        self,
        limits: IngestLimits,
        execute: Callable[[IngestJob], dict],
        *,
        name: str = "ingest",
        journal: Optional[JobJournal] = None,
    ):
        self.limits = limits.validate()
        self._execute = execute
        self._name = name
        self._journal = journal
        #: Records settled by a restart: terminal, read-only, pruned with the
        #: same window as the live registry so this cannot grow either.
        self._recovered: "OrderedDict[str, dict]" = OrderedDict()
        self._cond = threading.Condition()
        self._queue: deque[IngestJob] = deque()
        self._running: dict[str, IngestJob] = {}
        self._finished: "OrderedDict[str, IngestJob]" = OrderedDict()
        self._by_identity: dict[tuple, IngestJob] = {}
        self._busy_kbs: set[str] = set()
        self._workers: list[threading.Thread] = []
        self._stopping = False
        self._recent_seconds: deque[float] = deque(maxlen=20)
        self.stats: dict[str, int] = {
            "submitted": 0, "accepted": 0, "attached": 0, "rejected": 0,
            "succeeded": 0, "failed": 0, "timed_out": 0, "cancelled": 0,
            "peak_active": 0, "peak_queued": 0,
        }

    # ------------------------------------------------------------ admission
    def submit(
        self,
        *,
        kb_id: str,
        filename: str,
        temp_path: str,
        session_id: str,
        methods: tuple[str, ...] | list[str],
        deep_analysis: bool,
        content_sha: str,
    ) -> tuple[IngestJob, bool]:
        """Queue a job, attach to its twin, or refuse.

        Owns ``temp_path`` from here on in every outcome. Returns the job and
        whether this submission was attached to one already in flight. Raises
        :class:`IngestOverloaded` when the queue is full; the file is gone by
        then.
        """
        methods = tuple(methods)
        identity = (kb_id, content_sha, bool(deep_analysis), methods)
        with self._cond:
            self.stats["submitted"] += 1
            twin = self._by_identity.get(identity)
            if twin is not None and twin.active:
                twin.attached_count += 1
                self.stats["attached"] += 1
                job, attached = twin, True
                events.emit("ingest.job.attached", job_id=twin.job_id, kb_id=kb_id,
                            filename=filename, attached_uploads=twin.attached_count)
                T.metrics().count("ingest.attached")
            elif len(self._running) + len(self._queue) >= self.limits.admission_capacity:
                # Running plus waiting is the bound: with every worker busy
                # the queue holds at most INGEST_QUEUE_CAPACITY jobs, and an
                # idle worker is one more job that can be taken at once.
                self.stats["rejected"] += 1
                retry_after = self._retry_after_locked()
                overload = IngestOverloaded(
                    "The ingest queue is full "
                    f"({len(self._running)} running, {len(self._queue)} waiting); "
                    "try again shortly.",
                    retry_after_seconds=retry_after,
                )
                job, attached = None, False
                events.warn("ingest.job.rejected", kb_id=kb_id, filename=filename,
                            running=len(self._running), queued=len(self._queue),
                            queue_capacity=self.limits.queue_capacity,
                            retry_after_seconds=retry_after)
                T.metrics().count("ingest.rejected")
                T.metrics().record_error("overloaded", str(overload))
            else:
                job = IngestJob(
                    job_id=uuid.uuid4().hex[:12],
                    kb_id=kb_id,
                    filename=filename,
                    temp_path=temp_path,
                    session_id=session_id,
                    methods=methods,
                    deep_analysis=bool(deep_analysis),
                    chunking_mode="deep_analysis" if deep_analysis else "standard",
                    content_sha=content_sha,
                    identity=identity,
                )
                self._queue.append(job)
                self._by_identity[identity] = job
                self._journal_locked(job)
                self.stats["accepted"] += 1
                events.emit("ingest.job.accepted", job_id=job.job_id, kb_id=kb_id,
                            filename=filename, mode=job.chunking_mode,
                            methods=",".join(methods), queued=len(self._queue),
                            running=len(self._running))
                T.metrics().count("ingest.accepted")
                self.stats["peak_queued"] = max(self.stats["peak_queued"], len(self._queue))
                self._prune_finished_locked()
                self._ensure_workers_locked()
                self._cond.notify()
                attached = False
        if job is None:
            _discard_file(temp_path)
            raise overload
        if attached:
            _discard_file(temp_path)
        return job, attached

    def _retry_after_locked(self) -> float:
        if self._recent_seconds:
            typical = statistics.median(self._recent_seconds)
        else:
            typical = 30.0
        return float(max(5.0, min(300.0, round(typical))))

    # -------------------------------------------------------------- queries
    def get(self, job_id: str) -> Optional[IngestJob]:
        with self._cond:
            self._prune_finished_locked()
            return self._lookup_locked(job_id)

    def record_for(self, job_id: str) -> Optional[dict[str, Any]]:
        """One job as the API shows it, live or settled by a restart.

        This is what a status request asks: a client holding a ``job_id`` from
        before a restart gets the settled record, not a bare 404.
        """
        with self._cond:
            self._prune_finished_locked()
            job = self._lookup_locked(job_id)
            if job is not None:
                return job.snapshot(self._position_locked(job))
            record = self._recovered.get(job_id)
            return dict(record) if record is not None else None

    def _lookup_locked(self, job_id: str) -> Optional[IngestJob]:
        job = self._running.get(job_id) or self._finished.get(job_id)
        if job is None:
            for queued in self._queue:
                if queued.job_id == job_id:
                    return queued
        return job

    def position(self, job: IngestJob) -> Optional[int]:
        with self._cond:
            return self._position_locked(job)

    def _position_locked(self, job: IngestJob) -> Optional[int]:
        for index, queued in enumerate(self._queue, start=1):
            if queued is job:
                return index
        return None

    def describe(self, job: IngestJob) -> dict[str, Any]:
        with self._cond:
            return job.snapshot(self._position_locked(job))

    def list(self, kb_id: Optional[str] = None, *, active_only: bool = False) -> list[dict[str, Any]]:
        with self._cond:
            self._prune_finished_locked()
            jobs: list[IngestJob] = list(self._queue) + list(self._running.values())
            records = [job.snapshot(self._position_locked(job)) for job in jobs]
            if not active_only:
                records += [
                    job.snapshot(self._position_locked(job))
                    for job in self._finished.values()
                ]
                # Jobs a restart settled belong in the list too: they are the
                # ones a client is most likely to be asking about.
                records += [dict(record) for record in self._recovered.values()]
            return [
                record for record in records
                if kb_id is None or record.get("kb_id") == kb_id
            ]

    def snapshot(self) -> dict[str, Any]:
        """Capacity and counters, for the status endpoint and the load report."""
        with self._cond:
            self._prune_finished_locked()
            budget = budgets()
            return {
                "workers": self.limits.workers,
                "running": len(self._running),
                "queued": len(self._queue),
                "queue_capacity": self.limits.queue_capacity,
                "busy_knowledge_bases": sorted(self._busy_kbs),
                # ``provider`` is the Deep Analysis budget, kept under its
                # historical name; ``budgets`` carries every one of them.
                "provider": budget["deep_analysis"],
                "budgets": budget,
                "retained": {
                    "finished": len(self._finished),
                    "restart_settled": len(self._recovered),
                    "max": MAX_FINISHED,
                    "retention_seconds": self.limits.job_retention_seconds,
                },
                "stats": dict(self.stats),
                "limits": self.limits.to_dict(),
            }

    # ------------------------------------------------------------- waiting
    def wait(self, job: IngestJob, timeout: Optional[float]) -> bool:
        """Block until the job is terminal or ``timeout`` passes."""
        return job.done.wait(timeout)

    def drain(self, timeout: float = 30.0) -> bool:
        """Wait until nothing is queued or running. For tests and shutdown."""
        with self._cond:
            return self._cond.wait_for(
                lambda: not self._queue and not self._running, timeout=timeout
            )

    def close(self, timeout: float = 30.0) -> None:
        """Let running jobs finish, then stop the workers."""
        self.drain(timeout)
        with self._cond:
            self._stopping = True
            self._cond.notify_all()
        for worker in self._workers:
            worker.join(timeout)

    # -------------------------------------------------------------- cancel
    def cancel(self, job_id: str, reason: str = "cancelled by request") -> Optional[IngestJob]:
        """Stop a job. Queued: gone at once. Running: at its next seam.

        Returns the job (its ``status`` says which happened) or ``None`` for
        an unknown id. A job that has already finished is returned unchanged;
        a commit that completed is not undone by a cancel that came later.
        """
        with self._cond:
            job = self._lookup_locked(job_id)
            if job is None:
                return None
            if job.status == QUEUED:
                self._queue.remove(job)
                self._by_identity.pop(job.identity, None)
                job.status = CANCELLED
                job.error = reason
                job.error_category = "cancelled"
                self._finish_locked(job)
                path = job.temp_path
                events.emit("ingest.job.cancelled", job_id=job.job_id, kb_id=job.kb_id,
                            was="queued", reason=reason)
                T.metrics().count("ingest.cancelled")
            elif job.status == RUNNING:
                job.guard.cancel(reason)
                path = None
                events.emit("ingest.job.cancel_requested", job_id=job.job_id,
                            kb_id=job.kb_id, was="running", reason=reason)
            else:
                path = None
        if path:
            _discard_file(path)
            job.done.set()
        return job

    # ------------------------------------------------------------- workers
    def _ensure_workers_locked(self) -> None:
        alive = [worker for worker in self._workers if worker.is_alive()]
        self._workers = alive
        for index in range(len(alive), self.limits.workers):
            worker = threading.Thread(
                target=self._work, name=f"{self._name}-worker-{index + 1}", daemon=True
            )
            worker.start()
            self._workers.append(worker)

    def _next_locked(self) -> Optional[IngestJob]:
        for job in self._queue:
            if job.kb_id not in self._busy_kbs:
                return job
        return None

    def _work(self) -> None:
        while True:
            with self._cond:
                self._cond.wait_for(lambda: self._stopping or self._next_locked() is not None)
                job = self._next_locked()
                if job is None:
                    return  # stopping, and nothing runnable
                self._queue.remove(job)
                self._running[job.job_id] = job
                self._busy_kbs.add(job.kb_id)
                job.status = RUNNING
                job.started_at = datetime.now().isoformat(timespec="seconds")
                job._started_mono = time.monotonic()
                job.guard = JobGuard.for_timeout(self.limits.job_timeout_seconds)
                job.trace = T.JobTrace(job_id=job.job_id, kb_id=job.kb_id,
                                       mode=job.chunking_mode)
                # How long this job sat in the queue: the first number an
                # operator wants when uploads feel slow, because it separates
                # "the system is busy" from "the work itself is slow".
                job.trace.queue_seconds = job._started_mono - job._submitted_mono
                self._journal_locked(job)
                self.stats["peak_active"] = max(self.stats["peak_active"], len(self._running))
                events.emit("ingest.job.started", job_id=job.job_id, kb_id=job.kb_id,
                            mode=job.chunking_mode,
                            queue_seconds=round(job.trace.queue_seconds, 3),
                            running=len(self._running), queued=len(self._queue))
            self._run(job)
            with self._cond:
                self._running.pop(job.job_id, None)
                self._busy_kbs.discard(job.kb_id)
                if self._by_identity.get(job.identity) is job:
                    self._by_identity.pop(job.identity, None)
                self._finish_locked(job)
                self._cond.notify_all()
            job.done.set()

    def _run(self, job: IngestJob) -> None:
        started = time.monotonic()
        try:
            with use_guard(job.guard), T.use_trace(job.trace):
                result = self._execute(job)
            job.result = result
            job.doc_id = (result or {}).get("doc_id")
            job.status = SUCCEEDED
        except IngestInterrupted as stop:
            job.status = TIMED_OUT if stop.kind == "timed_out" else CANCELLED
            job.error = str(stop)
            job.exception = stop
            job.error_category = T.categorise(stop)
            logger.warning("ingest job %s %s: %s", job.job_id, job.status, stop)
        except Exception as error:  # noqa: BLE001 - a failed job is a state, not a crash
            job.status = FAILED
            job.error = str(error) or type(error).__name__
            job.exception = error
            job.error_category = T.categorise(error)
            logger.error("ingest job %s failed: %s", job.job_id, error, exc_info=True)
        finally:
            _discard_file(job.temp_path)
            self._record(job, time.monotonic() - started)

    def _record(self, job: IngestJob, seconds: float) -> None:
        """Close this job's trace and say what happened, once, in one place."""
        metrics = T.metrics()
        if job.trace is not None:
            job.trace.total_seconds = seconds
            job.trace.status = job.status
            job.trace.error_category = job.error_category
            metrics.finish(job.trace)
        metrics.count(f"ingest.{job.status}")
        fields = {
            "job_id": job.job_id, "kb_id": job.kb_id, "mode": job.chunking_mode,
            "doc_id": job.doc_id, "seconds": round(seconds, 3),
            "queue_seconds": round(job.trace.queue_seconds, 3) if job.trace else None,
        }
        if job.trace is not None:
            timing = job.trace.as_dict()
            fields.update({f"t_{name}": value for name, value in timing["stages"].items()})
            if timing["provider"]["calls"]:
                fields["provider_calls"] = timing["provider"]["calls"]
                fields["provider_seconds"] = timing["provider"]["seconds"]
            if timing["embedding"]["calls"]:
                fields["embedding_calls"] = timing["embedding"]["calls"]
        if job.status == SUCCEEDED:
            events.emit("ingest.job.succeeded", chunks=(job.result or {}).get("chunks_created"),
                        **fields)
        else:
            metrics.record_error(job.error_category, job.error or "")
            events.warn(f"ingest.job.{job.status}", error_category=job.error_category,
                        error=job.error, **fields)

    def _finish_locked(self, job: IngestJob) -> None:
        job.finished_at = datetime.now().isoformat(timespec="seconds")
        job._finished_mono = time.monotonic()
        if job._started_mono is not None:
            self._recent_seconds.append(job._finished_mono - job._started_mono)
        self.stats[job.status] = self.stats.get(job.status, 0) + 1
        _drop_traceback(job.exception)
        self._finished[job.job_id] = job
        self._journal_locked(job)
        self._prune_finished_locked()

    def _journal_locked(self, job: IngestJob) -> None:
        if self._journal is not None:
            self._journal.record(job.snapshot(self._position_locked(job)))

    def _prune_finished_locked(self) -> None:
        """Drop finished jobs that are past the retention window, and cap the
        registry whatever the window says.

        Called from every write *and* every read, so a process that stops
        receiving uploads still lets its last jobs go rather than holding them
        until the next submission that may never come.
        """
        now = time.monotonic()
        keep = self.limits.job_retention_seconds
        stale = [
            job_id for job_id, job in self._finished.items()
            if job._finished_mono is not None and now - job._finished_mono > keep
        ]
        for job_id in stale:
            job = self._finished.pop(job_id, None)
            if job is not None:
                self._forget_locked(job.job_id)
        while len(self._finished) > MAX_FINISHED:
            job_id, _ = self._finished.popitem(last=False)
            self._forget_locked(job_id)

        wall = time.time()
        expired = [
            job_id for job_id, record in self._recovered.items()
            if wall - float(record.get("journalled_at") or 0.0) > keep
        ]
        for job_id in expired:
            self._recovered.pop(job_id, None)
            self._forget_locked(job_id)
        while len(self._recovered) > MAX_FINISHED:
            job_id, _ = self._recovered.popitem(last=False)
            self._forget_locked(job_id)

    def _forget_locked(self, job_id: str) -> None:
        if self._journal is not None:
            self._journal.forget(job_id)

    def prune(self) -> None:
        """Apply the retention policy now. Called on every status read."""
        with self._cond:
            self._prune_finished_locked()

    # ------------------------------------------------------------- restart
    def recover(self, resolve_document=None) -> list[dict[str, Any]]:
        """Settle what a previous process left in the journal.

        Called once at start-up, before anything can be submitted. Returns the
        records that were still in flight, already settled -- so the caller can
        say how many uploads a restart interrupted.
        """
        if self._journal is None:
            return []
        records = self._journal.recover(
            active_states=ACTIVE,
            resolve_document=resolve_document,
            retention_seconds=self.limits.job_retention_seconds,
        )
        in_flight = []
        with self._cond:
            for record in sorted(records, key=lambda r: float(r.get("journalled_at") or 0.0)):
                job_id = str(record.get("job_id"))
                self._recovered[job_id] = record
                if record.get("restart_recovered"):
                    in_flight.append(record)
            self._prune_finished_locked()
        for record in in_flight:
            interrupted = record.get("status") == INTERRUPTED
            events.warn("ingest.job.restart_settled", job_id=record.get("job_id"),
                        kb_id=record.get("kb_id"), status=record.get("status"),
                        resolution=record.get("resolution"))
            T.metrics().count("ingest.restart_settled")
            if interrupted:
                T.metrics().count("ingest.interrupted")
                T.metrics().record_error("interrupted", str(record.get("error") or ""))
        return in_flight
