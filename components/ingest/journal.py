"""What a restart is allowed to forget about an ingest job, and what it is not.

A client that has been handed ``202 Accepted`` and a ``job_id`` holds a
promise. If the process restarts while that job is queued or running, the
job itself is gone -- it lived in memory, and nothing about this product
makes a half-finished parse worth resuming -- but the *promise* must still
be answerable. Answering it with ``404 Unknown job`` would be the one thing
a client cannot act on: it cannot tell "your upload never happened" from
"your upload finished and you missed it".

So every job writes three small records as it goes: when it is accepted,
when it starts, and when it reaches a terminal state. One file per job,
written to a temporary name and moved into place, so a reader sees a whole
record or none. On the next start-up, :meth:`JobJournal.recover` reads what
is there and settles every record that is still ``queued`` or ``running``:

* if the **ingest ledger** knows a document carrying that job's id, the job
  did finish -- the ledger write is the last thing a job does, and it is the
  definition of "this document is ingested" -- so the record becomes
  ``succeeded``, marked as recovered from the ledger rather than observed;
* otherwise the record becomes ``interrupted``: a terminal state that says
  exactly what is true, that the server stopped while this was in flight and
  the document was not registered.

That is the whole mechanism. No queue survives a restart, nothing is
re-run, no broker is involved, and the authority for "was this ingested" is
the file that was already the authority for it.

Retention is the same policy as the in-memory registry: a record older than
``INGEST_JOB_RETENTION`` is deleted, so the directory cannot grow without
bound any more than the registry can.

**A lost record is safe by construction.** Journalling is best effort -- it
must never fail an ingest it is only describing -- so it is worth being
precise about what a lost write costs. Losing the ``running`` record leaves
the ``queued`` one, which recovery treats identically because both are
"in flight". Losing the terminal record leaves an in-flight record, which
recovery settles against the ledger: the same answer, reached the same way.
Only the ``accepted`` record is load-bearing, and it is written before the
route ever returns a ``job_id``, on the thread that returns it. The
retry below exists to make even that unlikely rather than to make it correct.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from datetime import datetime
from typing import Any, Callable, Iterable, Optional

logger = logging.getLogger("chat_rag.ingest")

#: A job the process was running when it stopped. Terminal, and truthful:
#: nothing was committed, because the ledger write is a job's last act.
INTERRUPTED = "interrupted"

#: A job whose completion was read back from the ledger after a restart.
RECOVERED = "recovered_from_ledger"


class JobJournal:
    """One directory of small JSON records, one per job."""

    def __init__(self, directory: str):
        self.directory = directory
        self._lock = threading.Lock()

    # ------------------------------------------------------------- writing
    #: Attempts at the final rename, and the pause between them.
    #:
    #: Windows will not let ``os.replace`` overwrite a file that any handle
    #: has open, and this file has readers: the recovery pass, and anything
    #: watching the directory. The same contention cost the ingest ledger a
    #: write in Phase 1B, where the answer was a lock; here a lock would not
    #: help, because the reader is not always in this process. A few tries
    #: over a fraction of a second cover it, and a genuine failure is still
    #: only a lost description (see the module docstring).
    REPLACE_ATTEMPTS = 5
    REPLACE_PAUSE_SECONDS = 0.02

    def record(self, snapshot: dict[str, Any]) -> None:
        """Write (or replace) one job's record. Never raises: a journal that
        cannot be written must not fail the ingest it is describing."""
        job_id = snapshot.get("job_id")
        if not job_id:
            return
        tmp = None
        try:
            os.makedirs(self.directory, exist_ok=True)
            target = os.path.join(self.directory, f"{job_id}.json")
            tmp = f"{target}.{os.getpid()}.{threading.get_ident()}.tmp"
            with open(tmp, "w", encoding="utf-8") as handle:
                json.dump(snapshot, handle, ensure_ascii=False)
                handle.flush()
                os.fsync(handle.fileno())
            self._replace(tmp, target, job_id)
        except OSError as error:  # noqa: BLE001 - journalling is best effort
            logger.warning("could not journal ingest job %s: %s", job_id, error)
        finally:
            if tmp and os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except OSError:
                    pass

    def _replace(self, tmp: str, target: str, job_id: str) -> None:
        for attempt in range(self.REPLACE_ATTEMPTS):
            try:
                os.replace(tmp, target)
                return
            except PermissionError:
                # A reader holds the target open. It will not hold it long.
                if attempt == self.REPLACE_ATTEMPTS - 1:
                    logger.warning(
                        "could not replace the journal record for %s after %d attempts; "
                        "the previous record stands and a restart still settles it",
                        job_id, self.REPLACE_ATTEMPTS,
                    )
                    return
                time.sleep(self.REPLACE_PAUSE_SECONDS * (attempt + 1))

    def forget(self, job_id: str) -> None:
        path = os.path.join(self.directory, f"{job_id}.json")
        try:
            os.remove(path)
        except FileNotFoundError:
            pass
        except OSError as error:  # noqa: BLE001
            logger.warning("could not remove the journal record %s: %s", path, error)

    # ------------------------------------------------------------- reading
    def _records(self) -> Iterable[tuple[str, dict[str, Any]]]:
        try:
            names = sorted(os.listdir(self.directory))
        except FileNotFoundError:
            return []
        found = []
        for name in names:
            if not name.endswith(".json"):
                continue
            path = os.path.join(self.directory, name)
            try:
                with open(path, encoding="utf-8") as handle:
                    record = json.load(handle)
            except (OSError, ValueError) as error:  # noqa: BLE001
                logger.warning("unreadable journal record %s: %s", path, error)
                continue
            if isinstance(record, dict) and record.get("job_id"):
                found.append((path, record))
        return found

    def recover(
        self,
        *,
        active_states: frozenset[str],
        resolve_document: Optional[Callable[[str], Optional[dict]]] = None,
        retention_seconds: float = 3600.0,
        now: Optional[float] = None,
    ) -> list[dict[str, Any]]:
        """Settle every unfinished record and return what a restart still knows.

        ``resolve_document`` is asked, for a job that was in flight, whether
        the ledger holds a document that job wrote. It is injected rather than
        imported so this module knows nothing about the tracker.
        """
        now = time.time() if now is None else now
        settled: list[dict[str, Any]] = []
        with self._lock:
            for path, record in self._records():
                stamp = float(record.get("journalled_at") or 0.0)
                if retention_seconds and stamp and now - stamp > retention_seconds:
                    try:
                        os.remove(path)
                    except OSError:
                        pass
                    continue
                if record.get("status") in active_states:
                    record = self._settle(record, resolve_document)
                    self.record(record)
                settled.append(record)
        return settled

    @staticmethod
    def _settle(
        record: dict[str, Any],
        resolve_document: Optional[Callable[[str], Optional[dict]]],
    ) -> dict[str, Any]:
        job_id = str(record.get("job_id"))
        document = None
        if resolve_document is not None:
            try:
                document = resolve_document(job_id)
            except Exception as error:  # noqa: BLE001 - a failed lookup is "unknown"
                logger.warning("could not check the ledger for job %s: %s", job_id, error)
        settled = dict(record)
        settled["finished_at"] = settled.get("finished_at") or datetime.now().isoformat(
            timespec="seconds"
        )
        settled["restart_recovered"] = True
        if document is not None:
            # The ledger write is the last thing a job does. A document
            # carrying this job's id therefore means the job got all the way
            # through, whatever the journal last managed to say.
            settled["status"] = "succeeded"
            settled["resolution"] = RECOVERED
            settled["doc_id"] = document.get("doc_id") or settled.get("doc_id")
            settled["error"] = None
            settled["result"] = settled.get("result") or {
                "success": True,
                "doc_id": settled.get("doc_id"),
                "chunks_created": document.get("chunk_count"),
                "filename": settled.get("filename"),
                "chunking_mode": document.get("chunking_mode") or settled.get("chunking_mode"),
                "recovered_from_ledger": True,
            }
        else:
            settled["status"] = INTERRUPTED
            settled["resolution"] = INTERRUPTED
            settled["result"] = None
            settled["error"] = (
                "The server stopped while this upload was being processed. "
                "The document was not registered, so nothing was half-indexed; "
                "upload it again."
            )
        return settled

    def prune(self, retention_seconds: float, *, now: Optional[float] = None) -> int:
        """Delete records older than the retention window. Returns the count."""
        if not retention_seconds:
            return 0
        now = time.time() if now is None else now
        removed = 0
        with self._lock:
            for path, record in self._records():
                stamp = float(record.get("journalled_at") or 0.0)
                if stamp and now - stamp > retention_seconds:
                    try:
                        os.remove(path)
                        removed += 1
                    except OSError:
                        pass
        return removed
