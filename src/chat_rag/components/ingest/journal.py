"""What a restart is allowed to forget about an ingest job, and what it is not.

A client that has been handed ``202 Accepted`` and a ``job_id`` holds a
promise. If the process restarts while that job is queued or running, the
job itself is gone -- it lived in memory, and nothing about this product
makes a half-finished parse worth resuming -- but the *promise* must still
be answerable. Answering it with ``404 Unknown job`` would be the one thing
a client cannot act on: it cannot tell "your upload never happened" from
"your upload finished and you missed it".

So every job writes three small records as it goes: when it is accepted,
when it starts, and when it reaches a terminal state. On the next start-up,
:meth:`JobJournal.recover` reads what is there and settles every record that
is still ``queued`` or ``running``:

* if the **document ledger** knows a document carrying that job's id, the job
  did finish -- the ledger write is the last thing a job does, and it is the
  definition of "this document is ingested" -- so the record becomes
  ``succeeded``, marked as recovered from the ledger rather than observed;
* otherwise the record becomes ``interrupted``: a terminal state that says
  exactly what is true, that the server stopped while this was in flight and
  the document was not registered.

That is the whole mechanism. No queue survives a restart, nothing is
re-run, no broker is involved, and the authority for "was this ingested" is
the table that is already the authority for it.

Retention is the same policy as the in-memory registry: a record older than
``INGEST_JOB_RETENTION`` is deleted, so the table cannot grow without bound
any more than the registry can.

**Step 8 moved these records from one JSON file per job into the
``ingest_jobs`` table.** Three paragraphs of this module went with the files:
the write-to-a-scratch-name-and-rename, the retry loop around ``os.replace``
for Windows readers holding the target open, and the "a lost record is safe by
construction" argument that existed because those writes could fail. A row
written in a transaction is there or is not. What survives unchanged is the
policy above -- which states are settled, what settles them, and how long a
record is kept -- because that is the product's promise and not the storage's.

Journalling is still best effort in one direction only: a database error must
not fail an ingest it is merely describing, so a failed write is logged and
the ingest continues. Recovery then treats a missing record exactly as it
treats an in-flight one, by asking the ledger.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import Any, Callable, Optional

from chat_rag.storage import IngestJobRepository, session_scope

logger = logging.getLogger("chat_rag.ingest")

#: A job the process was running when it stopped. Terminal, and truthful:
#: nothing was committed, because the ledger write is a job's last act.
INTERRUPTED = "interrupted"

#: A job whose completion was read back from the ledger after a restart.
RECOVERED = "recovered_from_ledger"


class JobJournal:
    """The ``ingest_jobs`` table, as the job manager sees it."""

    def __init__(self, directory: Optional[str] = None):
        """``directory`` is accepted and unused.

        It named the directory of per-job JSON files until Step 8. The job
        manager and the restart tests still construct a journal with one;
        the records are rows either way.
        """
        self.directory = directory

    # ------------------------------------------------------------- writing
    def record(self, snapshot: dict[str, Any]) -> None:
        """Write (or replace) one job's record. Never raises: a journal that
        cannot be written must not fail the ingest it is describing."""
        job_id = snapshot.get("job_id")
        if not job_id:
            return
        try:
            with session_scope() as session:
                IngestJobRepository(session).record(snapshot)
        except Exception as error:  # noqa: BLE001 - journalling is best effort
            logger.warning("could not journal ingest job %s: %s", job_id, error)

    def forget(self, job_id: str) -> None:
        try:
            with session_scope() as session:
                IngestJobRepository(session).forget(job_id)
        except Exception as error:  # noqa: BLE001
            logger.warning("could not remove the journal record %s: %s", job_id, error)

    # ------------------------------------------------------------- reading
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

        The whole pass is one transaction: the records that expired go, the
        records that were in flight are settled, and either both happened or
        neither did. A restart interrupted half way through this used to leave
        some jobs settled and some not.
        """
        now = time.time() if now is None else now
        settled: list[dict[str, Any]] = []
        with session_scope() as session:
            repository = IngestJobRepository(session)
            if retention_seconds:
                repository.prune(now - retention_seconds)
            for record in repository.snapshots():
                if record.get("status") in active_states:
                    record = self._settle(record, resolve_document)
                    repository.record(record)
                settled.append(record)
        return settled

    def prune(self, retention_seconds: float, *, now: Optional[float] = None) -> int:
        """Delete records older than the retention window. Returns the count."""
        if not retention_seconds:
            return 0
        now = time.time() if now is None else now
        try:
            with session_scope() as session:
                return IngestJobRepository(session).prune(now - retention_seconds)
        except Exception as error:  # noqa: BLE001 - pruning is best effort
            logger.warning("could not prune the ingest journal: %s", error)
            return 0

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
