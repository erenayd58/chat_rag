"""Accepting a document, and the job that turns it into a searchable corpus.

An upload is two things kept deliberately apart. **Submission** validates,
stages the file and hands it to the job manager; it is fast, it makes no
provider call, and it either accepts or refuses. **Execution** is everything
after that -- parse, chunk (with any Deep Analysis model calls under the
provider budget), embed, write the store, record the ledger, stage the Viewer
-- and it runs on an ingest worker under the job's own deadline.

Two invariants govern execution and must survive any rewrite of it:

* the pipeline is **leased**, not merely fetched, for the whole job: an ingest
  outlives the request that asked for it, and nothing may evict the pipeline
  and close the store it is writing to;
* the **ledger write is the last act**, and a ledger that cannot be written
  takes the store rows back out again. A document exists exactly when the
  ledger says it does, which is what makes restart settlement truthful.

Nothing between the store write and the ledger write is interruptible. The
job's guard is asked at stage boundaries inside the pipeline, never in the
middle of one.
"""

from __future__ import annotations

import logging
import os
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Optional

from components.ingest import jobs as J
from components.observability import events, telemetry as T
from components.provenance import capture as capture_pipeline_snapshot
from components.viewer import analysis as viewer_analysis
from components.viewer import methods as viewer_methods
from config import paths
from core.exceptions import ConfigurationException, IndexIncompatibleException

from . import workspace
from .errors import InvalidRequest, NotFound

logger = logging.getLogger("RAG.ingest")


@dataclass
class Upload:
    """A document handed in for ingestion, without saying how it arrived.

    ``save`` writes the bytes to a path this module chooses. A Flask
    ``FileStorage``, a local file and a byte string all satisfy it, which is
    what keeps submission callable without a request.
    """

    filename: str
    save: Callable[[str], None]


@dataclass
class Accepted:
    """An upload the job manager took. The job owns the staged file now."""

    job: Any
    attached: bool
    #: True when a caller waited for the job and it settled inside the wait.
    waited: bool = False


def stage_file(upload: Upload) -> str:
    """Write an uploaded file where its ingest job will find it.

    Ingestion reads a path, not a stream: the parser opens the file, the
    ledger hashes it and the Viewer takes its content sha from it. The file
    outlives the call that received it -- the job that reads it runs on a
    worker -- so it goes to the staging directory the job manager owns and
    sweeps, and the job deletes it in every terminal state (see
    ``components/ingest/jobs.py``). Until it is handed to a job the caller
    owns it, which is why every early exit below deletes it itself.

    The name keeps the shape it has always had (``upload_<8 hex><ext>``): a
    document whose ingest produced no chunks still takes its fallback id from
    this file name.
    """
    extension = os.path.splitext(upload.filename)[1]
    directory = paths.upload_staging()
    os.makedirs(directory, exist_ok=True)
    temp_path = os.path.join(directory, f"upload_{uuid.uuid4().hex[:8]}{extension}")
    logger.info(f"Saving uploaded file to: {temp_path}")
    upload.save(temp_path)
    return temp_path


def discard_staged(temp_path: str) -> None:
    try:
        os.remove(temp_path)
    except FileNotFoundError:
        pass
    except OSError as error:  # noqa: BLE001 - a leftover file is not a failed upload
        logger.warning(f"Could not remove the uploaded temp file {temp_path}: {error}")


def resolve_methods(selection: Any, deep_flag: Optional[str] = None) -> list[str]:
    """Which chunking methods this upload asked to be analysed with.

    One upload, one parse, one canonical -- then every method the user ticked
    runs over that same canonical, so three methods cost one parse. The
    methods are an *analysis* choice; what gets indexed for retrieval is still
    the knowledge base's own chunker, and this does not change it.

    ``deep_flag`` is the older single-mode form, which still works: when it is
    present it replaces the selection entirely with Standard, plus Deep when
    it is on.
    """
    selected = viewer_methods.normalise(selection)
    raw = (deep_flag or '').strip().lower()
    if raw:
        wanted = raw in {'1', 'true', 'yes', 'on'}
        selected = viewer_methods.normalise(
            [viewer_methods.STANDARD] + ([viewer_methods.DEEP] if wanted else [])
        )
    return selected


def submit(services, *, upload: Upload, kb_id: Optional[str], session_id: str,
           methods: Any = None, deep_flag: Optional[str] = None) -> Accepted:
    """Validate an upload, stage it and queue its job.

    Raises :class:`~application.errors.InvalidRequest` or
    :class:`~application.errors.NotFound` before anything is written, and
    ``IngestOverloaded`` when the queue is full -- refused, never queued, and
    nothing is kept.
    """
    logger.info(f"Processing document: {upload.filename}")

    if not kb_id or not kb_id.strip():
        raise InvalidRequest('Knowledge base selection is required. Please select a '
                             'knowledge base before uploading documents.')
    kb = services.kb_manager.get(kb_id)
    if not kb:
        raise NotFound(f'Knowledge base "{kb_id}" not found')

    selected = resolve_methods(methods, deep_flag)
    # Deep Analysis (the amsc.deep.pipeline quality pipeline with an LLM
    # proposer and verifier) is a per-document, ingest-only decision -- never
    # a query-time toggle and never written into the knowledge base's chunker
    # config. A missing or failing model provider does not refuse the upload:
    # the deterministic quality contract runs alone and the document is
    # labelled with that status, never passed off as Standard.
    deep_analysis = viewer_methods.DEEP in selected

    pipeline = services.get_pipeline(session_id, kb_id)
    if deep_analysis and not hasattr(pipeline.chunker, 'chunk_text_deep'):
        # The one thing refused up front: a chunker with no Deep Analysis path.
        raise InvalidRequest(
            'Deep Analysis requires the structure-first chunker; this knowledge '
            f'base uses {pipeline.chunker.get_name()}.',
            details={'deep_analysis_unavailable': True},
        )

    temp_path = stage_file(upload)
    try:
        # Identity is the document's content: the same file uploaded again
        # while the first is in flight joins that job, and the Viewer files
        # the two under one document.
        content_sha = viewer_analysis.sha_of(temp_path)
    except Exception:
        discard_staged(temp_path)
        raise

    # From here the job owns the file, in every outcome.
    job, attached = services.ingest_jobs.submit(
        kb_id=kb_id,
        filename=upload.filename,
        temp_path=temp_path,
        session_id=session_id,
        methods=selected,
        deep_analysis=deep_analysis,
        content_sha=content_sha,
    )
    return Accepted(job=job, attached=attached)


def await_settlement(services, accepted: Accepted) -> Accepted:
    """Wait for a job, but only if a waiting slot is free.

    When it is not, the caller is answered the way a slow job is answered
    anyway: the job is accepted and running, and its id is handed back to
    poll. The compatibility path therefore degrades to the asynchronous one
    under load instead of taking the server down with it.
    """
    if not services.sync_waiters.acquire(blocking=False):
        logger.info(
            "answering %s asynchronously: %d synchronous waits already in progress",
            accepted.job.filename, services.settings.ingest_limits.sync_waiters,
        )
        return accepted
    try:
        services.ingest_jobs.wait(accepted.job, timeout=services.settings.ingest_sync_wait)
    finally:
        services.sync_waiters.release()
    accepted.waited = True
    return accepted


#: What a settled job means, as the product's own vocabulary. An adapter maps
#: these to its transport; nothing else needs to know a job status at all.
SUCCEEDED = 'succeeded'
FAILED = 'failed'
REINDEX_REQUIRED = 'reindex_required'
DEEP_UNAVAILABLE = 'deep_analysis_unavailable'
TIMED_OUT = 'timed_out'
CANCELLED = 'cancelled'
PENDING = 'pending'


@dataclass
class Outcome:
    kind: str
    job_id: str
    job: dict
    attached: bool = False
    result: Optional[dict] = None
    error: Optional[str] = None


def outcome(services, accepted: Accepted) -> Outcome:
    """What became of a job, in the product's terms rather than a status code."""
    job = accepted.job
    described = services.ingest_jobs.describe(job)
    common = dict(job_id=job.job_id, job=described, attached=accepted.attached)

    if job.status == J.SUCCEEDED:
        return Outcome(kind=SUCCEEDED, result=job.result, **common)
    if job.status == J.FAILED:
        error = job.exception
        if isinstance(error, IndexIncompatibleException):
            # The store holds vectors from another embedding model. Refused
            # rather than mixed; the knowledge base's Settings offer a re-index.
            return Outcome(kind=REINDEX_REQUIRED, error=str(error), **common)
        if isinstance(error, ConfigurationException):
            # A Deep Analysis request the backend cannot honour at all (a
            # chunker without a Deep Analysis path). Refused explicitly --
            # never a silent fall back to Standard.
            return Outcome(kind=DEEP_UNAVAILABLE, error=str(error), **common)
        return Outcome(kind=FAILED, error=job.error, **common)
    if job.status == J.TIMED_OUT:
        return Outcome(kind=TIMED_OUT, error=job.error, **common)
    if job.status == J.CANCELLED:
        return Outcome(kind=CANCELLED, error=job.error, **common)
    # Still queued or running when the synchronous wait ran out: the job goes
    # on, and the caller polls for it.
    return Outcome(kind=PENDING, **common)


# ------------------------------------------------------------- the job itself
def execute_job(services, job) -> dict:
    """One ingest job, on a worker thread: everything after submission.

    Returns the body the upload path has always answered with.
    """
    kb = services.kb_manager.get(job.kb_id)
    if not kb:
        raise RuntimeError(f'Knowledge base "{job.kb_id}" no longer exists')

    # Leased, not merely fetched: nothing may evict this pipeline -- closing
    # the store it is writing to -- while the job is running. It still comes
    # from the one seam, so the lease is taken on whatever that seam returned.
    with services.lease_pipeline(job.session_id, job.kb_id) as pipeline:
        return _run(services, job, kb, pipeline)


def _run(services, job, kb, pipeline) -> dict:
    chunking_mode = job.chunking_mode
    deep_analysis = job.deep_analysis
    selected = list(job.methods)

    if deep_analysis and not hasattr(pipeline.chunker, 'chunk_text_deep'):
        raise ConfigurationException(
            'Deep Analysis requires the structure-first chunker; this knowledge '
            f'base uses {pipeline.chunker.get_name()}.'
        )

    chunks = pipeline.ingest_document_from_file(
        file_path=job.temp_path, doc_title=job.filename, deep_analysis=deep_analysis,
    )

    tracker = services.documents()
    doc_id = os.path.basename(job.temp_path).replace('.', '_')
    # The real id comes from the chunks; the file name is only the fallback
    # for a document that produced none.
    if chunks:
        doc_id = chunks[0].doc_id
    job.doc_id = doc_id

    # The Deep Analysis report for this ingest (None on the Standard path):
    # status, model ids, chunk and smell counts before and after, regression
    # counts, proposer/verifier usage, checks. Counts and ids only -- never
    # prompts, never keys.
    deep_report = getattr(pipeline, 'last_deep_analysis_report', None)
    deep_summary = None
    deep_status = None
    if deep_report is not None:
        from components.chunker.deep_analysis import product_summary

        deep_summary = product_summary(deep_report)
        deep_status = deep_summary['status']

    # The configuration that just produced this corpus, captured after
    # ingestion succeeded and written in the same call that records the
    # document, so a failed ingest leaves neither. Read back by
    # `python -m cli report/inspect`, which otherwise can only describe
    # today's configuration rather than the one that ran.
    pipeline_snapshot = capture_pipeline_snapshot(
        pipeline, kb, kb_id=job.kb_id,
        vector_collection=services.kb_manager.collection(job.kb_id),
    )
    if pipeline_snapshot is not None:
        # The per-document ingest decision, next to the pipeline facts, so the
        # CLI can tell which mode -- and at which level of completion --
        # produced this corpus. The snapshot keeps the summary; the full
        # report lives once, in the document metadata below.
        pipeline_snapshot['ingest_options'] = {
            'chunking_mode': chunking_mode,
            'deep_analysis': deep_analysis,
            'deep_analysis_status': deep_status,
            'deep_analysis_summary': deep_summary,
        }

    doc_metadata = {
        'original_filename': job.filename,
        'upload_source': 'web_interface',
        'chunking_mode': chunking_mode,
        'ingest_job_id': job.job_id,
    }
    if deep_report is not None:
        doc_metadata['deep_analysis_status'] = deep_status
        doc_metadata['deep_analysis'] = deep_report

    with T.stage(T.LEDGER):
        recorded = tracker.mark_as_ingested(
            file_path=job.temp_path,
            doc_id=doc_id,
            chunk_count=len(chunks),
            metadata=doc_metadata,
            kb_id=job.kb_id,
            pipeline_snapshot=pipeline_snapshot,
            status='indexed',
            chunking_mode=chunking_mode,
        )
    if recorded is False:
        # The store holds the rows and the ledger does not know them: that is
        # the partial registration a job must never leave. Take the rows back
        # out and fail truthfully.
        rollback_indexed_chunks(pipeline, doc_id, chunks)
        raise RuntimeError(
            'The ingest ledger could not be written; the document was not '
            'registered and its chunks were removed from the index again.'
        )

    logger.info(f"Document processed successfully: {len(chunks)} chunks created")

    # Every other pipeline for this knowledge base is now holding a lexical
    # index that predates this document. Tell them; each rebuilds on its next
    # query. Without this a document ingested in one browser session is
    # missing from keyword search in another until the process restarts.
    stale = services.pipeline_cache.invalidate_indexes(job.kb_id, except_pipeline=pipeline)
    if stale:
        events.emit("pipeline.index.invalidated", kb_id=job.kb_id, pipelines=stale,
                    doc_id=doc_id)

    viewer_state = _stage_for_viewer(job, kb, pipeline, chunking_mode, deep_analysis, selected)

    message = 'Doküman yüklendi ve indekslendi'
    if len(selected) > 1:
        message = ('Doküman yüklendi · '
                   + ', '.join(viewer_methods.labels(selected)) + ' analizleri hazırlanıyor')
    elif deep_summary is not None:
        message = f"Doküman indekslendi — Deep Analysis: {deep_summary['label']}"
    return {
        'success': True,
        'message': message,
        'doc_id': doc_id,
        'chunks_created': len(chunks),
        'filename': job.filename,
        'chunking_mode': chunking_mode,
        # The product summary, not the full report: the browser shows status,
        # quality before/after and LLM usage; the report is on the document
        # record.
        'deep_analysis': deep_summary,
        # Where the Viewer's own analysis of this document got to.
        'viewer_analysis': (viewer_state or {}).get('status'),
    }


def _stage_for_viewer(job, kb, pipeline, chunking_mode, deep_analysis, selected):
    """Hand this ingest's own outputs to the Viewer packager.

    The canonical the chunker just normalised, and -- on a Deep Analysis
    upload -- the run it just produced. Both are already in memory, so the
    Viewer costs no second parse and no second provider call. Staging is
    serialisation only; the packaging runs on a worker, so this job does not
    wait for it, and a packaging problem must not fail an upload that already
    succeeded.
    """
    state = None
    try:
        with T.stage(T.VIEWER):
            chunker = getattr(pipeline, 'chunker', None)
            state = workspace.stage_analysis(
                job.doc_id,
                label=job.filename,
                kb_id=job.kb_id,
                kb_name=kb.get('name'),
                chunking_mode=chunking_mode,
                methods=selected,
                # Identity is the document's content: the same PDF uploaded
                # again is the same document, gaining variants rather than
                # becoming a second entry. Hashed once, at submission.
                content_sha=job.content_sha,
                units=getattr(chunker, 'last_canonical_units', None),
                deep_result=getattr(chunker, 'last_deep_result', None) if deep_analysis else None,
                # Measured by the pipeline at this very upload; None for paths
                # that never parsed (then the Viewer shows no time).
                parse_seconds=getattr(pipeline, 'last_parse_seconds', None),
            )
    except Exception as error:  # noqa: BLE001
        logger.warning(f"Could not stage {job.doc_id} for the viewer: {error}")
    finally:
        # The next ingest replaces them anyway; dropping them here keeps one
        # document's canonical out of memory afterwards.
        if getattr(pipeline, 'chunker', None) is not None:
            pipeline.chunker.last_canonical_units = None
            pipeline.chunker.last_deep_result = None
    return state


def rollback_indexed_chunks(pipeline, doc_id: str, chunks) -> None:
    """Undo a store write whose ledger record could not follow it."""
    if not chunks:
        return
    vector_db = getattr(pipeline, 'vector_db', None)
    delete = getattr(vector_db, 'delete_by_doc_id', None)
    if delete is None:
        logger.error(f"Cannot roll back {doc_id}: the store has no delete_by_doc_id")
        return
    try:
        delete(doc_id)
        retriever = getattr(pipeline, 'hybrid_retriever', None)
        if retriever is not None and hasattr(retriever, 'build_keyword_index'):
            retriever.build_keyword_index(vector_db.get_all_chunks())
    except Exception as error:  # noqa: BLE001 - reported, not hidden
        logger.error(f"Rolling back {doc_id} failed: {error}", exc_info=True)


# --------------------------------------------------------------- job queries
def list_jobs(services, kb_id: Optional[str] = None, *, active_only: bool = False) -> dict:
    """Jobs the process knows about, newest last, with the capacity picture."""
    return {
        'jobs': services.ingest_jobs.list(kb_id, active_only=active_only),
        'capacity': services.ingest_jobs.snapshot(),
    }


def job(services, job_id: str) -> dict:
    """One job, live or settled by a restart.

    A job id survives a restart: the journal records every job, and start-up
    settles anything that was in flight against the ledger. Not-found
    therefore means only one thing, that the job is older than the retention
    window.
    """
    record = services.ingest_jobs.record_for(job_id)
    if record is None:
        raise NotFound(
            'Unknown ingest job: it finished longer ago than jobs are kept '
            f'({int(services.settings.ingest_job_retention)}s). The document list '
            'shows what was ingested.',
            details={'unknown_job': True},
        )
    return record


def cancel(services, job_id: str) -> dict:
    """Cancel a job: a queued one at once, a running one at its next seam.

    A running job that has already written its document finishes as
    ``succeeded``; the answer's ``status`` says which happened.
    """
    cancelled = services.ingest_jobs.cancel(job_id)
    if cancelled is None:
        raise NotFound('Unknown ingest job', details={'unknown_job': True})
    return services.ingest_jobs.describe(cancelled)


def recover(services) -> list:
    """Settle whatever a previous process left in flight.

    Ingest jobs are not resumed -- a half-finished parse is not worth
    restarting and nothing was committed -- but they are settled, so a client
    holding a job id gets a truthful answer instead of a 404. The ledger
    decides: a document carrying the job's id means it finished.
    """
    from . import documents

    return services.ingest_jobs.recover(
        resolve_document=lambda job_id: documents.of_ingest_job(services, job_id)
    )
