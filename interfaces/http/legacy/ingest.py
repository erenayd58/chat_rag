"""HTTP for uploads and the jobs that carry them out.

The only place in this adapter with a table of its own: a settled job means
something in the product's terms (:mod:`application.ingest`), and those
meanings map onto status codes here so that nothing below has to know one.
"""

from __future__ import annotations

from flask import Blueprint, jsonify, request

from application import ingest as use_case
from application.errors import InvalidRequest

from ..context import flag, services, session_id
from .responses import install, ok

bp = Blueprint('ingest', __name__)
install(bp)

#: What each settled job answers with, and the flag its client branches on.
OUTCOMES = {
    use_case.SUCCEEDED: (200, None),
    use_case.REINDEX_REQUIRED: (409, 'reindex_required'),
    use_case.DEEP_UNAVAILABLE: (503, 'deep_analysis_unavailable'),
    use_case.FAILED: (500, None),
    use_case.TIMED_OUT: (504, 'timed_out'),
    use_case.CANCELLED: (409, 'cancelled'),
}


@bp.route('/api/documents/upload', methods=['POST'])
def upload_document():
    """Accept a document and ingest it as a job.

    The request validates, stages the file and submits a job; the parse, the
    chunking, any model calls, the embeddings and the store and ledger writes
    happen on an ingest worker under the configured limits. Two answers are
    possible:

    * ``async=1`` (what the console sends): **202** at once with the job to
      poll at ``GET /api/ingest/jobs/<job_id>``.
    * otherwise the request waits for the job -- up to ``INGEST_SYNC_WAIT``
      seconds -- and answers **200** with the document, or the job's own
      refusal. A job still running when the wait runs out is answered **202**
      with the job to poll, not cut off.

    A full queue is refused up front with **503**, ``overloaded: true`` and a
    ``Retry-After`` header; nothing is queued, nothing is kept.
    """
    if 'file' not in request.files:
        raise InvalidRequest('No file uploaded')
    uploaded = request.files['file']
    if uploaded.filename == '':
        raise InvalidRequest('No file selected')

    container = services()
    accepted = use_case.submit(
        container,
        upload=use_case.Upload(filename=uploaded.filename, save=uploaded.save),
        kb_id=request.form.get('kb_id'),
        session_id=session_id(),
        methods=request.form.getlist('methods') or request.form.get('methods'),
        deep_flag=request.form.get('deep_analysis'),
    )

    if flag(request.form.get('async')):
        return _pending(container, accepted), 202

    use_case.await_settlement(container, accepted)
    if not accepted.waited:
        # No waiting slot was free, so the job is answered the way a slow one
        # is: accepted, running, and handed back to poll.
        return _pending(container, accepted, busy=True), 202
    return _settled(use_case.outcome(container, accepted))


def _pending(container, accepted, *, busy: bool = False):
    body = {
        'success': True,
        'pending': True,
        'attached': accepted.attached,
        'job_id': accepted.job.job_id,
        'job': container.ingest_jobs.describe(accepted.job),
    }
    if busy:
        body['busy'] = True
    return jsonify(body)


def _settled(outcome):
    """One settled job, in the shapes this route has always given."""
    if outcome.kind == use_case.PENDING:
        return jsonify({'success': True, 'pending': True, 'attached': outcome.attached,
                        'job_id': outcome.job_id, 'job': outcome.job}), 202
    status, marker = OUTCOMES[outcome.kind]
    if outcome.kind == use_case.SUCCEEDED:
        return jsonify({**outcome.result, 'success': True, 'job_id': outcome.job_id,
                        'attached': outcome.attached, 'job': outcome.job}), status
    body = {'success': False, 'error': outcome.error, 'job_id': outcome.job_id}
    if marker:
        body[marker] = True
    return jsonify(body), status


@bp.route('/api/ingest/jobs', methods=['GET'])
def list_ingest_jobs():
    """Jobs the process knows about, newest last, with the capacity picture.

    ``kb_id`` narrows to one knowledge base; ``active=1`` leaves out finished
    jobs. Finished jobs stay for ``INGEST_JOB_RETENTION`` seconds; a restart
    forgets every job, and the ledger is the record of what was ingested.
    """
    return ok(**use_case.list_jobs(services(), request.args.get('kb_id') or None,
                                   active_only=flag(request.args.get('active'))))


@bp.route('/api/ingest/jobs/<job_id>', methods=['GET'])
def get_ingest_job(job_id):
    return ok(job=use_case.job(services(), job_id))


@bp.route('/api/ingest/jobs/<job_id>', methods=['DELETE'])
def cancel_ingest_job(job_id):
    return ok(job=use_case.cancel(services(), job_id))
