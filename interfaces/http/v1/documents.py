"""`/api/v1/documents` -- an upload, its corpus, and its chunking analysis.

Uploading is **always asynchronous** here: the answer is the job, and the job
is polled at ``/api/v1/ingest-jobs/{id}``. The legacy surface also has a
synchronous mode, which exists because it always did and which is the reason
that adapter needs a semaphore to stop a burst of uploads holding every
request thread. A contract written now does not inherit that.
"""

from __future__ import annotations

from flask import Blueprint, request

from application import documents as use_case
from application import ingest, workspace
from application.errors import InvalidRequest

from ..context import services, session_id
from . import resources
from .envelope import collection, install, page_request, resource

bp = Blueprint('v1_documents', __name__)
install(bp)


@bp.route('/documents', methods=['GET'])
def list_documents():
    """Every ingested document, newest first. ``?knowledge_base_id=`` narrows it.

    Each carries its ``analysis`` block, because "what may this document be
    asked about" is the first thing a screen listing documents needs and a
    second round trip per row is not a contract worth shipping.
    """
    offset, limit = page_request()
    kb_id = request.args.get('knowledge_base_id') or None
    records = use_case.list_all(services(), kb_id)
    states = workspace.analysis_states()
    items = [
        resources.document(record, state=states.get(record.get("doc_id")) or {})
        for record in records[offset:offset + limit]
    ]
    return collection(items, offset=offset, limit=limit, total=len(records))


@bp.route('/documents', methods=['POST'])
def upload_document():
    """Submit a document for ingestion.

    ``multipart/form-data``: ``file``, ``knowledge_base_id``, and ``methods``
    (repeated, or one comma-separated field) naming the chunking methods to
    analyse it with. One upload is one parse and one canonical, and every
    method runs over that same canonical -- so three methods cost one parse.
    What gets *indexed* for retrieval is still the knowledge base's own
    chunker; the methods are an analysis choice and do not change it.

    Always **202**, with the job. The same bytes submitted again while the
    first is still in flight are attached to that job rather than parsed
    twice; ``attached_uploads`` on the job says so.
    """
    if 'file' not in request.files:
        raise InvalidRequest('a file is required')
    uploaded = request.files['file']
    if not uploaded.filename:
        raise InvalidRequest('the uploaded file has no name')

    container = services()
    accepted = ingest.submit(
        container,
        upload=ingest.Upload(filename=uploaded.filename, save=uploaded.save),
        kb_id=request.form.get('knowledge_base_id'),
        session_id=session_id(),
        methods=request.form.getlist('methods') or request.form.get('methods'),
    )
    record = container.ingest_jobs.describe(accepted.job)
    return resource(resources.ingest_job(record), status=202,
                    headers={"Location": f"/api/v1/ingest-jobs/{record['job_id']}"})


@bp.route('/documents/<document_id>', methods=['GET'])
def get_document(document_id):
    record = use_case.get(services(), document_id)
    return resource(resources.document(
        record, state=workspace.analysis_state(document_id)))


@bp.route('/documents/<document_id>', methods=['DELETE'])
def delete_document(document_id):
    """Delete a document: its chunks, its ledger row and its own analysis.

    What it does **not** take is the content: another upload of the same bytes
    keeps the shared analysis and its variants, and only the last upload of a
    content takes that down with it.
    """
    use_case.delete(services(), document_id, session_id=session_id())
    return '', 204


@bp.route('/documents/<document_id>/chunks', methods=['GET'])
def document_chunks(document_id):
    """The chunks this document was *indexed* as -- the corpus a question
    searches, not an analysis variant."""
    offset, limit = page_request()
    found = use_case.chunks_of(services(), document_id,
                               kb_id=request.args.get('knowledge_base_id'),
                               session_id=session_id(), offset=offset, limit=limit)
    return collection([resources.chunk(row) for row in found["chunks"]],
                      offset=found["offset"], limit=found["limit"], total=found["total"])


@bp.route('/documents/<document_id>/units', methods=['GET'])
def document_units(document_id):
    """The parser's canonical reading of the document, before any chunker.

    Read-only and cache-backed -- the file is never parsed again. Read beside
    the chunks to tell a parser reading-order problem from a chunker one.
    """
    offset, limit = page_request()
    found = use_case.canonical_units(
        services(), document_id,
        kb_id=request.args.get('knowledge_base_id'), session_id=session_id(),
        page_from=request.args.get('page_from', type=int),
        page_to=request.args.get('page_to', type=int),
        unit_type=request.args.get('unit_type') or None,
        offset=offset, limit=limit,
    )
    return collection([resources.canonical_unit(row) for row in found["units"]],
                      offset=found["offset"], limit=found["limit"], total=found["total"],
                      pages=found["pages_in_document"])


# ------------------------------------------------------------- the analysis
@bp.route('/documents/<document_id>/analysis', methods=['GET'])
def get_analysis(document_id):
    """Where this document's chunking analysis got to.

    Always **200**, including ``status: "missing"``: a document that exists
    with no analysis is a fact, not a missing resource, and a client polling
    for a build should not have to read a 404 as progress.
    """
    return resource(resources.analysis(workspace.analysis_state(document_id)))


@bp.route('/documents/<document_id>/analysis', methods=['POST'])
def request_analysis(document_id):
    """Queue -- or retry -- the analysis. Queuing is all it does: the build
    runs on a worker, so this never waits on it."""
    return resource(resources.analysis(
        workspace.request_analysis(services(), document_id)), status=202)


@bp.route('/documents/<document_id>/analysis/methods', methods=['POST'])
def add_analysis_methods(document_id):
    """Add chunking variants to a document that is already here.

    This is how a second method reaches a document -- not by uploading the
    file again. The canonical is on disk, so nothing is parsed twice and no
    variant already built is rebuilt.
    """
    body = request.get_json(silent=True) or {}
    return resource(resources.analysis(
        workspace.add_methods(document_id, body.get("methods"))), status=202)


@bp.route('/documents/<document_id>/analysis/methods/<method>/chunks', methods=['GET'])
def analysis_method_chunks(document_id, method):
    """The rows one chunking method produced for this document.

    Three different refusals, deliberately kept apart, because they call for
    three different things from a client:

    * **400** the method is not one this deployment knows about;
    * **404** the method exists but is not one *this upload* selected -- the
      content may well have it, from another upload, and it is still not this
      document's to serve;
    * **409** ``not_ready`` -- selected, and not built yet. The body carries
      the analysis state so the client can keep polling.
    """
    found = workspace.chunk_rows(document_id, method)
    arm = found["arms"][method]
    offset, limit = page_request()
    rows = arm["rows"]
    return collection(rows[offset:offset + limit], offset=offset, limit=limit,
                      total=len(rows), method=method, engine=arm["kind"],
                      content_id=found.get("key"))
