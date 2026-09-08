"""`/api/v1/documents` -- an upload, its corpus, and its chunking analysis.

Uploading is **always asynchronous** here: the answer is the job, and the job
is polled at ``/api/v1/ingest-jobs/{job_id}``. The legacy surface also has a
synchronous mode, which exists because it always did and which is the reason
that adapter needs a semaphore to stop a burst of uploads holding every
request thread. A contract written now does not inherit that.
"""

from __future__ import annotations

import shutil
from typing import Annotated, Optional

from fastapi import APIRouter, Body, File, Form, Query, Request, Response, UploadFile, status

from application import documents as use_case
from application import ingest, workspace
from application.errors import InvalidRequest

from ..dependencies import Container, Page, SessionId, optional_number
from ..envelope import slice_of
from ..openapi import LOCATION_HEADER
from ..schemas import (
    Analysis, AnalysisChunks, AnalysisMethods, CanonicalUnit, CanonicalUnitCollection,
    Chunk, ChunkCollection, DocumentCollection, DocumentWithAnalysis, IngestJob,
)

router = APIRouter(prefix="/documents", tags=["documents"])

KnowledgeBaseFilter = Annotated[Optional[str], Query(
    alias="knowledge_base_id", description="narrow to one knowledge base")]


def _staged(upload: UploadFile):
    """How the ingest use case writes an uploaded file where its job finds it.

    ``application.ingest.Upload`` takes a ``save(path)`` callable and asks
    nothing about how the bytes arrived -- which is what lets the same
    submission run from a Flask ``FileStorage``, from this adapter's
    ``UploadFile`` and from a CLI with a local path.
    """
    def save(destination: str) -> None:
        upload.file.seek(0)
        with open(destination, "wb") as out:
            shutil.copyfileobj(upload.file, out)

    return ingest.Upload(filename=upload.filename or "", save=save)


@router.get("", response_model=DocumentCollection,
            summary="Every ingested document, newest first")
def list_documents(services: Container, page: Page,
                   knowledge_base_id: KnowledgeBaseFilter = None) -> DocumentCollection:
    """Each carries its ``analysis`` block, because "what may this document be
    asked about" is the first thing a screen listing documents needs and a
    second round trip per row is not a contract worth shipping."""
    records = use_case.list_all(services, knowledge_base_id or None)
    states = workspace.analysis_states()
    items = [
        DocumentWithAnalysis.read(record, state=states.get(record.get("doc_id")) or {})
        for record in slice_of(records, offset=page.offset, limit=page.limit)
    ]
    return DocumentCollection.of(items, offset=page.offset, limit=page.limit,
                                 total=len(records))


@router.post("", response_model=IngestJob, status_code=status.HTTP_202_ACCEPTED,
             responses={status.HTTP_202_ACCEPTED: {"headers": LOCATION_HEADER}},
             tags=["ingest jobs"], summary="Submit a document for ingestion")
def upload_document(
    services: Container, session: SessionId, request: Request, response: Response,
    file: Annotated[Optional[UploadFile], File(
        description="the document itself")] = None,
    knowledge_base_id: Annotated[Optional[str], Form()] = None,
    methods: Annotated[Optional[list[str]], Form(
        description="the chunking methods to analyse it with, one repeated "
                    "form field per method; an unknown or unavailable key is "
                    "dropped, and an empty selection falls back to the "
                    "default")] = None,
) -> IngestJob:
    """``multipart/form-data``, and always **202** with the job.

    One upload is one parse and one canonical, and every method runs over that
    same canonical -- so three methods cost one parse. What gets *indexed* for
    retrieval is still the knowledge base's own chunker; the methods are an
    analysis choice and do not change it.

    The same bytes submitted again while the first is still in flight are
    attached to that job rather than parsed twice; ``attached_uploads`` on the
    job says so.
    """
    if file is None:
        raise InvalidRequest("a file is required")
    if not file.filename:
        raise InvalidRequest("the uploaded file has no name")

    accepted = ingest.submit(
        services,
        upload=_staged(file),
        kb_id=knowledge_base_id,
        session_id=session,
        methods=methods,
    )
    record = services.ingest_jobs.describe(accepted.job)
    response.headers["Location"] = request.url_for(
        "get_job", job_id=record["job_id"]).path
    return IngestJob.of(record)


@router.get("/{document_id}", response_model=DocumentWithAnalysis, summary="Read one")
def get_document(services: Container, document_id: str) -> DocumentWithAnalysis:
    return DocumentWithAnalysis.read(
        use_case.get(services, document_id),
        state=workspace.analysis_state(document_id))


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT,
               response_class=Response, summary="Delete one")
def delete_document(services: Container, session: SessionId, document_id: str) -> Response:
    """Its chunks, its ledger row and its own analysis.

    What it does **not** take is the content: another upload of the same bytes
    keeps the shared analysis and its variants, and only the last upload of a
    content takes that down with it.
    """
    use_case.delete(services, document_id, session_id=session)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get("/{document_id}/chunks", response_model=ChunkCollection,
            summary="What this document was indexed as")
def document_chunks(services: Container, session: SessionId, page: Page, document_id: str,
                    knowledge_base_id: KnowledgeBaseFilter = None) -> ChunkCollection:
    """The corpus a question searches, not an analysis variant."""
    found = use_case.chunks_of(services, document_id, kb_id=knowledge_base_id,
                               session_id=session, offset=page.offset, limit=page.limit)
    return ChunkCollection.of([Chunk.of(row) for row in found["chunks"]],
                              offset=found["offset"], limit=found["limit"],
                              total=found["total"])


@router.get("/{document_id}/units", response_model=CanonicalUnitCollection,
            summary="The parser's canonical reading, before any chunker")
def document_units(
    services: Container, session: SessionId, page: Page, document_id: str,
    knowledge_base_id: KnowledgeBaseFilter = None,
    page_from: Annotated[Optional[str], Query(description="first page to read")] = None,
    page_to: Annotated[Optional[str], Query(description="last page to read")] = None,
    unit_type: Annotated[Optional[str], Query(description="one unit type only")] = None,
) -> CanonicalUnitCollection:
    """Read-only and cache-backed -- the file is never parsed again. Read
    beside the chunks to tell a parser reading-order problem from a chunker
    one."""
    found = use_case.canonical_units(
        services, document_id, kb_id=knowledge_base_id, session_id=session,
        page_from=optional_number(page_from), page_to=optional_number(page_to),
        unit_type=unit_type or None, offset=page.offset, limit=page.limit,
    )
    return CanonicalUnitCollection.of(
        [CanonicalUnit.of(row) for row in found["units"]],
        offset=found["offset"], limit=found["limit"], total=found["total"],
        pages=found["pages_in_document"])


# ------------------------------------------------------------- the analysis
@router.get("/{document_id}/analysis", response_model=Analysis, tags=["analysis"],
            summary="Where this document's chunking analysis got to")
def get_analysis(document_id: str) -> Analysis:
    """Always **200**, including ``status: "missing"``: a document that exists
    with no analysis is a fact, not a missing resource, and a client polling
    for a build should not have to read a 404 as progress."""
    return Analysis.of(workspace.analysis_state(document_id))


@router.post("/{document_id}/analysis", response_model=Analysis, tags=["analysis"],
             status_code=status.HTTP_202_ACCEPTED, summary="Queue or retry it")
def request_analysis(services: Container, document_id: str) -> Analysis:
    """Queuing is all it does: the build runs on a worker, so this never waits
    on it."""
    return Analysis.of(workspace.request_analysis(services, document_id))


@router.post("/{document_id}/analysis/methods", response_model=Analysis, tags=["analysis"],
             status_code=status.HTTP_202_ACCEPTED,
             summary="Add chunking variants to a document that is already here")
def add_analysis_methods(
    document_id: str,
    payload: Annotated[AnalysisMethods, Body(default_factory=AnalysisMethods)],
) -> Analysis:
    """This is how a second method reaches a document -- not by uploading the
    file again. The canonical is on disk, so nothing is parsed twice and no
    variant already built is rebuilt."""
    return Analysis.of(workspace.add_methods(document_id, payload.methods))


@router.get("/{document_id}/analysis/methods/{method}/chunks",
            response_model=AnalysisChunks, tags=["analysis"],
            summary="The rows one chunking method produced")
def analysis_method_chunks(page: Page, document_id: str, method: str) -> AnalysisChunks:
    """Three different refusals, deliberately kept apart, because they call for
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
    rows = arm["rows"]
    return AnalysisChunks.of(
        slice_of(rows, offset=page.offset, limit=page.limit),
        offset=page.offset, limit=page.limit, total=len(rows),
        method=method, engine=arm["kind"], content_id=found.get("key"))
