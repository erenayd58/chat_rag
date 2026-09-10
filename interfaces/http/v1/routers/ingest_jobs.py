"""`/api/v1/ingest-jobs` -- what became of a submitted upload.

A job id outlives the process that minted it. Every transition is journalled
and start-up settles anything that was in flight against the ingest ledger, so
a **404** here means one thing only: the job finished longer ago than jobs are
kept. It never means "we lost it".
"""

from __future__ import annotations

from typing import Annotated, Optional

from fastapi import APIRouter, Query

from chat_rag.application import ingest as use_case

from ..dependencies import Container, Page
from ..envelope import flag, slice_of
from ..schemas import IngestCapacity, IngestJob, IngestJobCollection

router = APIRouter(prefix="/ingest-jobs", tags=["ingest jobs"])


@router.get("", response_model=IngestJobCollection, summary="List them")
def list_jobs(
    services: Container, page: Page,
    knowledge_base_id: Annotated[Optional[str], Query(
        description="narrow to one knowledge base")] = None,
    active: Annotated[Optional[str], Query(
        description="leave out the finished ones")] = None,
) -> IngestJobCollection:
    found = use_case.list_jobs(services, knowledge_base_id or None,
                               active_only=flag(active))
    records = found["jobs"]
    capacity = found["capacity"]
    items = [IngestJob.of(record)
             for record in slice_of(records, offset=page.offset, limit=page.limit)]
    return IngestJobCollection.of(
        items, offset=page.offset, limit=page.limit, total=len(records),
        capacity=IngestCapacity(
            running=capacity["running"],
            queued=capacity["queued"],
            queue_capacity=capacity["queue_capacity"],
            workers=capacity["workers"],
        ))


@router.get("/{job_id}", response_model=IngestJob, summary="Read one")
def get_job(services: Container, job_id: str) -> IngestJob:
    return IngestJob.of(use_case.job(services, job_id))


@router.delete("/{job_id}", response_model=IngestJob, summary="Cancel one")
def cancel_job(services: Container, job_id: str) -> IngestJob:
    """A queued job at once, a running one at its next seam.

    A running job that has already written its document finishes as
    ``succeeded`` -- the returned ``status`` says which happened, so this is a
    resource and not a 204.
    """
    return IngestJob.of(use_case.cancel(services, job_id))
