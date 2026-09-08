"""A submitted upload, and what became of it.

A job id outlives the process that minted it: the journal records every
transition and start-up settles anything in flight against the ledger, so
``restart_settled`` is how a client tells "it finished while you were away"
from "it never happened". ``document_id`` is filled in the moment the ledger
knows the document, which is the job's last act.
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import Field

from .common import Collection, Schema


class JobError(Schema):
    """Why a job did not finish. ``type`` is the category a client branches on."""

    type: str = "internal"
    message: Optional[str] = None


class JobResult(Schema):
    """What a finished job produced."""

    document_id: Optional[str] = None
    chunk_count: Optional[int] = None
    chunking_mode: Optional[str] = None
    deep_analysis: Optional[Any] = None
    analysis_status: Optional[Any] = None


class IngestJob(Schema):
    id: Optional[str] = None
    status: Optional[str] = None
    knowledge_base_id: Optional[str] = None
    document_id: Optional[str] = None
    content_id: Optional[str] = None
    name: Optional[str] = None
    methods: list[str] = Field(default_factory=list)
    queue_position: Optional[int] = None
    attached_uploads: int = 0
    submitted_at: Optional[Any] = None
    started_at: Optional[Any] = None
    finished_at: Optional[Any] = None
    run_seconds: Optional[float] = None
    restart_settled: bool = False
    error: Optional[JobError] = None
    result: Optional[JobResult] = None

    @classmethod
    def of(cls, record: dict) -> "IngestJob":
        return cls(
            id=record.get("job_id"),
            status=record.get("status"),
            knowledge_base_id=record.get("kb_id"),
            document_id=record.get("doc_id"),
            content_id=record.get("content_sha256"),
            name=record.get("filename"),
            methods=list(record.get("methods") or []),
            queue_position=record.get("position"),
            attached_uploads=record.get("attached_uploads") or 0,
            submitted_at=record.get("submitted_at"),
            started_at=record.get("started_at"),
            finished_at=record.get("finished_at"),
            run_seconds=record.get("run_seconds"),
            restart_settled=bool(record.get("restart_recovered")),
            error=cls._error(record),
            result=cls._result(record),
        )

    @staticmethod
    def _error(record: dict) -> Optional[JobError]:
        if not record.get("error"):
            return None
        return JobError(type=record.get("error_category") or "internal",
                        message=record.get("error"))

    @staticmethod
    def _result(record: dict) -> Optional[JobResult]:
        result = record.get("result")
        if not result:
            return None
        return JobResult(
            document_id=result.get("doc_id"),
            chunk_count=result.get("chunks_created"),
            chunking_mode=result.get("chunking_mode"),
            deep_analysis=result.get("deep_analysis"),
            analysis_status=result.get("viewer_analysis"),
        )


class IngestCapacity(Schema):
    """One line of the queue, so a client can tell "slow" from "full"."""

    running: int
    queued: int
    queue_capacity: int
    workers: int


class IngestJobCollection(Collection[IngestJob]):
    capacity: IngestCapacity
