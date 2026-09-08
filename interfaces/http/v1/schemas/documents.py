"""One ingest -- one file, in exactly one knowledge base -- and its analysis.

Two identities are kept apart everywhere below, because conflating them is the
single most expensive mistake a schema can make:

    ``id``          the upload. One file, ingested into one knowledge base.
    ``content_id``  the bytes. Shared by every upload of the same document,
                    and what an analysis and its variants actually belong to.

``file_path`` -- the ledger's key, an absolute path on the machine that ran
the upload -- is not here and never leaves the server. It is the field a
PostgreSQL schema replaces with a row id, and no client has ever needed it.
"""

from __future__ import annotations

from typing import Any, Optional, Union

from pydantic import Field

from .common import Collection, Schema

#: The analysis states a document can be in. The same five the packager uses;
#: named here because they are part of the contract.
ANALYSIS_STATES = ("missing", "pending", "running", "ready", "failed")


class ContentAnalysis(Schema):
    """The **shared** analysis: every variant these bytes have, reusable by any
    upload of them, and the other uploads that share it."""

    requested_methods: list[str] = Field(default_factory=list)
    ready_methods: list[str] = Field(default_factory=list)
    shared_with_document_ids: list[str] = Field(default_factory=list)


class Analysis(Schema):
    """Where one upload's chunking analysis got to, truthfully.

    The two levels are kept apart on the wire because they are two different
    facts, and a screen that merges them lies in one direction or the other:
    ``selected_methods`` / ``ready_methods`` are **this upload's** -- ready is
    the intersection of what it selected with what has been built, the
    ``visible = selected ∩ ready`` rule -- while ``content`` is the shared
    analysis.
    """

    status: str = Field(description=f"one of {', '.join(ANALYSIS_STATES)}")
    content_id: Optional[str] = None
    selected_methods: list[str] = Field(default_factory=list)
    ready_methods: list[str] = Field(default_factory=list)
    failed_methods: list[str] = Field(default_factory=list)
    unit_count: Optional[int] = None
    deep_source: Optional[str] = None
    error: Optional[str] = None
    updated_at: Optional[str] = None
    content: ContentAnalysis

    @classmethod
    def of(cls, state: dict) -> "Analysis":
        selected = list(state.get("selected_methods") or [])
        return cls(
            status=state.get("status") or "missing",
            content_id=state.get("key"),
            selected_methods=selected,
            ready_methods=list(state.get("available_methods") or []),
            failed_methods=[m for m in (state.get("failed_methods") or [])
                            if m in selected],
            unit_count=state.get("unit_count"),
            deep_source=state.get("deep_source"),
            error=state.get("error") or None,
            updated_at=state.get("updated_at"),
            content=ContentAnalysis(
                requested_methods=list(state.get("requested") or []),
                ready_methods=list(state.get("ready_methods") or []),
                shared_with_document_ids=list(state.get("doc_ids") or []),
            ),
        )


class Document(Schema):
    """One ingest: one file, in exactly one knowledge base."""

    id: Optional[str] = None
    knowledge_base_id: Optional[str] = None
    name: str = ""
    content_id: Optional[str] = Field(
        default=None,
        description="the bytes, not the upload: two documents with one "
                    "content_id are the same file ingested twice",
    )
    size_bytes: int = 0
    chunk_count: int = 0
    chunking_mode: Optional[str] = None
    status: str = "indexed"
    ingested_at: Optional[str] = None
    ingest_job_id: Optional[str] = None

    @classmethod
    def of(cls, record: dict) -> "Document":
        metadata = record.get("metadata") or {}
        return cls(
            id=record.get("doc_id"),
            knowledge_base_id=record.get("kb_id"),
            name=metadata.get("original_filename") or record.get("file_name") or "",
            content_id=(record.get("file_hash") or "") or None,
            size_bytes=record.get("file_size") or 0,
            chunk_count=record.get("chunk_count") or 0,
            chunking_mode=record.get("chunking_mode"),
            status=record.get("status") or "indexed",
            ingested_at=record.get("ingested_at") or None,
            ingest_job_id=metadata.get("ingest_job_id"),
        )


class DocumentWithAnalysis(Document):
    """A document read on its own or in a list, carrying its analysis block.

    Carried rather than fetched separately because "what may this document be
    asked about" is the first thing a screen listing documents needs, and a
    second round trip per row is not a contract worth shipping.
    """

    analysis: Analysis

    @classmethod
    def read(cls, record: dict, *, state: dict) -> "DocumentWithAnalysis":
        return cls(**Document.of(record).model_dump(), analysis=Analysis.of(state))


class DocumentCollection(Collection[DocumentWithAnalysis]):
    pass


class AnalysisMethods(Schema):
    """The chunking variants to add to a document that is already here.

    This is how a second method reaches a document -- not by uploading the
    file again. Accepts the repeated form or one comma-separated string,
    because both are what the registry's keys arrive as.
    """

    methods: Optional[Union[list[str], str]] = None


class AnalysisChunks(Collection[dict[str, Any]]):
    """The rows one chunking method produced for this document.

    The rows are the packager's own JSONL, passed through: a live document is
    read over exactly the representation a frozen one is, and re-projecting
    them here would make the two differ.
    """

    method: str
    engine: Optional[str] = None
    content_id: Optional[str] = None
