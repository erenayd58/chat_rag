"""Stored chunks, ranked chunks, and the parser's canonical units.

``content`` is the document's own text, verbatim: it is what a citation
quotes, and a store may index whatever it likes but may not hand back a
re-rendered version. ``metadata`` is a pass-through field and says so in the
API doc -- it carries what the product produced, it is useful, and pinning it
would freeze internals this contract exists to leave free.
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import Field

from .common import Collection, Schema


class Chunk(Schema):
    """One stored chunk, as an inspection screen reads it."""

    id: Optional[str] = None
    document_id: Optional[str] = None
    content: str = ""
    chunk_index: Optional[int] = None
    total_chunks: Optional[int] = None
    section: Optional[str] = None
    chunking_mode: Optional[str] = None
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="what the chunker recorded; pass-through and not contractual",
    )

    @classmethod
    def of(cls, row: dict) -> "Chunk":
        metadata = row.get("metadata") or {}
        return cls(
            id=row.get("chunk_id"),
            document_id=metadata.get("doc_id"),
            content=row.get("content") or "",
            chunk_index=metadata.get("chunk_index"),
            total_chunks=metadata.get("total_chunks"),
            section=metadata.get("section_title") or metadata.get("heading"),
            chunking_mode=metadata.get("chunking_mode"),
            metadata=metadata,
        )


class ScoredChunk(Chunk):
    """A chunk a search returned, with the score that ranked it.

    ``score`` is comparable only within one answer: it is whatever the named
    method produced, and nothing here rescales it into a pretence of a
    universal number.
    """

    score: Optional[float] = None
    retrieval_method: Optional[str] = None

    @classmethod
    def ranked(cls, row: dict, *, method: str) -> "ScoredChunk":
        body = cls.of(row).model_dump()
        body["score"] = row.get("score", row.get("similarity_score"))
        body["retrieval_method"] = row.get("retrieval_method") or method
        return cls(**body)


class ChunkCollection(Collection[Chunk]):
    pass


class SearchResults(Collection[ScoredChunk]):
    """A search's answer: the ranked chunks, and which method ranked them."""

    method: Optional[str] = None
    knowledge_base_id: Optional[str] = None


class CanonicalUnit(Schema):
    """One unit of the parser's canonical reading of a document, before any
    chunker saw it."""

    id: Optional[str] = None
    order: Optional[int] = None
    type: Optional[str] = None
    text: str = ""
    heading_level: Optional[int] = None
    section_path: list[Any] = Field(default_factory=list)
    source: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def of(cls, row: dict) -> "CanonicalUnit":
        return cls(
            id=row.get("unit_id"),
            order=row.get("order"),
            type=row.get("type"),
            text=row.get("text") or "",
            heading_level=row.get("heading_level"),
            section_path=row.get("section_path") or [],
            source=row.get("source") or {},
        )


class CanonicalUnitCollection(Collection[CanonicalUnit]):
    """A page of units, plus every page number the document has, so a reader
    can jump without walking the collection."""

    pages: list[Any] = Field(default_factory=list)
