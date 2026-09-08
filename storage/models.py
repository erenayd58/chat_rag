"""The relational schema: five concepts and the edges the product maintains.

    KNOWLEDGE BASE   a named collection with its own chunker and its own
                     vector store
    DOCUMENT         one ingest -- one file, in exactly one knowledge base --
                     identified by its ``doc_id``
    CONTENT          the bytes, identified by their hash, shared by every
                     upload of them, and the owner of the analysis
    VARIANT          one chunking method run over one content, built once
    INGEST JOB       the record a restart answers a ``job_id`` from

and, beside them, the gold set: one confirmed answer per (knowledge base,
question).

Two things about this schema are deliberate and easy to get wrong:

**Document identity and content identity are two columns, not one.** The same
PDF uploaded into two knowledge bases is two documents and one content. Which
methods an upload *selected* belongs to the membership row; which variants a
content *has* belongs to the content. Collapsing them either loses an upload
or shares a choice that was never shared
(``tests/migration/test_domain_relations.py`` states the rule from outside).

**``kb_id`` on a document or a content is not a foreign key.** Deleting a
knowledge base here deliberately leaves its documents' ledger rows and their
analyses behind, still naming the id that is gone: the console groups them as
"knowledge base deleted" rather than dropping the record that a file was ever
ingested. ``ON DELETE CASCADE`` would silently lose that, and ``ON DELETE SET
NULL`` would lose the id the grouping is by. So the edge is maintained by the
domain, which is what the Step 6 characterisation tests hold it to. Every edge
that *is* referential -- a variant to its content, a membership to its
content -- is a real foreign key with a real cascade.

Nothing here stores a vector. Embeddings stay in the vector store until
Step 9; what a document row knows about them is how many chunks were written.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from sqlalchemy import (
    BigInteger, Boolean, CheckConstraint, DateTime, Float, ForeignKey, Index,
    Integer, String, Text, UniqueConstraint, func,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

#: Every identifier this product mints is short -- a uuid4's first eight
#: characters for a knowledge base, the chunker's document id for a document.
#: They are text, not uuids, because they are *already* public in ``/api/v1``
#: and re-minting them would change identities the contract pins.
ID = String(128)


class Base(DeclarativeBase):
    """One metadata object for the whole schema."""


class TimestampedMixin:
    """When a row appeared and when it last changed. Database time, not the
    application's: two processes writing one table should not disagree about
    what "now" is."""

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(),
        nullable=False,
    )


# --------------------------------------------------------------------------
# knowledge bases
# --------------------------------------------------------------------------
class KnowledgeBase(TimestampedMixin, Base):
    """A named collection with its own chunker, embedding and store."""

    __tablename__ = "knowledge_bases"

    id: Mapped[str] = mapped_column(ID, primary_key=True)
    name: Mapped[str] = mapped_column(Text, nullable=False)
    #: The name with case and inner spacing normalised away, which is the
    #: identity this product has always compared by. A unique index on it is
    #: what makes two concurrent creations of "Yillik raporlar" one row and
    #: one error, rather than a race the in-process check loses.
    name_key: Mapped[str] = mapped_column(Text, nullable=False)

    chunker_type: Mapped[str] = mapped_column(Text, nullable=False)
    chunker_params: Mapped[dict] = mapped_column(
        JSONB, nullable=False, default=dict, server_default="{}"
    )
    embedding_model_name: Mapped[Optional[str]] = mapped_column(Text)
    embedding_provider: Mapped[Optional[str]] = mapped_column(Text)
    vector_db_provider: Mapped[str] = mapped_column(
        Text, nullable=False, server_default="chroma"
    )
    #: Where this knowledge base keeps its vectors when it does not take the
    #: default. A path, and the only one left in the relational schema: it
    #: addresses the *vector store*, which is a directory until Step 9, and it
    #: is never an identity -- the primary key above is.
    vector_db_path: Mapped[Optional[str]] = mapped_column(Text)
    retrieval_method: Mapped[str] = mapped_column(
        Text, nullable=False, server_default="hybrid"
    )
    extra: Mapped[dict] = mapped_column(
        JSONB, nullable=False, default=dict, server_default="{}"
    )
    #: Anything a caller set that has no column of its own. The store this
    #: replaced was a JSON document and accepted any key; a record written by
    #: an older console -- ``embedding_provider`` is the one still in live
    #: stores -- must read back whole rather than lose a field to the schema.
    attributes: Mapped[dict] = mapped_column(
        JSONB, nullable=False, default=dict, server_default="{}"
    )

    __table_args__ = (
        UniqueConstraint("name_key", name="uq_knowledge_bases_name_key"),
        CheckConstraint("length(name) > 0", name="ck_knowledge_bases_name_not_blank"),
    )


# --------------------------------------------------------------------------
# content identity and its analysis
# --------------------------------------------------------------------------
class Content(TimestampedMixin, Base):
    """The bytes, and the analysis that belongs to them.

    One row per document *content*, whatever number of uploads point at it.
    ``content_key`` is the stable name the packaged artifacts sit under on
    disk -- the canonical units, the Deep run and the variants are files, and
    stay files until something makes them not be -- so the directory is
    derived from this row rather than being the record.
    """

    __tablename__ = "contents"

    id: Mapped[str] = mapped_column(ID, primary_key=True)
    #: The artifact directory's name. Unique: two contents sharing one
    #: directory would overwrite each other's canonical.
    content_key: Mapped[str] = mapped_column(Text, nullable=False)
    #: The document hash, when the upload knew it. Records written before
    #: content identity existed are keyed by their own upload id and have none.
    content_sha256: Mapped[Optional[str]] = mapped_column(Text)

    status: Mapped[str] = mapped_column(Text, nullable=False, server_default="pending")
    label: Mapped[Optional[str]] = mapped_column(Text)
    #: Not a foreign key. See the module docstring: an analysis outlives the
    #: deletion of the knowledge base its upload was in.
    kb_id: Mapped[Optional[str]] = mapped_column(ID, index=True)
    kb_name: Mapped[Optional[str]] = mapped_column(Text)
    chunking_mode: Mapped[Optional[str]] = mapped_column(Text)

    #: The union of every upload's choice: what this content should end up
    #: having, because a variant is built once and reused by all of them.
    requested_methods: Mapped[list] = mapped_column(
        ARRAY(Text), nullable=False, default=list, server_default="{}"
    )
    #: Recorded only once a build has run, which is why "not built yet" and
    #: "built and empty" are a null and an empty array rather than one value:
    #: the Step 6 contract reads the absence.
    ready_methods: Mapped[Optional[list]] = mapped_column(ARRAY(Text))
    failed_methods: Mapped[Optional[list]] = mapped_column(ARRAY(Text))

    unit_count: Mapped[Optional[int]] = mapped_column(Integer)
    parse_seconds: Mapped[Optional[float]] = mapped_column(Float)
    payload_bytes: Mapped[Optional[int]] = mapped_column(BigInteger)
    deep_source: Mapped[Optional[str]] = mapped_column(Text)
    error: Mapped[Optional[str]] = mapped_column(Text)
    traceback: Mapped[Optional[str]] = mapped_column(Text)

    memberships: Mapped[list["ContentDocument"]] = relationship(
        back_populates="content", cascade="all, delete-orphan",
        passive_deletes=True, lazy="selectin",
    )
    variants: Mapped[list["ContentVariant"]] = relationship(
        back_populates="content", cascade="all, delete-orphan",
        passive_deletes=True, lazy="selectin",
    )

    __table_args__ = (
        UniqueConstraint("content_key", name="uq_contents_content_key"),
        UniqueConstraint("content_sha256", name="uq_contents_sha256"),
    )


class ContentDocument(Base):
    """One upload's membership of one content, and what *it* asked for.

    The upload level. ``selected_methods`` is null for a record written before
    uploads carried their own choice, and null is not the same as empty: it
    means "fall back to the content's request", which is how those documents
    have always behaved.

    The composite primary key makes attaching an upload idempotent; the unique
    constraint on ``doc_id`` alone is the one that matters concurrently, since
    one upload belongs to exactly one content and two threads racing to attach
    it to two contents must not both win.
    """

    __tablename__ = "content_documents"

    content_id: Mapped[str] = mapped_column(
        ID, ForeignKey("contents.id", ondelete="CASCADE"), primary_key=True
    )
    doc_id: Mapped[str] = mapped_column(ID, primary_key=True)
    selected_methods: Mapped[Optional[list]] = mapped_column(ARRAY(Text))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    content: Mapped[Content] = relationship(back_populates="memberships")

    __table_args__ = (
        UniqueConstraint("doc_id", name="uq_content_documents_doc_id"),
    )


class ContentVariant(TimestampedMixin, Base):
    """One chunking method run over one content."""

    __tablename__ = "content_variants"

    content_id: Mapped[str] = mapped_column(
        ID, ForeignKey("contents.id", ondelete="CASCADE"), primary_key=True
    )
    method: Mapped[str] = mapped_column(String(64), primary_key=True)
    status: Mapped[str] = mapped_column(Text, nullable=False)
    #: How a Deep variant was produced -- an ingest's own run, or the
    #: deterministic contract that makes no provider call.
    source: Mapped[Optional[str]] = mapped_column(Text)
    chunk_count: Mapped[Optional[int]] = mapped_column(Integer)
    seconds: Mapped[Optional[float]] = mapped_column(Float)
    error: Mapped[Optional[str]] = mapped_column(Text)
    #: The rest of what a build recorded about this variant -- run status, run
    #: mode, model id, call count. One jsonb column rather than five nullable
    #: ones, because only Deep has them and nothing ever selects on them.
    details: Mapped[dict] = mapped_column(
        JSONB, nullable=False, default=dict, server_default="{}"
    )

    content: Mapped[Content] = relationship(back_populates="variants")


# --------------------------------------------------------------------------
# documents: the ingest ledger
# --------------------------------------------------------------------------
class Document(TimestampedMixin, Base):
    """One ingested file, in exactly one knowledge base.

    The row identity is ``id``; the product identity is ``doc_id``, which is
    what ``/api/v1`` has always addressed a document by. Neither is a path.
    ``file_name`` is what the upload was called, kept because every screen
    shows it; ``source_path`` is retained as a diagnostic only and is
    deliberately not unique, not indexed and not an address -- it named a
    staging file that is deleted the moment the job ends.
    """

    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(ID, primary_key=True)
    doc_id: Mapped[str] = mapped_column(ID, nullable=False)
    #: Not a foreign key; see the module docstring.
    kb_id: Mapped[Optional[str]] = mapped_column(ID, index=True)

    file_name: Mapped[str] = mapped_column(Text, nullable=False)
    source_path: Mapped[Optional[str]] = mapped_column(Text)
    file_hash: Mapped[Optional[str]] = mapped_column(Text, index=True)
    file_size: Mapped[int] = mapped_column(BigInteger, nullable=False, server_default="0")
    chunk_count: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    status: Mapped[str] = mapped_column(Text, nullable=False, server_default="indexed")
    chunking_mode: Mapped[Optional[str]] = mapped_column(Text)
    #: The application's own timestamp, in the ISO form every screen and the
    #: ``/api/v1`` schema already carry. Kept as text so a migrated row reads
    #: back exactly as it was written.
    ingested_at: Mapped[str] = mapped_column(Text, nullable=False)
    #: The upload's own metadata, as the console wrote it: original filename,
    #: upload source, the ingest job id, the Deep Analysis report.
    doc_metadata: Mapped[dict] = mapped_column(
        "metadata", JSONB, nullable=False, default=dict, server_default="{}"
    )
    #: The configuration that produced this corpus, captured after the ingest
    #: succeeded. Read back by ``python -m cli report``.
    pipeline_snapshot: Mapped[Optional[dict]] = mapped_column(JSONB)
    #: The ingest job that wrote this row, projected out of the metadata so a
    #: restart settles a ``job_id`` with an index lookup instead of a scan of
    #: every document ever ingested.
    ingest_job_id: Mapped[Optional[str]] = mapped_column(ID)

    __table_args__ = (
        UniqueConstraint("doc_id", name="uq_documents_doc_id"),
        Index("ix_documents_ingest_job_id", "ingest_job_id"),
        Index("ix_documents_ingested_at", "ingested_at"),
    )


# --------------------------------------------------------------------------
# ingest jobs
# --------------------------------------------------------------------------
class IngestJobRecord(TimestampedMixin, Base):
    """What a restart is allowed to answer a ``job_id`` with.

    A job itself lives in memory and is not resumed. This row is the promise
    made to a client holding its id: written when the job is accepted, when it
    starts and when it reaches a terminal state, and settled against the
    document ledger by the next process to start.
    """

    __tablename__ = "ingest_jobs"

    job_id: Mapped[str] = mapped_column(ID, primary_key=True)
    kb_id: Mapped[Optional[str]] = mapped_column(ID, index=True)
    status: Mapped[str] = mapped_column(Text, nullable=False, index=True)
    doc_id: Mapped[Optional[str]] = mapped_column(ID)
    filename: Mapped[Optional[str]] = mapped_column(Text)
    #: Wall clock, the retention window's key -- kept as the epoch seconds the
    #: journal has always recorded, so a record survives the migration
    #: verbatim and one retention rule still covers both.
    journalled_at: Mapped[float] = mapped_column(Float, nullable=False, server_default="0")
    restart_recovered: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default="false"
    )
    #: The whole journalled snapshot. The columns above are projections of it,
    #: for the two queries this table serves -- settle what is active, delete
    #: what is old -- and everything else a client polls for round-trips here
    #: rather than being flattened into columns nothing selects on.
    snapshot: Mapped[dict] = mapped_column(
        JSONB, nullable=False, default=dict, server_default="{}"
    )

    __table_args__ = (
        Index("ix_ingest_jobs_journalled_at", "journalled_at"),
    )


# --------------------------------------------------------------------------
# the gold set
# --------------------------------------------------------------------------
class GoldSetEntry(Base):
    """One confirmed answer: a (knowledge base, question) a human has marked.

    The only production-side ground truth this project has, and the input the
    regression CLI reads. Addressed by ``entry_id``, which is derived from the
    question and the document hash rather than from a chunk id -- chunk ids
    move whenever the chunker does.
    """

    __tablename__ = "gold_set_entries"

    entry_id: Mapped[str] = mapped_column(ID, primary_key=True)
    schema_version: Mapped[int] = mapped_column(Integer, nullable=False, server_default="1")
    kb_id: Mapped[str] = mapped_column(ID, nullable=False, index=True)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    document_id: Mapped[Optional[str]] = mapped_column(Text)
    document_title: Mapped[Optional[str]] = mapped_column(Text)
    document_sha256: Mapped[Optional[str]] = mapped_column(Text)
    correct_chunk_id: Mapped[Optional[str]] = mapped_column(Text)
    section: Mapped[Optional[str]] = mapped_column(Text)
    #: Free-form on purpose: a page list is sometimes numbers and sometimes
    #: the labels a scanned report prints. jsonb keeps whatever was marked.
    pages: Mapped[Optional[list]] = mapped_column(JSONB)
    unit_ids: Mapped[Optional[list]] = mapped_column(JSONB)
    evidence: Mapped[str] = mapped_column(Text, nullable=False, server_default="")
    retrieval_method: Mapped[Optional[str]] = mapped_column(Text)
    found_at_rank: Mapped[Optional[int]] = mapped_column(Integer)
    #: The application's ISO stamps, as the file store wrote them.
    created_at_text: Mapped[str] = mapped_column("created_at", Text, nullable=False)
    updated_at_text: Mapped[str] = mapped_column("updated_at", Text, nullable=False)


#: Every table this application owns, ordered so a truncation or a delete may
#: walk it front to back without tripping a foreign key.
ALL_TABLES = (
    "gold_set_entries",
    "ingest_jobs",
    "documents",
    "content_variants",
    "content_documents",
    "contents",
    "knowledge_bases",
)
