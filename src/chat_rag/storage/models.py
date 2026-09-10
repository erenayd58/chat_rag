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

Since Step 9 the vectors are here too, in the two tables at the bottom of
this module: a *collection* (one per knowledge base, plus the one the console
uses when none is selected) and the chunk rows it holds, each with its
embedding in a ``pgvector`` column. They are kept apart from the five
concepts above on purpose -- a chunk row is derived state, rebuilt by a
re-ingest or a re-index, and the only edge it has into the relational schema
is the one that must cascade: delete a knowledge base and its vectors go with
it, in the same transaction, by the database rather than by a caller
remembering to.
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


def _vector_column_type():
    """The ``vector`` column type, or ``JSONB`` where pgvector is not installed.

    ``pgvector`` is a production dependency and PostgreSQL is where the
    vectors live, so the first branch is what every deployment and every test
    takes. The fallback exists for the one thing this repository guarantees
    without a database or a full install -- importing the application, which
    ``tools/import_smoke.py`` and ``python -m cli manifest`` do on machines
    that have neither. A column that is never queried there cannot be wrong
    there; a failed import would be.
    """
    try:
        from pgvector.sqlalchemy import Vector
    except ImportError:  # pragma: no cover - the install is the normal case
        return JSONB
    # No argument: the column takes any width. See ``ChunkVector.embedding``.
    return Vector()


_VECTOR = _vector_column_type()


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
    #: Which store holds this knowledge base's vectors. ``pgvector`` is the
    #: only value this application writes or accepts; the column is kept
    #: because every record carries it and the provenance snapshot reports
    #: what a corpus was written with.
    vector_db_provider: Mapped[str] = mapped_column(
        Text, nullable=False, server_default="pgvector"
    )
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


# --------------------------------------------------------------------------
# the vectors
# --------------------------------------------------------------------------
class VectorCollection(TimestampedMixin, Base):
    """One searchable corpus, and which embedding space it is in.

    A collection is what a directory of Chroma files used to be: the unit a
    query is scoped to, and the unit a knowledge base owns. Its name *is* the
    knowledge base id, except for the one collection the console uses when no
    knowledge base is selected -- that one has no ``kb_id`` and is named by
    ``VECTOR_DB_COLLECTION`` (``documents``).

    The manifest columns are the ``embedding_index.json`` that used to sit
    beside the store, moved here for the same reason the vectors were: a
    knowledge base is a row now, not a directory, and a file beside a
    directory that no longer exists cannot describe it. They say which model
    wrote the vectors this collection holds, so a query embedded by a
    different one is refused rather than compared across spaces
    (``components/embedding/index_manifest.py`` is the comparison).

    ``kb_id`` **is** a foreign key, and it cascades -- unlike the ``kb_id`` on
    a document or a content, which deliberately outlives its knowledge base.
    The difference is what the row is: a ledger row records that a file was
    once ingested and is worth keeping after the collection it went into is
    gone; a vector is only meaningful inside the corpus it belongs to, and
    keeping it would be an orphan a later knowledge base could match against.
    """

    __tablename__ = "vector_collections"

    collection: Mapped[str] = mapped_column(ID, primary_key=True)
    kb_id: Mapped[Optional[str]] = mapped_column(
        ID, ForeignKey("knowledge_bases.id", ondelete="CASCADE"), index=True
    )

    embedding_provider: Mapped[Optional[str]] = mapped_column(Text)
    embedding_model: Mapped[Optional[str]] = mapped_column(Text)
    embedding_endpoint: Mapped[Optional[str]] = mapped_column(Text)
    embedding_dimension: Mapped[Optional[int]] = mapped_column(Integer)
    embedding_fingerprint: Mapped[Optional[str]] = mapped_column(Text)
    #: What the manifest recorded at the last write. Not a count of the rows
    #: below -- it is what the writer *said*, kept verbatim so a store whose
    #: rows changed underneath still reports the number its manifest carries,
    #: exactly as the file did.
    manifest_chunk_count: Mapped[Optional[int]] = mapped_column(Integer)
    #: The application's own ISO stamp, as the manifest file carried it.
    written_at: Mapped[Optional[str]] = mapped_column(Text)

    chunks: Mapped[list["ChunkVector"]] = relationship(
        back_populates="collection_row", cascade="all, delete-orphan",
        passive_deletes=True,
    )


class ChunkVector(TimestampedMixin, Base):
    """One chunk of one document: its text, its metadata and its embedding.

    The columns are the fields something other than the chunk itself reads --
    the document a chunk belongs to, its position in it, the heading it sits
    under, the method that produced it. Everything else the chunker wrote
    stays whole in ``metadata``, because two derived renderings the answer
    chain depends on (``search_text`` and ``table_view``) are carried there
    and a store that widened its record for them would have to widen it again
    for the next one.

    ``embedding`` has no declared width. The application supports more than
    one embedding space -- 384 dimensions from the local sentence-transformer,
    4096 from the demo's gateway model, and a 1-wide placeholder a
    lexical-only ingestion writes because nothing embeds its text -- and a
    ``vector(n)`` column would have to pick one and break the others. The
    width that was actually written is recorded beside it, which is what the
    manifest comparison reads.
    """

    __tablename__ = "chunk_vectors"

    collection: Mapped[str] = mapped_column(
        ID, ForeignKey("vector_collections.collection", ondelete="CASCADE"),
        primary_key=True,
    )
    #: The chunker's own id for this chunk, unique within its collection.
    #: Re-ingesting the same document overwrites the row rather than adding a
    #: second one with the same name.
    chunk_id: Mapped[str] = mapped_column(Text, primary_key=True)

    doc_id: Mapped[str] = mapped_column(Text, nullable=False, server_default="")
    doc_title: Mapped[str] = mapped_column(Text, nullable=False, server_default="")
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    total_chunks: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    section_title: Mapped[Optional[str]] = mapped_column(Text)
    #: Which chunking method produced this row (``metadata['chunker_type']``),
    #: projected out so a corpus can be described without reading every row's
    #: json.
    method: Mapped[Optional[str]] = mapped_column(String(64))

    #: The document's own text, byte for byte. A citation quotes this.
    content: Mapped[str] = mapped_column(Text, nullable=False, server_default="")
    chunk_metadata: Mapped[dict] = mapped_column(
        "metadata", JSONB, nullable=False, default=dict, server_default="{}"
    )

    #: Declared as ``vector`` with no width; see the class docstring. The type
    #: comes from :func:`_vector_column_type` so this module still imports on
    #: a machine with no pgvector installed (``python -m cli manifest`` must).
    embedding: Mapped[Optional[list]] = mapped_column(_VECTOR, nullable=True)
    embedding_dim: Mapped[Optional[int]] = mapped_column(Integer)

    collection_row: Mapped[VectorCollection] = relationship(back_populates="chunks")

    __table_args__ = (
        # The two access paths: everything in a collection (the lexical index
        # build, the browse, the re-index) and everything of one document in
        # it (the scoped query, the deletion, the rollback).
        Index("ix_chunk_vectors_collection_doc", "collection", "doc_id"),
    )


#: Every table this application owns, ordered so a truncation or a delete may
#: walk it front to back without tripping a foreign key.
ALL_TABLES = (
    "gold_set_entries",
    "ingest_jobs",
    "documents",
    "content_variants",
    "content_documents",
    "contents",
    "chunk_vectors",
    "vector_collections",
    "knowledge_bases",
)
