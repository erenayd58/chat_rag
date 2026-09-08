"""Every SQL statement this application runs, in one place.

The layering this file exists to keep honest:

    application / domain
           |
      repository  (here: the conceptual persistence API, unchanged)
           |
      PostgreSQL

Nothing above this module imports SQLAlchemy. The use cases in
:mod:`application`, the routers in :mod:`interfaces.http` and the domain
models in :mod:`core` all talk to the same four record stores they always
did -- ``KnowledgeBaseManager``, ``DocumentTracker``, the Viewer's analysis
module and ``JobJournal`` -- and those are now thin façades over the
repositories below. That is the whole shape of the migration: the callers did
not move, the storage did.

Each repository takes a :class:`~sqlalchemy.orm.Session` and does not own it.
The transaction boundary belongs to the caller, which is what makes
"create the knowledge base **and** its first membership, or neither" a
statement one can actually write (see :func:`storage.engine.session_scope`).

Rows in, dictionaries out. The dictionaries are the shapes the product has
always passed around -- a knowledge-base config, a ledger record, an analysis
state -- so a screen, a schema or a test that reads one cannot tell which
store produced it. That is deliberate: it is what let the persistence be
replaced without touching ``/api/v1``.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Callable, Iterable, Optional, Sequence

from sqlalchemy import delete, func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from .models import (
    Content, ContentDocument, ContentVariant, Document, GoldSetEntry,
    IngestJobRecord, KnowledgeBase,
)


def new_id() -> str:
    """A row identity. Opaque, minted here, never derived from a path."""
    return uuid.uuid4().hex


def _iso(value: Optional[datetime]) -> Optional[str]:
    """A timestamp in the form every screen and schema in this product reads.

    Local time, no offset, seconds -- the shape ``datetime.now().isoformat``
    produced when these records were files, kept so a client cannot tell the
    difference.
    """
    if value is None:
        return None
    if value.tzinfo is not None:
        value = value.astimezone().replace(tzinfo=None)
    return value.isoformat(timespec="seconds")


# ==========================================================================
# knowledge bases
# ==========================================================================
def name_key(name: str) -> str:
    """Identity of a knowledge base's name: case and inner spacing are not
    part of it. The unique index is on this, so the rule the product has
    always applied in Python is now the database's."""
    return " ".join(str(name or "").split()).casefold()


class KnowledgeBaseRepository:
    """The named collections, their chunker and their store configuration."""

    def __init__(self, session: Session):
        self.session = session

    # ------------------------------------------------------------- reading
    @staticmethod
    def _record(row: KnowledgeBase) -> dict[str, Any]:
        """One knowledge base as the console has always seen it.

        ``attributes`` is spread back in first, so a mapped column always wins
        over a stale copy of itself.
        """
        return {
            **(row.attributes or {}),
            "name": row.name,
            "chunker": {"type": row.chunker_type, "params": dict(row.chunker_params or {})},
            "embedding_model_name": row.embedding_model_name,
            **({"embedding_provider": row.embedding_provider}
               if row.embedding_provider else {}),
            "vector_db_provider": row.vector_db_provider,
            "vector_db_path": row.vector_db_path,
            "retrieval_method": row.retrieval_method,
            "extra": dict(row.extra or {}),
        }

    def list_ids(self) -> list[str]:
        return list(self.session.scalars(
            select(KnowledgeBase.id).order_by(KnowledgeBase.created_at, KnowledgeBase.id)
        ))

    def all(self) -> dict[str, dict[str, Any]]:
        """Every knowledge base, keyed by id, in creation order.

        Creation order rather than id order: the console lists them as they
        were made, which is how the JSON store's insertion order behaved and
        what the pagination contract in ``/api/v1`` pages through.
        """
        rows = self.session.scalars(
            select(KnowledgeBase).order_by(KnowledgeBase.created_at, KnowledgeBase.id)
        )
        return {row.id: self._record(row) for row in rows}

    def get(self, kb_id: str, *, lock: bool = False) -> Optional[dict[str, Any]]:
        """One knowledge base.

        ``lock`` takes the row for the length of the caller's transaction. It
        is what a deletion needs: two requests deleting the same knowledge base
        would otherwise both read it, both decide they were the one to remove
        its vector store, and both try -- the second onto a directory the first
        has already taken.
        """
        row = self.session.get(KnowledgeBase, kb_id, with_for_update=lock)
        return self._record(row) if row else None

    def find_by_name(self, name: str) -> Optional[str]:
        key = name_key(name)
        if not key:
            return None
        return self.session.scalar(
            select(KnowledgeBase.id).where(KnowledgeBase.name_key == key)
        )

    # ------------------------------------------------------------- writing
    def create(self, kb_id: str, config: dict[str, Any]) -> dict[str, Any]:
        """Insert one. The unique index on the name decides a race, not us."""
        row = KnowledgeBase(id=kb_id, **self._columns(config))
        self.session.add(row)
        self.session.flush()
        return self._record(row)

    def update(self, kb_id: str, updates: dict[str, Any]) -> Optional[dict[str, Any]]:
        row = self.session.get(KnowledgeBase, kb_id, with_for_update=True)
        if row is None:
            return None
        merged = {**self._record(row), **updates}
        for field, value in self._columns(merged).items():
            setattr(row, field, value)
        self.session.flush()
        return self._record(row)

    def delete(self, kb_id: str) -> bool:
        result = self.session.execute(
            delete(KnowledgeBase).where(KnowledgeBase.id == kb_id)
        )
        self.session.flush()
        return bool(result.rowcount)

    # --------------------------------------------------------------- shape
    #: The record fields that have a column of their own. Anything else a
    #: caller sets is kept in ``attributes``: the store it replaced was a JSON
    #: document and accepted any key, and a record written by an older console
    #: must still read back whole.
    MAPPED = frozenset({
        "name", "chunker", "embedding_model_name", "embedding_provider",
        "vector_db_provider", "vector_db_path", "retrieval_method", "extra",
    })

    @classmethod
    def _columns(cls, config: dict[str, Any]) -> dict[str, Any]:
        chunker = config.get("chunker") or {}
        name = str(config.get("name") or "")
        return {
            "name": name,
            "name_key": name_key(name),
            "chunker_type": str(chunker.get("type") or ""),
            "chunker_params": dict(chunker.get("params") or {}),
            "embedding_model_name": config.get("embedding_model_name"),
            "embedding_provider": config.get("embedding_provider"),
            "vector_db_provider": config.get("vector_db_provider") or "chroma",
            "vector_db_path": config.get("vector_db_path"),
            "retrieval_method": config.get("retrieval_method") or "hybrid",
            "extra": dict(config.get("extra") or {}),
            "attributes": {k: v for k, v in config.items() if k not in cls.MAPPED},
        }


# ==========================================================================
# documents: the ingest ledger
# ==========================================================================
class DocumentRepository:
    """One ingest, one row. Addressed by ``doc_id``, never by a path."""

    def __init__(self, session: Session):
        self.session = session

    @staticmethod
    def _record(row: Document) -> dict[str, Any]:
        """A ledger record in the shape every caller already reads.

        ``file_path`` is still answered because the legacy console screens
        display it, but it is no longer anything's identity: it is the
        staging name the upload arrived under, and the row is addressed by
        ``doc_id``.
        """
        return {
            "file_path": row.source_path or "",
            "file_name": row.file_name,
            "doc_id": row.doc_id,
            "chunk_count": row.chunk_count,
            "file_size": row.file_size,
            "ingested_at": row.ingested_at,
            "file_hash": row.file_hash or "",
            "kb_id": row.kb_id,
            "status": row.status,
            "chunking_mode": row.chunking_mode,
            "metadata": dict(row.doc_metadata or {}),
            "pipeline_snapshot": row.pipeline_snapshot,
        }

    # ------------------------------------------------------------- reading
    def list(self, kb_id: Optional[str] = None) -> list[dict[str, Any]]:
        """Every document, newest ingest first -- the order every screen shows."""
        query = select(Document)
        if kb_id is not None:
            query = query.where(Document.kb_id == kb_id)
        rows = self.session.scalars(query.order_by(Document.ingested_at.desc(),
                                                   Document.doc_id))
        return [self._record(row) for row in rows]

    def get_by_doc_id(self, doc_id: str) -> Optional[dict[str, Any]]:
        row = self.session.scalar(select(Document).where(Document.doc_id == doc_id))
        return self._record(row) if row else None

    def get_by_source_path(self, path: str) -> Optional[dict[str, Any]]:
        row = self.session.scalar(
            select(Document).where(Document.source_path == path)
            .order_by(Document.ingested_at.desc())
        )
        return self._record(row) if row else None

    def get_by_ingest_job(self, job_id: str) -> Optional[dict[str, Any]]:
        """The document a job wrote, if it got that far.

        An indexed lookup, where the ledger's answer used to be a scan of
        every record ever written. It is asked once per in-flight job at
        every start-up, so the difference is the difference between settling
        a restart in milliseconds and reading the whole ledger per job.
        """
        row = self.session.scalar(
            select(Document).where(Document.ingest_job_id == job_id)
            .order_by(Document.ingested_at.desc())
        )
        return self._record(row) if row else None

    def statistics(self, kb_id: Optional[str] = None) -> dict[str, Any]:
        """Counts and totals, computed by the database rather than by loading
        every record into this process to add them up."""
        where = [] if kb_id is None else [Document.kb_id == kb_id]
        totals = self.session.execute(
            select(
                func.count(Document.id),
                func.coalesce(func.sum(Document.chunk_count), 0),
                func.coalesce(func.sum(Document.file_size), 0),
                func.min(Document.ingested_at),
                func.max(Document.ingested_at),
            ).where(*where)
        ).one()
        count, chunks, size, oldest, latest = totals
        if not count:
            return {"total_documents": 0, "total_chunks": 0, "total_size_bytes": 0}
        return {
            "total_documents": int(count),
            "total_chunks": int(chunks),
            "total_size_bytes": int(size),
            "oldest_ingestion": oldest,
            "latest_ingestion": latest,
        }

    # ------------------------------------------------------------- writing
    def upsert(self, record: dict[str, Any]) -> dict[str, Any]:
        """Record one ingest, replacing any earlier record of the same upload.

        ``doc_id`` is the conflict target, so two workers finishing the same
        document at once leave one row rather than two -- the invariant the
        old ledger kept with a process-local lock and could not keep across
        processes at all.
        """
        metadata = dict(record.get("metadata") or {})
        values = {
            "id": new_id(),
            "doc_id": record["doc_id"],
            "kb_id": record.get("kb_id"),
            "file_name": record.get("file_name") or "",
            "source_path": record.get("file_path"),
            "file_hash": record.get("file_hash") or "",
            "file_size": int(record.get("file_size") or 0),
            "chunk_count": int(record.get("chunk_count") or 0),
            "status": record.get("status") or "indexed",
            "chunking_mode": record.get("chunking_mode"),
            "ingested_at": record.get("ingested_at") or datetime.now().isoformat(),
            "metadata": metadata,
            "pipeline_snapshot": record.get("pipeline_snapshot"),
            "ingest_job_id": metadata.get("ingest_job_id"),
        }
        # Against the table rather than the mapped class: the column is named
        # ``metadata``, which on a declarative class is SQLAlchemy's own
        # ``MetaData``. Naming the table leaves no room for that collision.
        statement = pg_insert(Document.__table__).values(**values)
        updates = {k: statement.excluded[k] for k in values if k not in ("id", "doc_id")}
        self.session.execute(
            statement.on_conflict_do_update(index_elements=["doc_id"], set_=updates)
        )
        self.session.flush()
        return self.get_by_doc_id(record["doc_id"]) or {}

    def delete_by_doc_id(self, doc_id: str) -> bool:
        result = self.session.execute(delete(Document).where(Document.doc_id == doc_id))
        self.session.flush()
        return bool(result.rowcount)

    def delete_by_source_path(self, path: str) -> bool:
        result = self.session.execute(
            delete(Document).where(Document.source_path == path)
        )
        self.session.flush()
        return bool(result.rowcount)


# ==========================================================================
# contents: identity, membership, analysis state and variants
# ==========================================================================
class ContentRepository:
    """The bytes, the uploads that point at them and the variants built.

    This is the store the Viewer's packaging reads and writes. Everything
    expensive it produces -- the canonical units, the Deep run tree, each
    method's ``chunks.jsonl``, the assembled payload -- stays a file, because
    those are large regenerable artifacts of document processing. What lives
    here is the *state*: which uploads share this content, what each of them
    asked for, which variants exist and how each build ended.
    """

    def __init__(self, session: Session):
        self.session = session

    # --------------------------------------------------------------- shape
    #: Variant fields with a column of their own; everything else a build
    #: records about a variant is kept verbatim in ``details``.
    VARIANT_COLUMNS = ("status", "source", "chunk_count", "seconds", "error")

    @classmethod
    def _variant_record(cls, row: ContentVariant) -> dict[str, Any]:
        record: dict[str, Any] = {"status": row.status}
        for field in ("source", "chunk_count", "seconds", "error"):
            value = getattr(row, field)
            if value is not None:
                record[field] = value
        record.update(row.details or {})
        return record

    @classmethod
    def _state(cls, row: Content) -> dict[str, Any]:
        """One content's analysis state, in the shape the packager has always
        written and every reader above it destructures."""
        selections = {
            member.doc_id: list(member.selected_methods)
            for member in row.memberships
            if member.selected_methods is not None
        }
        state: dict[str, Any] = {
            "key": row.content_key,
            "status": row.status,
            "doc_ids": sorted(member.doc_id for member in row.memberships),
            "requested": list(row.requested_methods or []),
            "selections": selections,
            "methods": {v.method: cls._variant_record(v) for v in row.variants},
            "updated_at": _iso(row.updated_at),
        }
        # A column that was never written is *absent*, not present-and-null.
        # The record this replaced was a JSON document that only carried the
        # fields something had set, and the difference is load-bearing: an
        # analysis with no ``ready_methods`` yet means "no build has run", which
        # is not the same answer as "a build ran and produced none" (the Step 6
        # contract reads exactly that distinction).
        optional = {
            "label": row.label, "kb_id": row.kb_id, "kb_name": row.kb_name,
            "chunking_mode": row.chunking_mode, "content_sha": row.content_sha256,
            "unit_count": row.unit_count, "parse_seconds": row.parse_seconds,
            "payload_bytes": row.payload_bytes, "deep_source": row.deep_source,
            "error": row.error, "traceback": row.traceback,
            "ready_methods": (list(row.ready_methods)
                              if row.ready_methods is not None else None),
            "failed_methods": (list(row.failed_methods)
                               if row.failed_methods is not None else None),
        }
        state.update({k: v for k, v in optional.items() if v is not None})
        return state

    # ------------------------------------------------------------- reading
    def get(self, key: str) -> Optional[dict[str, Any]]:
        row = self._row(key)
        return self._state(row) if row else None

    def all(self) -> dict[str, dict[str, Any]]:
        rows = self.session.scalars(select(Content).order_by(Content.content_key))
        return {row.content_key: self._state(row) for row in rows}

    def key_for_document(self, doc_id: str) -> Optional[str]:
        """Which content this upload belongs to.

        An index lookup on a unique column, where the file store walked every
        analysis directory on disk to answer the same question.
        """
        return self.session.scalar(
            select(Content.content_key)
            .join(ContentDocument, ContentDocument.content_id == Content.id)
            .where(ContentDocument.doc_id == doc_id)
        )

    def keys_with_status(self, statuses: Sequence[str]) -> list[str]:
        return list(self.session.scalars(
            select(Content.content_key)
            .where(Content.status.in_(list(statuses)))
            .order_by(Content.content_key)
        ))

    def _row(self, key: str, *, lock: bool = False) -> Optional[Content]:
        query = select(Content).where(Content.content_key == key)
        if lock:
            query = query.with_for_update()
        return self.session.scalar(query)

    # ------------------------------------------------------------- writing
    def upsert_state(
        self,
        key: str,
        *,
        merge: Optional[Callable[[dict[str, Any]], dict[str, Any]]] = None,
        fields: Optional[dict[str, Any]] = None,
        content_sha: Optional[str] = None,
    ) -> dict[str, Any]:
        """Apply one change to a content's state, under a row lock.

        The lock is the point. Two uploads of the same PDF landing together
        each read the record, add themselves to it and write it back; without
        serialising, the later write drops the earlier upload and its choice.
        The file store did this with a process-local lock, which was correct
        for one process and silently wrong for two. ``SELECT ... FOR UPDATE``
        is correct for any number.

        ``merge`` is for a field whose new value is a function of the old one
        -- the set of memberships, the union of requested methods, one
        upload's own selection -- and is handed the state as this transaction
        sees it, not as the caller read it earlier.
        """
        row = self._row(key, lock=True)
        if row is None:
            row = self._create(key, content_sha)
        state = self._state(row)
        changes = dict(fields or {})
        if merge is not None:
            changes = {**changes, **merge(state)}
        self._apply(row, changes)
        self.session.flush()
        self.session.refresh(row)
        return self._state(row)

    def _create(self, key: str, content_sha: Optional[str]) -> Content:
        """Insert this content, or adopt the one a racing writer just made.

        ``ON CONFLICT DO NOTHING`` followed by a locking read: whichever
        transaction inserts, both end up holding the same row, which is the
        duplicate-content invariant the schema is here to make impossible
        rather than unlikely.
        """
        self.session.execute(
            pg_insert(Content.__table__)
            .values(id=new_id(), content_key=key, content_sha256=content_sha or None,
                    status="pending", requested_methods=[])
            .on_conflict_do_nothing(index_elements=["content_key"])
        )
        row = self._row(key, lock=True)
        if row is None:  # pragma: no cover - the insert above guarantees a row
            raise RuntimeError(f"could not create or read the content {key!r}")
        return row

    #: State keys that are columns on the content row.
    _COLUMNS = {
        "status": "status", "label": "label", "kb_id": "kb_id",
        "kb_name": "kb_name", "chunking_mode": "chunking_mode",
        "content_sha": "content_sha256", "unit_count": "unit_count",
        "parse_seconds": "parse_seconds", "payload_bytes": "payload_bytes",
        "deep_source": "deep_source", "error": "error", "traceback": "traceback",
        "requested": "requested_methods", "ready_methods": "ready_methods",
        "failed_methods": "failed_methods",
    }

    def _apply(self, row: Content, changes: dict[str, Any]) -> None:
        for name, value in changes.items():
            column = self._COLUMNS.get(name)
            if column is not None:
                if name == "requested":
                    value = list(value or [])
                elif name in ("ready_methods", "failed_methods") and value is not None:
                    value = list(value)
                setattr(row, column, value)
            elif name == "doc_ids":
                self._set_memberships(row, value)
            elif name == "selections":
                self._set_selections(row, value)
            elif name == "methods":
                self._set_variants(row, value or {})
        # A change of any kind is a change to the record, so the row's own
        # ``updated_at`` moves even when only a child table was touched.
        row.updated_at = datetime.now().astimezone()

    def _set_memberships(self, row: Content, doc_ids: Iterable[str]) -> None:
        wanted = set(doc_ids or ())
        have = {member.doc_id: member for member in row.memberships}
        for doc_id in wanted - set(have):
            # Idempotent: another transaction may have attached this upload
            # already, and attaching it twice is not an error.
            self.session.execute(
                pg_insert(ContentDocument.__table__)
                .values(content_id=row.id, doc_id=doc_id)
                .on_conflict_do_nothing(index_elements=["doc_id"])
            )
        for doc_id in set(have) - wanted:
            self.session.delete(have[doc_id])
        self.session.flush()
        self.session.expire(row, ["memberships"])

    def _set_selections(self, row: Content, selections: dict[str, Sequence[str]]) -> None:
        selections = selections or {}
        for member in row.memberships:
            if member.doc_id in selections:
                member.selected_methods = list(selections[member.doc_id])
            else:
                # Absent from the map is "this upload recorded no choice",
                # which is a null and not an empty list; the two behave
                # differently and the Step 6 contract reads the difference.
                member.selected_methods = None
        self.session.flush()

    def _set_variants(self, row: Content, methods: dict[str, dict[str, Any]]) -> None:
        have = {variant.method: variant for variant in row.variants}
        for method, record in methods.items():
            record = dict(record or {})
            details = {k: v for k, v in record.items() if k not in self.VARIANT_COLUMNS}
            variant = have.get(method)
            if variant is None:
                variant = ContentVariant(content_id=row.id, method=method,
                                         status=str(record.get("status") or "missing"))
                self.session.add(variant)
            variant.status = str(record.get("status") or "missing")
            variant.source = record.get("source")
            variant.chunk_count = record.get("chunk_count")
            variant.seconds = record.get("seconds")
            variant.error = record.get("error")
            variant.details = details
        for method, variant in have.items():
            if method not in methods:
                self.session.delete(variant)
        self.session.flush()
        self.session.expire(row, ["variants"])

    # ------------------------------------------------------------ deleting
    def detach_document(self, key: str, doc_id: str) -> Optional[int]:
        """Remove one upload from a content. Returns how many remain.

        ``None`` when there was no such content. The caller decides what an
        empty content means -- here it means the last upload of it has gone
        and the analysis goes with it -- because that reference-count rule is
        the product's, not the schema's.
        """
        row = self._row(key, lock=True)
        if row is None:
            return None
        for member in list(row.memberships):
            if member.doc_id == doc_id:
                self.session.delete(member)
        self.session.flush()
        self.session.expire(row, ["memberships"])
        return len(row.memberships)

    def delete(self, key: str) -> bool:
        """Remove a content and, by cascade, its memberships and variants."""
        row = self._row(key, lock=True)
        if row is None:
            return False
        self.session.delete(row)
        self.session.flush()
        return True


# ==========================================================================
# ingest jobs
# ==========================================================================
class IngestJobRepository:
    """The journal a restart answers a ``job_id`` from."""

    def __init__(self, session: Session):
        self.session = session

    def record(self, snapshot: dict[str, Any]) -> None:
        """Write (or replace) one job's record."""
        job_id = snapshot.get("job_id")
        if not job_id:
            return
        values = {
            "job_id": str(job_id),
            "kb_id": snapshot.get("kb_id"),
            "status": str(snapshot.get("status") or ""),
            "doc_id": snapshot.get("doc_id"),
            "filename": snapshot.get("filename"),
            "journalled_at": float(snapshot.get("journalled_at") or 0.0),
            "restart_recovered": bool(snapshot.get("restart_recovered")),
            "snapshot": snapshot,
        }
        statement = pg_insert(IngestJobRecord.__table__).values(**values)
        self.session.execute(statement.on_conflict_do_update(
            index_elements=["job_id"],
            set_={k: statement.excluded[k] for k in values if k != "job_id"},
        ))
        self.session.flush()

    def forget(self, job_id: str) -> None:
        self.session.execute(
            delete(IngestJobRecord).where(IngestJobRecord.job_id == job_id)
        )
        self.session.flush()

    def snapshots(self) -> list[dict[str, Any]]:
        rows = self.session.scalars(
            select(IngestJobRecord).order_by(IngestJobRecord.journalled_at,
                                             IngestJobRecord.job_id)
        )
        return [dict(row.snapshot or {}) for row in rows]

    def prune(self, older_than: float) -> int:
        """Delete every record journalled before ``older_than``. One statement,
        where the directory it replaces listed and stat'd every file."""
        result = self.session.execute(
            delete(IngestJobRecord)
            .where(IngestJobRecord.journalled_at > 0)
            .where(IngestJobRecord.journalled_at < older_than)
        )
        self.session.flush()
        return int(result.rowcount or 0)


# ==========================================================================
# the gold set
# ==========================================================================
class GoldSetRepository:
    """Confirmed answers: one per (knowledge base, question)."""

    #: The entry fields with a column of their own, in the order the record
    #: has always been written.
    FIELDS = (
        "entry_id", "schema_version", "question", "kb_id", "document_id",
        "document_title", "document_sha256", "correct_chunk_id", "section",
        "pages", "unit_ids", "evidence", "retrieval_method", "found_at_rank",
        "created_at", "updated_at",
    )

    def __init__(self, session: Session):
        self.session = session

    @staticmethod
    def _record(row: GoldSetEntry) -> dict[str, Any]:
        return {
            "entry_id": row.entry_id,
            "schema_version": row.schema_version,
            "question": row.question,
            "kb_id": row.kb_id,
            "document_id": row.document_id,
            "document_title": row.document_title,
            "document_sha256": row.document_sha256,
            "correct_chunk_id": row.correct_chunk_id,
            "section": row.section,
            "pages": list(row.pages or []),
            "unit_ids": list(row.unit_ids or []),
            "evidence": row.evidence,
            "retrieval_method": row.retrieval_method,
            "found_at_rank": row.found_at_rank,
            "created_at": row.created_at_text,
            "updated_at": row.updated_at_text,
        }

    def all(self) -> dict[str, dict[str, Any]]:
        rows = self.session.scalars(select(GoldSetEntry).order_by(GoldSetEntry.entry_id))
        return {row.entry_id: self._record(row) for row in rows}

    def get(self, entry_id: str) -> Optional[dict[str, Any]]:
        row = self.session.get(GoldSetEntry, entry_id)
        return self._record(row) if row else None

    def upsert(self, entry: dict[str, Any]) -> dict[str, Any]:
        values = {
            "entry_id": entry["entry_id"],
            "schema_version": int(entry.get("schema_version") or 1),
            "kb_id": entry["kb_id"],
            "question": entry["question"],
            "document_id": entry.get("document_id"),
            "document_title": entry.get("document_title"),
            "document_sha256": entry.get("document_sha256"),
            "correct_chunk_id": entry.get("correct_chunk_id"),
            "section": entry.get("section"),
            "pages": list(entry.get("pages") or []),
            "unit_ids": list(entry.get("unit_ids") or []),
            "evidence": entry.get("evidence") or "",
            "retrieval_method": entry.get("retrieval_method"),
            "found_at_rank": entry.get("found_at_rank"),
            "created_at": entry["created_at"],
            "updated_at": entry["updated_at"],
        }
        statement = pg_insert(GoldSetEntry.__table__).values(**values)
        self.session.execute(statement.on_conflict_do_update(
            index_elements=["entry_id"],
            # ``created_at`` is not in the update set: re-marking a question
            # replaces the answer, not the day it was first confirmed.
            set_={k: statement.excluded[k] for k in values
                  if k not in ("entry_id", "created_at")},
        ))
        self.session.flush()
        return self.get(entry["entry_id"]) or {}

    def delete(self, entry_id: str) -> bool:
        result = self.session.execute(
            delete(GoldSetEntry).where(GoldSetEntry.entry_id == entry_id)
        )
        self.session.flush()
        return bool(result.rowcount)
