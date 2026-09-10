"""The document store, on PostgreSQL with pgvector.

This is the store the product runs. It answers the same twelve calls the
Chroma implementation answered, returns the same records, and computes the
same cosine distance in the same direction -- ``0`` identical, ``2`` opposite,
lower is better -- because everything above it reads that number: the dense
leg turns it into ``1 - distance``, the console into ``1 - distance / 2``, and
the RRF fusion assumes the list arrives nearest-first.

What changed underneath is what a *collection* is. It was a directory: one per
knowledge base, addressed by ``vector_db_path``, with an
``embedding_index.json`` beside it saying which model had written the vectors
inside. It is now a row in ``vector_collections`` and a ``WHERE`` clause over
``chunk_vectors``, named by the knowledge base's id. The isolation is the same
isolation; it is enforced by a key rather than by a path, which is what lets
deleting a knowledge base take its vectors with it in one transaction instead
of removing a directory and hoping no handle is open on it.

No SQL is written here. Every statement lives in
:class:`storage.repositories.ChunkVectorRepository`, and this class is what
turns a ``DocumentChunk`` into the row it stores and back again -- which is
the whole of what a store is in this application:

    pipeline / retrievers  ->  BaseVectorDB  ->  ChunkVectorRepository  ->  SQL

Transactions are per call, and a call is a unit of work: ``add_chunks`` writes
one ingest's chunks in one transaction, ``replace_all`` empties a collection
and refills it in one, and neither can leave a corpus half searchable.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from chat_rag.core.exceptions import VectorDBException
from chat_rag.core.models import DocumentChunk
from chat_rag.storage import ChunkVectorRepository
from chat_rag.storage.engine import DatabaseBound
from chat_rag.utils.logger import get_logger

from .base import BaseVectorDB

#: How many chunk rows go into one INSERT. The whole write is still one
#: transaction; this only bounds the size of a single statement, which matters
#: because a 4096-dimension vector is around 60 kB of parameter text and a
#: ten-thousand-chunk document would otherwise be one statement of half a
#: gigabyte.
INSERT_BATCH = 500

#: The chunk fields that are also written into metadata, because that is where
#: every caller reads them back from: a store hands metadata to the retrievers
#: whole and they look ``doc_id`` up in it.
_RECORD_FIELDS = (
    "doc_id", "doc_title", "chunk_index", "total_chunks", "section_title",
    "document_summary",
)


class PgVectorStore(BaseVectorDB, DatabaseBound):
    """One collection of chunks and their embeddings, in PostgreSQL."""

    #: Width of the placeholder a lexical-only ingestion stores. The number is
    #: arbitrary -- nothing reads these vectors -- and deliberately small so a
    #: collection that holds them is obviously not a dense index; the manifest
    #: comparison recognises exactly this width as "no dense index"
    #: (``components/embedding/index_manifest.py``). A collection that already
    #: holds real vectors keeps its own width instead, so a lexical re-ingest
    #: into a dense corpus cannot narrow it.
    LEXICAL_PLACEHOLDER_DIMENSION = 1

    def __init__(self, collection: str, kb_id: Optional[str] = None, *, database=None):
        """Name the collection; touch nothing.

        No statement runs here. Building a pipeline must not need a reachable
        database -- ``python -m cli manifest`` and ``tools/import_smoke.py``
        both construct one on machines that have none -- so the collection row
        is created by the first write instead.
        """
        self._database = database
        self.collection = str(collection)
        self.kb_id = kb_id
        self.logger = get_logger("PgVectorStore")

    def __repr__(self) -> str:  # pragma: no cover - diagnostics
        return f"PgVectorStore(collection={self.collection!r})"

    # ------------------------------------------------------------- plumbing
    @staticmethod
    def _failure(action: str, error: Exception) -> VectorDBException:
        return VectorDBException(f"Failed to {action}: {error}")

    def _metadata(self, chunk: DocumentChunk) -> Dict[str, Any]:
        """The metadata one chunk is stored with.

        The record's own fields are written flat beside whatever the chunker
        put in ``metadata``, and read back out on the way in -- which is the
        round trip the document-store contract pins. ``None`` values are
        omitted rather than stored: a missing key reads the same as an older
        record that never had one, and inventing a placeholder would make
        "the parser did not report this" indistinguishable from a value.
        """
        metadata: Dict[str, Any] = {
            "doc_id": chunk.doc_id,
            "doc_title": chunk.doc_title,
            "chunk_index": chunk.chunk_index,
            "total_chunks": chunk.total_chunks,
            "section_title": chunk.section_title,
            "document_summary": chunk.document_summary,
            "word_count": (chunk.metadata or {}).get("word_count", 0),
        }
        for key, value in (chunk.metadata or {}).items():
            metadata.setdefault(key, value)
        return {key: value for key, value in metadata.items() if value is not None}

    def _row(self, chunk: DocumentChunk,
             embedding: Optional[Sequence[float]]) -> Dict[str, Any]:
        return {
            "chunk_id": chunk.chunk_id,
            "content": chunk.content,
            "metadata": self._metadata(chunk),
            "embedding": None if embedding is None else list(embedding),
        }

    @staticmethod
    def _as_chunk(row: Dict[str, Any]) -> DocumentChunk:
        """A stored row as the object the retrievers and the CLI read.

        Metadata goes back whole, which is what lets ``retrieval_text`` and
        ``table_view`` reach the two derived renderings the answer chain
        depends on.
        """
        metadata = row.get("metadata") or {}
        return DocumentChunk(
            chunk_id=row["chunk_id"],
            content=row.get("content") or "",
            doc_id=metadata.get("doc_id", ""),
            doc_title=metadata.get("doc_title", ""),
            chunk_index=metadata.get("chunk_index", 0),
            total_chunks=metadata.get("total_chunks", 0),
            section_title=metadata.get("section_title"),
            document_summary=metadata.get("document_summary"),
            metadata=dict(metadata),
        )

    def _placeholder_vectors(self, count: int, width: Optional[int]) -> List[List[float]]:
        """``count`` copies of one constant vector, derived from no content.

        A unit basis vector rather than zeros: cosine distance is undefined
        for a zero-length vector, and a store that answered ``NaN`` would sort
        unpredictably rather than obviously wrongly.
        """
        width = width or self.LEXICAL_PLACEHOLDER_DIMENSION
        vector = [1.0] + [0.0] * (width - 1)
        return [list(vector) for _ in range(count)]

    def _check_width(self, repository: ChunkVectorRepository,
                     embeddings: Sequence[Sequence[float]]) -> None:
        """Refuse a write that would put two embedding spaces in one collection.

        A collection holds one width, as the store this replaces did -- it
        fixed the width at its first record and rejected anything else. Here
        the column would accept both and the *query* would fail instead, on a
        row nobody could point at. Refusing the write says which two widths
        disagreed, while there is still something to do about it.
        """
        widths = {len(vector) for vector in embeddings if vector is not None}
        if len(widths) > 1:
            raise VectorDBException(
                f"one batch carries vectors of {sorted(widths)} dimensions; "
                "a collection holds one embedding space"
            )
        stored = repository.stored_dimension(self.collection)
        if widths and stored is not None and stored not in widths:
            raise VectorDBException(
                f"this collection holds {stored}-dimensional vectors and the "
                f"write carries {widths.pop()}-dimensional ones; re-index it "
                "with the current embedding model first"
            )

    # --------------------------------------------------------------- writing
    def add_chunks(self, chunks: Sequence[DocumentChunk],
                   embeddings: Sequence[Sequence[float]], **kwargs: Any) -> None:
        """Write chunks and their vectors. One ingest, one transaction.

        ``embeddings`` empty is the lexical-only profile: the text and its
        metadata are what that profile stores, no model computes anything from
        it, and a constant placeholder goes in the vector column so the width
        recorded for the collection says plainly that it holds no dense index.
        """
        chunks = list(chunks)
        if not chunks:
            return
        embeddings = list(embeddings or [])
        try:
            with self._session() as session:
                repository = ChunkVectorRepository(session)
                repository.ensure_collection(self.collection, self.kb_id)
                if embeddings:
                    if len(embeddings) != len(chunks):
                        raise VectorDBException(
                            "add_chunks needs one embedding per chunk or none at all"
                        )
                    self._check_width(repository, embeddings)
                else:
                    embeddings = self._placeholder_vectors(
                        len(chunks), repository.stored_dimension(self.collection)
                    )
                rows = [self._row(chunk, vector)
                        for chunk, vector in zip(chunks, embeddings)]
                for start in range(0, len(rows), INSERT_BATCH):
                    repository.upsert(self.collection, rows[start:start + INSERT_BATCH])
        except VectorDBException:
            raise
        except Exception as error:
            raise self._failure("add chunks", error) from error

    def add_single_chunk(self, chunk: DocumentChunk,
                         embedding: Optional[Sequence[float]] = None) -> None:
        """One chunk, by the same path as a batch of them."""
        self.add_chunks([chunk], [embedding] if embedding is not None else [])

    def update_chunk(self, chunk_id: str, content: Optional[str] = None,
                     metadata: Optional[Dict[str, Any]] = None,
                     embedding: Optional[Sequence[float]] = None) -> None:
        """Change a chunk in place. Absent arguments leave their field alone."""
        try:
            with self._session() as session:
                ChunkVectorRepository(session).update(
                    self.collection, chunk_id, content=content,
                    metadata=metadata, embedding=embedding,
                )
        except Exception as error:
            raise self._failure(f"update chunk {chunk_id}", error) from error

    def replace_all(self, chunks: Sequence[DocumentChunk],
                    embeddings: Sequence[Sequence[float]],
                    batch_size: int = INSERT_BATCH) -> None:
        """Rewrite the whole collection with new vectors -- a re-index.

        Emptying and refilling rather than updating row by row, because the
        width may change: this is the call a knowledge base makes when its
        embedding model has, and half a collection in each space is exactly
        the state the manifest exists to prevent. One transaction, so a
        failure leaves the corpus as it was rather than empty.
        """
        chunks = list(chunks)
        embeddings = list(embeddings)
        if len(chunks) != len(embeddings):
            raise VectorDBException("replace_all needs one embedding per chunk")
        try:
            with self._session() as session:
                repository = ChunkVectorRepository(session)
                repository.ensure_collection(self.collection, self.kb_id)
                repository.clear(self.collection)
                rows = [self._row(chunk, vector)
                        for chunk, vector in zip(chunks, embeddings)]
                for start in range(0, len(rows), max(1, int(batch_size))):
                    repository.upsert(self.collection, rows[start:start + batch_size])
        except VectorDBException:
            raise
        except Exception as error:
            raise self._failure("re-index collection", error) from error

    def delete_chunk(self, chunk_id: str) -> None:
        try:
            with self._session() as session:
                ChunkVectorRepository(session).delete_chunk(self.collection, chunk_id)
        except Exception as error:
            raise self._failure(f"delete chunk {chunk_id}", error) from error

    def delete_by_doc_id(self, doc_id: str) -> None:
        """Every chunk of one document, and nothing else.

        Document deletion and ingest rollback both rely on this. A document
        that is not here is not an error -- a rollback runs after a failure
        that may have written nothing.
        """
        try:
            with self._session() as session:
                ChunkVectorRepository(session).delete_by_doc_id(self.collection, doc_id)
        except Exception as error:
            raise self._failure(f"delete document {doc_id}", error) from error

    def clear(self) -> int:
        """Empty the collection, keeping it. Used by the Chroma import tool."""
        try:
            with self._session() as session:
                return ChunkVectorRepository(session).clear(self.collection)
        except Exception as error:
            raise self._failure("clear the collection", error) from error

    # --------------------------------------------------------------- reading
    def query(self, query_embedding: Sequence[float], top_k: int = 10,
              filter_dict: Optional[Dict[str, Any]] = None,
              **kwargs: Any) -> List[Dict[str, Any]]:
        """The nearest ``top_k`` chunks, nearest first.

        Exact cosine distance, not an approximation: this schema carries no
        ANN index, because the embedding column has no fixed width and
        pgvector can only index one that has (``docs/database.md``). For a
        corpus of this size that is a scan of some thousands of rows, and it
        has one property an approximate index does not -- the ranking is the
        true ranking, so nothing downstream had to be re-tuned for recall.
        """
        try:
            with self._session() as session:
                return ChunkVectorRepository(session).search(
                    self.collection, query_embedding, top_k=top_k,
                    filter_dict=filter_dict,
                )
        except Exception as error:
            raise self._failure("query", error) from error

    def get_all_chunks(self) -> List[DocumentChunk]:
        try:
            with self._session() as session:
                rows = ChunkVectorRepository(session).rows(self.collection)
        except Exception as error:
            raise self._failure("retrieve chunks", error) from error
        chunks = [self._as_chunk(row) for row in rows]
        self.logger.info(f"Retrieved {len(chunks)} chunks from {self.collection}")
        return chunks

    def get_chunk_by_id(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        try:
            with self._session() as session:
                return ChunkVectorRepository(session).get(self.collection, chunk_id)
        except Exception as error:
            raise self._failure(f"get chunk {chunk_id}", error) from error

    def get_chunks_paginated(self, offset: int = 0, limit: int = 20,
                             filter_dict: Optional[Dict[str, Any]] = None
                             ) -> Dict[str, Any]:
        """One page of the corpus, plus the total the pager needs.

        Counted by the database rather than by loading every row to measure
        it, which is what the store this replaces had to do for a filtered
        page.
        """
        try:
            with self._session() as session:
                repository = ChunkVectorRepository(session)
                return {
                    "chunks": repository.rows(self.collection, filter_dict,
                                              offset=offset, limit=limit),
                    "total": repository.count(self.collection, filter_dict),
                    "offset": offset,
                    "limit": limit,
                }
        except Exception as error:
            raise self._failure("get paginated chunks", error) from error

    def search_chunks_by_text(self, search_text: str, offset: int = 0,
                              limit: int = 20) -> Dict[str, Any]:
        """The browse screen's phrase filter: a substring scan, not retrieval."""
        try:
            with self._session() as session:
                return ChunkVectorRepository(session).search_text(
                    self.collection, search_text, offset=offset, limit=limit
                )
        except Exception as error:
            raise self._failure("search chunks", error) from error

    def count(self) -> int:
        try:
            with self._session() as session:
                return ChunkVectorRepository(session).count(self.collection)
        except Exception as error:
            raise self._failure("count chunks", error) from error

    def get_name(self) -> str:
        return "PostgreSQL/pgvector"

    # -------------------------------------------------------------- manifest
    def _stored_dimension(self) -> Optional[int]:
        """The width of the vectors this collection holds, if it holds any.

        Read by the retriever to tell a dense corpus from one carrying a
        lexical profile's placeholders. Named as the Chroma store named it,
        because the retriever probes for it by name.
        """
        try:
            with self._session() as session:
                return ChunkVectorRepository(session).stored_dimension(self.collection)
        except Exception:
            # A status probe must not raise: it is asked while answering
            # ``/api/health`` and while deciding whether a query may use the
            # dense leg, and "unknown" is a usable answer to both.
            return None

    def read_manifest(self) -> Optional[Dict[str, Any]]:
        """Which embedding wrote these vectors, or ``None`` if unrecorded."""
        try:
            with self._session() as session:
                return ChunkVectorRepository(session).read_manifest(self.collection)
        except Exception:
            return None

    def write_manifest(self, manifest: Dict[str, Any]) -> Dict[str, Any]:
        """Record the embedding space this collection now holds."""
        try:
            with self._session() as session:
                return ChunkVectorRepository(session).write_manifest(
                    self.collection, manifest, self.kb_id
                )
        except Exception as error:
            raise self._failure("write the embedding manifest", error) from error
