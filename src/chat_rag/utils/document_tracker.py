"""The ingest ledger: what has been ingested, into which knowledge base.

Until Step 8 this was ``.ingested_documents.json``, a single object keyed by
the **absolute path of the uploaded file**. That key was the problem this
module now no longer has:

* the path was a staging file, deleted the moment the ingest job ended, so
  the ledger's primary key addressed something that did not exist;
* it was an absolute path on the machine that ran the upload, which made the
  ledger unmovable between a laptop and a container;
* and it was not the identity anything else used -- every route, both HTTP
  surfaces and the Viewer address a document by its ``doc_id``.

A document is now a row: ``documents.id`` is its database identity and
``documents.doc_id`` is the identity the product has always published. The
file name is kept because every screen shows it and the staging path is kept
as a diagnostic, but neither addresses anything.

The class kept its name and its methods, because roughly forty call sites and
the whole Step 6 contract suite are written against them. What it no longer
needs is the machinery that made a single JSON file safe under concurrent
writers: the re-read-inside-a-lock, the atomic rename, the quarantine of an
unparseable file, the refusal to write over a ledger that could not be read.
Those defended a whole-document rewrite against two processes; a row does not
need defending. Two overlapping uploads now write two rows, and the same
upload written twice is one row, because ``doc_id`` is unique and the write
is an upsert.
"""
import hashlib
import os
from datetime import datetime
from typing import Dict, List, Optional

from chat_rag.config import paths
from chat_rag.storage import DocumentRepository
from chat_rag.storage.engine import DatabaseBound


class DocumentTracker(DatabaseBound):
    """Tracks ingested documents to avoid re-processing"""

    def __init__(self, tracking_file: Optional[str] = None, *, database=None):
        """``tracking_file`` is accepted and unused.

        It named the JSON ledger until Step 8. The upload route and the
        workspace snapshot still build a tracker per request -- which is now
        free, because there is no file to read -- and the contract suites
        still point one at their own ``tmp_path``. Both keep working; the
        records come from PostgreSQL either way.
        """
        self._database = database
        self.tracking_file = tracking_file or paths.ingested_documents()

    # ------------------------------------------------------------- reading
    @property
    def ingested_docs(self) -> Dict[str, Dict]:
        """Every record, keyed by the upload's staging path (its ``doc_id``
        when it had none). Read from the database on each access.

        Kept because the shape is what a handful of callers destructure. It is
        no longer a cache: an instance that has been alive across another
        request's ingest sees that ingest, which is the lost update the file
        store spent a phase learning to avoid.
        """
        return {
            (record.get("file_path") or record["doc_id"]): record
            for record in self.get_all_documents()
        }

    def _compute_file_hash(self, file_path: str) -> str:
        """Compute hash of file content"""
        sha256_hash = hashlib.sha256()
        try:
            with open(file_path, "rb") as f:
                # Read in chunks to handle large files
                for byte_block in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(byte_block)
            return sha256_hash.hexdigest()
        except Exception as e:
            print(f"Warning: Could not compute hash for {file_path}: {e}")
            return ""

    def is_document_ingested(self, file_path: str) -> bool:
        """Has this exact file, at this path, already been ingested unchanged?"""
        abs_path = os.path.abspath(file_path)
        with self._session() as session:
            record = DocumentRepository(session).get_by_source_path(abs_path)
        if record is None or not os.path.exists(file_path):
            return False
        return self._compute_file_hash(file_path) == (record.get("file_hash") or "")

    def get_statistics(self, kb_id: Optional[str] = None) -> Dict:
        """Get statistics about ingested documents"""
        with self._session() as session:
            return DocumentRepository(session).statistics(kb_id)

    def get_all_documents(self, kb_id: Optional[str] = None) -> List[Dict]:
        """Get list of all ingested documents with their metadata"""
        with self._session() as session:
            return DocumentRepository(session).list(kb_id)

    def get_document_by_doc_id(self, doc_id: str) -> Optional[Dict]:
        """Get document information by doc_id"""
        with self._session() as session:
            return DocumentRepository(session).get_by_doc_id(doc_id)

    def get_document_by_ingest_job(self, job_id: str) -> Optional[Dict]:
        """The document a given ingest job wrote, if it got that far.

        An indexed lookup on a column projected out of the record's metadata.
        It used to be a scan of every document ever ingested, run once per
        in-flight job at every start-up.
        """
        with self._session() as session:
            return DocumentRepository(session).get_by_ingest_job(job_id)

    # ------------------------------------------------------------- writing
    def mark_as_ingested(
        self,
        file_path: str,
        doc_id: str,
        chunk_count: int,
        metadata: Optional[Dict] = None,
        kb_id: Optional[str] = None,
        pipeline_snapshot: Optional[Dict] = None,
        status: str = 'indexed',
        chunking_mode: Optional[str] = None
    ) -> bool:
        """Mark a document as ingested.

        True when the ledger now holds the record. A caller that has already
        written the document's vectors uses this to decide whether the
        document exists or has to be rolled back: a document the ledger does
        not know is not ingested, whatever the store holds.
        """
        abs_path = os.path.abspath(file_path)
        file_hash = self._compute_file_hash(file_path)

        if pipeline_snapshot is not None:
            # The snapshot has to name the bytes it describes, and the hash was
            # just computed: stamping it here reads no file a second time and
            # leaves no way for the two to disagree.
            pipeline_snapshot = {**pipeline_snapshot, 'document_sha256': file_hash}

        try:
            size = os.path.getsize(file_path)
        except OSError:
            size = 0

        record = {
            'doc_id': doc_id,
            'file_path': abs_path,
            'file_name': os.path.basename(abs_path),
            'file_hash': file_hash,
            'chunk_count': chunk_count,
            'file_size': size,
            'ingested_at': datetime.now().isoformat(),
            'kb_id': kb_id,
            'status': status,
            'chunking_mode': chunking_mode,
            'metadata': metadata or {},
            'pipeline_snapshot': pipeline_snapshot,
        }
        try:
            with self._session() as session:
                DocumentRepository(session).upsert(record)
        except Exception as e:  # noqa: BLE001 - the caller rolls back on False
            print(f"Warning: Could not record the ingested document: {e}")
            return False
        return True

    def remove_document(self, file_path: str) -> bool:
        """Remove a document from tracking by the path it was uploaded from.

        Kept for the callers that still hold a record and pass its
        ``file_path`` back. :meth:`remove_by_doc_id` is the one to use: the
        path is a leftover of the file ledger and names nothing that exists.
        """
        with self._session() as session:
            return DocumentRepository(session).delete_by_source_path(
                os.path.abspath(file_path)
            )

    def remove_by_doc_id(self, doc_id: str) -> bool:
        """Remove a document from tracking by its own identity."""
        with self._session() as session:
            return DocumentRepository(session).delete_by_doc_id(doc_id)
