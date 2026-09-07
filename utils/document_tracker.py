"""
Document tracking system to avoid re-ingesting documents
"""
import json
import os
import hashlib
import threading
from typing import Dict, Optional, List
from datetime import datetime

from config import paths


#: One lock per ledger file, shared by every tracker pointed at it.
#:
#: The upload route builds a fresh ``DocumentTracker`` per request and the
#: workspace snapshot builds one per page refresh, so several trackers for one
#: file are live in a single process at a time. Reads take this lock as well as
#: writes: a read outside it can land on a rename in progress -- on Windows the
#: open then fails outright -- and an unreadable ledger loads as an empty one,
#: which the next write would persist over every record that was really there.
_FILE_LOCKS: Dict[str, threading.RLock] = {}
_FILE_LOCKS_GUARD = threading.Lock()


def _lock_for(tracking_file: str) -> threading.RLock:
    """The lock guarding one ledger file, created on first use."""
    key = os.path.normcase(os.path.abspath(tracking_file))
    with _FILE_LOCKS_GUARD:
        lock = _FILE_LOCKS.get(key)
        if lock is None:
            lock = _FILE_LOCKS[key] = threading.RLock()
        return lock


class DocumentTracker:
    """Tracks ingested documents to avoid re-processing"""
    
    def __init__(self, tracking_file: Optional[str] = None):
        """Initialize document tracker"""
        # Defaults to the historical file unless a data directory is set.
        self.tracking_file = tracking_file or paths.ingested_documents()
        self.ingested_docs: Dict[str, Dict] = {}
        self._load_tracking_data()
    
    def _read_records(self) -> Dict[str, Dict]:
        """The ledger as it is on disk right now.

        Raises rather than guessing. What an unreadable ledger means is the
        caller's decision, and for a writer it must never mean "empty".
        """
        if not os.path.exists(self.tracking_file):
            return {}
        with open(self.tracking_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError("the ingest ledger is not a JSON object")
        return data

    def _load_tracking_data(self):
        """Load tracking data from file"""
        with _lock_for(self.tracking_file):
            try:
                self.ingested_docs = self._read_records()
            except Exception as e:
                # Constructing a tracker must not raise: the console builds one
                # per request. The file is left exactly as it is -- only a write
                # sets it aside, and only once the content is certainly at fault.
                print(f"Warning: Could not load tracking file: {e}")
                self.ingested_docs = {}

    def _write_records(self, records: Dict[str, Dict]) -> None:
        """Replace the ledger in one step.

        Written to a scratch name unique to this writer and moved into place,
        so a reader sees either the whole previous ledger or the whole new one,
        and an interrupted process leaves the previous file untouched instead
        of a half-written one.
        """
        # Historically this file sat in the working directory, which always
        # exists. A configured data directory does not, until something makes it.
        parent = os.path.dirname(os.path.abspath(self.tracking_file))
        if parent:
            os.makedirs(parent, exist_ok=True)
        tmp = f"{self.tracking_file}.{os.getpid()}.{threading.get_ident()}.tmp"
        try:
            with open(tmp, 'w', encoding='utf-8') as f:
                json.dump(records, f, indent=2)
                f.flush()
                # The rename is atomic on its own; this is what stops a power
                # failure from surviving the rename but not the contents.
                os.fsync(f.fileno())
            os.replace(tmp, self.tracking_file)
        finally:
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except OSError:
                    pass

    def _set_aside_unreadable(self, error: Exception) -> Dict[str, Dict]:
        """Keep a ledger whose contents cannot be parsed, and start a new one.

        The previous behaviour was to load nothing and let the next write
        replace the unreadable file, which destroyed whatever a person might
        still have recovered from it. Renaming costs nothing and keeps it.
        """
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        spoiled = f"{self.tracking_file}.corrupt-{stamp}"
        try:
            os.replace(self.tracking_file, spoiled)
            print(f"Warning: the ingest ledger could not be parsed ({error}); "
                  f"it has been kept as {spoiled} and a new one started")
        except OSError as move_error:
            print(f"Warning: the ingest ledger could not be parsed ({error}) "
                  f"and could not be set aside ({move_error})")
        return {}

    def _mutate(self, change) -> bool:
        """Apply one change to the ledger on disk, under that file's lock.

        The records are re-read inside the lock rather than taken from this
        instance's memory. Two uploads that each built a tracker before either
        saved would otherwise write their own snapshot in turn, and the second
        would drop the first document -- the lost update this exists to stop.

        ``change`` may return ``False`` to say it changed nothing, which leaves
        the file alone.
        """
        with _lock_for(self.tracking_file):
            try:
                records = self._read_records()
            except OSError as e:
                # The ledger is there and this process could not read it.
                # Writing now would replace records that still exist, so the
                # change is refused rather than applied over them.
                print(f"Warning: Could not read tracking file, leaving it alone: {e}")
                return False
            except Exception as e:
                records = self._set_aside_unreadable(e)
            applied = change(records)
            if applied is not False:
                try:
                    self._write_records(records)
                except Exception as e:
                    print(f"Warning: Could not save tracking file: {e}")
                    applied = False
            # This instance now reflects the ledger, not only its own change.
            self.ingested_docs = records
            return applied is not False

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
        """Check if document has already been ingested"""
        abs_path = os.path.abspath(file_path)
        
        if abs_path not in self.ingested_docs:
            return False
        
        # Check if file still exists
        if not os.path.exists(file_path):
            return False
        
        # Check if file has been modified (compare hash)
        current_hash = self._compute_file_hash(file_path)
        stored_hash = self.ingested_docs[abs_path].get('file_hash', '')
        
        return current_hash == stored_hash
    
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
        """Mark a document as ingested"""
        abs_path = os.path.abspath(file_path)
        file_hash = self._compute_file_hash(file_path)

        if pipeline_snapshot is not None:
            # The snapshot has to name the bytes it describes, and the hash was
            # just computed: stamping it here reads no file a second time and
            # leaves no way for the two to disagree.
            pipeline_snapshot = {**pipeline_snapshot, 'document_sha256': file_hash}

        record = {
            'doc_id': doc_id,
            'file_hash': file_hash,
            'chunk_count': chunk_count,
            'file_size': os.path.getsize(file_path),
            'ingested_at': datetime.now().isoformat(),
            'kb_id': kb_id,  # Store KB ID with document
            'status': status,
            'chunking_mode': chunking_mode,
            'metadata': metadata or {},
            'pipeline_snapshot': pipeline_snapshot
        }

        def add(records: Dict[str, Dict]) -> None:
            records[abs_path] = record

        # True when the ledger on disk now holds the record. A caller that has
        # already written the document's vectors uses this to decide whether
        # the document exists or has to be rolled back: a document the ledger
        # does not know is not ingested, whatever the store holds.
        return self._mutate(add)
    
    def remove_document(self, file_path: str):
        """Remove a document from tracking"""
        abs_path = os.path.abspath(file_path)

        def drop(records: Dict[str, Dict]) -> bool:
            # False leaves the ledger alone when there was nothing to remove.
            return records.pop(abs_path, None) is not None

        self._mutate(drop)
    
    def get_statistics(self, kb_id: Optional[str] = None) -> Dict:
        """Get statistics about ingested documents"""
        # Filter by KB if kb_id is provided
        filtered_docs = self.ingested_docs
        if kb_id is not None:
            filtered_docs = {
                path: doc for path, doc in self.ingested_docs.items()
                if doc.get('kb_id') == kb_id
            }
        
        if not filtered_docs:
            return {
                'total_documents': 0,
                'total_chunks': 0,
                'total_size_bytes': 0
            }
        
        return {
            'total_documents': len(filtered_docs),
            'total_chunks': sum(doc['chunk_count'] for doc in filtered_docs.values()),
            'total_size_bytes': sum(doc['file_size'] for doc in filtered_docs.values()),
            'oldest_ingestion': min(doc['ingested_at'] for doc in filtered_docs.values()),
            'latest_ingestion': max(doc['ingested_at'] for doc in filtered_docs.values())
        }
    
    def get_all_documents(self, kb_id: Optional[str] = None) -> List[Dict]:
        """Get list of all ingested documents with their metadata"""
        documents = []
        for file_path, doc_data in self.ingested_docs.items():
            doc_kb_id = doc_data.get('kb_id')
            
            # Filter by KB if kb_id is provided
            if kb_id is not None and doc_kb_id != kb_id:
                continue
                
            documents.append({
                'file_path': file_path,
                'file_name': os.path.basename(file_path),
                'doc_id': doc_data.get('doc_id', ''),
                'chunk_count': doc_data.get('chunk_count', 0),
                'file_size': doc_data.get('file_size', 0),
                'ingested_at': doc_data.get('ingested_at', ''),
                'file_hash': doc_data.get('file_hash', ''),
                'kb_id': doc_kb_id,
                # Records written before these fields existed count as
                # indexed with an unknown chunking mode.
                'status': doc_data.get('status', 'indexed'),
                'chunking_mode': doc_data.get('chunking_mode'),
                'metadata': doc_data.get('metadata', {}),
                'pipeline_snapshot': doc_data.get('pipeline_snapshot')
            })

        # Sort by ingestion date (newest first)
        documents.sort(key=lambda x: x['ingested_at'], reverse=True)
        return documents

    def get_document_by_doc_id(self, doc_id: str) -> Optional[Dict]:
        """Get document information by doc_id"""
        for file_path, doc_data in self.ingested_docs.items():
            if doc_data.get('doc_id') == doc_id:
                return {
                    'file_path': file_path,
                    'file_name': os.path.basename(file_path),
                    'doc_id': doc_data.get('doc_id', ''),
                    'chunk_count': doc_data.get('chunk_count', 0),
                    'file_size': doc_data.get('file_size', 0),
                    'ingested_at': doc_data.get('ingested_at', ''),
                    'file_hash': doc_data.get('file_hash', ''),
                    'status': doc_data.get('status', 'indexed'),
                    'chunking_mode': doc_data.get('chunking_mode'),
                    'metadata': doc_data.get('metadata', {}),
                    'pipeline_snapshot': doc_data.get('pipeline_snapshot')
                }
        return None

