# /Users/murseltasgin/projects/chat_rag/utils/document_tracker.py
"""
Document tracking system to avoid re-ingesting documents
"""
import json
import os
import hashlib
from typing import Dict, Set, Optional, List
from datetime import datetime

from config import paths


class DocumentTracker:
    """Tracks ingested documents to avoid re-processing"""
    
    def __init__(self, tracking_file: Optional[str] = None):
        """
        Initialize document tracker
        
        Args:
            tracking_file: Path to the tracking file
        """
        # Defaults to the historical file unless a data directory is set.
        self.tracking_file = tracking_file or paths.ingested_documents()
        self.ingested_docs: Dict[str, Dict] = {}
        self._load_tracking_data()
    
    def _load_tracking_data(self):
        """Load tracking data from file"""
        if os.path.exists(self.tracking_file):
            try:
                with open(self.tracking_file, 'r') as f:
                    self.ingested_docs = json.load(f)
            except Exception as e:
                print(f"Warning: Could not load tracking file: {e}")
                self.ingested_docs = {}
        else:
            self.ingested_docs = {}
    
    def _save_tracking_data(self):
        """Save tracking data to file"""
        try:
            # Historically this file sat in the working directory, which always
            # exists. A configured data directory does not, until something
            # makes it.
            parent = os.path.dirname(os.path.abspath(self.tracking_file))
            if parent:
                os.makedirs(parent, exist_ok=True)
            with open(self.tracking_file, 'w') as f:
                json.dump(self.ingested_docs, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save tracking file: {e}")
    
    def _compute_file_hash(self, file_path: str) -> str:
        """
        Compute hash of file content
        
        Args:
            file_path: Path to file
        
        Returns:
            SHA256 hash of file content
        """
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
        """
        Check if document has already been ingested
        
        Args:
            file_path: Path to the document
        
        Returns:
            True if document is already ingested and unchanged
        """
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
    ):
        """
        Mark a document as ingested

        Args:
            file_path: Path to the document
            doc_id: Document ID
            chunk_count: Number of chunks created
            metadata: Additional metadata
            kb_id: Knowledge base ID this document belongs to
            pipeline_snapshot: Immutable record of the configuration that
                produced this corpus (see components.provenance). It is written
                in the same call that records the document, so an ingest that
                failed -- and therefore never reached this method -- cannot
                leave a snapshot behind.
            status: Lifecycle state of the record. Ingestion is synchronous
                today, so a record only ever exists as 'indexed';
                'processing' and 'failed' are reserved for asynchronous
                ingestion.
            chunking_mode: The per-document ingest choice ('standard' or
                'deep_analysis'). None on records written by callers that
                predate the field.
        """
        abs_path = os.path.abspath(file_path)
        file_hash = self._compute_file_hash(file_path)

        if pipeline_snapshot is not None:
            # The snapshot has to name the bytes it describes, and the hash was
            # just computed: stamping it here reads no file a second time and
            # leaves no way for the two to disagree.
            pipeline_snapshot = {**pipeline_snapshot, 'document_sha256': file_hash}

        self.ingested_docs[abs_path] = {
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

        self._save_tracking_data()
    
    def get_ingested_files(self) -> Set[str]:
        """
        Get set of all ingested file paths
        
        Returns:
            Set of absolute file paths
        """
        return set(self.ingested_docs.keys())
    
    def get_file_info(self, file_path: str) -> Optional[Dict]:
        """
        Get information about an ingested file
        
        Args:
            file_path: Path to the file
        
        Returns:
            Dictionary with file information or None
        """
        abs_path = os.path.abspath(file_path)
        return self.ingested_docs.get(abs_path)
    
    def remove_document(self, file_path: str):
        """
        Remove a document from tracking
        
        Args:
            file_path: Path to the document
        """
        abs_path = os.path.abspath(file_path)
        if abs_path in self.ingested_docs:
            del self.ingested_docs[abs_path]
            self._save_tracking_data()
    
    def get_statistics(self, kb_id: Optional[str] = None) -> Dict:
        """
        Get statistics about ingested documents
        
        Args:
            kb_id: Optional knowledge base ID to filter statistics
        
        Returns:
            Dictionary with statistics
        """
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
    
    def clear_all(self):
        """Clear all tracking data"""
        self.ingested_docs = {}
        self._save_tracking_data()

    def get_all_documents(self, kb_id: Optional[str] = None) -> List[Dict]:
        """
        Get list of all ingested documents with their metadata
        
        Args:
            kb_id: Optional knowledge base ID to filter documents

        Returns:
            List of document dictionaries
        """
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
        """
        Get document information by doc_id

        Args:
            doc_id: Document ID to search for

        Returns:
            Document dictionary or None if not found
        """
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

