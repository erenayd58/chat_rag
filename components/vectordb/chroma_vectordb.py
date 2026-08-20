# /Users/murseltasgin/projects/chat_rag/components/vectordb/chroma_vectordb.py
"""
ChromaDB vector database implementation
"""
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings

from utils.logger import RAGLogger
from .base import BaseVectorDB
from core.models import DocumentChunk
from core.exceptions import VectorDBException
from utils.logger import get_logger

class ChromaVectorDB(BaseVectorDB):
    """ChromaDB vector database implementation"""
    
    def __init__(
        self,
        path: str = "./chroma_db",
        collection_name: str = "documents",
        hnsw_m: int = 64,
        hnsw_ef_construction: int = 200,
        hnsw_ef_search: int = 100,
        space: str = "cosine"
    ):
        """
        Initialize ChromaDB client
        
        Args:
            path: Path to store the database
            collection_name: Name of the collection
        """
        self.path = path
        self.collection_name = collection_name
        self.hnsw_m = hnsw_m
        self.hnsw_ef_construction = hnsw_ef_construction
        self.hnsw_ef_search = hnsw_ef_search
        self.space = space
        self.logger = get_logger("ChromaVectorDB")
        
        try:
            self.client = chromadb.PersistentClient(
                path=path,
                settings=Settings(anonymized_telemetry=False)
            )
            # Configure collection; many Chroma versions only accept hnsw:space here
            # Index params like M/ef are server-level, not per-collection, in most releases
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata={
                    "hnsw:space": self.space
                }
            )
        except Exception as e:
            raise VectorDBException(f"Failed to initialize ChromaDB: {e}")
    
    def add_chunks(
        self,
        chunks: List[DocumentChunk],
        embeddings: List[List[float]],
        **kwargs
    ) -> None:
        """Add chunks with embeddings to the database"""
        try:
            chunk_ids = []
            documents = []
            metadatas = []
            
            for chunk in chunks:
                chunk_ids.append(chunk.chunk_id)
                documents.append(chunk.content)
                
                metadata = {
                    'doc_id': chunk.doc_id,
                    'doc_title': chunk.doc_title,
                    'chunk_index': chunk.chunk_index,
                    'total_chunks': chunk.total_chunks,
                    'section_title': chunk.section_title,
                    'document_summary': chunk.document_summary,
                    'word_count': chunk.metadata.get('word_count', 0) if chunk.metadata else 0
                }
                # Add any additional metadata
                if chunk.metadata:
                    for key, value in chunk.metadata.items():
                        if key not in metadata:
                            metadata[key] = value
                # Chroma metadata values cannot be None. Preserve known values
                # and omit unavailable parser/boundary metadata rather than
                # inventing placeholders.
                metadata = {
                    key: value for key, value in metadata.items() if value is not None
                }
                
                metadatas.append(metadata)
            
            self.collection.add(
                ids=chunk_ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas
            )
        except Exception as e:
            raise VectorDBException(f"Failed to add chunks: {e}")
    
    def query(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        filter_dict: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Query the vector database"""
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=filter_dict,
                **kwargs
            )
            
            # Format results
            formatted_results = []
            if results['ids'] and results['ids'][0]:
                for i in range(len(results['ids'][0])):
                    formatted_results.append({
                        'chunk_id': results['ids'][0][i],
                        'content': results['documents'][0][i],
                        'distance': results['distances'][0][i],
                        'metadata': results['metadatas'][0][i]
                    })
            
            return formatted_results
        except Exception as e:
            raise VectorDBException(f"Query failed: {e}")
    
    def get_all_chunks(self) -> List[DocumentChunk]:
        """Retrieve all chunks from the database"""
        try:
            results = self.collection.get()
            chunks = []
            
            if results['ids']:
                for i, chunk_id in enumerate(results['ids']):
                    metadata = results['metadatas'][i]
                    chunk = DocumentChunk(
                        chunk_id=chunk_id,
                        content=results['documents'][i],
                        doc_id=metadata.get('doc_id', ''),
                        doc_title=metadata.get('doc_title', ''),
                        chunk_index=metadata.get('chunk_index', 0),
                        total_chunks=metadata.get('total_chunks', 0),
                        section_title=metadata.get('section_title'),
                        document_summary=metadata.get('document_summary'),
                        metadata=metadata
                    )
                    chunks.append(chunk)

            self.logger.info(f"Retrieved {len(chunks)} chunks from ChromaDB")
            
            return chunks
        except Exception as e:
            raise VectorDBException(f"Failed to retrieve chunks: {e}")
    
    def delete_by_doc_id(self, doc_id: str) -> None:
        """Delete all chunks for a document"""
        try:
            results = self.collection.get(where={"doc_id": doc_id})
            if results['ids']:
                self.collection.delete(ids=results['ids'])
        except Exception as e:
            raise VectorDBException(f"Failed to delete document: {e}")
    
    def get_name(self) -> str:
        """Get the vector database name"""
        return "ChromaDB"
    
    def count(self) -> int:
        """Get the total number of chunks in the database"""
        try:
            return self.collection.count()
        except Exception as e:
            raise VectorDBException(f"Failed to count chunks: {e}")

    def get_chunk_by_id(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific chunk by ID with its embedding"""
        try:
            results = self.collection.get(
                ids=[chunk_id],
                include=['documents', 'metadatas', 'embeddings']
            )

            if results['ids'] and len(results['ids']) > 0:
                # Check if embeddings exist and is not empty
                has_embeddings = (
                    results.get('embeddings') is not None and 
                    len(results.get('embeddings', [])) > 0 and
                    results['embeddings'][0] is not None
                )
                
                return {
                    'chunk_id': results['ids'][0],
                    'content': results['documents'][0],
                    'metadata': results['metadatas'][0],
                    'embedding': results['embeddings'][0] if has_embeddings else None
                }
            return None
        except Exception as e:
            raise VectorDBException(f"Failed to get chunk: {e}")

    def update_chunk(
        self,
        chunk_id: str,
        content: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        embedding: Optional[List[float]] = None
    ) -> None:
        """Update a chunk's content, metadata, or embedding"""
        try:
            # Get existing chunk
            existing = self.get_chunk_by_id(chunk_id)
            if not existing:
                raise VectorDBException(f"Chunk {chunk_id} not found")

            # Delete old chunk
            self.collection.delete(ids=[chunk_id])

            # Add updated chunk
            self.collection.add(
                ids=[chunk_id],
                documents=[content if content is not None else existing['content']],
                metadatas=[metadata if metadata is not None else existing['metadata']],
                embeddings=[embedding if embedding is not None else existing['embedding']]
            )
        except Exception as e:
            raise VectorDBException(f"Failed to update chunk: {e}")

    def delete_chunk(self, chunk_id: str) -> None:
        """Delete a specific chunk by ID"""
        try:
            self.collection.delete(ids=[chunk_id])
        except Exception as e:
            raise VectorDBException(f"Failed to delete chunk: {e}")

    def get_chunks_paginated(
        self,
        offset: int = 0,
        limit: int = 20,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Get paginated chunks with optional filtering"""
        try:
            # Get total count
            if filter_dict:
                # For filtered queries, we need to get all and count
                results = self.collection.get(where=filter_dict, include=['documents', 'metadatas'])
                total = len(results['ids']) if results['ids'] else 0

                # Apply pagination manually
                start = offset
                end = min(offset + limit, total)

                paginated_results = {
                    'ids': results['ids'][start:end] if results['ids'] else [],
                    'documents': results['documents'][start:end] if results['documents'] else [],
                    'metadatas': results['metadatas'][start:end] if results['metadatas'] else []
                }
            else:
                total = self.collection.count()
                # Get all and paginate
                results = self.collection.get(include=['documents', 'metadatas'])

                start = offset
                end = min(offset + limit, total)

                paginated_results = {
                    'ids': results['ids'][start:end] if results['ids'] else [],
                    'documents': results['documents'][start:end] if results['documents'] else [],
                    'metadatas': results['metadatas'][start:end] if results['metadatas'] else []
                }

            chunks = []
            if paginated_results['ids']:
                for i in range(len(paginated_results['ids'])):
                    chunks.append({
                        'chunk_id': paginated_results['ids'][i],
                        'content': paginated_results['documents'][i],
                        'metadata': paginated_results['metadatas'][i]
                    })

            return {
                'chunks': chunks,
                'total': total,
                'offset': offset,
                'limit': limit
            }
        except Exception as e:
            raise VectorDBException(f"Failed to get paginated chunks: {e}")

    def search_chunks_by_text(
        self,
        search_text: str,
        offset: int = 0,
        limit: int = 20
    ) -> Dict[str, Any]:
        """Search chunks by text content (keyword search)"""
        try:
            # Get all chunks and filter by text
            results = self.collection.get(include=['documents', 'metadatas'])

            matching_chunks = []
            if results['ids']:
                search_lower = search_text.lower()
                for i in range(len(results['ids'])):
                    if search_lower in results['documents'][i].lower():
                        matching_chunks.append({
                            'chunk_id': results['ids'][i],
                            'content': results['documents'][i],
                            'metadata': results['metadatas'][i]
                        })

            total = len(matching_chunks)
            start = offset
            end = min(offset + limit, total)

            return {
                'chunks': matching_chunks[start:end],
                'total': total,
                'offset': offset,
                'limit': limit
            }
        except Exception as e:
            raise VectorDBException(f"Failed to search chunks: {e}")

    def add_single_chunk(
        self,
        chunk_id: str,
        content: str,
        embedding: List[float],
        metadata: Dict[str, Any]
    ) -> None:
        """Add a single chunk to the database"""
        try:
            self.collection.add(
                ids=[chunk_id],
                documents=[content],
                embeddings=[embedding],
                metadatas=[metadata]
            )
        except Exception as e:
            raise VectorDBException(f"Failed to add chunk: {e}")

