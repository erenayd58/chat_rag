"""
Base vector database abstraction
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from core.models import DocumentChunk


class BaseVectorDB(ABC):
    """Base class for vector database providers"""
    
    @abstractmethod
    def add_chunks(
        self,
        chunks: List[DocumentChunk],
        embeddings: List[List[float]],
        **kwargs
    ) -> None:
        """
        Add chunks with embeddings to the database
        
        Args:
            chunks: List of document chunks
            embeddings: List of embedding vectors
            **kwargs: Additional provider-specific parameters
        """
        pass
    
    @abstractmethod
    def query(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        filter_dict: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Query the vector database
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            filter_dict: Optional metadata filter
            **kwargs: Additional parameters
        
        Returns:
            List of results with chunks and scores
        """
        pass
    
    @abstractmethod
    def get_all_chunks(self) -> List[DocumentChunk]:
        """
        Retrieve all chunks from the database
        
        Returns:
            List of all document chunks
        """
        pass
    
    @abstractmethod
    def delete_by_doc_id(self, doc_id: str) -> None:
        """
        Delete all chunks for a document
        
        Args:
            doc_id: Document ID to delete
        """
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Get the vector database name"""
        pass
    
    @abstractmethod
    def count(self) -> int:
        """Get the total number of chunks in the database"""
        pass

