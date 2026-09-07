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
        """Add chunks with embeddings to the database"""
        pass
    
    @abstractmethod
    def query(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        filter_dict: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Query the vector database"""
        pass
    
    @abstractmethod
    def get_all_chunks(self) -> List[DocumentChunk]:
        """Retrieve all chunks from the database"""
        pass
    
    @abstractmethod
    def delete_by_doc_id(self, doc_id: str) -> None:
        """Delete all chunks for a document"""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Get the vector database name"""
        pass
    
    @abstractmethod
    def count(self) -> int:
        """Get the total number of chunks in the database"""
        pass

