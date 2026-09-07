"""
Base retriever abstraction
"""
from abc import ABC, abstractmethod
from typing import List
from core.models import DocumentChunk, RetrievalResult


class BaseRetriever(ABC):
    """Base class for retrieval methods"""
    
    @abstractmethod
    def vector_search(self, query: str, top_k: int = 10) -> List[RetrievalResult]:
        """Perform vector similarity search"""
        pass
    
    @abstractmethod
    def keyword_search(self, query: str, top_k: int = 10) -> List[RetrievalResult]:
        """Perform keyword-based search"""
        pass
    
    @abstractmethod
    def hybrid_search(self, query: str, top_k: int = 10) -> List[RetrievalResult]:
        """Combine the dense and lexical legs into one ranking"""
        pass

    @abstractmethod
    def build_keyword_index(self, chunks: List[DocumentChunk]) -> None:
        """Build keyword search index"""
        pass

