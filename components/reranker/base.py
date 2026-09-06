"""
Base reranker abstraction
"""
from abc import ABC, abstractmethod
from typing import List
from core.models import RetrievalResult


class BaseReranker(ABC):
    """Base class for reranking strategies"""
    
    @abstractmethod
    def rerank(
        self,
        query: str,
        results: List[RetrievalResult],
        top_k: int = 5
    ) -> List[RetrievalResult]:
        """
        Rerank results based on relevance to query
        
        Args:
            query: Search query
            results: List of retrieval results
            top_k: Number of top results to return
        
        Returns:
            Reranked list of results
        """
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Get the reranker name"""
        pass

