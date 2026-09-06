"""
Base chunker abstraction
"""
from abc import ABC, abstractmethod
from typing import List
from core.models import DocumentChunk


class BaseChunker(ABC):
    """Base class for text chunkers"""
    
    @abstractmethod
    def chunk_text(
        self,
        text: str,
        doc_id: str,
        doc_title: str,
        document_summary: str = None,
        **kwargs
    ) -> List[DocumentChunk]:
        """
        Chunk text into document chunks
        
        Args:
            text: Text to chunk
            doc_id: Document ID
            doc_title: Document title
            document_summary: Optional document summary
            **kwargs: Additional chunking parameters
        
        Returns:
            List of document chunks
        """
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Get the chunker name"""
        pass
    
    @abstractmethod
    def get_config(self) -> dict:
        """Get the chunker configuration"""
        pass

