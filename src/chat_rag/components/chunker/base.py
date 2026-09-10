"""
Base chunker abstraction
"""
from abc import ABC, abstractmethod
from typing import List
from chat_rag.core.models import DocumentChunk


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
        """Chunk text into document chunks"""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Get the chunker name"""
        pass
    
    @abstractmethod
    def get_config(self) -> dict:
        """Get the chunker configuration"""
        pass

