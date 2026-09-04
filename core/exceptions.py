# /Users/murseltasgin/projects/chat_rag/core/exceptions.py
"""
Custom exceptions for the RAG system
"""


class RAGException(Exception):
    """Base exception for RAG system"""
    pass


class ChunkerException(RAGException):
    """Exception raised by chunker components"""
    pass


class EmbeddingException(RAGException):
    """Exception raised by embedding components"""
    pass


class VectorDBException(RAGException):
    """Exception raised by vector database components"""
    pass


class LLMException(RAGException):
    """Exception raised by LLM components"""
    pass


class RetrieverException(RAGException):
    """Exception raised by retriever components"""
    pass


class ConfigurationException(RAGException):
    """Exception raised for configuration errors"""
    pass



class IndexIncompatibleException(RAGException):
    """The vector store was written by a different embedding model and must
    be re-indexed before new vectors can be added to it."""
    pass


class IngestInterrupted(RAGException):
    """An ingest job was stopped on purpose before it committed anything.

    Raised at a stage boundary (never in the middle of a store write) by the
    job guard when the job's deadline has passed or it was cancelled. ``kind``
    is the terminal job state it maps to: ``timed_out`` or ``cancelled``.
    """

    def __init__(self, kind: str, message: str = ""):
        super().__init__(message or f"ingest {kind}")
        self.kind = kind


class IngestOverloaded(RAGException):
    """The ingest queue is full; the upload was refused, not queued.

    ``retry_after_seconds`` is the manager's estimate of when a slot frees.
    """

    def __init__(self, message: str, retry_after_seconds: float = 30.0):
        super().__init__(message)
        self.retry_after_seconds = retry_after_seconds
