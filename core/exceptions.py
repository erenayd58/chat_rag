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
