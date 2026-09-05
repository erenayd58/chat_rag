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


class QueryTimeout(RAGException):
    """A query ran past its deadline and was stopped at the next seam.

    Raised by the query guard where the ingest guard raises
    :class:`IngestInterrupted`: before an outbound call, while waiting for an
    answer slot, and by the deadline-aware transports between attempts. Work
    already inside a stage runs to the end of that stage first.
    """

    def __init__(self, message: str = ""):
        super().__init__(message or "the query's deadline passed before it finished")


class QueryOverloaded(RAGException):
    """The query was refused, not queued: no capacity for it right now.

    ``reason`` says which limit refused it -- ``admission`` (every query slot
    is in use) or ``answer_capacity`` (no answer-model slot came free within
    the query's wait) -- because they call for different sizing decisions.
    ``retry_after_seconds`` is an estimate of when to try again.
    """

    def __init__(self, message: str, *, reason: str = "admission",
                 retry_after_seconds: float = 5.0):
        super().__init__(message)
        self.reason = reason
        self.retry_after_seconds = retry_after_seconds


#: The exceptions that mean *stop this work*, never *this step failed*.
#:
#: Both enhancement paths in this system are written to degrade rather than
#: fail: a clarification the model could not produce falls back to the
#: original question, a strategy it could not choose falls back to hybrid,
#: a document summary it could not write falls back to the title. That is
#: deliberate product behaviour and it stays.
#:
#: It must not extend to these. A deadline that has passed, a queue or a
#: budget that refused, an operator's cancellation -- none of them say "this
#: step failed, carry on without it". They say the work must stop, and a
#: bare ``except Exception`` that swallows one turns a limit into a silent
#: degradation: the query keeps running past its deadline, making further
#: calls that will be refused in turn, and answers from heuristics as though
#: nothing happened. Every fallback handler in the enhancement paths
#: re-raises these first.
RESOURCE_CONTROL_EXCEPTIONS = (
    IngestInterrupted,
    IngestOverloaded,
    QueryTimeout,
    QueryOverloaded,
)
