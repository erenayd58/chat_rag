# /Users/murseltasgin/projects/chat_rag/core/models.py
"""
Data models and schemas used across the application
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime
import numpy as np


@dataclass
class DocumentChunk:
    """Represents a chunk of a document with metadata and context"""
    chunk_id: str
    content: str
    doc_id: str
    doc_title: str
    chunk_index: int
    total_chunks: int
    section_title: Optional[str] = None
    previous_context: Optional[str] = None
    next_context: Optional[str] = None
    document_summary: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    embedding: Optional[np.ndarray] = None

    @property
    def search_text(self) -> Optional[str]:
        """A retrieval-only rendering of a table this chunk carries.

        Produced by the Deep Analysis chunker and carried in metadata, so it
        survives a round trip through the vector store without widening this
        record. Retrieval may read it; the answer context and every citation
        read ``content``, which is the document's own text.
        """
        return (self.metadata or {}).get("search_text") or None


@dataclass
class RetrievalResult:
    """Represents a retrieval result with relevance score"""
    chunk: DocumentChunk
    score: float
    retrieval_method: str
    rank: int


@dataclass
class ConversationTurn:
    """Represents a single turn in conversation"""
    user_query: str
    clarified_query: Optional[str]
    retrieved_context: Optional[str]
    assistant_response: Optional[str]
    timestamp: str
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class SearchStrategy:
    """Represents the search strategy for a query"""
    recommended_strategy: str  # 'vector', 'bm25', or 'hybrid'
    reasoning: str
    query_type: str
    expected_answer_type: str
    suggested_top_k: int
    use_query_expansion: bool
    use_reranking: bool


@dataclass
class QueryClarification:
    """Represents query clarification result"""
    clarified_query: str
    needs_clarification: bool
    entities: List[str]
    resolution_notes: str
    confidence: str = "medium"


@dataclass
class SearchQuery:
    """Represents a search query with metadata"""
    text: str
    type: str  # 'semantic', 'keyword', 'alternative', 'original'
    purpose: str

