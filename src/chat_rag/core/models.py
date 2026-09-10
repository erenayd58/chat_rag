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

    @property
    def retrieval_text(self) -> str:
        """What retrieval indexes for this chunk, on either leg.

        The rendering is indexed *beside* ``content``, never instead of it, so
        every term that matched before still matches and the sentences a table
        sits under stay in its vector. A chunk with no rendering -- every
        Standard and Markdown chunk -- indexes exactly ``content``, byte for
        byte, as it always did.
        """
        search_text = self.search_text
        if not search_text:
            return self.content
        return self.content + "\n" + search_text

    @property
    def table_view(self) -> Optional[str]:
        """A readable rendering of the one table this chunk carries.

        Produced by the Deep Analysis chunker, and only where the table's own
        structure made every label/value/period pairing certain. It is a
        reading aid for the answer model: it rides in the context *beneath*
        ``content`` and clearly marked as derived, it is never indexed, and a
        citation still points at ``content``, which is the document's own text.
        """
        return (self.metadata or {}).get("table_view") or None


@dataclass
class RetrievalResult:
    """Represents a retrieval result with relevance score"""
    chunk: DocumentChunk
    score: float
    retrieval_method: str
    rank: int
