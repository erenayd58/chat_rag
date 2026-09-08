"""The API's own types.

The public schema of `/api/v1`, declared as Pydantic models and kept
deliberately apart from the application's records and the stores' rows. The
current implementation keys a document by an absolute file path, a knowledge
base by a vector-store directory and an analysis by a directory named for a content
hash; none of that appears in this package, so none of it has to survive the
move to PostgreSQL and pgvector.

The rule that makes that stick is ``extra="forbid"`` on
:class:`~interfaces.http.v1.schemas.common.Schema`: a response is built by
naming fields, so a field nobody named cannot arrive by accident. The two
deliberate exceptions -- a chunk's ``metadata`` and an answer's
``diagnostics`` -- are typed as open dictionaries and published as
pass-through in ``docs/api-v1.md``.
"""

from __future__ import annotations

from .chunks import (
    CanonicalUnit, CanonicalUnitCollection, Chunk, ChunkCollection, ScoredChunk,
    SearchResults,
)
from .common import ApiError, Collection, ErrorResponse, Page, Schema
from .documents import (
    ANALYSIS_STATES, Analysis, AnalysisChunks, AnalysisMethods, ContentAnalysis, Document,
    DocumentCollection, DocumentWithAnalysis,
)
from .ingest_jobs import (
    IngestCapacity, IngestJob, IngestJobCollection, JobError, JobResult,
)
from .knowledge_bases import (
    EmbeddingIndex, EmbeddingReindex, KnowledgeBase, KnowledgeBaseCollection,
    KnowledgeBaseCreate, KnowledgeBaseUpdate,
)
from .meta import (
    Capacity, ChunkingMethod, ChunkingMethodCollection, Health, ModelChain, RetrievalMethod,
    RetrievalMethodCollection,
)
from .queries import Answer, AnswerTiming, Citation, QueryRequest, SearchRequest

__all__ = [
    "ANALYSIS_STATES", "Analysis", "AnalysisChunks", "AnalysisMethods", "Answer",
    "AnswerTiming", "ApiError", "CanonicalUnit", "CanonicalUnitCollection", "Capacity",
    "Chunk", "ChunkCollection", "ChunkingMethod", "ChunkingMethodCollection", "Citation",
    "Collection", "ContentAnalysis", "Document", "DocumentCollection",
    "DocumentWithAnalysis", "EmbeddingIndex", "EmbeddingReindex", "ErrorResponse",
    "Health", "IngestCapacity", "IngestJob", "IngestJobCollection", "JobError",
    "JobResult", "KnowledgeBase", "KnowledgeBaseCollection", "KnowledgeBaseCreate",
    "KnowledgeBaseUpdate", "ModelChain", "Page", "QueryRequest", "RetrievalMethod",
    "RetrievalMethodCollection", "Schema", "ScoredChunk", "SearchRequest",
    "SearchResults",
]
