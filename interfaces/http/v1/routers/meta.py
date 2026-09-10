"""What this deployment can offer, and whether it is well.

Discovery has to work on a machine that can run nothing, which is the shape of
every route here: an unavailable option is listed with its reason rather than
hidden, because a picker that silently drops one cannot explain why it is
missing and "this machine has no embedding model for it" is the answer the
user needs.
"""

from __future__ import annotations

from typing import Annotated, Optional

from fastapi import APIRouter, Query

from chat_rag.application import catalogue, ops

from ..dependencies import Container, SessionId
from ..schemas import (
    ChunkingMethod, ChunkingMethodCollection, Health, ModelChain, RetrievalMethod,
    RetrievalMethodCollection,
)

router = APIRouter(tags=["meta"])

KnowledgeBaseFilter = Annotated[Optional[str], Query(
    alias="knowledge_base_id",
    description="whose pipeline to report on; the process default when absent",
)]


@router.get("/meta/chunking-methods", response_model=ChunkingMethodCollection,
            summary="Every chunking method this deployment knows about")
def chunking_methods() -> ChunkingMethodCollection:
    """A projection of the library's registry, with no catalogue of its own.

    A method registered in the library appears here on the next request; there
    is no list in this repository to edit, which is what makes
    "implementation + registration + tests" the whole cost of adding one.
    """
    items = [ChunkingMethod.of(entry) for entry in catalogue.chunking_method_facts()]
    return ChunkingMethodCollection.of(items, offset=0, limit=len(items),
                                       total=len(items))


@router.get("/meta/retrieval-methods", response_model=RetrievalMethodCollection,
            summary="What this knowledge base's retriever can actually serve")
def retrieval_methods(services: Container, session: SessionId,
                      knowledge_base_id: KnowledgeBaseFilter = None
                      ) -> RetrievalMethodCollection:
    """Reporting only -- no search runs. A profile that computes no embeddings
    says so here rather than failing a search later."""
    found = catalogue.retrieval(services, session_id=session,
                               kb_id=knowledge_base_id or None)
    items = [RetrievalMethod.of(entry) for entry in found.get("methods") or []]
    return RetrievalMethodCollection.of(items, offset=0, limit=len(items),
                                        total=len(items),
                                        default=found.get("default"))


@router.get("/meta/models", response_model=ModelChain,
            summary="The configured model chain: chunking, embedding, answer")
def models(services: Container, session: SessionId,
           knowledge_base_id: KnowledgeBaseFilter = None) -> ModelChain:
    """Names, ids and endpoints only. Never a key."""
    return ModelChain(chain=catalogue.model_chain(services, session_id=session,
                                                  kb_id=knowledge_base_id or None))


@router.get("/health", response_model=Health, tags=["health"],
            summary="Liveness, readiness and one line of capacity")
def health(services: Container) -> Health:
    """Deliberately small and always answerable, including while degraded --
    which it reports rather than fails on. It is polled every few seconds by
    something that only needs to know whether to look further."""
    return Health.of(ops.health(services))
