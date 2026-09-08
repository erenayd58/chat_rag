"""`/api/v1/knowledge-bases` -- the collection a document is ingested into."""

from __future__ import annotations

from typing import Annotated, Optional

from fastapi import APIRouter, Body, Query, Request, Response, status

from application import chunks as chunk_use_case
from application import knowledge_bases as use_case

from ..dependencies import Container, Page, SessionId
from ..envelope import slice_of
from ..openapi import LOCATION_HEADER
from ..schemas import (
    Chunk, ChunkCollection, EmbeddingIndex, EmbeddingReindex, KnowledgeBase,
    KnowledgeBaseCollection, KnowledgeBaseCreate, KnowledgeBaseUpdate,
)

router = APIRouter(prefix="/knowledge-bases", tags=["knowledge bases"])


@router.get("", response_model=KnowledgeBaseCollection, summary="List them")
def list_knowledge_bases(services: Container, page: Page) -> KnowledgeBaseCollection:
    records = use_case.list_all(services)
    items = [KnowledgeBase.of(record)
             for record in slice_of(records, offset=page.offset, limit=page.limit)]
    return KnowledgeBaseCollection.of(items, offset=page.offset, limit=page.limit,
                                      total=len(records))


@router.post("", response_model=KnowledgeBase, status_code=status.HTTP_201_CREATED,
             responses={status.HTTP_201_CREATED: {"headers": LOCATION_HEADER}},
             summary="Create one")
def create_knowledge_base(
    services: Container, request: Request, response: Response,
    payload: Annotated[KnowledgeBaseCreate, Body(default_factory=KnowledgeBaseCreate)],
) -> KnowledgeBase:
    """Its chunker, embedding model and storage are fixed here and not editable
    afterwards, because the corpus that gets ingested depends on them --
    changing one later would describe the vectors wrongly."""
    record = use_case.create(services, payload.as_payload())
    # Built from the routing table rather than written out, so the version
    # prefix stays declared in exactly one place.
    response.headers["Location"] = request.url_for(
        "get_knowledge_base", kb_id=record["kb_id"]).path
    return KnowledgeBase.of(record)


@router.get("/{kb_id}", response_model=KnowledgeBase, summary="Read one")
def get_knowledge_base(services: Container, kb_id: str) -> KnowledgeBase:
    return KnowledgeBase.of(use_case.get(services, kb_id))


@router.patch("/{kb_id}", response_model=KnowledgeBase, summary="Rename one")
def update_knowledge_base(
    services: Container, kb_id: str,
    payload: Annotated[KnowledgeBaseUpdate, Body(default_factory=KnowledgeBaseUpdate)],
) -> KnowledgeBase:
    """Only ``name`` and ``extra``: everything else was decided at creation."""
    return KnowledgeBase.of(use_case.update(services, kb_id, payload.changes()))


@router.delete("/{kb_id}", status_code=status.HTTP_204_NO_CONTENT,
               response_class=Response, summary="Delete one, and its corpus")
def delete_knowledge_base(services: Container, kb_id: str) -> Response:
    """Its documents' ledger rows and analyses deliberately survive -- a user
    does not lose the record that a file was ever ingested. Its vectors do
    not: they go in the same transaction as the record. **409** when the
    corpus could not be cleared, in which case neither half went."""
    use_case.delete(services, kb_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get("/{kb_id}/embedding-index", response_model=EmbeddingIndex,
            summary="Do the stored vectors belong to the configured model")
def embedding_index(services: Container, session: SessionId, kb_id: str) -> EmbeddingIndex:
    return EmbeddingIndex(**use_case.embedding_index(services, kb_id, session_id=session))


@router.post("/{kb_id}/embedding-index/rebuild", response_model=EmbeddingReindex,
             summary="Re-embed every stored chunk with the current model")
def rebuild_embedding_index(services: Container, session: SessionId,
                            kb_id: str) -> EmbeddingReindex:
    """Synchronous, and the one long write this API does inline: it is an
    operator action on a knowledge base nobody is querying, not a user
    action."""
    return EmbeddingReindex(
        **use_case.reindex_embeddings(services, kb_id, session_id=session))


@router.get("/{kb_id}/chunks", response_model=ChunkCollection,
            summary="Browse the corpus, a page at a time")
def browse_chunks(
    services: Container, session: SessionId, page: Page, kb_id: str,
    search: Annotated[Optional[str], Query(
        description="filter by phrase; a substring scan inside the store")] = None,
) -> ChunkCollection:
    """A substring scan inside the store: no embedding call and no index build,
    which is why -- unlike the searches -- it runs under no query bound."""
    found = chunk_use_case.browse(
        services, kb_id=kb_id, session_id=session,
        offset=page.offset, limit=page.limit, search_text=search or "",
    )
    return ChunkCollection.of([Chunk.of(row) for row in found["chunks"]],
                              offset=found["offset"], limit=found["limit"],
                              total=found["total"])
