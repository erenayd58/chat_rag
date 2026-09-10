"""A knowledge base: a named collection with its own chunker and its own store.

The record store owns the configuration; the vector store owns the vectors;
the pipeline cache owns the live handles onto that store. Deleting one is
therefore three acts in a fixed order, and the order is the reason this is a
use case rather than three calls a route makes in whatever sequence it was
written in.
"""

from __future__ import annotations

import gc
import logging
from typing import Any

from chat_rag.components.observability import events
from chat_rag.core.exceptions import ConfigurationException

from .errors import Conflict, InvalidRequest, NotFound

logger = logging.getLogger("RAG.knowledge_bases")


def list_all(services) -> list[dict]:
    return services.kb_manager.list()


def get(services, kb_id: str) -> dict:
    kb = services.kb_manager.get(kb_id)
    if not kb:
        raise NotFound('Knowledge base not found')
    return {'kb_id': kb_id, **kb}


def create(services, payload: dict) -> dict:
    """Create one from a client payload.

    The manager validates: a duplicate name, an unknown chunker or a provider
    this deployment cannot serve is the caller's problem, and nothing is
    written.
    """
    try:
        return services.kb_manager.create_from_payload(payload or {})
    except ValueError as error:
        raise InvalidRequest(str(error)) from error


def update(services, kb_id: str, updates: dict) -> dict:
    """Change the presentation fields of a knowledge base.

    Only ``name`` and ``extra``: chunker, embedding model and storage are
    fixed at creation because the ingested corpus depends on them.
    """
    if not services.kb_manager.get(kb_id):
        raise NotFound('Knowledge base not found')
    data = updates or {}
    changes: dict[str, Any] = {}
    if 'name' in data:
        name = str(data.get('name') or '').strip()
        if not name:
            raise InvalidRequest('name is required')
        existing = services.kb_manager.find_by_name(name)
        if existing is not None and existing != kb_id:
            raise InvalidRequest(f'A knowledge base named {name!r} already exists')
        changes['name'] = name
    if 'extra' in data:
        if not isinstance(data['extra'], dict):
            raise InvalidRequest('extra must be an object')
        changes['extra'] = data['extra']
    if not changes:
        raise InvalidRequest('Nothing to update')
    return services.kb_manager.update(kb_id, changes)


def delete(services, kb_id: str) -> dict:
    """Delete a knowledge base: its record and the vectors that belong to it.

    Cached pipelines are dropped first. That used to be load-bearing -- a live
    Chroma client held the store's sqlite file open and on Windows an open
    handle was enough to make the directory undeletable. It is hygiene now: a
    cached pipeline for a knowledge base that no longer exists would answer
    from a lexical index built before the delete, and dropping it is how the
    next query rebuilds against an empty collection.
    """
    if not services.kb_manager.get(kb_id):
        raise NotFound('Knowledge base not found')

    release_store_handles(services, kb_id)

    result = services.kb_manager.delete_with_storage(kb_id)
    if not result.get('deleted'):
        reason = result.get('reason', 'delete failed')
        if reason == 'not found':
            raise NotFound(reason)
        raise Conflict(reason)
    return result


def release_store_handles(services, kb_id: str) -> None:
    """Close and forget every cached pipeline for this knowledge base.

    Dropping the pipeline from the cache does not close anything, so each
    store is closed explicitly first.
    """
    dropped = services.pipeline_cache.discard_kb(kb_id)
    events.emit("pipeline.cache.discarded", kb_id=kb_id, pipelines=dropped)
    gc.collect()


# --------------------------------------------------------------- the vectors
def embedding_index(services, kb_id: str, *, session_id: str) -> dict:
    """Whether the knowledge base's vectors belong to the current embedding."""
    if not services.kb_manager.get(kb_id):
        raise NotFound('Knowledge base not found')
    return services.get_pipeline(session_id, kb_id).embedding_index_status()


def reindex_embeddings(services, kb_id: str, *, session_id: str) -> dict:
    """Re-embed every stored chunk with the current embedding model."""
    if not services.kb_manager.get(kb_id):
        raise NotFound('Knowledge base not found')
    pipeline = services.get_pipeline(session_id, kb_id)
    try:
        result = pipeline.reindex_embeddings()
    except ConfigurationException as error:
        # No embedding model this deployment can run: a configuration the
        # caller has to fix, not a fault of the store.
        raise InvalidRequest(str(error)) from error
    logger.info(f"Re-indexed knowledge base {kb_id}: {result}")
    return {'result': result, 'index': pipeline.embedding_index_status()}
