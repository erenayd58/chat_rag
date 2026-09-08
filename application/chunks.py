"""Chunk inspection, and the Lab's read-only searches over a stored corpus.

Two kinds of work, kept together because they read the same rows. Browsing and
editing a chunk touches the store and nothing else. Searching runs the front
half of a query -- embedding the question, scoring, building the lexical index
if this pipeline has not built it yet -- and therefore runs under the query
bounds (:func:`application.query.bounded`), not beside them.

Every search here refuses a method the configured retriever cannot serve
rather than letting a retriever exception reach the caller as a fault: a
lexical-only profile has no vectors, and saying so is the answer.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Optional

from components.retriever import (
    method_is_available, retrieval_capabilities, unavailable_reason,
)
from core.exceptions import RESOURCE_CONTROL_EXCEPTIONS

from . import query as query_bounds
from .errors import ApplicationError, InvalidRequest, NotFound, ProcessingFailed

logger = logging.getLogger("RAG.chunks")


def _search_metadata(services, pipeline, kb_id: Optional[str], *, method: str,
                     query: str) -> dict:
    """What a search result says about where it came from.

    Attached to the answer and to every row in it: the review screen shows a
    result next to the knowledge base, embedding model and store that produced
    it, so two runs can be told apart.
    """
    kb = services.kb_manager.get(kb_id) if kb_id else None
    return {
        'kb_id': kb_id,
        'kb_name': kb.get('name') if kb else 'Unknown KB',
        'embedding_model': pipeline.settings.embedding_model_name,
        'vector_db': pipeline.vector_db.get_name(),
        'search_method': method,
        'search_query': query,
    }


def _require_kb(kb_id: Optional[str]) -> str:
    if not kb_id or not kb_id.strip():
        raise InvalidRequest('Knowledge base selection is required for search')
    return kb_id


def _require_query(text: str) -> str:
    text = (text or '').strip()
    if not text:
        raise InvalidRequest('Query is required')
    return text


# ------------------------------------------------------------------ browsing
def browse(services, *, kb_id: Optional[str], session_id: str, offset: int = 0,
           limit: int = 20, search_text: str = "") -> dict:
    """A page of stored chunks, or the keyword matches for ``search_text``.

    Unbounded on purpose: this is a substring scan inside the store, with no
    embedding call and no index build, so it is not the front half of a query
    the way the searches below are.
    """
    search_text = (search_text or '').strip()
    if search_text:
        _require_kb(kb_id)

    pipeline = services.get_pipeline(session_id, kb_id)
    if not search_text:
        result = pipeline.vector_db.get_chunks_paginated(offset=offset, limit=limit)
        return {
            'chunks': result['chunks'], 'total': result['total'],
            'offset': result['offset'], 'limit': result['limit'],
            'search_metadata': result.get('search_metadata'),
        }

    result = pipeline.vector_db.search_chunks_by_text(
        search_text=search_text, offset=offset, limit=limit,
    )
    metadata = _search_metadata(services, pipeline, kb_id, method='keyword',
                                query=search_text)
    for chunk in result['chunks']:
        chunk['search_metadata'] = metadata
    return {
        'chunks': result['chunks'], 'total': result['total'],
        'offset': result['offset'], 'limit': result['limit'],
        'search_metadata': metadata,
    }


def read(services, chunk_id: str, *, kb_id: Optional[str], session_id: str) -> dict:
    """One chunk, with the first ten dimensions of its embedding for display."""
    chunk = services.get_pipeline(session_id, kb_id).vector_db.get_chunk_by_id(chunk_id)
    if not chunk:
        raise NotFound('Chunk not found')
    embedding = chunk.get('embedding')
    return {
        'chunk_id': chunk['chunk_id'],
        'content': chunk['content'],
        'metadata': chunk['metadata'],
        'embedding_snippet': embedding[:10] if embedding else None,
        'embedding_dimension': len(embedding) if embedding else 0,
    }


def update(services, chunk_id: str, *, content: Optional[str], metadata: Optional[dict],
           kb_id: Optional[str], session_id: str) -> None:
    """Edit a chunk. Changing its content re-embeds it, or the vector would
    no longer stand for the text."""
    if content is None and metadata is None:
        raise InvalidRequest('Content or metadata is required')
    pipeline = services.get_pipeline(session_id, kb_id)
    embedding = None
    if content is not None:
        embedding = pipeline.embedding_model.encode(content).tolist()
    pipeline.vector_db.update_chunk(chunk_id=chunk_id, content=content,
                                    metadata=metadata, embedding=embedding)


def delete(services, chunk_id: str, *, kb_id: Optional[str], session_id: str) -> None:
    services.get_pipeline(session_id, kb_id).vector_db.delete_chunk(chunk_id)


# ------------------------------------------------------------- the searches
def search_vector(services, *, query: str, kb_id: Optional[str], session_id: str,
                  offset: int = 0, limit: int = 20) -> dict:
    """Dense search over the stored vectors, paginated after scoring."""
    with query_bounds.bounded(services, mode='lab.search_vector',
                              session_id=session_id, kb_id=kb_id) as run:
        with _reported('Vector search'):
            text = _require_query(query)
            _require_kb(kb_id)
            metadata = _search_metadata(services, run.pipeline, kb_id,
                                        method='vector', query=text)
            embedding = run.pipeline.embedding_model.encode(text).tolist()
            hits = run.pipeline.vector_db.query(query_embedding=embedding,
                                                top_k=offset + limit)
            rows = [{
                'chunk_id': hit['chunk_id'],
                'content': hit['content'],
                'metadata': hit['metadata'],
                # Cosine distance runs 0 (identical) to 2 (opposite) --
                # pgvector's ``<=>`` and the range every store this product
                # has shipped reported; the raw distance is kept for
                # debugging.
                'similarity_score': 1 - (hit['distance'] / 2),
                'distance': hit['distance'],
                'search_metadata': metadata,
            } for hit in hits[offset:offset + limit]]
            return {'chunks': rows, 'total': len(hits), 'offset': offset,
                    'limit': limit, 'search_metadata': metadata}


def search_bm25(services, *, query: str, kb_id: Optional[str], session_id: str,
                offset: int = 0, limit: int = 20) -> dict:
    """Lexical search over the knowledge base's BM25 index, paginated after
    scoring -- the index is built on demand if this pipeline has none."""
    with query_bounds.bounded(services, mode='lab.search_bm25',
                              session_id=session_id, kb_id=kb_id) as run:
        with _reported('BM25 search'):
            text = _require_query(query)
            _require_kb(kb_id)
            metadata = _search_metadata(services, run.pipeline, kb_id,
                                        method='bm25', query=text)
            hits = run.pipeline.hybrid_retriever.keyword_search(text, top_k=offset + limit)
            rows = [{
                'chunk_id': hit.chunk.chunk_id,
                'content': hit.chunk.content,
                'metadata': hit.chunk.metadata,
                'score': hit.score,
                'retrieval_method': 'bm25',
                'search_term': text,
                'search_metadata': metadata,
            } for hit in hits[offset:offset + limit]]
            return {'chunks': rows, 'total': len(hits), 'offset': offset,
                    'limit': limit, 'search_metadata': metadata}


def experiment_search(services, *, query: str, method: str, kb_id: Optional[str],
                      session_id: str, top_k: int = 20) -> dict:
    """One retrieval, by the method the reviewer picked.

    The capability check comes first and is the same one the review screen
    reads, so 'vector' on a lexical-only profile is refused with the reason
    rather than surfacing a retriever exception as a fault.
    """
    with query_bounds.bounded(services, mode='lab.experiment_search',
                              session_id=session_id, kb_id=kb_id) as run:
        with _reported('Experiment search'):
            text = _require_query(query)
            retriever = run.pipeline.hybrid_retriever
            if not method_is_available(retriever, method):
                raise InvalidRequest(
                    unavailable_reason(retriever, method),
                    details={'unsupported_method': True,
                             'capabilities': retrieval_capabilities(retriever)},
                )

            if method == 'vector':
                embedding = run.pipeline.embedding_model.encode(text).tolist()
                rows = [{
                    'chunk_id': hit['chunk_id'],
                    'content': hit['content'],
                    'metadata': hit['metadata'],
                    'score': 1 - (hit['distance'] / 2),
                    'retrieval_method': 'vector',
                    'search_term': text,
                } for hit in run.pipeline.vector_db.query(query_embedding=embedding,
                                                          top_k=top_k)]
            elif method in ('bm25', 'hybrid'):
                search = (retriever.keyword_search if method == 'bm25'
                          else retriever.hybrid_search)
                rows = [{
                    'chunk_id': hit.chunk.chunk_id,
                    'content': hit.chunk.content,
                    'metadata': hit.chunk.metadata,
                    'score': hit.score,
                    'retrieval_method': hit.retrieval_method,
                    'search_term': text,
                } for hit in search(text, top_k)]
            else:
                raise InvalidRequest('Unknown retrieval method')

            return {'chunks': rows, 'retrieval_method': method, 'query': text}


@contextmanager
def _reported(what: str):
    """Anything unexpected inside a bounded search is a server fault.

    The searches used to catch ``Exception`` and answer 500 themselves, which
    is what let the bound record the query as failed rather than as a clean
    exit. Converting it here keeps that -- and keeps the resource-control
    exceptions, which mean *stop*, travelling untouched to the bound that
    raised them.
    """
    try:
        yield
    except (ApplicationError, *RESOURCE_CONTROL_EXCEPTIONS):
        raise
    except Exception as error:  # noqa: BLE001 - reported as a fault, not hidden
        logger.error(f"{what} failed: {error}", exc_info=True)
        raise ProcessingFailed(str(error)) from error
