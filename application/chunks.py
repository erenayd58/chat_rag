"""Browsing a stored corpus, and searching it.

Two kinds of work, kept together because they read the same rows. Browsing
touches the store and nothing else. Searching runs the front half of a query
-- embedding the question, scoring, building the lexical index if this
pipeline has not built it yet -- and therefore runs under the query bounds
(:func:`application.query.bounded`), not beside them.

:func:`search` refuses a method the configured retriever cannot serve rather
than letting a retriever exception reach the caller as a fault: a lexical-only
profile has no vectors, and saying so is the answer.

There used to be more here, for a screen that is gone. Reading, editing and
deleting one indexed chunk were the Flask-era Lab's own affordances, never
promoted to the contract -- editing an indexed chunk changes the corpus behind
the ingest ledger's back -- and two of the three searches were separate
endpoints that differed only in which retriever leg they called, which is a
parameter (``docs/legacy-removal.md``).
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


# -------------------------------------------------------------- the search
def experiment_search(services, *, query: str, method: str, kb_id: Optional[str],
                      session_id: str, top_k: int = 20) -> dict:
    """One retrieval, by the method the caller picked.

    One function for every method, because the three differ only in which leg
    of the retriever they call. The capability check comes first and is the
    same one ``GET /api/v1/meta/retrieval-methods`` reads, so 'vector' on a
    lexical-only profile is refused with the reason rather than surfacing a
    retriever exception as a fault.
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
