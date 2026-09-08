"""A document: one ingested file, in exactly one knowledge base.

Three stores hold a piece of it -- PostgreSQL holds the ledger row and the
analysis record, the vector store holds the chunks, the workspace directory
holds the packaged artifacts -- and the rules about which of them a deletion
reaches are the product's, not any one store's. They live here, which is why
the schema drawn in Step 8 could be read off them
(``tests/migration/test_domain_relations.py`` is the same rules from the
outside, and held the schema to them).
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from . import workspace
from .errors import NotFound

logger = logging.getLogger("RAG.documents")


def list_all(services, kb_id: Optional[str] = None) -> list[dict]:
    return services.documents().get_all_documents(kb_id=kb_id)


def get(services, doc_id: str) -> dict:
    """One ingested document, by its own id."""
    record = services.documents().get_document_by_doc_id(doc_id)
    if not record:
        raise NotFound('Document not found')
    return record


def statistics(services, *, kb_id: Optional[str], session_id: str) -> dict:
    """Ledger counts, plus what the store itself holds.

    The store's own count is best-effort: it needs a built pipeline, and a
    knowledge base whose store cannot be opened must not make the documents
    screen fail -- the ledger numbers are still true.
    """
    stats = services.documents().get_statistics(kb_id=kb_id)

    vector_db_chunks = 0
    try:
        pipeline = (services.get_pipeline(session_id, kb_id) if kb_id
                    else services.default_pipeline)
        vector_db_chunks = len(pipeline.vector_db.get_all_chunks())
    except Exception:  # noqa: BLE001 - a countless store is not a broken screen
        pass

    return {
        'total_documents': stats['total_documents'],
        'total_chunks': stats['total_chunks'],
        'total_size_mb': round(stats['total_size_bytes'] / (1024 * 1024), 2),
        'vector_db_chunks': vector_db_chunks,
        'oldest_ingestion': stats.get('oldest_ingestion'),
        'latest_ingestion': stats.get('latest_ingestion'),
        'kb_id': kb_id,
    }


#: What "every chunk of this document" means when a caller does not page.
#: The legacy route asks for one page this size and hands the whole thing to
#: the browser; a paging caller passes its own window instead.
WHOLE_DOCUMENT = 10000


def chunks_of(services, doc_id: str, *, kb_id: Optional[str], session_id: str,
              offset: int = 0, limit: int = WHOLE_DOCUMENT) -> dict:
    """The stored chunks of one document, from its own knowledge base's store."""
    pipeline = services.get_pipeline(session_id, kb_id)
    stored = pipeline.vector_db.get_chunks_paginated(
        offset=offset, limit=limit, filter_dict={'doc_id': doc_id}
    )
    return {'chunks': stored['chunks'], 'total': stored['total'],
            'offset': stored.get('offset', offset), 'limit': stored.get('limit', limit)}


def canonical_units(services, doc_id: str, *, kb_id: Optional[str], session_id: str,
                    page_from: Optional[int] = None, page_to: Optional[int] = None,
                    unit_type: Optional[str] = None, offset: int = 0,
                    limit: int = 100) -> dict:
    """The parser's canonical units for a document, before chunking.

    Read-only and cache-backed: the PDF is never re-parsed and nothing is
    reordered, merged or cleaned here. Used side by side with the chunk view
    to tell a parser reading-order problem from a chunker one.
    """
    from components.parsers.canonical_units_store import (
        find_cache_file, load_units, select_units, summarize_source,
    )

    pipeline = services.get_pipeline(session_id, kb_id)
    stored = pipeline.vector_db.get_chunks_paginated(
        offset=0, limit=10000, filter_dict={'doc_id': doc_id}
    )
    wanted = workspace.unit_ids_of(stored.get('chunks') or [])
    if not wanted:
        raise NotFound('No canonical unit ids on the chunks of this document. '
                       'Only documents ingested through the structured parser '
                       'expose a parser view.')

    cache_file = find_cache_file(wanted)
    if cache_file is None:
        raise NotFound('No canonical unit cache found for this document. '
                       'Re-upload it with the structured parser available.')

    units = load_units(cache_file)
    window, total, pages = select_units(
        units, page_from=page_from, page_to=page_to, unit_type=unit_type,
        offset=offset, limit=limit,
    )

    rows = [{
        'index': index,
        'order': row.get('order'),
        'unit_id': row.get('unit_id'),
        'type': row.get('type'),
        'heading_level': row.get('heading_level'),
        'section_path': row.get('section_path') or [],
        'text': row.get('text') or '',
        'source': summarize_source(row),
    } for index, row in enumerate(window, start=offset + 1)]

    all_pages = sorted({
        (r.get('source') or {}).get('page')
        for r in units
        if (r.get('source') or {}).get('page') is not None
    })
    return {
        'doc_id': doc_id,
        'source': str(cache_file),
        'total_units_in_document': len(units),
        'total': total,
        'returned': len(rows),
        'offset': offset,
        'limit': limit,
        'pages_in_selection': pages,
        'pages_in_document': all_pages,
        'units': rows,
    }


def delete(services, doc_id: str, *, kb_id: Optional[str] = None,
           session_id: str = 'global') -> None:
    """Delete a document: its chunks, its ledger row and its analysis.

    The three edges a document owns outright. What it does **not** own is the
    content: another upload of the same PDF keeps the shared analysis, and
    only the last upload of a content takes it down with it -- a rule the
    packager enforces, not this function.
    """
    ledger = services.documents()
    record = ledger.get_document_by_doc_id(doc_id)
    owning_kb = (record.get('kb_id') if record else None) or kb_id

    pipeline = (services.get_pipeline(session_id, owning_kb) if owning_kb
                else services.default_pipeline)
    pipeline.vector_db.delete_by_doc_id(doc_id)

    if record:
        # By the document's own identity. It used to be by the path the file
        # was uploaded from, which was the ledger's key and is now neither a
        # key nor a file that still exists.
        ledger.remove_by_doc_id(doc_id)

    workspace.discard(doc_id)


def of_ingest_job(services, job_id: str) -> Optional[dict[str, Any]]:
    """The ledger record a job wrote, if it got that far.

    The ledger write is the last act of an ingest, so a document stamped with
    this job's id is proof the job completed -- which is what lets a restart
    tell "it finished and you missed it" from "it never happened".
    """
    return services.documents().get_document_by_ingest_job(job_id)
