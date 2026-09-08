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

from components.observability import events

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


#: ``statistics()`` was here -- the body of ``GET /api/stats``, a count of
#: documents, chunks and bytes for one knowledge base. It has no ``/api/v1``
#: answer and needs none: ``GET /api/v1/documents`` carries every number in it
#: per document, and ``GET /api/v1/health`` carries the capacity half
#: (``docs/api-v1.md``). It went with the surface that asked for it.


def owning_knowledge_base(services, doc_id: str, kb_id: Optional[str]) -> Optional[str]:
    """Which knowledge base's store holds this document.

    A document belongs to exactly one, and the ledger row says which -- so a
    caller does not have to, and a read that omits the filter must not be
    answered out of the process default's store, which holds nothing and would
    look like a document with no chunks.

    The record wins over what the caller passed, the same way it does in
    :func:`delete`, which applies this rule to the record it has already read:
    there is one true answer and it is not the query string's.
    """
    record = services.documents().get_document_by_doc_id(doc_id)
    return (record.get('kb_id') if record else None) or kb_id


#: What "every chunk of this document" means when a caller does not page. A
#: paging caller passes its own window instead.
WHOLE_DOCUMENT = 10000


def chunks_of(services, doc_id: str, *, kb_id: Optional[str], session_id: str,
              offset: int = 0, limit: int = WHOLE_DOCUMENT) -> dict:
    """The stored chunks of one document, from its own knowledge base's store."""
    pipeline = services.get_pipeline(
        session_id, owning_knowledge_base(services, doc_id, kb_id))
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

    pipeline = services.get_pipeline(
        session_id, owning_knowledge_base(services, doc_id, kb_id))
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
    # The same rule :func:`owning_knowledge_base` applies for a read, on the
    # record this function has already had to fetch.
    owning_kb = (record.get('kb_id') if record else None) or kb_id

    pipeline = (services.get_pipeline(session_id, owning_kb) if owning_kb
                else services.default_pipeline)
    pipeline.vector_db.delete_by_doc_id(doc_id)
    forget_lexical_indexes(services, pipeline, owning_kb, doc_id=doc_id)

    if record:
        # By the document's own identity. It used to be by the path the file
        # was uploaded from, which was the ledger's key and is now neither a
        # key nor a file that still exists.
        ledger.remove_by_doc_id(doc_id)

    workspace.discard(doc_id)


def forget_lexical_indexes(services, pipeline, kb_id: Optional[str], *,
                           doc_id: str) -> None:
    """Tell every pipeline of this knowledge base that its corpus shrank.

    The lexical index is built once per pipeline, from the store, and is never
    consulted against it again -- so a document deleted out of the store stays
    in the BM25 index of every pipeline holding one, and a keyword or hybrid
    search keeps returning its chunks until the process restarts. An ingest
    already tells the others (``application.ingest``); a deletion has to tell
    them too, and unlike an ingest it has to tell **itself**, because there is
    nothing here that rebuilds its own index the way a completed ingest does.

    Invalidation and not a rebuild: a delete should not pay for re-reading the
    whole corpus, and ``ensure_index`` builds again on the next search.
    """
    if kb_id:
        stale = services.pipeline_cache.invalidate_indexes(kb_id)
        events.emit("pipeline.index.invalidated", kb_id=kb_id, pipelines=stale,
                    doc_id=doc_id)
    # The pipeline the deletion ran through is not necessarily in the cache --
    # a document whose knowledge base is unknown is deleted through the
    # process default, which is nobody's cache entry.
    retriever = getattr(pipeline, 'hybrid_retriever', None)
    dropper = getattr(retriever, 'invalidate_index', None)
    if callable(dropper):
        try:
            dropper()
        except Exception as error:  # noqa: BLE001 - a stale index is not a failed delete
            logger.warning(f"could not invalidate the index after deleting {doc_id}: {error}")


def of_ingest_job(services, job_id: str) -> Optional[dict[str, Any]]:
    """The ledger record a job wrote, if it got that far.

    The ledger write is the last act of an ingest, so a document stamped with
    this job's id is proof the job completed -- which is what lets a restart
    tell "it finished and you missed it" from "it never happened".
    """
    return services.documents().get_document_by_ingest_job(job_id)
