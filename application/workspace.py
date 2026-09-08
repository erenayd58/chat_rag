"""The Viewer's view of this console: its workspace, and each document's analysis.

The Viewer reads a packaged Deep Analysis tree; an ingest here already produces
everything expensive that tree needs. What lives in this module is the console
side of that boundary -- staging an ingest's own outputs, reporting where the
packaging got to, and building the read model the Viewer's workspace panel
shows. No provider call is ever made on this path: a Deep Analysis upload is
reused as it stands, and a Standard upload gets the deterministic quality
contract (``use_llm=False``), which is free.

The packaging itself belongs to :mod:`components.viewer.analysis` -- one
background worker, one directory per content, a state file per document -- and
is not re-implemented here. This module decides *what* to ask it for.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Optional, Sequence

from components.viewer import analysis
from components.viewer import methods as viewer_methods

from .errors import InvalidRequest, NotFound, NotReady

logger = logging.getLogger("RAG.workspace")

#: The pipeline key the packaging worker resolves canonical units under. Fixed
#: rather than the browser's session id: the worker outlives the request that
#: queued it, and a per-session key would build a second pipeline per browser.
ANALYSIS_SESSION = "viewer-analysis"


# ------------------------------------------------------- the companion server
def probe_viewer(url: str, timeout: float = 1.5) -> dict:
    """Is the companion Viewer server answering at ``url``? A local probe only."""
    import urllib.request
    import urllib.error

    if not url:
        return {'configured': False, 'url': '', 'reachable': False}
    health = url.rstrip('/') + '/api/health'
    try:
        with urllib.request.urlopen(health, timeout=timeout) as response:
            payload = json.loads(response.read().decode('utf-8') or '{}')
    except (urllib.error.URLError, OSError, ValueError):
        return {'configured': True, 'url': url, 'reachable': False}
    documents = payload.get('documents') if isinstance(payload, dict) else None
    return {
        'configured': True,
        'url': url,
        'reachable': True,
        'documents': len(documents) if isinstance(documents, (list, dict)) else None,
    }


# ------------------------------------------------------------ canonical units
def recover_canonical_units(services, doc_id: str, kb_id: Optional[str] = None):
    """The canonical units of an already-ingested document, from the parser's
    own cache -- the PDF is not parsed again.

    Documents ingested before the Viewer packaging existed have no staged
    canonical; their chunks still carry the unit ids that identify the cache
    entry the parser wrote, which is enough to pin an analysis to exactly the
    corpus that was chunked.

    Called on the packaging worker, never inside a request: it reads a whole
    document's chunks out of the vector store and then walks the parser cache.
    """
    from components.parsers.canonical_units_store import find_cache_file, load_units

    pipeline = services.get_pipeline(ANALYSIS_SESSION, kb_id)
    stored = pipeline.vector_db.get_chunks_paginated(
        offset=0, limit=10000, filter_dict={'doc_id': doc_id}
    )
    wanted = unit_ids_of(stored.get('chunks') or [])
    if not wanted:
        return None
    cache_file = find_cache_file(wanted)
    return load_units(cache_file) if cache_file else None


def unit_ids_of(chunk_rows: Sequence[dict]) -> list:
    """Every canonical unit id the stored chunks name.

    Three spellings, because three generations of the metadata wrote it
    differently and all three are still in live stores.
    """
    wanted: list = []
    for chunk in chunk_rows:
        metadata = chunk.get('metadata') or {}
        ids = metadata.get('unit_ids')
        if not ids:
            for key in ('unit_ids_json', 'amsc_unit_ids_json'):
                raw = metadata.get(key)
                if raw:
                    try:
                        ids = json.loads(raw)
                    except Exception:
                        ids = None
                    if ids:
                        break
        if ids:
            wanted.extend(ids)
    return wanted


def install_unit_resolver(services) -> None:
    """Give the packaging worker its way back to the parser cache."""
    analysis.set_unit_resolver(
        lambda doc_id, kb_id=None: recover_canonical_units(services, doc_id, kb_id)
    )


# --------------------------------------------------------------- the lifecycle
def stage_analysis(doc_id: str, *, label: str, kb_id: Optional[str] = None,
                   kb_name: Optional[str] = None, chunking_mode: Optional[str] = None,
                   units=None, deep_result=None, methods=None,
                   content_sha: Optional[str] = None,
                   parse_seconds: Optional[float] = None) -> dict:
    """Hand one document's ingest outputs to the Viewer packager.

    With ``units`` this is the upload path: the ingest's own canonical (and,
    on a Deep Analysis upload, its own run) are written and queued, together
    with every other chunking method the upload asked for. Without them it is
    the catch-up path for a document ingested earlier: only the request is
    recorded, and the worker recovers the canonical.
    """
    if units is None:
        return analysis.request_build(doc_id=doc_id, label=label, kb_id=kb_id,
                                      kb_name=kb_name, chunking_mode=chunking_mode,
                                      content_sha=content_sha, wanted=methods)
    return analysis.stage(
        doc_id=doc_id, label=label, units=units, deep_result=deep_result, methods=methods,
        kb_id=kb_id, kb_name=kb_name, chunking_mode=chunking_mode, content_sha=content_sha,
        parse_seconds=parse_seconds,
    )


def prepare_missing(services) -> list:
    """Queue an analysis for every tracked document that has none yet."""
    known = analysis.states()
    queued = []
    kb_names = {kb.get('kb_id'): kb.get('name') for kb in services.kb_manager.list()}
    for doc in services.documents().get_all_documents():
        doc_id = doc.get('doc_id')
        if not doc_id:
            continue
        status = (known.get(doc_id) or {}).get('status', analysis.STATUS_MISSING)
        if status in (analysis.STATUS_READY, analysis.STATUS_RUNNING, analysis.STATUS_PENDING):
            continue
        if status == analysis.STATUS_FAILED:
            continue  # a failed build is retried on request, not on every refresh
        try:
            state = stage_analysis(
                doc_id,
                label=_label_of(doc, doc_id),
                kb_id=doc.get('kb_id'),
                kb_name=kb_names.get(doc.get('kb_id')),
                chunking_mode=doc.get('chunking_mode'),
            )
        except Exception as e:  # noqa: BLE001 - one bad document must not stop the rest
            logger.warning(f"Could not stage {doc_id} for the viewer: {e}")
            continue
        if state.get('status') != analysis.STATUS_MISSING:
            queued.append(doc_id)
    return queued


def request_analysis(services, doc_id: str) -> dict:
    """Queue -- or retry -- the analysis of a document that is already here."""
    doc = services.documents().get_document_by_doc_id(doc_id)
    if not doc:
        raise NotFound('Document not found')
    kb = services.kb_manager.get(doc.get('kb_id')) or {}
    return stage_analysis(
        doc_id,
        label=_label_of(doc, doc_id),
        kb_id=doc.get('kb_id'), kb_name=kb.get('name'),
        chunking_mode=doc.get('chunking_mode'),
    )


def analysis_state(doc_id: str) -> dict:
    return analysis.read_state(doc_id)


def analysis_states() -> dict[str, dict]:
    """Every document's analysis state, in one read.

    A list of documents needs one of these per row, and asking per document
    would walk the analysis directory once per row.
    """
    return analysis.states()


def add_methods(doc_id: str, wanted) -> dict:
    """Add chunking variants to a document that is already here.

    This is how a second method reaches a document -- not by uploading the
    PDF again. The canonical is on disk, so nothing is parsed twice and no
    variant already built is rebuilt.
    """
    try:
        return analysis.add_methods(doc_id, viewer_methods.normalise(wanted))
    except FileNotFoundError as error:
        raise NotFound(str(error)) from error


def payload(doc_id: str) -> dict:
    """The finished viewer payload for one live document."""
    built = analysis.payload(doc_id)
    if built is None:
        state = analysis.read_state(doc_id)
        raise NotReady(f"no viewer payload for {doc_id} ({state.get('status')})", state=state)
    return built


def chunk_rows(doc_id: str, method: str = "") -> dict:
    """One live document's chunk rows, per chunking method.

    The Viewer's own server reads this to build a retrieval index over a
    document that lives here, so "Dokumana sor" works on a freshly uploaded
    PDF instead of sending the reader back to this console's chat. Rows only:
    the same JSONL the chunker wrote and the benchmark reads, so a live
    document is retrieved over exactly the representation a frozen one is.

    ``method`` selects one; empty returns every ready one.
    """
    state = analysis.read_state(doc_id)
    if state.get('status') == analysis.STATUS_MISSING:
        raise NotReady(f'no viewer analysis for {doc_id}', state=state)

    wanted = (method or '').strip()
    # What this upload may be asked about: its own selection, narrowed to the
    # variants that are built. Another upload of the same PDF may have more of
    # them; they are not this document's answer.
    available = state.get('available_methods') or []
    chosen = [wanted] if wanted else list(available)
    unknown = [m for m in chosen if m not in viewer_methods.METHODS]
    if unknown:
        raise InvalidRequest(f"unknown chunking method {unknown[0]!r}")
    # A method the content has but this upload did not select is not this
    # document's to serve, and saying so is more use than an empty arm list.
    refused = [m for m in chosen if m not in available]
    chosen = [m for m in chosen if m in available]
    if refused and not chosen:
        # Two different refusals, and conflating them costs a client the
        # difference between "wait" and "stop asking". A method this upload
        # *selected* is coming; one it did not select is not this document's
        # to serve, however much the shared content has it packaged.
        if refused[0] in (state.get('selected_methods') or []):
            raise NotReady(
                f"{refused[0]!r} has not been built for this document yet",
                state=state,
            )
        raise NotFound(
            f"{refused[0]!r} is not one of this document's analysis methods "
            f"({', '.join(available) or 'none ready'})",
            details={'state': state},
        )

    arms = {}
    for name in chosen:
        rows = analysis.chunk_rows(doc_id, name)
        if rows is None:
            continue
        arms[name] = {
            'kind': viewer_methods.METHODS[name].engine,
            'label': viewer_methods.label(name),
            'chunk_count': len(rows),
            'rows': rows,
        }
    if not arms:
        raise NotReady(f'no packaged chunks for {doc_id}', state=state)
    return {
        'doc_id': doc_id,
        'label': state.get('label') or doc_id,
        'key': state.get('key'),
        'arms': arms,
    }


def discard(doc_id: str) -> bool:
    """Drop a document's live Viewer analysis.

    Only this console's live directory is reachable from here; the chunk
    repository's frozen benchmark trees are not, and are never touched.
    """
    return analysis.discard(doc_id)


def boundary_model_stats() -> dict:
    return analysis.boundary_model_stats()


# ------------------------------------------------------------- the read model
def snapshot(services, *, console_url: str = "") -> dict:
    """The console's knowledge bases and their documents, as one read-only
    snapshot.

    This is the single source of truth the Viewer's workspace panel reads, so
    a knowledge base created here -- or a document ingested into it -- shows
    up over there without anyone copying state by hand. Names, counts and
    ingest metadata only: no paths outside the file name, no keys.
    """
    documents = services.documents().get_all_documents()
    viewer_states = analysis.states()
    by_kb: dict = {}
    for doc in documents:
        doc_id = doc.get('doc_id') or ''
        # What the Viewer can do with this document right now. It is read from
        # the packager's own records, so the two screens cannot disagree about
        # whether an analysis exists.
        viewer = viewer_states.get(doc_id) or {'status': analysis.STATUS_MISSING}
        entry = {
            'doc_id': doc_id,
            'name': _label_of(doc, ''),
            'chunk_count': doc.get('chunk_count') or 0,
            'file_size': doc.get('file_size') or 0,
            'ingested_at': doc.get('ingested_at') or '',
            'status': doc.get('status') or 'indexed',
            'chunking_mode': doc.get('chunking_mode'),
            'file_hash': (doc.get('file_hash') or '')[:16],
            'viewer': {
                'status': viewer.get('status'),
                'deep_source': viewer.get('deep_source'),
                'unit_count': viewer.get('unit_count'),
                'error': viewer.get('error'),
                'updated_at': viewer.get('updated_at'),
                # Two levels, deliberately apart. ``requested`` and
                # ``ready_methods`` are what *this upload* asked for and what
                # of it is built -- the Viewer offers exactly these, so it
                # never shows a method this upload did not choose.
                'requested': viewer.get('selected_methods') or [],
                'ready_methods': viewer.get('available_methods') or [],
                # ``content_*`` is what the shared analysis holds: every
                # variant this content has, reusable by any upload of it.
                'content_requested': viewer.get('requested') or [],
                'content_ready_methods': viewer.get('ready_methods') or [],
                'failed_methods': [m for m in (viewer.get('failed_methods') or [])
                                   if m in (viewer.get('selected_methods') or [])],
                'methods': {m: spec for m, spec in (viewer.get('methods') or {}).items()
                            if m in (viewer.get('selected_methods') or [])},
                # The same PDF uploaded twice is one analysis; this is how a
                # caller can tell that two records are one document.
                'analysis_key': viewer.get('key'),
                'shared_with': [d for d in (viewer.get('doc_ids') or []) if d != doc_id],
            },
        }
        by_kb.setdefault(doc.get('kb_id') or '', []).append(entry)

    knowledge_bases = []
    for kb in services.kb_manager.list():
        kb_id = kb.get('kb_id')
        docs = by_kb.pop(kb_id, [])
        knowledge_bases.append({
            'kb_id': kb_id,
            'name': kb.get('name') or kb_id,
            'chunker': (kb.get('chunker') or {}).get('type'),
            'retrieval_method': kb.get('retrieval_method'),
            'vector_db_provider': kb.get('vector_db_provider'),
            'embedding_model_name': kb.get('embedding_model_name'),
            'documents': docs,
            'document_count': len(docs),
            'chunk_count': sum(d['chunk_count'] for d in docs),
        })
    # Documents whose knowledge base was deleted still exist in the tracker.
    # Hiding them would make the panel disagree with the console, but one card
    # per vanished id buries the live bases under a wall of hex, so they are
    # reported as a single group; each document keeps its own former id.
    orphans = [dict(doc, kb_id=kb_id) for kb_id, docs in by_kb.items() for doc in docs]
    if orphans:
        knowledge_bases.append({
            'kb_id': None,
            'name': 'Bilgi tabani silinmis kayitlar',
            'chunker': None,
            'retrieval_method': None,
            'vector_db_provider': None,
            'embedding_model_name': None,
            'documents': orphans,
            'document_count': len(orphans),
            'chunk_count': sum(d['chunk_count'] for d in orphans),
            'orphan': True,
        })

    return {
        'generated_at': datetime.now().isoformat(timespec='seconds'),
        'console_url': console_url,
        'knowledge_bases': knowledge_bases,
        'totals': {
            'knowledge_bases': len(knowledge_bases),
            'documents': len(documents),
            'chunks': sum(d.get('chunk_count') or 0 for d in documents),
            'viewer_ready': sum(
                1 for d in documents
                if (viewer_states.get(d.get('doc_id') or '') or {}).get('status') == analysis.STATUS_READY
            ),
        },
    }


def _label_of(doc: dict, fallback: str) -> str:
    return (doc.get('metadata') or {}).get('original_filename') or doc.get('file_name') or fallback


def resume_incomplete() -> list:
    """Pick up whatever packaging a previous process was in the middle of."""
    return analysis.resume_incomplete()
