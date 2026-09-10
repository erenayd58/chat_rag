"""The Viewer's side of this console: staging an analysis, and reading it back.

The Viewer reads a packaged Deep Analysis tree; an ingest here already produces
everything expensive that tree needs. What lives in this module is the console
side of that boundary -- staging an ingest's own outputs, and reporting where
the packaging got to. No provider call is ever made on this path: a Deep
Analysis upload is reused as it stands, and a Standard upload gets the
deterministic quality contract (``use_llm=False``), which is free.

The Viewer is a screen of this console's own front end now, so it reads the
documents and knowledge bases every other screen reads. There is no workspace
snapshot here any more, and no bulk "prepare everything" queue: the screen
queues the document a reader actually opened
(``POST /api/v1/documents/<id>/analysis``).

The packaging itself belongs to :mod:`components.viewer.analysis` -- one
background worker, one directory per content, a state file per document -- and
is not re-implemented here. This module decides *what* to ask it for.
"""

from __future__ import annotations

import json
import logging
from typing import Optional, Sequence

from chat_rag.components.viewer import analysis
from chat_rag.components.viewer import methods as viewer_methods

from .errors import InvalidRequest, NotFound, NotReady

logger = logging.getLogger("RAG.workspace")

#: The pipeline key the packaging worker resolves canonical units under. Fixed
#: rather than the browser's session id: the worker outlives the request that
#: queued it, and a per-session key would build a second pipeline per browser.
ANALYSIS_SESSION = "viewer-analysis"


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
    from chat_rag.components.parsers.canonical_units_store import find_cache_file, load_units

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


def _label_of(doc: dict, fallback: str) -> str:
    return (doc.get('metadata') or {}).get('original_filename') or doc.get('file_name') or fallback


def resume_incomplete() -> list:
    """Pick up whatever packaging a previous process was in the middle of."""
    return analysis.resume_incomplete()
