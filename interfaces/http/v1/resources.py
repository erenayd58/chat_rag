"""What each product concept looks like on the wire.

One projection per resource, in one file, because the projections *are* the
contract: the field names, the identity of each thing, and -- just as much --
what is deliberately not there. The current implementation stores a document
against an absolute file path, a knowledge base against a Chroma directory and
an analysis against a directory named for a content hash. None of that reaches
a client, so none of it has to survive the move to PostgreSQL and pgvector.

Two identities are kept apart everywhere below, because conflating them is the
single most expensive mistake a schema can make:

    ``id``          the upload. One file, ingested into one knowledge base.
    ``content_id``  the bytes. Shared by every upload of the same document,
                    and what an analysis and its variants actually belong to.

Two fields are pass-through and say so in the API doc: a chunk's ``metadata``
and a query's ``diagnostics``. They carry what the product produced, they are
useful, and they are explicitly not contractual -- pinning them would freeze
internals this contract exists to leave free.
"""

from __future__ import annotations

from typing import Any, Optional

#: The analysis states a document can be in. The same five the packager uses;
#: named here because they are part of the contract.
ANALYSIS_STATES = ("missing", "pending", "running", "ready", "failed")


def knowledge_base(record: dict) -> dict:
    """A named collection with its own chunker and its own corpus.

    No storage location and no store provider: where the vectors live is a
    deployment's business, and it is the thing the pgvector migration changes.
    The embedding model *is* here -- it is a product fact, because the vectors
    belong to it and re-indexing is how you change it.
    """
    return {
        "id": record.get("kb_id"),
        "name": record.get("name"),
        "chunker": record.get("chunker") or {},
        "retrieval_method": record.get("retrieval_method"),
        "embedding_model": record.get("embedding_model_name"),
        "extra": record.get("extra") or {},
    }


def chunking_method(entry: dict) -> dict:
    """One chunking method, exactly as the library's registry describes it.

    Nothing is added and nothing is filtered: this is a projection of the
    registry's own record, which is what makes a newly registered method
    appear here with no edit in this repository. ``available`` is this
    machine's answer, not the library's -- a method needing an embedder it
    cannot load is offered as unavailable with the reason, never hidden.
    """
    return {
        "key": entry["key"],
        "label": entry["label"],
        "summary": entry["summary"],
        "engine": entry["engine"],
        "available": entry["available"],
        "unavailable_reason": entry.get("reason") or None,
        "uses_model": entry["uses_model"],
        "default": entry["default"],
        # An orchestration runs over a baseline partition rather than being
        # one; a picker that shows the two the same way misleads.
        "orchestration": bool(entry.get("orchestration")),
        "baseline": entry.get("baseline"),
    }


def analysis(state: dict) -> dict:
    """Where one upload's chunking analysis got to, truthfully.

    The two levels are kept apart on the wire because they are two different
    facts, and a screen that merges them lies in one direction or the other:

    * ``selected_methods`` / ``ready_methods`` are **this upload's**. Ready is
      the intersection of what it selected with what has been built -- the
      ``visible = selected ∩ ready`` rule -- so a client can render exactly
      what this document may be asked about and nothing else.
    * ``content`` is the **shared analysis**: every variant these bytes have,
      reusable by any upload of them, and the other uploads that share it.
    """
    status = state.get("status") or "missing"
    selected = list(state.get("selected_methods") or [])
    return {
        "status": status,
        "content_id": state.get("key"),
        "selected_methods": selected,
        "ready_methods": list(state.get("available_methods") or []),
        "failed_methods": [m for m in (state.get("failed_methods") or []) if m in selected],
        "unit_count": state.get("unit_count"),
        "deep_source": state.get("deep_source"),
        "error": state.get("error") or None,
        "updated_at": state.get("updated_at"),
        "content": {
            "requested_methods": list(state.get("requested") or []),
            "ready_methods": list(state.get("ready_methods") or []),
            "shared_with_document_ids": list(state.get("doc_ids") or []),
        },
    }


def document(record: dict, *, state: Optional[dict] = None) -> dict:
    """One ingest: one file, in exactly one knowledge base.

    ``file_path`` is the ledger's key and never leaves the server -- it names
    a staging directory on the machine that ran the upload, and it is the
    field a PostgreSQL schema replaces with a row id.
    """
    body = {
        "id": record.get("doc_id"),
        "knowledge_base_id": record.get("kb_id"),
        "name": (record.get("metadata") or {}).get("original_filename")
                or record.get("file_name") or "",
        # The bytes, not the upload: two documents with one content_id are the
        # same PDF ingested twice.
        "content_id": (record.get("file_hash") or "") or None,
        "size_bytes": record.get("file_size") or 0,
        "chunk_count": record.get("chunk_count") or 0,
        "chunking_mode": record.get("chunking_mode"),
        "status": record.get("status") or "indexed",
        "ingested_at": record.get("ingested_at") or None,
        "ingest_job_id": (record.get("metadata") or {}).get("ingest_job_id"),
    }
    if state is not None:
        body["analysis"] = analysis(state)
    return body


def ingest_job(record: dict) -> dict:
    """A submitted upload, and what became of it.

    A job id outlives the process that minted it: the journal records every
    transition and start-up settles anything in flight against the ledger, so
    ``restart_settled`` is how a client tells "it finished while you were away"
    from "it never happened". ``document_id`` is filled in the moment the
    ledger knows the document, which is the job's last act.
    """
    return {
        "id": record.get("job_id"),
        "status": record.get("status"),
        "knowledge_base_id": record.get("kb_id"),
        "document_id": record.get("doc_id"),
        "content_id": record.get("content_sha256"),
        "name": record.get("filename"),
        "methods": list(record.get("methods") or []),
        "queue_position": record.get("position"),
        "attached_uploads": record.get("attached_uploads") or 0,
        "submitted_at": record.get("submitted_at"),
        "started_at": record.get("started_at"),
        "finished_at": record.get("finished_at"),
        "run_seconds": record.get("run_seconds"),
        "restart_settled": bool(record.get("restart_recovered")),
        "error": _job_error(record),
        "result": _job_result(record),
    }


def _job_error(record: dict) -> Optional[dict]:
    if not record.get("error"):
        return None
    return {"type": record.get("error_category") or "internal",
            "message": record.get("error")}


def _job_result(record: dict) -> Optional[dict]:
    result = record.get("result")
    if not result:
        return None
    return {
        "document_id": result.get("doc_id"),
        "chunk_count": result.get("chunks_created"),
        "chunking_mode": result.get("chunking_mode"),
        "deep_analysis": result.get("deep_analysis"),
        "analysis_status": result.get("viewer_analysis"),
    }


def chunk(row: dict) -> dict:
    """One stored chunk, as an inspection screen reads it.

    ``content`` is the document's own text, verbatim: it is what a citation
    quotes, and a store may index whatever it likes but may not hand back a
    re-rendered version. ``metadata`` is pass-through and not contractual.
    """
    metadata = row.get("metadata") or {}
    return {
        "id": row.get("chunk_id"),
        "document_id": metadata.get("doc_id"),
        "content": row.get("content") or "",
        "chunk_index": metadata.get("chunk_index"),
        "total_chunks": metadata.get("total_chunks"),
        "section": metadata.get("section_title") or metadata.get("heading"),
        "chunking_mode": metadata.get("chunking_mode"),
        "metadata": metadata,
    }


def scored_chunk(row: dict, *, method: str) -> dict:
    """A chunk a search returned, with the score that ranked it.

    ``score`` is comparable only within one answer: it is whatever the named
    method produced, and nothing here rescales it into a pretence of a
    universal number.
    """
    body = chunk(row)
    body["score"] = row.get("score", row.get("similarity_score"))
    body["retrieval_method"] = row.get("retrieval_method") or method
    return body


def canonical_unit(row: dict) -> dict:
    """One unit of the parser's canonical reading of a document, before any
    chunker saw it."""
    return {
        "id": row.get("unit_id"),
        "order": row.get("order"),
        "type": row.get("type"),
        "text": row.get("text") or "",
        "heading_level": row.get("heading_level"),
        "section_path": row.get("section_path") or [],
        "source": row.get("source") or {},
    }


def citation(source: dict) -> dict:
    """One source an answer was allowed to use, and whether it used it.

    ``label`` is what the answer cites (``[S1]``), ``used`` says whether the
    text actually cited it, and ``content`` is the chunk verbatim so a client
    can show the passage rather than a summary of it.
    """
    return {
        "label": source.get("label"),
        "chunk_id": source.get("chunk_id"),
        "document_id": source.get("doc_id"),
        "document": source.get("document"),
        "section": source.get("heading") or source.get("section"),
        "pages": source.get("pages") or [],
        "chunking_mode": source.get("chunking_mode"),
        "used": bool(source.get("used")),
        "score": source.get("score"),
        "content": source.get("content") or source.get("content_preview") or "",
    }


def answer(result: dict, *, knowledge_base_id: Optional[str]) -> dict:
    """One answered question.

    ``diagnostics`` is the pipeline's own metadata, passed through and
    explicitly not contractual: it is how a retrieval change is investigated,
    and freezing it would freeze the internals this contract exists to leave
    free. Everything above it is the part a client may depend on.
    """
    metadata: dict[str, Any] = result.get("metadata") or {}
    generation = metadata.get("answer") or {}
    timing = metadata.get("query") or {}
    return {
        "answer": result.get("answer") or "",
        "citations": [citation(source) for source in (result.get("sources") or [])],
        "knowledge_base_id": knowledge_base_id,
        "retrieval_method": metadata.get("retrieval_method"),
        # True when the answer actually cited at least one of the sources it
        # was given. A client showing an answer without this cannot tell a
        # grounded answer from a guess.
        "grounded": bool(generation.get("grounded")),
        "timing": {
            "query_id": timing.get("query_id"),
            "total_seconds": timing.get("total_seconds"),
            "stages": timing.get("stages") or {},
        },
        "diagnostics": metadata,
    }
