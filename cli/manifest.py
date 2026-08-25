"""One description of what produced an output, preferring the ingest record.

A corpus is produced once and read for months, so the configuration worth
reporting is the one that ran, not the one installed today. A successful
ingest writes an immutable snapshot (``components.provenance``) and this module
reads it back. Only when a document predates that -- or was ingested by a path
that never captured one -- does it fall back to describing the live wiring, and
then it says so out loud rather than passing today's settings off as history.

The derivation itself lives in ``components.provenance`` precisely so ingest
and inspection cannot drift apart; what is here is the resolution, the
document-level facts, and the assembled manifest.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence

from components.provenance import snapshot as provenance

SCHEMA_VERSION = 1

#: Re-exported so callers and tests have one name for each of these.
FEATURE_SOURCES = provenance.FEATURE_SOURCES
IMPORTANT_DEPENDENCIES = provenance.IMPORTANT_DEPENDENCIES
features_of = provenance.features_of
version_facts = provenance.version_facts
version_warnings = provenance.version_warnings

#: Said whenever a manifest describes the pipeline as it is now rather than as
#: it was when the corpus was built.
FALLBACK_WARNING = (
    "ingest-time configuration unavailable; current runtime config shown"
)


def pipeline_facts(kb: Dict[str, Any]) -> Dict[str, Any]:
    """How this knowledge base is wired *right now*.

    Used for the retrieval side of a run, which happens now, and as the
    fallback for a document ingested before snapshots existed.
    """
    from . import runtime

    return provenance.pipeline_facts(
        runtime.pipeline_for(kb), kb,
        storage_path=runtime.kb_manager.storage_path(kb["kb_id"]),
    )


def current_features(kb: Dict[str, Any]) -> Dict[str, Optional[bool]]:
    from . import runtime

    return provenance.features_of(
        provenance.structured_parser(runtime.pipeline_for(kb))
    )


# ------------------------------------------------------------------ document


def _page_of(unit: Dict[str, Any]) -> Optional[int]:
    page = (unit.get("source") or {}).get("page")
    return int(page) if isinstance(page, (int, float)) else None


def pages_covered(units: Sequence[Dict[str, Any]], views: Sequence[Any]) -> int:
    """Distinct pages the extracted corpus actually covers.

    Lower than the document's page count when a page yielded nothing -- a
    cover, a divider -- which is worth seeing rather than rounding away.
    """
    pages = {p for p in (_page_of(unit) for unit in units) if p is not None}
    if not pages:
        pages = {p for view in views for p in view.pages if isinstance(p, int)}
    return len(pages)


def page_count(views: Sequence[Any], units: Sequence[Dict[str, Any]]) -> Optional[int]:
    """The document's own page count, as the parser recorded it at ingest.

    The uploaded file is not reopened: the parser already counted the pages and
    the count travelled into the chunk metadata. Falling back to the pages the
    canonical stream covers keeps a number available for corpora stored before
    that metadata existed.
    """
    for view in views:
        recorded = view.extra.get("page_count")
        if isinstance(recorded, int) and recorded > 0:
            return recorded
    return pages_covered(units, views) or None


def document_facts(
    document: Optional[Dict[str, Any]],
    units: Sequence[Dict[str, Any]],
    views: Sequence[Any],
) -> Optional[Dict[str, Any]]:
    if not document:
        return None
    document_metadata = document.get("metadata") or {}
    return {
        "document_id": document.get("doc_id"),
        # The stored name is a temp upload name; the one a person recognises
        # is the one the browser sent.
        "filename": document_metadata.get("original_filename")
        or document.get("file_name"),
        # Recorded at ingest. Hashing the file again here would prove nothing
        # about the corpus that is actually stored.
        "sha256": document.get("file_hash") or None,
        "pages": page_count(views, units),
        "file_size": document.get("file_size"),
        "ingested_at": document.get("ingested_at"),
    }


# ---------------------------------------------------------------- resolution


def resolve_configuration(
    kb: Dict[str, Any], document: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    """The ingest-time configuration if it was captured, else today's.

    Returns the ``pipeline``/``features``/``versions`` blocks together with a
    ``provenance`` block naming which of the two a reader is looking at, and
    the warnings that go with it.
    """
    stored = (document or {}).get("pipeline_snapshot")
    if provenance.is_usable(stored):
        return {
            "pipeline": stored["pipeline"],
            "features": stored.get("features") or {},
            "versions": stored.get("versions") or {},
            "provenance": {
                "source": "ingest-snapshot",
                "captured_at": stored.get("captured_at"),
                "snapshot_schema_version": stored.get("schema_version"),
                "document_sha256": stored.get("document_sha256"),
            },
            "warnings": [],
        }

    warnings = [FALLBACK_WARNING]
    if stored is not None:
        warnings.append(
            "a pipeline snapshot is stored for this document but this build "
            "cannot read it (schema "
            f"{stored.get('schema_version') if isinstance(stored, dict) else '?'})"
        )
    return {
        "pipeline": pipeline_facts(kb),
        "features": current_features(kb),
        "versions": version_facts(),
        "provenance": {
            "source": "current-runtime",
            "captured_at": None,
            "snapshot_schema_version": None,
            "document_sha256": None,
        },
        "warnings": warnings,
    }


# ------------------------------------------------------------------ manifest


def build_manifest(
    kb: Dict[str, Any],
    *,
    document: Optional[Dict[str, Any]] = None,
    units: Sequence[Dict[str, Any]] = (),
    views: Sequence[Any] = (),
    extra_warnings: Sequence[str] = (),
) -> Dict[str, Any]:
    """The canonical snapshot. Callers pass what they already loaded."""
    resolved = resolve_configuration(kb, document)
    versions = resolved["versions"]

    referenced = {
        unit_id.split("#", 1)[0] for view in views for unit_id in view.unit_ids
    }
    warnings = list(extra_warnings) + list(resolved["warnings"])
    warnings += provenance.version_warnings(versions)
    if resolved["pipeline"].get("parser") is None:
        warnings.append(
            "no structured PDF parser is recorded; canonical-stream features "
            "cannot be reported"
        )

    # A snapshot names the bytes it was captured against; if the document has
    # since been replaced under the same record, the snapshot describes a
    # different corpus and saying nothing would be the worst outcome.
    recorded_sha = resolved["provenance"].get("document_sha256")
    stored_sha = (document or {}).get("file_hash")
    if recorded_sha and stored_sha and recorded_sha != stored_sha:
        warnings.append(
            f"the ingest snapshot was captured against {recorded_sha[:12]} but "
            f"the tracked document is {stored_sha[:12]}"
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "pipeline-manifest",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "knowledge_base": {"kb_id": kb["kb_id"], "name": kb.get("name")},
        "document": document_facts(document, units, views),
        "provenance": resolved["provenance"],
        "pipeline": resolved["pipeline"],
        "counts": {
            "canonical_units": len(units),
            "chunks": len(views),
            "canonical_units_referenced_by_chunks": len(referenced),
            "pages_with_canonical_units": pages_covered(units, views),
        },
        "features": resolved["features"],
        "versions": versions,
        "warnings": warnings,
    }
