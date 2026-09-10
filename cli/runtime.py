"""Shared plumbing for the CLI: it borrows production wiring, never copies it.

Every knowledge base, pipeline and retriever the CLI touches is built by the
same functions the web app uses, so an evaluation run measures what the app
would actually return. Nothing here re-implements parsing, chunking or scoring.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

# Production wiring. The CLI composes the very same application the web
# adapter does -- one container, the same settings resolution, the same
# pipeline cache and the same managers -- and imports no web framework to get
# it.
from chat_rag.application.services import build_settings_for_kb, default_services  # noqa: F401
from chat_rag.components.provenance import git_sha  # noqa: F401  (re-exported)
from chat_rag.components.retriever import retrieval_capabilities
from chat_rag.utils import DocumentTracker

CLI_SESSION = "cli"

services = default_services()
kb_manager = services.kb_manager
gold_manager = services.gold_manager


def get_pipeline(session_id: str, kb_id: Optional[str] = None):
    """The application's own pipeline seam, so an evaluation measures what the
    console would actually return."""
    return services.get_pipeline(session_id, kb_id)


class CliError(Exception):
    """A problem the user can fix, reported without a traceback."""


# --------------------------------------------------------------- knowledge base


def resolve_kb(reference: str) -> Dict[str, Any]:
    """Find a knowledge base by id or by name."""
    if not reference:
        raise CliError("--kb is required (knowledge base id or name)")
    record = kb_manager.get(reference)
    if record:
        return {"kb_id": reference, **record}
    kb_id = kb_manager.find_by_name(reference)
    if kb_id:
        return {"kb_id": kb_id, **kb_manager.get(kb_id)}
    known = ", ".join(sorted(kb["name"] for kb in kb_manager.list())) or "none"
    raise CliError(f"No knowledge base named or numbered {reference!r}. Known: {known}")


def pipeline_for(kb: Dict[str, Any]):
    return get_pipeline(CLI_SESSION, kb["kb_id"])


# ------------------------------------------------------------------- retrieval


def capabilities_of(kb: Dict[str, Any]) -> Dict[str, Any]:
    """What this knowledge base's retriever can serve, from the app's own rules."""
    return retrieval_capabilities(pipeline_for(kb).hybrid_retriever)


def search(kb: Dict[str, Any], query: str, top_k: int, method: str) -> List[Any]:
    """Run production retrieval, refusing a method this retriever cannot serve.

    The capability rules are the same ones the review screen uses, so the CLI
    cannot accidentally run a dense search on a lexical-only profile.
    """
    pipeline = pipeline_for(kb)
    retriever = pipeline.hybrid_retriever
    capabilities = retrieval_capabilities(retriever)
    entry = next((m for m in capabilities["methods"] if m["name"] == method), None)
    if entry is None:
        raise CliError(f"Unknown retrieval method {method!r}")
    if not entry["available"]:
        raise CliError(
            f"'{method}' is not available here: "
            f"{entry.get('reason', 'unsupported')}\n"
            f"Available: {', '.join(m['name'] for m in capabilities['methods'] if m['available'])}"
        )
    if method == "bm25":
        return retriever.keyword_search(query, top_k)
    if method == "vector":
        return retriever.vector_search(query, top_k)
    return retriever.hybrid_search(query, top_k)


def default_method(kb: Dict[str, Any]) -> str:
    capabilities = retrieval_capabilities(pipeline_for(kb).hybrid_retriever)
    if not capabilities["default"]:
        raise CliError("This retriever offers no usable search method")
    return capabilities["default"]


# ----------------------------------------------------------------------- chunks


def _load_json(value: Optional[str], fallback):
    if not value:
        return fallback
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return fallback


@dataclass(frozen=True)
class ChunkView:
    """A stored chunk, flattened the way both the CLI and structural QA read it."""

    chunk_id: str
    text: str
    doc_id: str
    heading: Optional[str]
    section_paths: List[List[str]]
    pages: List[Any]
    unit_ids: List[str]
    token_count: int
    #: "metadata" when the chunker's own heading was stored, "section_title"
    #: when the knowledge base predates that and only the derived string is
    #: available. Heading-dependent QA rules over-report on the latter.
    heading_source: Optional[str] = None
    #: The display string derived from the section path, as stored.
    section_title: Optional[str] = None
    #: Everything else the store holds for this chunk, minus the serialised
    #: fields already expanded above.
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def of(cls, chunk) -> "ChunkView":
        metadata = chunk.metadata or {}
        # ``heading`` is the chunker's own; ``section_title`` is a display
        # string derived from the section path. Older knowledge bases were
        # ingested before the heading was persisted and only have the latter.
        stored = metadata.get("heading")
        heading = stored or chunk.section_title
        return cls(
            chunk_id=chunk.chunk_id,
            text=chunk.content or "",
            doc_id=chunk.doc_id or metadata.get("doc_id") or "",
            heading=heading,
            section_paths=_load_json(metadata.get("section_paths_json"), []),
            pages=_load_json(metadata.get("pages_json"), []),
            unit_ids=_load_json(metadata.get("unit_ids_json"), []),
            token_count=int(metadata.get("token_count") or 0),
            heading_source=("metadata" if stored else
                            ("section_title" if chunk.section_title else None)),
            section_title=chunk.section_title,
            extra={
                key: value for key, value in metadata.items()
                if key not in _EXPANDED_METADATA
            },
        )

    def as_qa_row(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "heading": self.heading,
            "section_paths": self.section_paths,
            "pages": self.pages,
            "unit_ids": self.unit_ids,
            "token_count": self.token_count,
        }

    def as_export_row(self) -> Dict[str, Any]:
        """The stored chunk, flattened for an exportable corpus file."""
        return {
            **self.as_qa_row(),
            "document_id": self.doc_id,
            "section_title": self.section_title,
            "metadata": self.extra,
        }


#: Serialised metadata keys ChunkView already exposes as fields; keeping them
#: in ``extra`` too would publish the same value under two names.
_EXPANDED_METADATA = frozenset({
    "heading", "section_paths_json", "pages_json", "unit_ids_json", "token_count",
})


def all_chunks(kb: Dict[str, Any]) -> List[ChunkView]:
    return [ChunkView.of(chunk) for chunk in pipeline_for(kb).vector_db.get_all_chunks()]


# ----------------------------------------------------------- canonical units


@dataclass(frozen=True)
class CanonicalMatch:
    """The canonical stream a stored corpus was demonstrably built from."""

    path: str
    units: List[Dict[str, Any]]
    verified_units: int


def _collapse(text: Any) -> str:
    return " ".join(str(text or "").split())


def resolve_canonical_units(
    views: Sequence[ChunkView], cache_dir: Optional[str] = None
) -> Optional[CanonicalMatch]:
    """Find the canonical file these chunks were built from, and prove it.

    The parser keys its cache by a hash of the PDF bytes, which cannot be
    recomputed once the uploaded temp file is gone, so a document has to be
    matched to its cache entry by what its chunks carry. Matching on unit ids
    alone is not enough: ids are positional, so a stream from an older parser
    version shares almost all of them and differs only where a unit was added
    or removed. Every candidate therefore has to reproduce the stored chunk
    text -- each referenced unit's text must appear verbatim in the chunk that
    cites it. An older stream fails that on the units after its first
    divergence, so a near-miss is rejected rather than silently exported.

    Returns ``None`` when nothing qualifies; the caller decides whether that
    is a warning or a failure.
    """
    try:
        from chat_rag.components.parsers.canonical_units_store import default_cache_dir, load_units
    except ImportError:
        return None

    directory = cache_dir or str(default_cache_dir())
    if not os.path.isdir(directory):
        return None

    wanted = {u.split("#", 1)[0] for view in views for u in view.unit_ids if u}
    if not wanted:
        return None

    best: Optional[CanonicalMatch] = None
    for name in sorted(os.listdir(directory)):
        if not name.endswith(".jsonl"):
            continue
        path = os.path.join(directory, name)
        try:
            rows = load_units(path)
        except Exception:
            continue
        texts = {row.get("unit_id"): _collapse(row.get("text")) for row in rows}
        if not wanted.issubset(texts):
            continue

        checked = 0
        for view in views:
            body = _collapse(view.text)
            for unit_id in view.unit_ids:
                # A fragment id names part of a unit, so the whole unit's text
                # is not expected to appear; the id still has to exist.
                if "#" in unit_id:
                    continue
                unit_text = texts.get(unit_id.split("#", 1)[0])
                if not unit_text:
                    continue
                if unit_text not in body:
                    checked = -1
                    break
                checked += 1
            if checked < 0:
                break
        if checked <= 0:
            continue

        # Several streams can agree; prefer the one with no unexplained
        # surplus, then the lexicographically first, so the answer is stable.
        candidate = CanonicalMatch(path=path, units=rows, verified_units=checked)
        if best is None or len(rows) < len(best.units):
            best = candidate
    return best


# -------------------------------------------------------------------- documents


def documents_for(kb: Dict[str, Any]) -> List[Dict[str, Any]]:
    return DocumentTracker().get_all_documents(kb_id=kb["kb_id"])


def document_sha(doc_id: str) -> Optional[str]:
    """The sha256 the ingest tracker already recorded. No file is re-read."""
    tracked = DocumentTracker().get_document_by_doc_id(doc_id)
    return (tracked or {}).get("file_hash") or None


# ------------------------------------------------------------------- gold sets


def load_gold(path: str) -> Dict[str, Any]:
    """Read a frozen gold set, tolerating the older bare-list shape."""
    if not os.path.isfile(path):
        raise CliError(f"Gold set not found: {path}")
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, list):
        return {"schema_version": 1, "entries": payload, "frozen_at": None}
    if not isinstance(payload, dict) or "entries" not in payload:
        raise CliError(f"{path} does not look like a gold set")
    return payload


def run_environment(kb: Dict[str, Any]) -> Dict[str, Any]:
    """What produced a set of numbers, from the same facts the manifest uses.

    Retrieval happens now, so this describes the pipeline as it is configured
    now. A report's manifest describes the corpus, and prefers the snapshot
    captured when that corpus was ingested; the two are different questions and
    a report names any disagreement between them.
    """
    from .manifest import pipeline_facts, version_facts

    facts = pipeline_facts(kb)
    return {
        "kb_id": kb["kb_id"],
        "kb_name": kb.get("name"),
        "chunker": facts["chunker"],
        "parser": facts["parser"],
        "parser_backend": facts["parser_backend"],
        "normalization_version": facts["normalization_version"],
        "retrieval_profile": facts["retrieval_profile"],
        "retriever": facts["retriever"],
        "uses_embeddings": facts["uses_embeddings"],
        "requires_document_embeddings": facts["requires_document_embeddings"],
        "embedding_model": facts["configured_embedding_model"],
        "vector_db_provider": facts["vector_db_provider"],
        # A snapshot captured before Step 9 named a directory; one captured
        # after names a collection. Whichever the record carries is what the
        # report shows.
        "vector_collection": facts.get("vector_collection")
        or facts.get("storage_path"),
        "git_sha": version_facts()["chat_rag_git_sha"],
    }
