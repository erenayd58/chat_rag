"""What the pipeline was when a document was ingested.

A corpus is produced once and read for months. Describing it with today's
configuration answers the wrong question: the parser flags, the chunker and the
library versions that matter are the ones that ran, not the ones installed now.
So a successful ingest captures an immutable snapshot of them and it travels
with the document.

This lives in ``components`` rather than in the CLI because both sides need the
same derivation: ingest writes the snapshot, and inspection falls back to
deriving the same shape from the live objects when a document predates it.
Deriving it twice, in two places, is exactly how the two would drift apart.

Nothing here parses, chunks or searches. It reads what the already-built
objects say about themselves.
"""

from __future__ import annotations

import importlib.metadata as metadata
import json
import os
import platform
import subprocess
import sys
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

SCHEMA_VERSION = 1

#: Dependencies that can change what the pipeline produces. Deliberately not a
#: dump of the environment: a hundred pinned packages hide the eight that
#: matter.
IMPORTANT_DEPENDENCIES = (
    "amsc-poc",
    "pymupdf4llm",
    "pymupdf",
    "pgvector",
    "rank-bm25",
    "sentence-transformers",
    "tiktoken",
    "onnxruntime",
)

#: The canonical-stream repairs, read off the parser that is actually
#: registered rather than restated here. A flag that is renamed or removed in
#: the parser shows up as ``null`` instead of quietly reporting the old value.
FEATURE_SOURCES: Dict[str, Callable[[Any], Optional[bool]]] = {
    # Spread pages are read column-major; that ordering is the profile's.
    "column_order": lambda p: bool(
        getattr(getattr(p, "_spread_profile", None), "reading_order", None)
    ),
    "running_headers": lambda p: bool(getattr(p, "RUNNING_HEADER_MIN_PAGES", 0)),
    "visual_grid": lambda p: bool(getattr(p, "RECONSTRUCT_VISUAL_GRIDS", False)),
    "lead_in_headings": lambda p: bool(getattr(p, "DEMOTE_LEAD_IN_HEADINGS", False)),
    "numbered_headings": lambda p: bool(getattr(p, "PROMOTE_MISSED_HEADINGS", False)),
    "table_captions": lambda p: bool(getattr(p, "DEMOTE_CAPTION_HEADINGS", False)),
    "split_headings": lambda p: bool(getattr(p, "REJOIN_SPLIT_HEADINGS", False)),
    "sentence_headings": lambda p: bool(getattr(p, "DEMOTE_SENTENCE_HEADINGS", False)),
}


# ------------------------------------------------------------------ versions


#: Set at build time where the checkout is not available -- an image does not
#: ship .git, so a container would otherwise have to report its own commit as
#: unknown.
GIT_SHA_ENV = "CHAT_RAG_GIT_SHA"


def git_sha(path: Optional[str] = None) -> Optional[str]:
    """HEAD of the repository at ``path`` (the working directory by default)."""
    if path is None:
        stamped = (os.getenv(GIT_SHA_ENV) or "").strip()
        if stamped:
            return stamped
    command = ["git"] + (["-C", path] if path else []) + ["rev-parse", "HEAD"]
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, timeout=10, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def _dependency_versions() -> Dict[str, Optional[str]]:
    found: Dict[str, Optional[str]] = {}
    for name in IMPORTANT_DEPENDENCIES:
        try:
            found[name] = metadata.version(name)
        except Exception:
            # Absent or unreadable is a fact about the environment, not a
            # reason to fail: report it as null.
            found[name] = None
    return found


def _amsc_provenance() -> Dict[str, Optional[str]]:
    """Which build of the chunking library is installed, and from where.

    A wheel built from a pinned GitHub commit records that commit; an editable
    install records a directory, and its commit has to be read from that
    checkout instead.
    """
    info: Dict[str, Optional[str]] = {
        "amsc_version": None,
        "amsc_source": None,
        "amsc_git_sha": None,
    }
    try:
        info["amsc_version"] = metadata.version("amsc-poc")
    except Exception:
        pass

    try:
        raw = metadata.distribution("amsc-poc").read_text("direct_url.json")
        direct_url = json.loads(raw) if raw else {}
    except Exception:
        direct_url = {}

    commit = (direct_url.get("vcs_info") or {}).get("commit_id")
    if commit:
        info["amsc_source"] = direct_url.get("url")
        info["amsc_git_sha"] = commit
        return info

    try:
        import amsc

        package_dir = os.path.dirname(os.path.abspath(amsc.__file__))
    except Exception:
        return info
    info["amsc_source"] = direct_url.get("url") or package_dir
    info["amsc_git_sha"] = git_sha(package_dir)
    return info


def version_facts() -> Dict[str, Any]:
    facts: Dict[str, Any] = {
        "chat_rag_git_sha": git_sha(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "important_dependencies": _dependency_versions(),
    }
    facts.update(_amsc_provenance())
    return facts


def version_warnings(versions: Dict[str, Any]) -> List[str]:
    """A missing commit is reported, never inferred and never fatal."""
    warnings: List[str] = []
    for key, what in (("chat_rag_git_sha", "chat_rag"), ("amsc_git_sha", "amsc")):
        if not versions.get(key):
            warnings.append(f"{what} git commit could not be determined")
    missing = [
        name
        for name, version in (versions.get("important_dependencies") or {}).items()
        if not version
    ]
    if missing:
        warnings.append("version unavailable for: " + ", ".join(missing))
    return warnings


# ------------------------------------------------------------------ pipeline


def structured_parser(pipeline: Any) -> Any:
    """The parser the pipeline would actually use for a PDF, or None."""
    factory = getattr(pipeline, "parser_factory", None)
    if factory is None or not callable(getattr(factory, "get_parser", None)):
        return None
    try:
        return factory.get_parser("document.pdf")
    except Exception:
        return None


def features_of(parser: Any) -> Dict[str, Optional[bool]]:
    if parser is None:
        return {name: None for name in FEATURE_SOURCES}
    resolved: Dict[str, Optional[bool]] = {}
    for name, read in FEATURE_SOURCES.items():
        try:
            resolved[name] = read(parser)
        except Exception:
            resolved[name] = None
    return resolved


def pipeline_facts(
    pipeline: Any, kb: Dict[str, Any], *, vector_collection: Optional[str] = None
) -> Dict[str, Any]:
    """How a pipeline is wired, read from the objects themselves."""
    retriever = getattr(pipeline, "hybrid_retriever", None)
    parser = structured_parser(pipeline)

    # The retriever's declared contract, and whether this configuration
    # therefore computes any embeddings. A model can be configured and never
    # used -- on the lexical profile it always is -- so the two are separate
    # facts and the model name is reported either way.
    requires = bool(getattr(retriever, "requires_document_embeddings", True))
    configured_model = (kb or {}).get("embedding_model_name")

    return {
        "parser": parser.get_name() if parser is not None else None,
        "parser_backend": getattr(parser, "parser_backend", None),
        "normalization_version": getattr(parser, "NORMALIZATION_VERSION", None),
        "chunker": ((kb or {}).get("chunker") or {}).get("type"),
        "retrieval_profile": getattr(pipeline, "retrieval_profile", None),
        "retriever": type(retriever).__name__ if retriever is not None else None,
        "uses_embeddings": bool(requires and configured_model),
        "requires_document_embeddings": requires,
        "configured_embedding_model": configured_model,
        "vector_db_provider": (kb or {}).get("vector_db_provider"),
        # Which collection in the vector store this corpus is in. It was a
        # filesystem path until Step 9 and is a key now, so the field was
        # renamed rather than quietly redefined: a snapshot captured before
        # that has ``storage_path`` and no ``vector_collection``, and the
        # report reads whichever it finds instead of pretending a directory
        # name is a collection.
        "vector_collection": vector_collection,
    }


# ------------------------------------------------------------------ capture


def build_snapshot(
    pipeline: Any,
    kb: Dict[str, Any],
    *,
    kb_id: Optional[str] = None,
    vector_collection: Optional[str] = None,
) -> Dict[str, Any]:
    """Capture the configuration that just produced a corpus.

    ``document_sha256`` is left unset here and stamped by the tracker from the
    hash it already computes for the file, so the snapshot names the bytes it
    describes without the file being read a second time.

    Capturing must never be able to fail an ingest that otherwise succeeded, so
    every field degrades to ``None`` rather than raising.
    """
    parser = structured_parser(pipeline)
    return {
        "schema_version": SCHEMA_VERSION,
        "captured_at": datetime.now().isoformat(timespec="seconds"),
        "kb_id": kb_id,
        "document_sha256": None,
        "pipeline": pipeline_facts(pipeline, kb,
                                   vector_collection=vector_collection),
        "features": features_of(parser),
        "versions": version_facts(),
    }


def capture(
    pipeline: Any,
    kb: Dict[str, Any],
    *,
    kb_id: Optional[str] = None,
    vector_collection: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """``build_snapshot`` that returns None instead of propagating a failure.

    An ingest that produced chunks has succeeded. Losing the description of how
    it did so is a gap in provenance -- reported later as an unavailable
    snapshot -- not a reason to discard the work.
    """
    try:
        return build_snapshot(
            pipeline, kb, kb_id=kb_id, vector_collection=vector_collection
        )
    except Exception:
        return None


def is_usable(snapshot: Any) -> bool:
    """Whether a stored snapshot can be read by this version of the code."""
    return (
        isinstance(snapshot, dict)
        and snapshot.get("schema_version") == SCHEMA_VERSION
        and isinstance(snapshot.get("pipeline"), dict)
    )
