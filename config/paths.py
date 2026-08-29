"""Where runtime state lives.

Every persistent thing this application writes -- the knowledge base records,
the ingest ledger, the runtime gold set, the vector stores, the logs -- has
always been a path relative to the working directory. That is right for local
development and wrong for a container, where the source tree is rebuilt on
every image change and only a mounted directory survives.

Rather than move those paths, this module adds one switch. With
``CHAT_RAG_DATA_DIR`` unset every function returns exactly the path the code
used before, so a local checkout keeps writing to the same files it always
has. With it set -- which is what the compose file does -- the same state is
gathered under one directory that can be mounted.

Nothing here creates directories or reads files; it only resolves names.
"""

from __future__ import annotations

import os
from typing import Optional

#: Set this to gather all runtime state under one directory. Unset means the
#: historical, working-directory-relative layout.
DATA_DIR_ENV = "CHAT_RAG_DATA_DIR"


def data_root() -> Optional[str]:
    value = (os.getenv(DATA_DIR_ENV) or "").strip()
    return value or None


def _resolve(relative: str, default: str) -> str:
    root = data_root()
    return os.path.join(root, *relative.split("/")) if root else default


def knowledge_bases() -> str:
    return _resolve("state/knowledge_bases.json", "./.knowledge_bases.json")


def ingested_documents() -> str:
    return _resolve("state/ingested_documents.json", ".ingested_documents.json")


def gold_set() -> str:
    return _resolve("state/gold_set.json", "./.gold_set.json")


def logs() -> str:
    return _resolve("logs", "logs")


def vector_store_root(provider: str = "chroma") -> str:
    """The directory a provider's per-knowledge-base stores sit under."""
    name = "faiss_db" if provider == "faiss" else "chroma_db"
    return _resolve("faiss" if provider == "faiss" else "chroma", f"./{name}")


def vector_store(provider: str = "chroma", kb_id: Optional[str] = None) -> str:
    """Where one knowledge base keeps its vectors, by default.

    A knowledge base with an explicit ``vector_db_path`` overrides this; the
    default is what both the pipeline builder and the deletion guard resolve,
    and they must agree or a store is orphaned.
    """
    root = vector_store_root(provider)
    return os.path.join(root, kb_id) if kb_id else root


def canonical_cache() -> str:
    """The parser's canonical-unit cache.

    Already selectable with ``STRUCTURED_PARSER_CACHE``, which the parser reads
    directly; this mirrors the same default so a data directory covers it too.
    """
    configured = (os.getenv("STRUCTURED_PARSER_CACHE") or "").strip()
    if configured:
        return configured
    return _resolve("cache/canonical-units", ".cache/canonical-units")


def viewer_live_analysis() -> str:
    """Where this console packages its documents for the Viewer v2.

    One directory per ingested document, holding the canonical units the
    chunking ran on, the packaged Deep Analysis arm and the viewer payload
    built from them. Regenerable from an ingest; never a frozen artifact.
    """
    return _resolve("viewer-live", "./artifacts/viewer-live")


def embedding_cache() -> str:
    """Per-text vector cache of the OpenAI-compatible embedding provider
    (one ``.npy`` per exact text, per model). Regenerable."""
    return _resolve("cache/embeddings", ".cache/embeddings")
