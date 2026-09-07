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

The one thing this module does beyond resolving names is decide which
*sources* of a path setting are allowed to speak, which is the subject of
``load_env_file`` below. Nothing here creates directories or parses documents.
"""

from __future__ import annotations

import os
import tempfile
from typing import Dict, List, Optional

#: Set this to gather all runtime state under one directory. Unset means the
#: historical, working-directory-relative layout.
DATA_DIR_ENV = "CHAT_RAG_DATA_DIR"

#: The fallback vector store, used when no knowledge base is selected.
VECTOR_DB_PATH_ENV = "VECTOR_DB_PATH"

#: The structured parser's canonical-unit cache.
PARSER_CACHE_ENV = "STRUCTURED_PARSER_CACHE"

#: Variables that name where runtime *state* lives, as opposed to which model
#: to call or how big a chunk is. These are the ones a data root owns.
STATE_PATH_ENV = (VECTOR_DB_PATH_ENV, PARSER_CACHE_ENV)

#: Settings the .env file supplied, and the state paths it was not allowed to
#: supply because a data root was already declared. Diagnostics only.
_from_env_file: Dict[str, str] = {}
_ignored_from_env_file: Dict[str, str] = {}


def data_root() -> Optional[str]:
    value = (os.getenv(DATA_DIR_ENV) or "").strip()
    return value or None


# --------------------------------------------------------------- the contract


def load_env_file(path: str) -> Dict[str, str]:
    """Apply a ``.env`` file, without letting it move a declared data root.

    The precedence, highest first:

    1. the real process environment -- what a container, a compose file, a
       test or an operator's shell actually set;
    2. this file, for everything that is not a state path;
    3. this file, for state paths, but *only* when no data root is declared.

    Rule 3 is the whole point. ``.env`` is a developer's local file and it
    describes the developer's local layout: ``VECTOR_DB_PATH=./chroma_db``
    means "the store in my checkout". A deployment, a smoke check or a test
    that declares ``CHAT_RAG_DATA_DIR`` has said where its state lives, and
    a file left over from local development must not quietly move it back --
    which is exactly how a smoke run came to open the developer's real Chroma
    store. An operator who genuinely wants a store outside the data root still
    has rule 1: set the variable in the environment, where it is visible.

    Returns the settings that were applied, and records the ones that were
    refused (see :func:`diagnostics`).
    """
    from dotenv import dotenv_values

    _from_env_file.clear()
    _ignored_from_env_file.clear()
    if not os.path.isfile(path):
        return {}

    values = {k: v for k, v in dotenv_values(path).items() if v is not None}
    # The data root decides how every other key is treated, so it is applied
    # before them rather than in file order.
    ordered = ([DATA_DIR_ENV] if DATA_DIR_ENV in values else []) + [
        key for key in values if key != DATA_DIR_ENV
    ]

    for key in ordered:
        if key in os.environ:
            continue  # rule 1
        if key in STATE_PATH_ENV and data_root() is not None:
            _ignored_from_env_file[key] = values[key]  # rule 3
            continue
        os.environ[key] = values[key]
        _from_env_file[key] = values[key]
    return dict(_from_env_file)


def diagnostics() -> List[str]:
    """Human-readable lines about anything surprising in the path setup.

    Printed by the entrypoints at start-up, so "which store am I on" is
    answered by the log rather than by reading two files and an env dump.
    """
    lines: List[str] = []
    root = data_root()
    for key, value in sorted(_ignored_from_env_file.items()):
        lines.append(
            f"{key}={value} in .env ignored: {DATA_DIR_ENV}={root} owns this path"
        )
    override = (os.getenv(VECTOR_DB_PATH_ENV) or "").strip()
    if root and override and not _within(override, root):
        lines.append(
            f"{VECTOR_DB_PATH_ENV}={override} is outside {DATA_DIR_ENV}={root}; "
            "the fallback store will not travel with the data directory"
        )
    return lines


def _within(path: str, root: str) -> bool:
    try:
        return os.path.commonpath(
            [os.path.abspath(path), os.path.abspath(root)]
        ) == os.path.abspath(root)
    except ValueError:  # different drives on Windows
        return False


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


def vector_store_root() -> str:
    """The directory the per-knowledge-base stores sit under."""
    return _resolve("chroma", "./chroma_db")


def fallback_vector_store() -> str:
    """The store used when no knowledge base is selected.

    ``VECTOR_DB_PATH`` names it outright when set -- the one path override
    this application has always had. It is resolved here rather than read
    straight out of the environment in ``Settings`` so that one place decides
    it, and so :func:`load_env_file` can keep a stale ``.env`` from supplying
    it. Per-knowledge-base stores are unaffected: they come from
    :func:`vector_store`, which this override has never applied to.
    """
    override = (os.getenv(VECTOR_DB_PATH_ENV) or "").strip()
    return override or vector_store_root()


def vector_store(kb_id: Optional[str] = None) -> str:
    """Where one knowledge base keeps its vectors, by default.

    A knowledge base with an explicit ``vector_db_path`` overrides this; the
    default is what both the pipeline builder and the deletion guard resolve,
    and they must agree or a store is orphaned.
    """
    root = vector_store_root()
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
    """Where this console packages its documents for the Viewer.

    One directory per ingested document, holding the canonical units the
    chunking ran on, the packaged Deep Analysis arm and the viewer payload
    built from them. Regenerable from an ingest; never a frozen artifact.
    """
    return _resolve("viewer-live", "./artifacts/viewer-live")


def boundary_embedding_cache() -> str:
    """Per-text vector cache of the semantic boundary model.

    Kept apart from the retrieval embedding cache on purpose: the two use
    different models for different jobs, and one must never answer for the
    other.
    """
    return _resolve("cache/boundary-embeddings", ".cache/boundary-embeddings")


def embedding_cache() -> str:
    """Per-text vector cache of the OpenAI-compatible embedding provider
    (one ``.npy`` per exact text, per model). Regenerable."""
    return _resolve("cache/embeddings", ".cache/embeddings")


def ingest_journal() -> str:
    """Where ingest jobs write the record a restart answers from.

    Small JSON files, one per job, pruned on the same retention window as the
    in-memory registry (``components/ingest/journal.py``). It travels with the
    data root when one is set, because a client's ``job_id`` should survive a
    container restart the same way its documents do.
    """
    return _resolve("state/ingest-jobs", ".ingest-jobs")


def upload_staging() -> str:
    """Where uploaded files wait for their ingest job.

    An upload used to live in the system temp directory for exactly one
    request. An ingest job outlives the request that submitted it, so the file
    has to outlive it too, and it has to be somewhere a restart can sweep: a
    job exists only in memory, so any file still here when the process starts
    belongs to no job and is removed. With a data root the directory sits
    under it; without one it is a subdirectory of the system temp directory,
    resolved at call time so a test can point ``tempfile`` elsewhere.
    """
    root = data_root()
    if root:
        return os.path.join(root, "uploads")
    return os.path.join(tempfile.gettempdir(), "chat_rag-uploads")
