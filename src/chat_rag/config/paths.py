"""Where the *files* this application writes live.

Since Step 9 that is a smaller set again. The knowledge bases, the ingest
ledger, the gold set, the ingest journal, the Viewer's analysis records and --
as of this step -- the chunks and their embeddings are all rows in PostgreSQL
(``config/database.py`` owns that connection string). What is left here is
what is genuinely a file: the parser's canonical-unit cache, the packaged
Viewer artifacts, the embedding caches, the upload staging directory and the
logs.

Every one of those has always been a path relative to the working directory.
That is right for local development and wrong for a container, where the
source tree is rebuilt on every image change and only a mounted directory
survives.

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
import sys
import tempfile
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional

#: Set this to gather all runtime state under one directory. Unset means the
#: historical, working-directory-relative layout.
DATA_DIR_ENV = "CHAT_RAG_DATA_DIR"

#: The structured parser's canonical-unit cache.
PARSER_CACHE_ENV = "STRUCTURED_PARSER_CACHE"

#: Variables that name where runtime *state* lives, as opposed to which model
#: to call or how big a chunk is. These are the ones a data root owns.
#:
#: One name, since Step 9 took the vector store out of the filesystem. The
#: rule below is still worth keeping for it: a developer's ``.env`` describes
#: a checkout, and a deployment that declared a data root has already said
#: where its state goes.
STATE_PATH_ENV = (PARSER_CACHE_ENV,)

#: Settings the .env file supplied, and the state paths it was not allowed to
#: supply because a data root was already declared. Diagnostics only.
_from_env_file: Dict[str, str] = {}
_ignored_from_env_file: Dict[str, str] = {}


@dataclass(frozen=True)
class PathSettings:
    """Where this process keeps its files, as a value rather than a lookup.

    Two settings decide every path below: a data root that gathers all of them
    under one mountable directory, and one explicit override for the parser
    cache. Everything else is derived, so a caller configures a layout by
    constructing this rather than by arranging the environment a module will
    read later.

    Frozen, and built by nothing at import: :func:`paths_from_env` is the only
    thing in this repository that turns an environment into one of these.
    """

    #: Set this to gather all runtime state under one directory. ``None``
    #: means the historical, working-directory-relative layout -- which is
    #: what a local checkout has always had, and what every default below
    #: falls back to.
    data_root: Optional[str] = None
    #: The structured parser's canonical-unit cache, when it is named
    #: explicitly. It is the one path that may sit outside the data root, and
    #: :func:`diagnostics` says so when it does.
    parser_cache: Optional[str] = None

    # ----------------------------------------------------------- derivation
    def _resolve(self, relative: str, default: str) -> str:
        root = self.data_root
        return os.path.join(root, *relative.split("/")) if root else default

    # ------------------------------------------------------ pre-Step-8 state
    #
    # These four named the JSON files that held the knowledge bases, the
    # ingest ledger, the gold set and the ingest journal. PostgreSQL holds all
    # four now (``storage/``), and nothing in the application writes to these
    # paths any more. They are kept for one real job:
    # ``tools/import_legacy_state.py`` finds an existing installation's files
    # here, and the record stores keep the value so a diagnostic can still say
    # which layout a process was configured for.

    def knowledge_bases(self) -> str:
        return self._resolve("state/knowledge_bases.json", "./.knowledge_bases.json")

    def ingested_documents(self) -> str:
        return self._resolve("state/ingested_documents.json", ".ingested_documents.json")

    def gold_set(self) -> str:
        return self._resolve("state/gold_set.json", "./.gold_set.json")

    def ingest_journal(self) -> str:
        """Where ingest jobs used to write the record a restart answers from.

        Small JSON files, one per job. The ``ingest_jobs`` table holds them
        since Step 8; this is kept alongside the three above so an existing
        installation can be imported, and so the journal can report what it
        was configured for.
        """
        return self._resolve("state/ingest-jobs", ".ingest-jobs")

    # ------------------------------------------------------------- the rest
    def logs(self) -> str:
        return self._resolve("logs", "logs")

    def canonical_cache(self) -> str:
        """The parser's canonical-unit cache.

        An explicit :attr:`parser_cache` wins outright; otherwise this mirrors
        the same default the parser has always used, so a data directory
        covers it too.
        """
        configured = (self.parser_cache or "").strip()
        if configured:
            return configured
        return self._resolve("cache/canonical-units", ".cache/canonical-units")

    def viewer_live_analysis(self) -> str:
        """Where this console packages its documents for the Viewer.

        One directory per ingested document, holding the canonical units the
        chunking ran on, the packaged Deep Analysis arm and the viewer payload
        built from them. Regenerable from an ingest; never a frozen artifact.
        """
        return self._resolve("viewer-live", "./artifacts/viewer-live")

    def boundary_embedding_cache(self) -> str:
        """Per-text vector cache of the semantic boundary model.

        Kept apart from the retrieval embedding cache on purpose: the two use
        different models for different jobs, and one must never answer for the
        other.
        """
        return self._resolve("cache/boundary-embeddings", ".cache/boundary-embeddings")

    def embedding_cache(self) -> str:
        """Per-text vector cache of the OpenAI-compatible embedding provider
        (one ``.npy`` per exact text, per model). Regenerable."""
        return self._resolve("cache/embeddings", ".cache/embeddings")

    def upload_staging(self) -> str:
        """Where uploaded files wait for their ingest job.

        An upload used to live in the system temp directory for exactly one
        request. An ingest job outlives the request that submitted it, so the
        file has to outlive it too, and it has to be somewhere a restart can
        sweep: a job exists only in memory, so any file still here when the
        process starts belongs to no job and is removed. With a data root the
        directory sits under it; without one it is a subdirectory of the
        system temp directory, resolved at call time so a test can point
        ``tempfile`` elsewhere.
        """
        if self.data_root:
            return os.path.join(self.data_root, "uploads")
        return os.path.join(tempfile.gettempdir(), "chat_rag-uploads")

    def to_dict(self) -> Dict[str, Optional[str]]:
        return {"data_root": self.data_root, "parser_cache": self.parser_cache}


def paths_from_env(env: Optional[Mapping[str, str]] = None) -> PathSettings:
    """Read the two path settings from an environment. The only reader."""
    env = os.environ if env is None else env
    return PathSettings(
        data_root=(env.get(DATA_DIR_ENV) or "").strip() or None,
        parser_cache=(env.get(PARSER_CACHE_ENV) or "").strip() or None,
    )


# --------------------------------------------------- whose layout is current
#
# The module-level functions below are what the application calls, and each
# one resolves at the moment it is called rather than at import. That is
# deliberate and unchanged: a data root declared after import -- by a test, by
# a smoke tool, by the ``.env`` file being applied -- has always taken effect,
# and a value cached at import would silently ignore it.
#
# What :func:`current` answers *with* changed twice. L2 made it one reader
# producing a value (:func:`paths_from_env`), so a caller could build the same
# value itself and hand it to :class:`~chat_rag.config.Settings`. This step
# makes it resolve through the engine when there is one, the way
# ``storage.session_scope()``, ``limits.provider_budget()`` and
# ``telemetry.metrics()`` already do: inside an activation the answer is that
# engine's, and outside one it is the environment's, exactly as before.
#
# Without this, ``Settings.paths`` was a value the runtime reported and nothing
# read -- so a second engine given its own data root still wrote its packaged
# analyses, its staged uploads and its parser cache into the first one's
# directories, because every writer resolved through the process environment.


def _activated_runtime() -> Optional[Any]:
    """The activated engine's runtime, if there is one.

    Read out of ``sys.modules`` rather than imported, for two reasons and both
    matter. **Cycles:** ``chat_rag.runtime`` imports ``chat_rag.config``, and
    this module is reached *while* that package is still initialising --
    ``config/__init__.py`` applies the ``.env`` file on the way past -- so an
    import here would be a partially initialised module. **Cost:** this is
    asked once per parse, per state read and per staged upload, and a module
    that has never been imported cannot have an activated runtime, so the
    lookup is also the answer.
    """
    runtime = sys.modules.get("chat_rag.runtime")
    return None if runtime is None else runtime.active()


def current() -> PathSettings:
    """The path layout this call belongs to.

    The activated engine's if there is one, and the process environment's
    otherwise. Both answers are read now rather than remembered, so nothing
    here freezes a layout at import.
    """
    engine = _activated_runtime()
    return engine.paths if engine is not None else paths_from_env()


def data_root() -> Optional[str]:
    return current().data_root


def knowledge_bases() -> str:
    return current().knowledge_bases()


def ingested_documents() -> str:
    return current().ingested_documents()


def gold_set() -> str:
    return current().gold_set()


def ingest_journal() -> str:
    return current().ingest_journal()


def logs() -> str:
    return current().logs()


def canonical_cache() -> str:
    return current().canonical_cache()


def viewer_live_analysis() -> str:
    return current().viewer_live_analysis()


def boundary_embedding_cache() -> str:
    return current().boundary_embedding_cache()


def embedding_cache() -> str:
    return current().embedding_cache()


def upload_staging() -> str:
    return current().upload_staging()


# --------------------------------------------------------------- the contract


def load_env_file(path: str) -> Dict[str, str]:
    """Apply a ``.env`` file, without letting it move a declared data root.

    The precedence, highest first:

    1. the real process environment -- what a container, a compose file, a
       test or an operator's shell actually set;
    2. this file, for everything that is not a state path;
    3. this file, for state paths, but *only* when no data root is declared.

    Rule 3 is the whole point. ``.env`` is a developer's local file and it
    describes the developer's local layout:
    ``STRUCTURED_PARSER_CACHE=./.cache/canonical-units`` means "the cache in
    my checkout". A deployment, a smoke check or a test that declares
    ``CHAT_RAG_DATA_DIR`` has said where its state lives, and a file left over
    from local development must not quietly move it back -- which is exactly
    how a smoke run came to write into the developer's own checkout. An
    operator who genuinely wants a cache outside the data root still has rule
    1: set the variable in the environment, where it is visible.

    Returns the settings that were applied, and records the ones that were
    refused (see :func:`diagnostics`).

    This is the one place in the package that writes to ``os.environ``, and
    the reason is the rule above: the file is applied *to* the environment so
    that one precedence order holds for every reader, including readers that
    are not this application -- a library reading its own variable sees the
    file too. Everything else in ``chat_rag`` only ever reads, and only from a
    ``*_from_env`` function.
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
        # The *environment's* data root, never an engine's. Applying a file to
        # the process environment is something that happens before any engine
        # exists -- ``config/__init__.py`` does it on the way past -- so asking
        # :func:`current` here would be asking a module that is still being
        # imported which engine is active.
        if key in STATE_PATH_ENV and paths_from_env().data_root is not None:
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
    settings = current()
    root = settings.data_root
    for key, value in sorted(_ignored_from_env_file.items()):
        lines.append(
            f"{key}={value} in .env ignored: {DATA_DIR_ENV}={root} owns this path"
        )
    override = (settings.parser_cache or "").strip()
    if root and override and not _within(override, root):
        lines.append(
            f"{PARSER_CACHE_ENV}={override} is outside {DATA_DIR_ENV}={root}; "
            "the parser cache will not travel with the data directory"
        )
    return lines


def _within(path: str, root: str) -> bool:
    try:
        return os.path.commonpath(
            [os.path.abspath(path), os.path.abspath(root)]
        ) == os.path.abspath(root)
    except ValueError:  # different drives on Windows
        return False
