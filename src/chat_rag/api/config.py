"""How a caller configures an :class:`~chat_rag.api.Engine`.

:class:`~chat_rag.config.Settings` is already a value rather than a read of the
environment (L2), and it is what the engine runs on. This is not a second
configuration system on top of it -- it is the *stated* subset: the dozen
settings a caller embedding this engine actually chooses, with names that mean
something outside this repository, and one rule for everything they do not
mention.

The rule
--------

A field left at ``None`` is not "the default". It means *this caller did not
say*, and what happens then is exactly what happened before there was a
facade: :meth:`Settings.from_env` reads it, under the precedence
``config`` has always had --

    the real process environment  >  accepted values from ``.env``  >  the
    application default written on the field

so an engine embedded in a program that already configures this deployment
keeps that deployment's configuration, and only the fields named here move.
``read_environment=False`` swaps the base for the dataclass defaults
(``Settings()``), which is what a test wants: a configuration that depends on
nothing outside the call.

The escape hatch
----------------

``settings=`` takes a whole :class:`Settings`. Everything below is a
convenience over the fields of that object, and a caller who needs one this
does not name should not have to wait for a release to reach it -- so the full
object is accepted and the named fields are applied on top of it.

Where the files go
------------------

``data_dir`` is real, and it is the setting that makes two engines in one
process genuinely separate. It gathers everything this engine writes -- the
packaged Viewer analyses, the staged uploads, the parser's canonical-unit
cache, both embedding caches and the legacy state paths -- under one
directory, and ``config.paths`` resolves through the activated engine, so a
second engine's writes land under *its* root rather than in the first one's
directories.

Left unset it means what it has always meant: the process environment's answer
(``CHAT_RAG_DATA_DIR``, else the historical working-directory-relative
layout), read at the moment a path is needed rather than frozen here.

Two things it does not cover, both on purpose. **The database** is not a file:
it is ``database_url``, and two engines under different data roots still share
a database unless they are told otherwise. **The log file** belongs to the
process rather than to an engine -- ``chat_rag.utils.logger`` is called by an
entry point, not by a container -- so it keeps reading the environment.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from typing import Any, Mapping, Optional

from chat_rag.config import Settings
from chat_rag.config.database import DatabaseSettings, normalize_url
from chat_rag.config.ingest import IngestLimits
from chat_rag.config.paths import PathSettings
from chat_rag.config.query import QueryLimits


@dataclass(frozen=True)
class EngineConfig:
    """The settings an embedding caller states; everything else is inherited.

    ``EngineConfig()`` is the deployment's own configuration, unchanged -- the
    same one ``python -m asgi`` would serve with.
    """

    # ------------------------------------------------------------- storage
    #: ``postgresql+psycopg://user:password@host:port/name``. Required, by the
    #: application rather than by this class: the relational records are
    #: PostgreSQL and there is no mode that runs without them. Unset here and
    #: unset in the environment, the engine refuses at the first call that
    #: needs a row, naming ``DATABASE_URL``.
    database_url: Optional[str] = None

    # --------------------------------------------------------------- files
    #: Where everything this engine writes goes: packaged Viewer analyses,
    #: staged uploads, the parser's canonical-unit cache, both embedding
    #: caches. Unset means the process environment's answer, read when a path
    #: is needed. Two engines given two roots share no file.
    data_dir: Optional[str] = None
    #: The parser's canonical-unit cache, when it belongs somewhere other than
    #: under :attr:`data_dir`. The one path that may sit outside the root --
    #: it is worth sharing between engines, because a cache entry is keyed by
    #: the document's content and re-parsing a PDF costs minutes.
    parser_cache: Optional[str] = None

    # ------------------------------------------------------------- retrieval
    #: ``bm25_only`` (no provider at all), ``hybrid_rrf`` (the final chain) or
    #: ``benchmark_aligned`` (the frozen Phase 4/5 configuration).
    retrieval_profile: Optional[str] = None
    #: How many passages a question retrieves when it does not say.
    top_k: Optional[int] = None
    #: The indexing chunker a knowledge base is created with when its own
    #: creation call does not name one.
    chunker: Optional[str] = None

    # ---------------------------------------------------------------- models
    embedding_provider: Optional[str] = None
    embedding_model: Optional[str] = None
    answer_provider: Optional[str] = None
    answer_model: Optional[str] = None

    # ---------------------------------------------------------------- bounds
    #: Ingest workers, and how many jobs may wait behind them. An upload that
    #: finds both full is refused rather than queued.
    ingest_workers: Optional[int] = None
    ingest_queue_capacity: Optional[int] = None
    #: How many callers may be inside a question at once, and how long one
    #: has in total before it is abandoned.
    query_max_active: Optional[int] = None
    query_timeout_seconds: Optional[float] = None

    # ----------------------------------------------------------- the base
    #: Whether the fields left unset are read from the environment. False
    #: gives the application defaults instead, which is a configuration that
    #: depends on nothing outside this call.
    read_environment: bool = True
    #: A whole :class:`Settings` to build on, for anything the fields above do
    #: not name. Applied first; the named fields win over it.
    settings: Optional[Settings] = None

    # ------------------------------------------------------------------ build
    def build(self) -> Settings:
        """The :class:`Settings` an engine composed from this runs on.

        Constructed once, through :func:`dataclasses.replace`, so that
        ``Settings.__post_init__`` validates the result and re-derives the
        mirrors it publishes (``query_timeout``, ``pipeline_cache_max`` and
        the rest). Setting attributes on a built ``Settings`` would skip both,
        and a caller who asked for a query deadline would not get one.
        """
        base = self.settings
        if base is None:
            base = Settings.from_env() if self.read_environment else Settings()

        changes: dict[str, Any] = {}
        _state(changes, "retrieval_profile", self.retrieval_profile)
        _state(changes, "default_top_k", self.top_k)
        _state(changes, "chunker_type", self.chunker)
        _state(changes, "embedding_provider", self.embedding_provider)
        _state(changes, "embedding_model_name", self.embedding_model)
        _state(changes, "answer_provider", self.answer_provider)
        _state(changes, "answer_model", self.answer_model)

        database = _database(base.database, self.database_url)
        if database is not None:
            changes["database"] = database
        layout = _paths(base.paths, self.data_dir, self.parser_cache)
        if layout is not None:
            changes["paths"] = layout
        ingest = _ingest(base.ingest_limits, self.ingest_workers,
                         self.ingest_queue_capacity)
        if ingest is not None:
            changes["ingest_limits"] = ingest
        query = _query(base.query_limits, self.query_max_active,
                       self.query_timeout_seconds)
        if query is not None:
            changes["query_limits"] = query

        return replace(base, **changes) if changes else base


def _state(changes: dict[str, Any], field: str, value: Any) -> None:
    """Record a field the caller actually stated. ``None`` states nothing."""
    if value is not None:
        changes[field] = value


def _database(base: DatabaseSettings, url: Optional[str]) -> Optional[DatabaseSettings]:
    """The database settings, with a stated URL applied and validated.

    Normalised the same way the environment reader normalises one, so a bare
    ``postgresql://`` gets the psycopg 3 driver this application is written
    against rather than whichever DBAPI happens to be installed.
    """
    if url is None:
        return None
    return replace(base, url=normalize_url(url)).validate()


def _paths(base: PathSettings, data_dir: Optional[str],
           parser_cache: Optional[str]) -> Optional[PathSettings]:
    """The layout, with a stated root applied.

    Absolute, because a data root is where files *are* and a relative one
    would mean "wherever this process happens to be standing" -- which is the
    working-directory-relative layout this setting exists to replace, spelled
    less clearly. The unset case still gets that layout, from the environment,
    which is where it belongs.
    """
    changes: dict[str, Any] = {}
    if data_dir is not None:
        changes["data_root"] = os.path.abspath(data_dir)
    if parser_cache is not None:
        changes["parser_cache"] = os.path.abspath(parser_cache)
    return replace(base, **changes) if changes else None


def _ingest(base: IngestLimits, workers: Optional[int],
            queue_capacity: Optional[int]) -> Optional[IngestLimits]:
    """The ingest bounds with the stated ones applied, and validated.

    Validated here because ``Settings.__post_init__`` does not: the limits
    are checked by whoever builds them, which for the product is
    ``limits_from_env``. A caller who asks for zero workers should be refused
    by name at construction, not by a job manager that starts no threads.
    """
    changes: dict[str, Any] = {}
    _state(changes, "workers", workers)
    _state(changes, "queue_capacity", queue_capacity)
    return replace(base, **changes).validate() if changes else None


def _query(base: QueryLimits, max_active: Optional[int],
           timeout_seconds: Optional[float]) -> Optional[QueryLimits]:
    changes: dict[str, Any] = {}
    _state(changes, "max_active", max_active)
    _state(changes, "timeout_seconds", timeout_seconds)
    return replace(base, **changes).validate() if changes else None


def describe(settings: Settings) -> Mapping[str, Any]:
    """The non-secret configuration an engine is running with.

    The same method the start-up banner and ``/api/ops/metrics`` read, so
    "what is this engine configured to do" has one answer whichever surface
    asks it -- and, for the same reason, never a credential.
    """
    return settings.effective_configuration()
