"""The engine, as one object a program holds.

Everything under this file already worked without a web server: the use cases
take plain arguments (L1), the settings are a value rather than a read of the
environment (L2), and the state that used to be process-wide belongs to a
``Runtime`` a container owns (L3). What was missing was somebody to *hold* one
-- a caller had to compose a ``Services``, invent a session id, remember which
calls need an activation, and read dictionaries. That is this class.

Three things it owns, and each is the point of one of the steps before it:

**Its own ``Services`` and ``Runtime``.** ``build_services(settings)`` composes
the same container ``asgi.py`` composes: the record stores, the pipeline cache,
the ingest workers, the admission counter, and a runtime holding the connection
pool, the three provider budgets, the metrics registry, the packaging queue and
the analysis engine. Two engines in one process share none of it -- and, given
``EngineConfig(data_dir=...)``, none of the files either: ``config.paths``
resolves through the activated engine, so the packaged analyses, the staged
uploads and the caches land under this engine's root.

It is also **not the process default**. The default is the runtime that
callers who were never handed one resolve to -- Alembic, ``tools/migrate.py``,
a CLI command -- and the product's container installs itself as it on purpose.
A library engine does not: being the first container in somebody else's
process is an accident of ordering, not a mandate to speak for it.

**Its own session id.** The session is the pipeline cache's key and nothing
else -- not identity, not authorisation. ``/api/v1`` takes it from the
transport because a browser has one; a library caller has no such thing and
should not have to invent one, so an engine is its own session and every call
this facade makes passes it.

**The activation.** ``Services.activate()`` is what makes ``session_scope()``,
``provider_budget()``, ``metrics()`` and the packager's queue resolve to *this*
engine's rather than to whichever was built first in the process. The use cases
that reach past their arguments already carry ``@in_engine``; the ones that do
not -- every :mod:`chat_rag.application.workspace` call, which is the whole
Viewer read model and the analysis lifecycle -- are wrapped here. Doing it in
one place is the reason :meth:`_active` exists rather than being spelled out at
each call site.

What it is not
--------------

Not a server, and not an owner of any process-wide decision. It installs no log
handler, sets no thread-pool variable and reads no ``.env`` of its own; the
entry points do that (``chat_rag.process``, ``chat_rag.utils.logger``), because
a library that reconfigures the process on construction has spoken for a
program that has not.
"""

from __future__ import annotations

import uuid
from typing import Any, Mapping, Optional

from chat_rag.application import catalogue as catalogue_use_case
from chat_rag.application import documents as documents_use_case
from chat_rag.application import ingest as ingest_use_case
from chat_rag.application import ops as ops_use_case
from chat_rag.application import workspace as workspace_use_case
from chat_rag.application.services import Services, build_services
from chat_rag.config import Settings

from .config import EngineConfig, describe
from .resources import Document, KnowledgeBase, KnowledgeBases
from .results import Health, Method


class Engine:
    """One configured RAG engine: ingest, analyse, search, answer.

    Use it as a context manager, so what it holds is given back::

        with Engine(EngineConfig(database_url=...)) as engine:
            kb = engine.knowledge_bases.create("Reports")
            document = kb.ingest("report.pdf")
            answer = kb.ask("What changed?")

    Constructing one composes the container and nothing more: no connection is
    opened, no model is loaded and no thread is started until something is
    asked for. PostgreSQL is required, and the refusal comes at the first call
    that needs a row, naming ``DATABASE_URL`` -- not at construction, so an
    engine can be built on a machine that cannot reach one.
    """

    def __init__(self, config: Optional[EngineConfig] = None, *,
                 session_id: Optional[str] = None,
                 install_process_default: bool = False):
        self._config = config if config is not None else EngineConfig()
        self._settings = self._config.build()
        # Not the process default, unless asked. Being the first
        # ``build_services`` in a process is an accident of ordering, and
        # taking the default on it would hand this engine's pool, budgets,
        # counters and packaging queue to code that never asked for one --
        # and then dispose that pool when this engine closed. Say so
        # explicitly when a program also runs something that never sees an
        # engine: ``tools/migrate.py``, Alembic, a CLI command.
        self._services = build_services(
            self._settings, install_default=install_process_default)
        # The pipeline cache's key, and nothing else. Its own, so two engines
        # in one process cache their pipelines apart; overridable, because a
        # program embedding this may already have a session of its own and
        # sharing the cache entry is then the point.
        self._session_id = session_id or f"engine-{uuid.uuid4().hex}"
        self._closed = False
        self.knowledge_bases = KnowledgeBases(self)

    # ------------------------------------------------------------- identity
    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def config(self) -> EngineConfig:
        return self._config

    @property
    def settings(self) -> Settings:
        """The configuration this engine is running on, resolved."""
        return self._settings

    @property
    def services(self) -> Services:
        """The container underneath. The escape hatch to the use cases.

        Public on purpose and not the way to use this library: everything the
        facade offers is one of these calls, and reaching past it means taking
        on the two things the facade does -- passing the session id, and
        running inside :meth:`activate`.
        """
        return self._services

    def __repr__(self) -> str:
        state = "closed" if self._closed else "open"
        return f"<Engine {self._session_id} {state}>"

    # ----------------------------------------------------------- activation
    def activate(self):
        """Make this engine the one deep code resolves through, for a block.

        Public because a caller reaching :attr:`services` needs it, and used
        internally by every method below. Nesting is safe and restores exactly.
        """
        return self._services.activate()

    def _active(self):
        """:meth:`activate`, with the one check worth making first.

        A call on a closed engine would otherwise reach a disposed connection
        pool and fail as a database error, which names the wrong thing.
        """
        if self._closed:
            raise RuntimeError(
                "this Engine is closed; build another one rather than reusing it")
        return self._services.activate()

    # -------------------------------------------------------------- reading
    def health(self) -> Health:
        """What this engine can do right now. Cheap, and always answerable.

        Answerable *while degraded* on purpose: it reports the state rather
        than failing on it, which is what makes it usable as a probe.
        """
        with self._active():
            return Health.of(ops_use_case.health(self._services))

    def metrics(self, *, recent: int = 10) -> Mapping[str, Any]:
        """Counters, utilisation, latency and cache state over a bounded window.

        What to read once :meth:`health` says to look closer. Bounded by
        construction: the trace window has a fixed length and the recent-job
        list is capped, so this answer cannot grow with uptime.
        """
        with self._active():
            return ops_use_case.metrics(self._services, recent=recent)

    def configuration(self) -> Mapping[str, Any]:
        """The effective non-secret configuration. Never a credential."""
        return describe(self._settings)

    def chunking_methods(self) -> list[Method]:
        """The chunking methods this deployment can run, and what is true of each.

        Read from the library's registry, so a method registered there appears
        here without a second registration. An unavailable one carries its
        reason instead of quietly disappearing.
        """
        with self._active():
            return [Method.of(entry)
                    for entry in catalogue_use_case.chunking_method_facts()]

    # ------------------------------------------------------------ documents
    def documents(self, knowledge_base_id: Optional[str] = None) -> list[Document]:
        """Every ingested document, or those of one knowledge base."""
        with self._active():
            records = documents_use_case.list_all(self._services, knowledge_base_id)
        return [Document(self, record) for record in records]

    def document(self, document_id: str) -> Document:
        with self._active():
            return Document(self, documents_use_case.get(self._services, document_id))

    def ingest_jobs(self, knowledge_base_id: Optional[str] = None, *,
                    active_only: bool = False) -> Mapping[str, Any]:
        """The jobs this engine knows about, with its ingest capacity."""
        with self._active():
            return ingest_use_case.list_jobs(self._services, knowledge_base_id,
                                             active_only=active_only)

    # ------------------------------------------------------------- start-up
    def recover(self) -> Mapping[str, Any]:
        """Pick up what a previous process left, the way a server start does.

        Two different things, and neither is a resumption of work that was
        committed. Interrupted **ingest** jobs are *settled* against the
        ledger, so a caller holding a job id from before the restart is
        answered truthfully rather than with "unknown"; interrupted **Viewer**
        packaging is genuinely resumed, because everything it needs was
        written to disk before it started.

        Not done on construction: reading somebody else's leftovers is a
        decision for the program that owns the database, not a side effect of
        building an object.
        """
        with self._active():
            return {
                "settled_ingest_jobs": ingest_use_case.recover(self._services),
                "resumed_analyses": workspace_use_case.resume_incomplete(),
            }

    # -------------------------------------------------------------- closing
    def close(self) -> None:
        """Give back what this engine holds. Safe to call more than once.

        Running ingest jobs are let finish first and the connection pool goes
        last, because a job's final act is a ledger write and disposing the
        pool under it would lose the one write an ingest must not lose.
        """
        if self._closed:
            return
        self._closed = True
        self._services.close()

    def __enter__(self) -> "Engine":
        return self

    def __exit__(self, *exception) -> bool:
        self.close()
        return False


def open_engine(config: Optional[EngineConfig] = None, **settings: Any) -> Engine:
    """An :class:`Engine` from keyword settings, for a caller with one or two.

    ``open_engine(database_url=...)`` rather than
    ``Engine(EngineConfig(database_url=...))``. Exactly the same object; the
    keywords are :class:`EngineConfig` fields and are refused by name if they
    are not.
    """
    if config is not None and settings:
        raise TypeError("pass either a config or keyword settings, not both")
    return Engine(config if config is not None else EngineConfig(**settings))


__all__ = ["Engine", "KnowledgeBase", "open_engine"]
