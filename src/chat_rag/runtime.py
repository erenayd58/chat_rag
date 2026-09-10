"""What a running engine owns: the state that used to be a process global.

Seven things in this application outlived any one call and belonged to nobody:
the database engine and its pool, three provider budgets, the metrics
registry, the Viewer packager's queue and worker, and the analysis-query
engine. Each was a module-level variable built on first use, which is exactly
right for a program that is one process and one configuration -- and exactly
wrong for a library, where the second :class:`~chat_rag.application.services.Services`
in a process would silently share the first one's connection pool, spend its
provider slots and record into its counters.

They are :class:`Runtime` now, and a ``Services`` owns one.

How deep code still finds it
----------------------------

Threading a runtime through every function that measures a stage or takes a
provider slot would touch most of the package for no gain, so it is published
instead, on a :class:`~contextvars.ContextVar`::

    with runtime.activate(services.runtime):
        ...                       # current() is this runtime in here

:func:`current` is what the module-level accessors resolve through --
``storage.session_scope()``, ``limits.provider_budget()``,
``telemetry.metrics()`` and the rest kept their names and their signatures,
and only changed where they look.

**The process default is the compatibility shim, and it is load-bearing.**
The first ``build_services()`` in a process installs its runtime as the
default (:func:`install_default`), so every call site that has not been handed
a runtime -- a CLI command, ``tools/migrate.py``, Alembic, a test that reaches
a repository directly -- behaves exactly as it did when these were globals.
A *second* ``Services`` never becomes the default: its runtime is reachable
only by activation, which is what keeps the two apart.

A context variable is not inherited by a thread, so the workers that outlive a
call activate for themselves: the ingest job manager's worker, the Viewer
packager's worker. Both are started by the ``Services`` whose runtime they
carry, and both wrap their work in :func:`activate`.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Iterator, Optional

from chat_rag.config import Settings
from chat_rag.config.paths import PathSettings


class Runtime:
    """One engine's share of everything that is not a value.

    Built lazily, piece by piece: importing this module must not open a
    connection, load a model or start a thread, and neither must constructing
    a ``Runtime``. Each accessor below builds its piece on first use, under
    this object's own lock.
    """

    def __init__(self, settings: Optional[Settings] = None) -> None:
        #: The configuration this runtime was built for. ``Settings`` is a
        #: value (L2); everything else here is not.
        self.settings = settings if settings is not None else Settings.from_env()
        #: Whether this runtime was *given* its configuration or read it. The
        #: database is the one piece that behaves differently: a configured
        #: engine is exactly what it was configured with, while an
        #: environment-derived one keeps tracking the environment across a
        #: dispose, which is what the module global did and what a test that
        #: changes ``DATABASE_URL`` still relies on.
        self._configured = settings is not None
        self._lock = threading.RLock()
        self._database: Any = None
        self._provider_budget: Any = None
        self._embedding_budget: Any = None
        self._answer_budget: Any = None
        self._metrics: Any = None
        self._packager: Any = None
        self._analysis: Any = None

    # ------------------------------------------------------------- config
    @property
    def paths(self) -> PathSettings:
        """Where this engine's files go. The configuration's answer, not the
        environment's, so a second engine can be given a second data root."""
        return self.settings.paths

    # ----------------------------------------------------------- database
    @property
    def database(self):
        """This engine's connection pool, built on first use.

        Lazily, for the reason ``storage.engine`` has always been lazy:
        importing a module must not need a database. ``python -m cli
        manifest``, the import smoke and every test that never touches a row
        must work on a machine with no PostgreSQL.
        """
        with self._lock:
            if self._database is None:
                from chat_rag.storage.engine import Database

                self._database = Database(
                    self.settings.database if self._configured else None)
            return self._database

    # ------------------------------------------------------------ budgets
    #
    # Three, because they bound three different external services and none may
    # starve the others: Deep Analysis calls a chat endpoint, a re-index
    # saturates an embeddings endpoint, and a burst of questions calls the
    # answer model. Sized from this runtime's settings rather than from the
    # environment, so a second engine can be given smaller ones.
    @property
    def provider_budget(self):
        with self._lock:
            if self._provider_budget is None:
                from chat_rag.components.ingest.limits import ProviderBudget

                self._provider_budget = ProviderBudget(
                    self.settings.ingest_limits.provider_max_inflight)
            return self._provider_budget

    @provider_budget.setter
    def provider_budget(self, budget) -> None:
        with self._lock:
            self._provider_budget = budget

    @property
    def embedding_budget(self):
        with self._lock:
            if self._embedding_budget is None:
                from chat_rag.components.ingest.limits import ProviderBudget

                self._embedding_budget = ProviderBudget(
                    self.settings.ingest_limits.embedding_max_inflight)
            return self._embedding_budget

    @embedding_budget.setter
    def embedding_budget(self, budget) -> None:
        with self._lock:
            self._embedding_budget = budget

    @property
    def answer_budget(self):
        with self._lock:
            if self._answer_budget is None:
                from chat_rag.components.ingest.limits import ProviderBudget

                self._answer_budget = ProviderBudget(
                    self.settings.query_limits.answer_max_inflight)
            return self._answer_budget

    @answer_budget.setter
    def answer_budget(self, budget) -> None:
        with self._lock:
            self._answer_budget = budget

    # ------------------------------------------------------------ metrics
    @property
    def metrics(self):
        """This engine's counters and its window of recent traces."""
        with self._lock:
            if self._metrics is None:
                from chat_rag.components.observability.telemetry import MetricsRegistry

                self._metrics = MetricsRegistry()
            return self._metrics

    @metrics.setter
    def metrics(self, registry) -> None:
        with self._lock:
            self._metrics = registry

    # ----------------------------------------------------------- packager
    @property
    def packager(self):
        """The Viewer packager's queue, worker thread and per-document locks.

        One worker per runtime, started on the first build this runtime is
        asked for -- never at import, and never shared with another engine's
        queue.
        """
        with self._lock:
            if self._packager is None:
                from chat_rag.components.viewer.analysis import PackagerState

                self._packager = PackagerState()
            return self._packager

    # ---------------------------------------------------- analysis engine
    @property
    def analysis(self):
        """The Viewer's retrieval engine over one document's analysis arms,
        and the record of which arms it has been given."""
        with self._lock:
            if self._analysis is None:
                from chat_rag.application.analysis_query import AnalysisEngineHolder

                self._analysis = AnalysisEngineHolder()
            return self._analysis

    # ----------------------------------------------------------- shutdown
    def close(self) -> None:
        """Release what this runtime holds. Safe to call more than once.

        The packager's worker is a daemon thread over an in-memory queue and
        is left to the process; what has to be given back is the connection
        pool, because a second engine in the same process would otherwise hold
        its own pool open for the life of the interpreter.
        """
        with self._lock:
            database, self._database = self._database, None
        if database is not None:
            database.dispose()


# --------------------------------------------------------------- publication
#
# A ContextVar rather than a thread-local: it is inherited by a task spawned
# inside an activation and restored exactly on exit, which a thread-local
# nested inside itself is not.
_active: ContextVar[Optional[Runtime]] = ContextVar("chat_rag_runtime", default=None)

_default_lock = threading.Lock()
_default: Optional[Runtime] = None


@contextmanager
def activate(runtime: Runtime) -> Iterator[Runtime]:
    """Make ``runtime`` the one :func:`current` answers with, for this block."""
    token = _active.set(runtime)
    try:
        yield runtime
    finally:
        _active.reset(token)


def install_default(runtime: Runtime, *, force: bool = False) -> Runtime:
    """Make ``runtime`` the process default, for callers with no activation.

    The *first* ``Services`` in a process installs itself here, which is what
    makes the module-level accessors behave exactly as the globals they
    replaced. A second one does not, so it cannot take the first one's callers
    with it -- and ``force`` exists for a test that means to.
    """
    global _default
    with _default_lock:
        if _default is None or force:
            _default = runtime
        return _default


def default() -> Runtime:
    """The process default, built from the environment if nobody installed one.

    That fallback is what keeps a tool, a migration or a test that never
    builds a ``Services`` working: they used to reach a module global built on
    first use, and this is the same thing with a name.
    """
    global _default
    with _default_lock:
        if _default is None:
            _default = Runtime()
        return _default


def current() -> Runtime:
    """The runtime this call belongs to.

    The activated one if there is one, and the process default otherwise.
    Every module-level accessor that used to read a global resolves here.
    """
    runtime = _active.get()
    return runtime if runtime is not None else default()


def active() -> Optional[Runtime]:
    """The activated runtime, or ``None``. For a caller that needs to know
    whether it is inside one rather than which."""
    return _active.get()


def reset_default() -> None:
    """Forget the process default. For a test, and for nothing else."""
    global _default
    with _default_lock:
        _default = None
