"""One engine per *runtime*, one session per unit of work.

The rule this module exists to enforce is the one that is easiest to break by
accident: **a repository call does not build an engine**. An engine owns a
connection pool, and building one per call would open a TCP connection, a
PostgreSQL backend and a TLS handshake for every knowledge-base lookup on
every page load, then leave them to the garbage collector. So the engine is
built once, lazily, and :meth:`Database.session_scope` hands out sessions from
it.

Lazily, because importing a module must not need a database. ``python -m cli
manifest``, ``tools/import_smoke.py`` and every test that never touches a
record all import application code, and none of them should fail on a machine
with no PostgreSQL. The first call that actually needs a row is where a
missing or unreachable database is reported, by name.

What changed in L3 is *who* it belongs to. It was a module global -- one
engine per process, whatever else was running in it -- and it is
:class:`Database` now, owned by a :class:`~chat_rag.runtime.Runtime` and
therefore by one ``Services``. Two engines in one process no longer share a
pool by accident.

The module-level functions below are unchanged in name and signature and
resolve through :func:`chat_rag.runtime.current`, so every existing caller --
Alembic, ``tools/migrate.py``, a repository, a test -- keeps working and gets
the process default when nothing has been activated.

Transactions are the caller's, stated with a ``with``::

    with session_scope() as session:
        ...                       # commits at the end, rolls back on an error

Nothing here writes SQL and nothing here knows what a knowledge base is; that
is :mod:`chat_rag.storage.repositories`. This module knows about connections.
"""

from __future__ import annotations

import logging
import threading
from contextlib import contextmanager
from typing import Any, Iterator, Optional

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

from chat_rag.config.database import DatabaseSettings, database_from_env

logger = logging.getLogger("RAG.storage")


class DatabaseNotConfigured(RuntimeError):
    """``DATABASE_URL`` is not set.

    Its own type because the answer is a deployment instruction rather than a
    stack trace: the caller turns it into a start-up refusal or a degraded
    health reason, not a 500 with a traceback in it.
    """


class DatabaseUnavailable(RuntimeError):
    """The database is configured and this process cannot reach it."""


class Database:
    """One connection pool, and the sessions that come out of it.

    Constructed with settings, it is exactly those settings: an engine
    configured by construction, which is what a second ``Services`` in a
    process needs in order not to be the first one's.

    Constructed with none, it reads the environment on first use and forgets
    what it read when it is disposed -- which is what the module global did,
    and what a test that changes ``DATABASE_URL`` and disposes still relies on.
    The process-default runtime is built that way for exactly that reason.
    """

    def __init__(self, settings: Optional[DatabaseSettings] = None) -> None:
        self._lock = threading.RLock()
        self._given = settings
        self._settings: Optional[DatabaseSettings] = settings
        self._engine: Optional[Engine] = None
        self._sessions: Optional[sessionmaker] = None

    # ------------------------------------------------------------- config
    def configured_settings(self) -> DatabaseSettings:
        """The database configuration this engine is using.

        Remembered once, so the pool cannot be sized from one set of values
        and reported from another.
        """
        with self._lock:
            if self._settings is None:
                self._settings = database_from_env()
            return self._settings

    # ------------------------------------------------------------ the pool
    def engine(self) -> Engine:
        """This runtime's engine, built on first use."""
        with self._lock:
            if self._engine is not None:
                return self._engine
            settings = self.configured_settings()
            if not settings.configured:
                raise DatabaseNotConfigured(
                    "DATABASE_URL is not set. PostgreSQL is where this application "
                    "keeps its knowledge bases, documents, analyses and ingest "
                    "jobs; see docs/configuration.md and env.example."
                )
            self._engine = create_engine(
                settings.url,
                pool_size=settings.pool_size,
                max_overflow=settings.max_overflow,
                pool_timeout=settings.pool_timeout,
                pool_recycle=settings.pool_recycle,
                # Hands out no connection the server has already closed. The
                # cost is one round trip per checkout; the alternative is a
                # request failing on a stale connection after an idle night.
                pool_pre_ping=True,
                connect_args={"connect_timeout": int(settings.connect_timeout)},
                echo=settings.echo,
                future=True,
            )
            self._sessions = sessionmaker(
                bind=self._engine, expire_on_commit=False, future=True)
            logger.info("database engine ready: %s", settings.sanitized_url)
            return self._engine

    def session_factory(self) -> sessionmaker:
        with self._lock:
            self.engine()
            assert self._sessions is not None  # built beside the engine
            return self._sessions

    @contextmanager
    def session_scope(self, session: Optional[Session] = None) -> Iterator[Session]:
        """One transaction, committed at the end of the block or rolled back.

        Passing an existing ``session`` joins the caller's transaction instead
        of opening a second one, which is what lets a repository method be
        used on its own *and* inside a larger unit of work without either
        duplicating the other's commit.
        """
        if session is not None:
            yield session
            return
        made = self.session_factory()()
        try:
            yield made
            made.commit()
        except Exception:
            made.rollback()
            raise
        finally:
            # Returns the connection to the pool. Not closing here is exactly
            # how a pool leaks: the session would hold its connection until it
            # was collected, and under load that is every connection at once.
            made.close()

    def dispose(self) -> None:
        """Close every pooled connection and forget the engine.

        Called on shutdown, and by a test that has changed ``DATABASE_URL``:
        an environment-tracking database forgets what it read, so the next
        call builds a new engine against the new configuration. One built from
        explicit settings keeps them -- it was configured, not discovered.
        """
        with self._lock:
            if self._engine is not None:
                self._engine.dispose()
            self._engine = None
            self._sessions = None
            self._settings = self._given

    # ------------------------------------------------------------ reporting
    def describe(self) -> dict[str, Any]:
        """The database configuration, without touching the database.

        What ``effective_configuration()`` reports, and therefore what the
        start-up banner prints: a *configuration* answer must not depend on a
        round trip. Whether the database can be reached is a different
        question, asked by :meth:`health` and answered by
        ``/api/ops/metrics``.
        """
        try:
            return self.configured_settings().to_dict()
        except ValueError as error:  # a bad DATABASE_POOL_* value
            return {"configured": False, "url": "", "error": str(error)}

    def health(self) -> dict[str, Any]:
        """Whether this process can reach its database right now.

        Never raises: it is read by ``/api/health``, which has to answer while
        the database is down -- that is the moment it is asked.
        """
        try:
            settings = self.configured_settings()
        except ValueError as error:  # a bad DATABASE_POOL_* value
            return {"configured": False, "reachable": False, "error": str(error)}
        if not settings.configured:
            return {"configured": False, "reachable": False,
                    "error": "DATABASE_URL is not set"}
        report: dict[str, Any] = {"configured": True, "url": settings.sanitized_url}
        try:
            with self.engine().connect() as connection:
                connection.execute(text("SELECT 1"))
            report["reachable"] = True
        except (SQLAlchemyError, OSError) as error:
            report["reachable"] = False
            # The connection string can carry a password; the exception text
            # can carry the whole DSN. Only the class name and the sanitized
            # URL go into a body ``/api/ops/metrics`` serves unauthenticated.
            report["error"] = type(error).__name__
        report.update(self.pool_status())
        return report

    def pool_status(self) -> dict[str, Any]:
        """How many connections are checked out, and how many the pool may hold.

        The number an operator looks at when "the database is slow" turns out
        to be "every connection is held by a request waiting on a model call".
        """
        with self._lock:
            engine = self._engine
        if engine is None:
            return {"pool": None}
        pool = engine.pool
        status: dict[str, Any] = {}
        for name in ("size", "checkedin", "checkedout", "overflow"):
            reader = getattr(pool, name, None)
            if callable(reader):
                try:
                    status[name] = reader()
                except Exception:  # noqa: BLE001 - a status read must not raise
                    pass
        return {"pool": status or None}

    def require_reachable(self) -> None:
        """Raise unless this process can reach its database.

        Used at start-up, where the right behaviour is a controlled refusal
        with the reason named rather than a server that binds a port and fails
        every request.
        """
        try:
            with self.engine().connect() as connection:
                connection.execute(text("SELECT 1"))
        except DatabaseNotConfigured:
            raise
        except (SQLAlchemyError, OSError) as error:
            raise DatabaseUnavailable(
                f"cannot reach the database at "
                f"{self.configured_settings().sanitized_url}: {type(error).__name__}"
            ) from error


class DatabaseBound:
    """A store that knows which database it belongs to.

    Every record store used to open its transactions with the module-level
    ``session_scope()``, which meant the process's one engine whoever was
    asking. A ``Services`` hands its stores its own :class:`Database` now, so
    two engines in one process write to two databases rather than to
    whichever one was built first.

    Handed none, it resolves the current runtime's -- which is what a store
    built directly by a test or a tool gets, and is exactly the old
    behaviour.
    """

    #: Set by :meth:`bind_database` or by a subclass's constructor.
    _database: Optional["Database"] = None

    def bind_database(self, database: Optional["Database"]) -> None:
        self._database = database

    def _session(self, session: Optional[Session] = None):
        database = self._database
        if database is None:
            from chat_rag import runtime

            database = runtime.current().database
        return database.session_scope(session)


# ------------------------------------------------------- the module surface
#
# Same names, same signatures, one indirection: whichever runtime this call
# belongs to. Activated if a ``Services`` activated one, and the process
# default otherwise -- which is the first ``Services`` built, or an
# environment-derived one for a caller that never builds a ``Services`` at all.


def _database() -> Database:
    from chat_rag import runtime

    return runtime.current().database


def configured_settings() -> DatabaseSettings:
    return _database().configured_settings()


def engine() -> Engine:
    return _database().engine()


def session_factory() -> sessionmaker:
    return _database().session_factory()


@contextmanager
def session_scope(session: Optional[Session] = None) -> Iterator[Session]:
    with _database().session_scope(session) as scoped:
        yield scoped


def dispose() -> None:
    _database().dispose()


def describe() -> dict[str, Any]:
    return _database().describe()


def health() -> dict[str, Any]:
    return _database().health()


def pool_status() -> dict[str, Any]:
    return _database().pool_status()


def require_reachable() -> None:
    _database().require_reachable()
