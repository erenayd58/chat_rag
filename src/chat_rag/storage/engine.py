"""One engine per process, one session per unit of work.

The rule this module exists to enforce is the one that is easiest to break by
accident: **a repository call does not build an engine**. An engine owns a
connection pool, and building one per call would open a TCP connection, a
PostgreSQL backend and a TLS handshake for every knowledge-base lookup on
every page load, then leave them to the garbage collector. So the engine is
built once, lazily, on the configuration this process was started with, and
:func:`session_scope` hands out sessions from it.

Lazily, because importing a module must not need a database. ``python -m cli
manifest``, ``tools/import_smoke.py`` and every test that never touches a
record all import application code, and none of them should fail on a machine
with no PostgreSQL. The first call that actually needs a row is where a
missing or unreachable database is reported, by name.

Transactions are the caller's, stated with a ``with``::

    with session_scope() as session:
        ...                       # commits at the end, rolls back on an error

Nothing here writes SQL and nothing here knows what a knowledge base is; that
is :mod:`storage.repositories`. This module knows about connections.
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


_lock = threading.RLock()
_engine: Optional[Engine] = None
_sessions: Optional[sessionmaker] = None
_settings: Optional[DatabaseSettings] = None


def configured_settings() -> DatabaseSettings:
    """The database configuration this process is using.

    Read from the environment on first use and remembered, so the pool cannot
    be sized from one set of values and reported from another.
    """
    global _settings
    with _lock:
        if _settings is None:
            _settings = database_from_env()
        return _settings


def engine() -> Engine:
    """The one engine this process shares, built on first use."""
    global _engine, _sessions
    with _lock:
        if _engine is not None:
            return _engine
        settings = configured_settings()
        if not settings.configured:
            raise DatabaseNotConfigured(
                "DATABASE_URL is not set. PostgreSQL is where this application "
                "keeps its knowledge bases, documents, analyses and ingest "
                "jobs; see docs/configuration.md and env.example."
            )
        _engine = create_engine(
            settings.url,
            pool_size=settings.pool_size,
            max_overflow=settings.max_overflow,
            pool_timeout=settings.pool_timeout,
            pool_recycle=settings.pool_recycle,
            # Hands out no connection the server has already closed. The cost
            # is one round trip per checkout; the alternative is a request
            # failing on a stale connection after an idle night.
            pool_pre_ping=True,
            connect_args={"connect_timeout": int(settings.connect_timeout)},
            echo=settings.echo,
            future=True,
        )
        _sessions = sessionmaker(bind=_engine, expire_on_commit=False, future=True)
        logger.info("database engine ready: %s", settings.sanitized_url)
        return _engine


def session_factory() -> sessionmaker:
    with _lock:
        engine()
        assert _sessions is not None  # built beside the engine, under the lock
        return _sessions


@contextmanager
def session_scope(session: Optional[Session] = None) -> Iterator[Session]:
    """One transaction, committed at the end of the block or rolled back.

    Passing an existing ``session`` joins the caller's transaction instead of
    opening a second one, which is what lets a repository method be used
    on its own *and* inside a larger unit of work without either duplicating
    the other's commit.
    """
    if session is not None:
        yield session
        return
    made = session_factory()()
    try:
        yield made
        made.commit()
    except Exception:
        made.rollback()
        raise
    finally:
        # Returns the connection to the pool. Not closing here is exactly how
        # a pool leaks: the session would hold its connection until it was
        # collected, and under load that is every connection at once.
        made.close()


def dispose() -> None:
    """Close every pooled connection and forget the engine.

    Called on shutdown, and by a test that has changed ``DATABASE_URL``: the
    next call builds a new engine against the new configuration.
    """
    global _engine, _sessions, _settings
    with _lock:
        if _engine is not None:
            _engine.dispose()
        _engine = None
        _sessions = None
        _settings = None


def describe() -> dict[str, Any]:
    """The database configuration, without touching the database.

    What ``effective_configuration()`` reports, and therefore what the start-up
    banner prints: a *configuration* answer must not depend on a round trip.
    Whether the database can be reached is a different question, asked by
    :func:`health` and answered by ``/api/ops/metrics``.
    """
    try:
        return configured_settings().to_dict()
    except ValueError as error:  # a bad DATABASE_POOL_* value
        return {"configured": False, "url": "", "error": str(error)}


def health() -> dict[str, Any]:
    """Whether this process can reach its database right now.

    Never raises: it is read by ``/api/health``, which has to answer while the
    database is down -- that is the moment it is asked.
    """
    try:
        settings = configured_settings()
    except ValueError as error:  # a bad DATABASE_POOL_* value
        return {"configured": False, "reachable": False, "error": str(error)}
    if not settings.configured:
        return {"configured": False, "reachable": False,
                "error": "DATABASE_URL is not set"}
    report: dict[str, Any] = {"configured": True, "url": settings.sanitized_url}
    try:
        with engine().connect() as connection:
            connection.execute(text("SELECT 1"))
        report["reachable"] = True
    except (SQLAlchemyError, OSError) as error:
        report["reachable"] = False
        # The connection string can carry a password; the exception text can
        # carry the whole DSN. Only the class name and the sanitized URL go
        # into a body ``/api/ops/metrics`` serves unauthenticated.
        report["error"] = type(error).__name__
    report.update(pool_status())
    return report


def pool_status() -> dict[str, Any]:
    """How many connections are checked out, and how many the pool may hold.

    The number an operator looks at when "the database is slow" turns out to
    be "every connection is held by a request waiting on a model call".
    """
    with _lock:
        if _engine is None:
            return {"pool": None}
    pool = _engine.pool
    status: dict[str, Any] = {}
    for name in ("size", "checkedin", "checkedout", "overflow"):
        reader = getattr(pool, name, None)
        if callable(reader):
            try:
                status[name] = reader()
            except Exception:  # noqa: BLE001 - a status read must not raise
                pass
    return {"pool": status or None}


def require_reachable() -> None:
    """Raise unless this process can reach its database.

    Used at start-up, where the right behaviour is a controlled refusal with
    the reason named rather than a server that binds a port and fails every
    request.
    """
    try:
        with engine().connect() as connection:
            connection.execute(text("SELECT 1"))
    except DatabaseNotConfigured:
        raise
    except (SQLAlchemyError, OSError) as error:
        raise DatabaseUnavailable(
            f"cannot reach the database at {configured_settings().sanitized_url}: "
            f"{type(error).__name__}"
        ) from error
