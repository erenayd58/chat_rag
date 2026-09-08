"""Where the relational state lives, and how many connections may reach it.

Until Step 8 this application had no database. Its durable records -- the
knowledge bases, the ingest ledger, the Viewer's analysis state, the ingest
journal and the gold set -- were JSON files resolved by :mod:`config.paths`,
and the working directory was their address. That is why this module exists
beside ``paths`` rather than inside it: ``paths`` still owns the *files* this
application writes (vector stores, canonical caches, packaged variants, logs),
and this owns the one connection string everything relational goes through.

The precedence is the package's, unchanged: the real process environment wins,
then ``.env``, then the default written on the dataclass field below.

``DATABASE_URL`` has no default on purpose. Every other setting in this
package can fall back to something sensible; a database cannot, because the
only fallback would be a hard-coded credential in a tracked file. Unset, the
engine refuses to build and says so by name -- at the first call that needs
it, not at import, so ``python -m cli manifest`` and the import smoke still
run on a machine with no database at all.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Mapping, Optional
from urllib.parse import urlsplit, urlunsplit

#: The connection string. ``postgresql+psycopg://user:password@host:port/name``.
URL_ENV = "DATABASE_URL"

#: The driver this application is written against. SQLAlchemy accepts a bare
#: ``postgresql://`` and picks whatever DBAPI is installed; naming psycopg 3
#: explicitly is what stops a machine with psycopg2 lying around from running
#: the application on a driver nobody tested it on.
DEFAULT_DRIVER = "postgresql+psycopg"


def _number(env: Mapping[str, str], name: str, default, kind):
    """One environment number, parsed or refused by name."""
    raw = (env.get(name) or "").strip()
    if not raw:
        return kind(default)
    try:
        return kind(raw)
    except ValueError:
        raise ValueError(
            f"{name}={raw!r} is not a {'whole number' if kind is int else 'number'}"
        ) from None


@dataclass(frozen=True)
class DatabaseSettings:
    """The connection string and the pool sized around it.

    One default each, written here and nowhere else.
    """

    #: Empty means "not configured". Checked when the engine is built.
    url: str = ""
    #: Connections kept open per process. The runtime is one process with
    #: ``WAITRESS_THREADS`` request threads plus ``INGEST_WORKERS`` job
    #: workers and one packaging thread, so five checked-out connections with
    #: five more available under a burst covers every thread that can be
    #: inside a session at once without holding a connection per thread.
    pool_size: int = 5
    max_overflow: int = 5
    #: Seconds a caller waits for a connection before the pool gives up. A
    #: request that cannot get one fails as a 503 rather than hanging.
    pool_timeout: float = 30.0
    #: Seconds after which an idle connection is replaced. Below the default
    #: idle timeout of every managed PostgreSQL this is likely to run behind,
    #: so a connection is never handed out after the server has dropped it.
    pool_recycle: float = 1800.0
    #: Seconds to wait for the TCP connection itself. Without it a database
    #: that is merely unreachable takes the process's start-up with it.
    connect_timeout: float = 10.0
    #: Log every statement. Development only; the log is not redacted.
    echo: bool = False

    @property
    def configured(self) -> bool:
        return bool(self.url)

    def validate(self) -> "DatabaseSettings":
        if self.pool_size < 1:
            raise ValueError("DATABASE_POOL_SIZE must be at least 1")
        if self.max_overflow < 0:
            raise ValueError("DATABASE_MAX_OVERFLOW cannot be negative")
        for name, value in (("DATABASE_POOL_TIMEOUT", self.pool_timeout),
                            ("DATABASE_POOL_RECYCLE", self.pool_recycle),
                            ("DATABASE_CONNECT_TIMEOUT", self.connect_timeout)):
            if value <= 0:
                raise ValueError(f"{name} must be greater than zero")
        if self.url and not self.url.startswith("postgresql"):
            raise ValueError(
                f"{URL_ENV} must be a PostgreSQL URL (postgresql+psycopg://...); "
                "this application's persistence is PostgreSQL and nothing else"
            )
        return self

    # ------------------------------------------------------------ reporting
    @property
    def sanitized_url(self) -> str:
        """The connection string with the credential taken out.

        Reported by ``/api/ops/metrics`` and printed at start-up, both of
        which are unauthenticated or written to a log file, so what goes into
        them may name the host and the database and nothing else.
        """
        return sanitize_url(self.url)

    def to_dict(self) -> dict[str, Any]:
        return {
            "configured": self.configured,
            "url": self.sanitized_url,
            "pool_size": self.pool_size,
            "max_overflow": self.max_overflow,
            "pool_timeout_seconds": self.pool_timeout,
            "pool_recycle_seconds": self.pool_recycle,
            "connect_timeout_seconds": self.connect_timeout,
        }


def sanitize_url(url: str) -> str:
    """``postgresql+psycopg://user:pw@host/db`` -> ``postgresql+psycopg://host/db``."""
    if not url:
        return ""
    try:
        parts = urlsplit(url)
    except ValueError:  # pragma: no cover - urlsplit is forgiving
        return "(unparseable)"
    host = parts.hostname or ""
    if parts.port:
        host = f"{host}:{parts.port}"
    return urlunsplit((parts.scheme, host, parts.path, "", ""))


def normalize_url(url: str) -> str:
    """Give a bare ``postgresql://`` URL the driver this application uses."""
    url = (url or "").strip()
    if url.startswith("postgresql://"):
        return DEFAULT_DRIVER + url[len("postgresql") :]
    return url


def database_from_env(env: Optional[Mapping[str, str]] = None) -> DatabaseSettings:
    """The database configuration this process runs with."""
    env = os.environ if env is None else env
    return DatabaseSettings(
        url=normalize_url(env.get(URL_ENV) or ""),
        pool_size=_number(env, "DATABASE_POOL_SIZE", 5, int),
        max_overflow=_number(env, "DATABASE_MAX_OVERFLOW", 5, int),
        pool_timeout=_number(env, "DATABASE_POOL_TIMEOUT", 30.0, float),
        pool_recycle=_number(env, "DATABASE_POOL_RECYCLE", 1800.0, float),
        connect_timeout=_number(env, "DATABASE_CONNECT_TIMEOUT", 10.0, float),
        echo=(env.get("DATABASE_ECHO") or "").strip().lower() in {"1", "true", "yes", "on"},
    ).validate()
