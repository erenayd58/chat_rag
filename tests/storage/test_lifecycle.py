"""The connection pool: one engine, no leak, a controlled failure, no secret.

The four ways a database integration goes wrong operationally, in the order
they bite:

1. **an engine per call.** Every repository call opening its own pool is a TCP
   connection and a PostgreSQL backend per knowledge-base lookup, garbage
   collected eventually. It looks fine on one developer's machine and takes the
   server's connection limit at ten users.
2. **a session nobody closes.** The connection goes back to the pool when the
   session is closed, not when it is finished with. Under load, "eventually" is
   every connection at once.
3. **an unreachable database at start-up.** The application must refuse and say
   which host, not bind a port and fail every request with a stack trace.
4. **a credential in a body.** ``/api/ops/metrics`` is unauthenticated and a
   PostgreSQL DSN carries a password.
"""

from __future__ import annotations

import sys
import threading

import pytest

from chat_rag import storage
from chat_rag.storage import session_scope
from chat_rag.storage.engine import DatabaseNotConfigured, DatabaseUnavailable


@pytest.fixture(autouse=True)
def _restore_the_engine():
    """Give the next test the engine this session's configuration builds.

    Several tests here point the process at a database that is not there. The
    environment is restored for them (``tests/conftest.py``), but the engine is
    a process-wide object built from the environment as it was at the time, so
    it has to be dropped as well -- including when a test fails part way.
    """
    yield
    storage.dispose()


def _checked_out() -> int:
    return storage.pool_status()["pool"]["checkedout"]


# ==========================================================================
# one engine
# ==========================================================================
def test_the_engine_is_built_once_and_shared():
    first = storage.engine()
    for _ in range(20):
        with session_scope() as session:
            session.execute(__import__("sqlalchemy").text("SELECT 1"))
    assert storage.engine() is first, "a repository call built a second engine"


def test_importing_the_storage_package_opens_no_connection(monkeypatch):
    """``python -m cli manifest`` and ``tools/import_smoke.py`` import the whole
    application on machines with no PostgreSQL at all. The first call that
    needs a row is where a missing database is reported -- not the import."""
    storage.dispose()
    monkeypatch.delenv("DATABASE_URL", raising=False)

    # Reading the configuration is free, and says what is wrong without trying.
    assert storage.describe()["configured"] is False
    assert storage.health() == {"configured": False, "reachable": False,
                                "error": "DATABASE_URL is not set"}
    with pytest.raises(DatabaseNotConfigured):
        storage.engine()


# ==========================================================================
# no leak
# ==========================================================================
def test_every_session_returns_its_connection_to_the_pool():
    storage.engine()
    before = _checked_out()

    for _ in range(50):
        with session_scope() as session:
            session.execute(__import__("sqlalchemy").text("SELECT 1"))

    assert _checked_out() == before, "a session held its connection"


def test_a_session_that_raises_still_returns_its_connection():
    """The leak that only shows up once something starts failing."""
    storage.engine()
    before = _checked_out()

    for _ in range(20):
        with pytest.raises(RuntimeError):
            with session_scope():
                raise RuntimeError("the unit of work failed")

    assert _checked_out() == before


def test_concurrent_units_of_work_do_not_exhaust_the_pool():
    """More callers than the pool holds. Each waits for a connection and gives
    it back; none of them fails, and none of them keeps one afterwards."""
    storage.engine()
    before = _checked_out()
    errors: list[BaseException] = []

    def work() -> None:
        try:
            for _ in range(10):
                with session_scope() as session:
                    session.execute(__import__("sqlalchemy").text("SELECT 1"))
        except BaseException as error:  # noqa: BLE001
            errors.append(error)

    threads = [threading.Thread(target=work) for _ in range(12)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=60)

    assert not errors, errors
    assert _checked_out() == before


# ==========================================================================
# a controlled failure
# ==========================================================================
def test_an_unreachable_database_is_a_named_refusal_not_a_traceback(monkeypatch):
    storage.dispose()
    monkeypatch.setenv(
        "DATABASE_URL",
        "postgresql+psycopg://nobody:nothing@127.0.0.1:1/absent",
    )
    monkeypatch.setenv("DATABASE_CONNECT_TIMEOUT", "1")

    with pytest.raises(DatabaseUnavailable) as raised:
        storage.require_reachable()

    message = str(raised.value)
    assert "127.0.0.1:1/absent" in message, "it must say which host it tried"
    assert "nothing" not in message, "and not what it tried with"

    # And health answers instead of raising, because that is the moment it is
    # asked: the probe polling a degraded service.
    report = storage.health()
    assert report["configured"] is True and report["reachable"] is False
    assert "nothing" not in repr(report)


def test_the_entrypoints_refuse_to_serve_without_a_database(monkeypatch):
    """A start-up refusal with an exit status, not a server that binds and
    then fails every request."""
    from runtime import bootstrap

    storage.dispose()
    monkeypatch.delenv("DATABASE_URL", raising=False)
    with pytest.raises(SystemExit) as raised:
        bootstrap.require_database()
    assert raised.value.code == 2


# ==========================================================================
# no secret
# ==========================================================================
def test_no_report_of_the_database_carries_its_credential(monkeypatch):
    storage.dispose()
    monkeypatch.setenv(
        "DATABASE_URL",
        "postgresql+psycopg://reporter:s3cr3t-p4ssw0rd@db.example.com:5432/prod",
    )

    described = storage.describe()
    assert described["url"] == "postgresql+psycopg://db.example.com:5432/prod"
    assert "s3cr3t" not in repr(described) and "reporter" not in repr(described)

    from chat_rag.config import Settings

    effective = Settings().effective_configuration()
    assert "s3cr3t" not in repr(effective)
    assert effective["database"]["url"].endswith("/prod")


def test_the_configuration_report_costs_no_round_trip(monkeypatch):
    """The start-up banner and ``/api/ops/metrics`` both read it, and neither
    should be able to hang on a database that is merely slow."""
    storage.dispose()
    monkeypatch.setenv("DATABASE_URL",
                       "postgresql+psycopg://nobody:nothing@127.0.0.1:1/absent")

    # ``storage.engine`` is the re-exported accessor, not the submodule of the
    # same name; the module itself is reached through sys.modules.
    engine_module = sys.modules["chat_rag.storage.engine"]

    def refuse(*args, **kwargs):
        raise AssertionError("describe() opened a connection")

    monkeypatch.setattr(engine_module, "create_engine", refuse)
    assert storage.describe()["url"].endswith("/absent")
