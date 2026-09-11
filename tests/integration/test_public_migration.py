"""An installed library can build the schema it expects.

The migrations have shipped in the wheel since L5, but applying them was a
step outside the package: ``alembic upgrade head`` needs the repository's
``alembic.ini`` and ``tools/migrate.py`` needs the repository. A program that
``pip install``-ed ``chat-rag`` had the migrations and no supported way to run
them. :meth:`Engine.migrate` and :func:`migrate_database` are that way, and
this module proves them against an **empty** PostgreSQL rather than the
database the session has already built (``fresh_database``).

What is asserted is the whole claim and not a mock of it: an empty database,
then a schema the application's own repositories work on, then the same call
again reporting that there was nothing to do. The refusals are the published
ones -- a caller catching ``chat_rag.Unavailable`` catches an unreachable
database here the way it catches an unconfigured provider elsewhere.
"""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine, inspect
from sqlalchemy.exc import OperationalError

from chat_rag import Engine, EngineConfig, Settings, migrate_database
from chat_rag.api import Migration, ProcessingFailed, Unavailable
from chat_rag.config.database import DatabaseSettings
from chat_rag.storage.models import ALL_TABLES
from chat_rag.storage import schema


def _tables(url: str) -> set[str]:
    engine = create_engine(url)
    try:
        return set(inspect(engine).get_table_names())
    finally:
        engine.dispose()


# ------------------------------------------------------------ the engine's
def test_an_engine_creates_its_schema_and_then_works_on_it(fresh_database):
    """empty PostgreSQL -> engine.migrate() -> a knowledge base in it."""
    assert _tables(fresh_database) == set(), "the fixture handed over a used database"

    with Engine(EngineConfig(database_url=fresh_database,
                             retrieval_profile="bm25_only")) as engine:
        first = engine.migrate()
        assert isinstance(first, Migration)
        assert first.outcome == "created"
        assert first.before is None
        assert first.after == schema.head_revision()
        assert first.changed is True

        present = _tables(fresh_database)
        assert set(ALL_TABLES) <= present, sorted(set(ALL_TABLES) - present)
        assert "alembic_version" in present

        # And it is a working database: the engine that migrated it uses it.
        kb = engine.knowledge_bases.create("Reports")
        assert [k.id for k in engine.knowledge_bases.list()] == [kb.id]


def test_migrating_a_database_at_head_reports_that_and_changes_nothing(fresh_database):
    """The restart case. Its own answer, not silence and not an error."""
    with Engine(EngineConfig(database_url=fresh_database)) as engine:
        assert engine.migrate().outcome == "created"
        again = engine.migrate()
        assert again.outcome == "current"
        assert again.changed is False
        assert again.before == again.after == schema.head_revision()


def test_a_closed_engine_refuses_to_migrate(fresh_database):
    engine = Engine(EngineConfig(database_url=fresh_database))
    engine.close()
    with pytest.raises(RuntimeError, match="closed"):
        engine.migrate()


# ---------------------------------------------------------- the function
def test_migrate_database_needs_no_engine_afterwards(fresh_database):
    """The deployment-step spelling: one call, the schema, nothing held."""
    from chat_rag import runtime

    outcome = migrate_database(database_url=fresh_database)
    assert outcome.outcome == "created"
    assert set(ALL_TABLES) <= _tables(fresh_database)
    assert migrate_database(database_url=fresh_database).outcome == "current"
    # The engine built for the call was never the process default: a program
    # that migrates first must not find its pool handed to whatever runs next.
    assert runtime.active() is None


def test_migrate_database_takes_a_config_or_keywords_but_not_both(fresh_database):
    with pytest.raises(TypeError):
        migrate_database(EngineConfig(database_url=fresh_database),
                         database_url=fresh_database)


# ------------------------------------------------------------ the refusals
def test_an_unconfigured_database_is_refused_by_name():
    """The refusal is the published one, and it names what to set."""
    with Engine(EngineConfig(read_environment=False)) as engine:
        with pytest.raises(Unavailable, match="DATABASE_URL"):
            engine.migrate()


def test_an_unreachable_database_is_refused_without_its_password():
    """Nothing about the request is wrong -- the same call succeeds once the
    database answers -- so it is ``Unavailable``. The message carries the host
    and never the credential."""
    nowhere = "postgresql+psycopg://user:s3cret@127.0.0.1:1/nothing"
    # The full ``Settings`` is the escape hatch for a field ``EngineConfig``
    # does not name -- here the connect timeout, so a port that drops packets
    # rather than refusing them does not cost the suite ten seconds.
    settings = Settings(database=DatabaseSettings(url=nowhere, connect_timeout=1.0))
    with Engine(EngineConfig(settings=settings)) as engine:
        with pytest.raises(Unavailable) as refused:
            engine.migrate()
    assert "127.0.0.1:1" in str(refused.value)
    assert "s3cret" not in str(refused.value)


def test_a_migration_that_fails_is_a_processing_failure_with_its_cause(
        fresh_database, monkeypatch):
    """A server without the ``vector`` extension is the real case; the
    failure is raised as the application's own, with the database error
    chained so the operator sees which statement refused."""
    def refuse(database):
        raise OperationalError("CREATE EXTENSION vector", {}, Exception("not available"))

    monkeypatch.setattr(schema, "upgrade", refuse)
    with Engine(EngineConfig(database_url=fresh_database)) as engine:
        with pytest.raises(ProcessingFailed) as failed:
            engine.migrate()
    assert "OperationalError" in str(failed.value)
    assert isinstance(failed.value.__cause__, OperationalError)
    assert _tables(fresh_database) == set()
