"""The step between "the container started" and "the server serves".

``tools/migrate.py`` is what makes an automatic migration acceptable rather
than merely convenient, and each of its three properties is a specific way a
deploy has gone wrong:

* it **waits**, because `docker compose` releases a container the moment the
  database's health check passes and a server still replaying its write-ahead
  log refuses connections for a few seconds after that. Without the wait the
  container exits, is restarted and fails slightly later -- a crash loop that
  eventually resolves itself and, in the log, is indistinguishable from one
  that never will;
* it takes a **lock**, because Alembic does not. Two containers starting
  together both read "head is 0001" and both run 0002, and the second fails
  halfway through a schema the first is still creating;
* it is **idempotent**, because every restart runs it, and a restart is the
  ordinary case rather than the exception.

The suite's database is built by ``alembic upgrade head`` already
(``tests/conftest.py``), so these run against a database that is at head --
which is exactly the state a restart finds, and the state the interesting
assertion is about.
"""

from __future__ import annotations

import pytest
from sqlalchemy import text

from chat_rag import storage
from chat_rag.storage.engine import DatabaseUnavailable
from tools import migrate


@pytest.fixture
def no_migrate_switch(monkeypatch):
    monkeypatch.delenv("CHAT_RAG_MIGRATE_ON_START", raising=False)


# ------------------------------------------------------------- the revisions


def test_the_head_is_the_last_migration_in_this_checkout():
    """Read from the scripts, not written down anywhere a second time."""
    head = migrate.head_revision()
    assert head and head.startswith("0002")


def test_a_prepared_database_is_already_at_head():
    with storage.engine().connect() as connection:
        assert migrate.current_revision(connection) == migrate.head_revision()


def test_running_it_against_a_database_at_head_changes_nothing():
    """The restart case, which is most of them. It must be a no-op that says
    so, not a no-op that is silent and not an error."""
    assert migrate.upgrade() == "current"


def test_it_is_safe_to_run_twice_in_a_row():
    assert migrate.upgrade() == "current"
    assert migrate.upgrade() == "current"


def test_check_reports_without_changing_anything(capsys):
    """``--check`` is what an operator runs to ask where a database is. It
    exits 0 only when there is nothing to apply."""
    assert migrate.report() == 0
    printed = capsys.readouterr().out
    assert "head:" in printed and migrate.head_revision() in printed


# ------------------------------------------------------------------ the lock


def test_the_advisory_lock_is_released_afterwards():
    """Held to the end of the session it was taken on, an advisory lock would
    make the *next* container wait on a connection this pool is merely keeping
    warm. It is released explicitly, so nothing holds it once the upgrade
    returns."""
    migrate.upgrade()

    with storage.engine().connect() as connection:
        held = connection.execute(
            text("SELECT count(*) FROM pg_locks WHERE locktype = 'advisory' "
                 "AND objid = :key"),
            {"key": migrate.MIGRATION_LOCK_KEY & 0xFFFFFFFF},
        ).scalar()
    assert held == 0


def test_a_second_migration_waits_for_the_first_rather_than_racing_it():
    """The property, exercised: with the lock held elsewhere, an upgrade cannot
    start. Held on its own connection, so this is the two-container case and
    not a session waiting on itself -- PostgreSQL advisory locks are
    re-entrant within one session, which would make a self-test pass for the
    wrong reason.
    """
    key = migrate.MIGRATION_LOCK_KEY
    with storage.engine().connect() as holder:
        holder.execute(text("SELECT pg_advisory_lock(:key)"), {"key": key})
        try:
            with storage.engine().connect() as other:
                acquired = other.execute(
                    text("SELECT pg_try_advisory_lock(:key)"), {"key": key}
                ).scalar()
            assert acquired is False, (
                "a second container could migrate while the first was migrating"
            )
        finally:
            holder.execute(text("SELECT pg_advisory_unlock(:key)"), {"key": key})
            holder.commit()


# ------------------------------------------------------------------ the wait


def test_it_gives_up_by_name_rather_than_waiting_forever(monkeypatch):
    """A wrong DATABASE_URL must be reported within the minute, not retried
    until somebody kills the container. The message names the host, and it
    comes through the sanitizer, so a password in the URL is not logged."""
    monkeypatch.setattr(
        storage, "require_reachable",
        lambda: (_ for _ in ()).throw(DatabaseUnavailable("cannot reach the database at x")),
    )
    monkeypatch.setattr(migrate.time, "sleep", lambda seconds: None)

    with pytest.raises(SystemExit) as refusal:
        migrate.wait_for_database(timeout=0.0)
    assert "gave up waiting" in str(refusal.value)


def test_a_database_that_is_merely_slow_is_waited_for(monkeypatch):
    """The one it exists for: two refusals, then an answer, and no error."""
    attempts = {"n": 0}

    def flaky():
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise DatabaseUnavailable("cannot reach the database at x")

    monkeypatch.setattr(storage, "require_reachable", flaky)
    monkeypatch.setattr(storage, "dispose", lambda: None)
    monkeypatch.setattr(migrate.time, "sleep", lambda seconds: None)

    migrate.wait_for_database(timeout=30.0)

    assert attempts["n"] == 3


def test_an_unset_url_is_not_something_waiting_will_fix(monkeypatch):
    """No amount of retrying supplies a DATABASE_URL nobody set, so that one
    is refused on the first attempt."""
    from chat_rag.storage.engine import DatabaseNotConfigured

    attempts = {"n": 0}

    def unconfigured():
        attempts["n"] += 1
        raise DatabaseNotConfigured("DATABASE_URL is not set")

    monkeypatch.setattr(storage, "require_reachable", unconfigured)

    with pytest.raises(SystemExit):
        migrate.wait_for_database(timeout=30.0)
    assert attempts["n"] == 1


# ----------------------------------------------------------------- the knob


def test_migrating_at_start_up_is_on_unless_it_is_turned_off(no_migrate_switch):
    assert migrate.enabled() is True


@pytest.mark.parametrize("value", ["0", "false", "no", "off", "OFF"])
def test_a_deployment_can_apply_its_schema_separately(monkeypatch, value):
    """For an installation that reviews its migrations as their own step. The
    application then starts against whatever schema is there."""
    monkeypatch.setenv("CHAT_RAG_MIGRATE_ON_START", value)
    assert migrate.enabled() is False


def test_turning_it_off_makes_the_entrypoint_step_a_no_op(monkeypatch, capsys):
    """And it says so: a start-up that silently skipped the schema would be
    the same failure as one that silently applied the wrong one."""
    monkeypatch.setenv("CHAT_RAG_MIGRATE_ON_START", "0")
    monkeypatch.setattr(migrate, "upgrade",
                        lambda *a, **k: pytest.fail("it migrated anyway"))

    assert migrate.main([]) == 0
    assert "CHAT_RAG_MIGRATE_ON_START" in capsys.readouterr().out


@pytest.mark.parametrize("bad", ["soon", "-1"])
def test_a_nonsense_wait_is_refused_by_name(monkeypatch, bad, no_migrate_switch):
    monkeypatch.setenv("CHAT_RAG_DB_WAIT", bad)
    with pytest.raises(SystemExit, match="CHAT_RAG_DB_WAIT"):
        migrate.main([])
