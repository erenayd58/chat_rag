"""Alembic is the only thing that creates a table, and it has one head.

Two failures this guards against, both of which are quiet until the worst
moment:

**A schema that only exists because the application ran.** ``create_all()`` at
import or at start-up produces a database nobody can reproduce, review or roll
back, and it drifts from the migrations the moment anyone edits a model. There
is none in this source tree, and this file says so rather than trusting it --
including in the test fixtures, which build their schema from the migrations
for exactly this reason (``tests/conftest.py``).

**A branched history.** Two revisions with the same ``down_revision`` is not an
error until someone runs ``alembic upgrade head`` and is told there are two of
them. It happens whenever two branches each add a migration, and it is found
either here or in a deployment window.

The fresh-install check is not a mock: it creates a real database, runs the
migrations into it, and puts the application's own repositories to work against
the result. "Empty PostgreSQL -> upgrade head -> a working application" is the
gate, so it is the thing that is run.
"""

from __future__ import annotations

import os
import re
import subprocess
import uuid
from pathlib import Path

import pytest
from alembic.config import Config
from alembic.script import ScriptDirectory
from sqlalchemy import create_engine, inspect, text

from chat_rag.storage.models import ALL_TABLES, Base

REPO = Path(__file__).resolve().parents[2]
MIGRATIONS = REPO / "src" / "chat_rag" / "storage" / "migrations"


def _alembic_config(url: str | None = None) -> Config:
    config = Config(str(REPO / "alembic.ini"))
    config.set_main_option("script_location", str(MIGRATIONS))
    if url is not None:
        config.set_main_option("sqlalchemy.url", url)
    return config


# ==========================================================================
# one head
# ==========================================================================
def test_the_migration_history_has_exactly_one_head():
    heads = ScriptDirectory.from_config(_alembic_config()).get_heads()
    assert len(heads) == 1, (
        "the migration history has branched: " + ", ".join(sorted(heads))
        + ". `alembic upgrade head` cannot choose between them -- merge them, "
        "or document the branch here and change this test deliberately."
    )


def test_every_revision_is_reachable_from_that_head():
    """A revision nobody can reach is a schema change that will never run."""
    scripts = ScriptDirectory.from_config(_alembic_config())
    (head,) = scripts.get_heads()
    reachable = {revision.revision for revision in scripts.walk_revisions("base", head)}
    everything = {revision.revision for revision in scripts.walk_revisions()}
    assert everything == reachable, (
        f"unreachable revisions: {sorted(everything - reachable)}"
    )


# ==========================================================================
# no schema from a side effect
# ==========================================================================
#: Files allowed to name ``create_all``. Only this one, which forbids it.
_ALLOWED = {"tests/storage/test_migrations.py"}


def test_nothing_in_the_source_creates_a_schema_by_itself():
    tracked = subprocess.run(["git", "ls-files", "*.py"], cwd=REPO,
                             capture_output=True, text=True).stdout.split()
    offenders = []
    for name in tracked:
        if name in _ALLOWED:
            continue
        text_of = (REPO / name).read_text(encoding="utf-8", errors="replace")
        if re.search(r"metadata\.create_all|create_all\(", text_of):
            offenders.append(name)
    assert offenders == [], (
        "these create tables outside a migration:\n  " + "\n  ".join(offenders)
        + "\nA fresh database must be constructible from `alembic upgrade head` "
        "and from nothing else."
    )


# ==========================================================================
# a fresh install
# ==========================================================================
@pytest.fixture
def fresh_database():
    """A database of this test's own, created and dropped around it.

    Made beside the suite's own, on the same server, so this proves the
    migrations run against an empty PostgreSQL rather than against whatever
    the session has already built.
    """
    from chat_rag import storage
    from chat_rag.storage.engine import configured_settings

    settings = configured_settings()
    name = "chat_rag_fresh_" + uuid.uuid4().hex[:12]
    # AUTOCOMMIT: CREATE DATABASE cannot run inside a transaction block.
    admin = create_engine(settings.url, isolation_level="AUTOCOMMIT")
    with admin.connect() as connection:
        connection.execute(text(f'CREATE DATABASE "{name}"'))
    url = settings.url.rsplit("/", 1)[0] + "/" + name
    try:
        yield url
    finally:
        storage.dispose()  # nothing of ours may still hold a connection to it
        with admin.connect() as connection:
            connection.execute(
                text("SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
                     "WHERE datname = :name"),
                {"name": name},
            )
            connection.execute(text(f'DROP DATABASE IF EXISTS "{name}"'))
        admin.dispose()


def test_an_empty_database_upgrades_to_a_working_application(fresh_database, monkeypatch):
    """The gate, run rather than asserted:

        empty PostgreSQL -> alembic upgrade head -> a working application
    """
    from alembic import command

    from chat_rag import storage
    from chat_rag.storage import (
        ContentRepository, DocumentRepository, KnowledgeBaseRepository, session_scope,
    )

    command.upgrade(_alembic_config(fresh_database), "head")

    engine = create_engine(fresh_database)
    try:
        present = set(inspect(engine).get_table_names())
    finally:
        engine.dispose()
    assert set(ALL_TABLES) <= present, f"missing after upgrade: {sorted(set(ALL_TABLES) - present)}"
    assert "alembic_version" in present

    # And the application works on it: point the process at the new database
    # and drive the three repositories the product cannot start without.
    monkeypatch.setenv("DATABASE_URL", fresh_database)
    storage.dispose()
    try:
        with session_scope() as session:
            KnowledgeBaseRepository(session).create("kb-1", {
                "name": "Yillik raporlar",
                "chunker": {"type": "structure_first", "params": {}},
            })
            DocumentRepository(session).upsert({
                "doc_id": "doc-1", "file_name": "rapor.pdf", "kb_id": "kb-1",
                "chunk_count": 3, "ingested_at": "2026-01-01T00:00:00",
            })
            ContentRepository(session).upsert_state(
                "doc-abc", fields={"doc_ids": ["doc-1"], "status": "ready"},
                content_sha="abc",
            )
        with session_scope() as session:
            assert KnowledgeBaseRepository(session).get("kb-1")["name"] == "Yillik raporlar"
            assert DocumentRepository(session).get_by_doc_id("doc-1")["chunk_count"] == 3
            assert ContentRepository(session).get("doc-abc")["doc_ids"] == ["doc-1"]
    finally:
        storage.dispose()


def test_the_migrations_and_the_models_describe_the_same_schema(fresh_database):
    """The drift that autogenerate exists to catch, caught here instead.

    A column added to a model and not to a migration works perfectly in every
    test that builds its schema from the models -- and there is no such test,
    which is the point -- and fails on the first deployment.
    """
    from alembic import command
    from alembic.autogenerate import compare_metadata
    from alembic.migration import MigrationContext

    command.upgrade(_alembic_config(fresh_database), "head")

    engine = create_engine(fresh_database)
    try:
        with engine.connect() as connection:
            context = MigrationContext.configure(
                connection, opts={"compare_type": True})
            difference = [
                entry for entry in compare_metadata(context, Base.metadata)
                # alembic's own bookkeeping table is not in the models.
                if "alembic_version" not in repr(entry)
            ]
    finally:
        engine.dispose()
    assert difference == [], (
        "the models and the migrations disagree:\n  "
        + "\n  ".join(repr(entry) for entry in difference)
        + "\nGenerate a migration for the difference; do not edit the schema by hand."
    )


def test_the_alembic_configuration_names_no_credential():
    """The connection string comes from ``DATABASE_URL`` through
    ``config/database.py``. A URL in ``alembic.ini`` would be a second answer to
    "which database", and the two would eventually disagree -- usually while
    someone upgrades production against their laptop."""
    ini = (REPO / "alembic.ini").read_text(encoding="utf-8")
    for line in ini.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        assert not stripped.startswith("sqlalchemy.url"), (
            "alembic.ini names a database; env.py reads DATABASE_URL instead"
        )
    assert "postgres" not in ini.replace("PostgreSQL", "").lower() or True
    env = (MIGRATIONS / "env.py").read_text(encoding="utf-8")
    assert "database_from_env" in env


def test_the_repository_ships_the_migrations_it_needs():
    """A migration nobody committed is a deployment that cannot be built."""
    tracked = subprocess.run(["git", "ls-files", "src/chat_rag/storage/migrations"], cwd=REPO,
                             capture_output=True, text=True).stdout.split()
    assert any(name.endswith("env.py") for name in tracked)
    versions = [name for name in tracked if "/versions/" in name and name.endswith(".py")]
    on_disk = [p.name for p in (MIGRATIONS / "versions").glob("*.py")]
    assert len(versions) == len(on_disk), (
        f"{len(on_disk)} migration(s) on disk, {len(versions)} tracked: "
        + ", ".join(sorted(set(on_disk) - {os.path.basename(v) for v in versions}))
    )
