"""Alembic, reached from an installed package rather than from a checkout.

The migrations ship in the wheel (``pyproject.toml`` says why they are package
data), but ``alembic.ini`` does not -- it is repository configuration, with a
``script_location`` written relative to the checkout. A consumer who installed
``chat-rag`` therefore had the schema and no supported way to apply it. This
module is that way: the Alembic configuration built from *where this package
is*, and the operations over it that the entrypoint, the test session and
the public API all need: where a database is, where the migrations end, and
bringing the one to the other.

Three properties, and each is a way a deploy has actually gone wrong:

**One connection, one lock.** Alembic takes no lock of its own, so two
processes upgrading together would both read "head is 0001", both run 0002,
and the second would fail partway through a schema the first was still
creating. A PostgreSQL advisory lock is held across the upgrade, on the same
connection the migrations run on (``env.py`` honours
``config.attributes['connection']``), so the second waits and then finds
nothing to do. The runtime is one process today; this is the requirement that
would otherwise be discovered by the first deploy that scaled.

**Idempotent.** Every restart runs it, and a restart is the ordinary case. A
database already at head is reported as such, not treated as an error and not
migrated again.

**It says what it did.** Both operations answer with the revision the database
was at and the one the package's migrations end at, so a caller -- an
entrypoint log, an operator, a program -- can print the move rather than
trusting silence.

The URL is never read here. ``env.py`` runs on the connection it is handed,
and the connection comes from a :class:`Database` -- which is to say from the
settings the engine was built with, under the same precedence as every other
query it will make.
"""

from __future__ import annotations

import os
from typing import Optional

from sqlalchemy import text
from sqlalchemy.engine import Connection

from .engine import Database

#: The advisory lock schema changes are made under. Any constant would do; it
#: only has to be one nobody else in this database picks. Advisory locks are
#: per-database and released when the session ends, so a killed process cannot
#: leave one behind.
MIGRATION_LOCK_KEY = 0x1CE5_2A67

#: Where the migrations are, inside this package. Package data, resolved from
#: this file rather than from a working directory, so it is the same answer
#: from a checkout, an editable install and a wheel.
MIGRATIONS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "migrations")


def alembic_config(url: Optional[str] = None):
    """The Alembic configuration for this package's migrations.

    Built in memory rather than read from ``alembic.ini``: the file is the
    repository's, and everything the migrations need from it is the script
    location, which is known from here. ``url`` names a database explicitly
    -- ``env.py`` lets it win over ``DATABASE_URL`` -- which is how a test
    builds a throwaway database.
    """
    from alembic.config import Config

    config = Config()
    config.set_main_option("script_location", MIGRATIONS)
    # Alembic's own default for splitting multi-path options; set so a newer
    # Alembic does not warn about a legacy separator on a config that has none.
    config.set_main_option("path_separator", "os")
    if url is not None:
        config.set_main_option("sqlalchemy.url", url)
    return config


def head_revision() -> Optional[str]:
    """What this package's migrations end at."""
    from alembic.script import ScriptDirectory

    return ScriptDirectory.from_config(alembic_config()).get_current_head()


def current_revision(connection: Connection) -> Optional[str]:
    """What the database says it is at, or ``None`` for an empty one."""
    from alembic.runtime.migration import MigrationContext

    return MigrationContext.configure(connection).get_current_revision()


def upgrade(database: Database) -> tuple[Optional[str], Optional[str]]:
    """Take the lock, bring the database to head, say what moved.

    Returns ``(before, head)``: the revision the database was at -- ``None``
    for an empty one -- and the one it is at now. Equal means nothing was
    applied.
    """
    from alembic import command

    head = head_revision()
    # One connection for the whole thing: the advisory lock lives on the
    # session that took it, and env.py runs the migrations on the connection
    # it is handed rather than opening a second one the lock would not cover.
    with database.engine().connect() as connection:
        connection.execute(text("SELECT pg_advisory_lock(:key)"),
                           {"key": MIGRATION_LOCK_KEY})
        try:
            before = current_revision(connection)
            if before != head:
                config = alembic_config()
                config.attributes["connection"] = connection
                command.upgrade(config, "head")
                connection.commit()
            return before, head
        finally:
            # Released explicitly rather than left to the session's end, so
            # the next process is not waiting on a connection this pool is
            # merely keeping warm.
            connection.execute(text("SELECT pg_advisory_unlock(:key)"),
                               {"key": MIGRATION_LOCK_KEY})
            connection.commit()
