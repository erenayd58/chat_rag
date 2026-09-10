"""How Alembic reaches this application's database.

One rule: the URL is the application's, read through ``config`` so the ``.env``
precedence that decides where the application writes is the same precedence
that decides where a migration runs. A connection string in ``alembic.ini``
would be a second answer to that question, and the two would eventually
disagree -- usually while someone is upgrading production against their
laptop's database.
"""

from __future__ import annotations

import os
import sys

from alembic import context
from sqlalchemy import engine_from_config, pool

# ``chat_rag`` is normally an installed distribution and needs nothing here.
# A checkout that has not been installed still has to migrate, so the directory
# holding the package -- ``src`` -- is put on the path as a fallback. This file
# is ``src/chat_rag/storage/migrations/env.py``: four levels up.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))))

from chat_rag import config  # noqa: F401,E402  - applies .env exactly as the application does
from chat_rag.config.database import database_from_env  # noqa: E402
from chat_rag.storage.models import Base  # noqa: E402

target_metadata = Base.metadata


def _url() -> str:
    """Which database to migrate.

    ``DATABASE_URL`` normally, so the migrations and the application cannot be
    pointed at two different places. A caller that sets ``sqlalchemy.url`` on
    the config object it passes in has named one explicitly and wins -- that is
    how a test builds a throwaway database, and how an operator migrates a
    second deployment without exporting anything.
    """
    explicit = (context.config.get_main_option("sqlalchemy.url", None) or "").strip()
    if explicit:
        return explicit
    settings = database_from_env()
    if not settings.configured:
        raise SystemExit(
            "DATABASE_URL is not set. Alembic runs against the same database "
            "the application does; see docs/database.md."
        )
    return settings.url


def run_migrations_offline() -> None:
    """Emit SQL to stdout instead of running it, for a reviewed deployment."""
    context.configure(
        url=_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        compare_type=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = context.config.attributes.get("connection", None)
    if connectable is not None:
        # A caller (the test fixture, a smoke check) handed us its own
        # connection; run inside its transaction rather than opening a second.
        context.configure(connection=connectable, target_metadata=target_metadata,
                          compare_type=True)
        with context.begin_transaction():
            context.run_migrations()
        return

    section = context.config.get_section(context.config.config_ini_section) or {}
    section["sqlalchemy.url"] = _url()
    engine = engine_from_config(section, prefix="sqlalchemy.", poolclass=pool.NullPool)
    with engine.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata,
                          compare_type=True)
        with context.begin_transaction():
            context.run_migrations()
    engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
