"""Bring the database up to head, safely, before the server serves.

    python -m tools.migrate                 # wait for it, lock it, upgrade it
    python -m tools.migrate --check         # report the revisions, change nothing

This is what the container runs before ``python -m asgi`` (see
``docker-entrypoint.sh``). Running it by hand against a reachable database is
the same thing ``alembic upgrade head`` was, with three properties a bare
``alembic`` invocation does not have -- each of which is a way a deploy has
actually gone wrong rather than a precaution in the abstract.

**It waits.** ``docker compose`` starts this container once the database
reports healthy, but a database is healthy the moment it accepts a connection,
and a restarted server still replaying its write-ahead log refuses one for a
few seconds after that. Failing there gives a container that exits, is
restarted, and fails again slightly later -- a crash loop that resolves itself
and looks, in the log, exactly like one that will not. So a connection refused
is retried for ``CHAT_RAG_DB_WAIT`` seconds and only then is it an error.

**It takes a lock.** Alembic does not. Two application containers starting
together would both read "head is 0001", both run 0002, and the second would
fail partway through a schema the first was still creating. A PostgreSQL
advisory lock is held across the upgrade, so the second waits and then finds
there is nothing to do. The runtime is one process today (``asgi.py`` says
why), which makes this cheap insurance rather than a live requirement -- but it
is the requirement that would otherwise be discovered during the first deploy
that scaled.

**It says what it did.** A migration that runs as a side effect of a deploy is
only acceptable if the deploy log shows the revision it moved from and the one
it moved to. That is the objection ``alembic.ini`` raises against automatic
migrations, and this is the answer to it: the two revisions are printed, and
"already at head" is printed as its own outcome rather than as silence.

Set ``CHAT_RAG_MIGRATE_ON_START=0`` to skip it entirely -- for a deployment
that applies its schema through a reviewed, separate step. The application
then starts against whatever schema is there, and fails loudly on the first
query if that schema is behind.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

logger = logging.getLogger("RAG.migrate")

#: The advisory lock this repository uses for schema changes. Any constant
#: would do; it only has to be one nobody else in this database picks. Advisory
#: locks are per-database and are released when the session ends, so a killed
#: container cannot leave one behind.
MIGRATION_LOCK_KEY = 0x1CE5_2A67

#: How long a connection refused is treated as "not up yet" rather than as a
#: wrong address. Long enough for a database recovering its write-ahead log,
#: short enough that a genuinely wrong DATABASE_URL is reported within a
#: minute rather than looking like a slow start.
DEFAULT_WAIT_SECONDS = 60.0

#: Between attempts. Small: the common case is one or two.
RETRY_INTERVAL_SECONDS = 1.0


def _seconds(name: str, default: float) -> float:
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        raise SystemExit(f"{name}={raw!r} is not a number of seconds") from None
    if value < 0:
        raise SystemExit(f"{name}={raw!r} cannot be negative")
    return value


def wait_for_database(timeout: float | None = None) -> None:
    """Block until the database answers, or give up by name.

    ``DatabaseNotConfigured`` is not retried: no amount of waiting supplies a
    ``DATABASE_URL`` that was never set.
    """
    import storage
    from storage.engine import DatabaseNotConfigured, DatabaseUnavailable

    deadline = time.monotonic() + (DEFAULT_WAIT_SECONDS if timeout is None else timeout)
    attempt = 0
    while True:
        attempt += 1
        try:
            storage.require_reachable()
            if attempt > 1:
                print(f"[migrate] the database answered on attempt {attempt}")
            return
        except DatabaseNotConfigured as error:
            raise SystemExit(f"[migrate] {error}") from error
        except DatabaseUnavailable as error:
            if time.monotonic() >= deadline:
                raise SystemExit(
                    f"[migrate] gave up waiting for the database: {error}"
                ) from error
            if attempt == 1:
                print(f"[migrate] waiting for the database: {error}")
            # The pool now holds a connection that failed. Drop it, so the next
            # attempt dials again instead of being handed the same dead socket.
            storage.dispose()
            time.sleep(RETRY_INTERVAL_SECONDS)


def _alembic_config():
    from alembic.config import Config

    settings = Config(os.path.join(REPO_ROOT, "alembic.ini"))
    settings.set_main_option(
        "script_location", os.path.join(REPO_ROOT, "storage", "migrations")
    )
    return settings


def current_revision(connection) -> str | None:
    """What the database says it is at, or ``None`` for an empty one."""
    from alembic.runtime.migration import MigrationContext

    return MigrationContext.configure(connection).get_current_revision()


def head_revision() -> str | None:
    """What this checkout's migrations end at."""
    from alembic.script import ScriptDirectory

    return ScriptDirectory.from_config(_alembic_config()).get_current_head()


def upgrade(timeout: float | None = None) -> str:
    """Take the lock, upgrade to head, report what moved.

    Returns ``'created'``, ``'upgraded'`` or ``'current'``.
    """
    from alembic import command
    from sqlalchemy import text

    import storage

    wait_for_database(timeout)

    head = head_revision()
    # One connection for the whole thing: the advisory lock lives on the
    # session that took it, and env.py runs the migrations on the connection it
    # is handed (`context.attributes['connection']`) rather than opening a
    # second one that the lock would not cover.
    with storage.engine().connect() as connection:
        connection.execute(text("SELECT pg_advisory_lock(:key)"),
                           {"key": MIGRATION_LOCK_KEY})
        try:
            before = current_revision(connection)
            if before == head:
                print(f"[migrate] already at head ({head})")
                return "current"

            print(f"[migrate] {before or 'an empty database'} -> {head}")
            settings = _alembic_config()
            settings.attributes["connection"] = connection
            command.upgrade(settings, "head")
            connection.commit()
            print(f"[migrate] done: now at {head}")
            return "created" if before is None else "upgraded"
        finally:
            # Released explicitly rather than left to the session's end, so the
            # next container is not waiting on a connection this pool is merely
            # keeping warm.
            connection.execute(text("SELECT pg_advisory_unlock(:key)"),
                               {"key": MIGRATION_LOCK_KEY})
            connection.commit()


def report(timeout: float | None = None) -> int:
    """Say where the database is and where the migrations end. Change nothing."""
    import storage

    wait_for_database(timeout)
    head = head_revision()
    with storage.engine().connect() as connection:
        before = current_revision(connection)
    print(f"[migrate] database: {before or '(empty)'}")
    print(f"[migrate] head:     {head}")
    return 0 if before == head else 1


def enabled() -> bool:
    """Whether a start-up is expected to migrate. On unless told otherwise."""
    raw = (os.environ.get("CHAT_RAG_MIGRATE_ON_START") or "").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true",
                        help="report the revisions and change nothing")
    parser.add_argument("--wait", type=float, default=None, metavar="SECONDS",
                        help="how long to wait for the database "
                             f"(default: CHAT_RAG_DB_WAIT, else {DEFAULT_WAIT_SECONDS:.0f})")
    arguments = parser.parse_args(argv)

    timeout = arguments.wait
    if timeout is None:
        timeout = _seconds("CHAT_RAG_DB_WAIT", DEFAULT_WAIT_SECONDS)

    if arguments.check:
        return report(timeout)

    if not enabled():
        print("[migrate] CHAT_RAG_MIGRATE_ON_START is off; the schema is "
              "somebody else's step")
        return 0

    upgrade(timeout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
