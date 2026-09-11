"""The schema: where a database is, and bringing it to head.

Nothing in this application creates a table as a side effect; a fresh database
is built by the migrations and by nothing else (``alembic.ini`` says why). That
made applying them a step outside the library -- ``alembic upgrade head`` from
a checkout, ``tools/migrate.py`` from the container -- and a program that
installed the wheel had the migrations and no supported way to run them. This
is the use case, so the facade can offer it.

Two refusals, both the application's own so a caller can catch them where it
catches everything else this library raises. A database that is not configured
or cannot be reached is :class:`~application.errors.Unavailable`: nothing
about the request is wrong, and the same call succeeds once the database
answers. A migration that was attempted and failed -- a server without the
``vector`` extension is the usual one -- is
:class:`~application.errors.ProcessingFailed`, with the cause chained.
"""

from __future__ import annotations

import logging
from typing import Any

from sqlalchemy.exc import SQLAlchemyError

from chat_rag.storage import schema
from chat_rag.storage.engine import DatabaseNotConfigured, DatabaseUnavailable

from .errors import ProcessingFailed, Unavailable

logger = logging.getLogger("RAG.schema")


def _reachable(services) -> Any:
    """This container's database, once it has answered."""
    database = services.runtime.database
    try:
        database.require_reachable()
    except (DatabaseNotConfigured, DatabaseUnavailable) as error:
        raise Unavailable(str(error)) from error
    return database


def upgrade(services) -> dict:
    """Bring this container's database to head, and say what that took.

    ``outcome`` is one of three words: ``created`` for an empty database that
    now has the schema, ``upgraded`` for one that was behind, ``current`` for
    one that was already at head -- reported as its own answer rather than as
    silence, because every restart runs this and a start-up that quietly
    applied the wrong thing would look exactly like one that applied nothing.
    """
    database = _reachable(services)
    try:
        before, head = schema.upgrade(database)
    except SQLAlchemyError as error:
        raise ProcessingFailed(
            f"the schema could not be brought to head: {type(error).__name__}: {error}"
        ) from error
    if before == head:
        outcome = "current"
    else:
        outcome = "created" if before is None else "upgraded"
        logger.info("schema %s: %s -> %s", outcome, before or "(empty)", head)
    return {"before": before, "after": head, "outcome": outcome}
