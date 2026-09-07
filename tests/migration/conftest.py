"""The migration contract suite: what must stay true when the implementation changes.

This directory is deliberately separate from ``tests/unit`` and
``tests/integration``. Those suites answer "does this code work?". This one
answers a different question, asked repeatedly during a platform migration:

    the framework, the module layout, the persistence, the vector store and
    the frontend are all going to be replaced -- what must the replacement
    still do?

So every test here is written against an **observable contract** rather than
against the implementation that currently satisfies it. A test that would
fail merely because Flask became FastAPI, because ``.knowledge_bases.json``
became a table, or because Chroma became pgvector, does not belong here --
it belongs in the suite for the thing it is testing. What belongs here is
the behaviour a user, an API client or the sibling repository would notice.

Run it alone:

    python -m pytest tests/migration -q
    python -m pytest -m migration -q        # the same set, by marker

It is fast on purpose (no provider, no model download, no network, one small
in-memory corpus) so it can be run on every step of the migration rather
than at the end of it.
"""

from __future__ import annotations

import pytest


def pytest_collection_modifyitems(items):
    """Everything in this directory carries the ``migration`` marker."""
    for item in items:
        if "tests/migration/" in item.nodeid.replace("\\", "/"):
            item.add_marker(pytest.mark.migration)
