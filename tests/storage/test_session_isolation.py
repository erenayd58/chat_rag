"""The suite's own isolation: a test's background work ends with the test.

``tests/conftest.py`` gives every test empty tables by truncating them, and a
truncate cannot share the database with a writer. The writers this process
has are its own worker threads -- the ingest job manager's and the Viewer
packager's -- and each belongs to the runtime of the container that started
it. The session promises that none of them is still working when a test
ends, whatever the test itself did about it, and that the truncate never
runs into one. This file holds the session to that promise.

Two tests, in this order, on purpose. The first starts a packaging build on
a container of its own and returns at once, the way a test that only cared
about the upload would; the build takes a moment and then writes a row. The
second test starts with empty tables and finds no row and no build in
flight. Without the session's wait, the first test's worker would write into
the second test's database -- or the second test's truncate would find the
worker holding ``contents`` and PostgreSQL would report a deadlock, which is
exactly how the flake this guards against showed up.

The build is replaced at the module attribute the worker resolves it from,
so no chunker runs; what is exercised is the wait, not the packaging.
"""

from __future__ import annotations

import threading
import time

from chat_rag.application.services import build_services
from chat_rag.components.viewer import analysis
from chat_rag.storage import ContentRepository, session_scope

KEY = "doc-session-isolation-probe"
BUILD_SECONDS = 0.75

#: What the first test's build did, for the second test to read.
_probe = {"container": None, "wrote_at": None}


def test_a_build_left_running_finishes_before_the_test_is_over(monkeypatch):
    """Returns with the build still running, as an inattentive test would."""
    written = threading.Event()

    def slow_build(key: str) -> dict:
        time.sleep(BUILD_SECONDS)
        state = analysis._set_state(key, status=analysis.STATUS_READY)
        _probe["wrote_at"] = time.monotonic()
        written.set()
        return state

    monkeypatch.setattr(analysis, "build", slow_build)

    services = build_services()
    _probe["container"] = services
    with services.activate():
        assert analysis.enqueue(KEY) == analysis.STATUS_PENDING
        with analysis.state().lock:
            assert KEY in analysis.state().inflight, "the build was not accepted"
    # Deliberately no wait: the row lands after this function has returned.
    assert not written.is_set(), (
        "the probe build finished before the test did; raise BUILD_SECONDS so "
        "the second test is a test of the wait")


def test_the_next_test_starts_with_empty_tables_and_an_idle_packager():
    services = _probe["container"]
    assert services is not None, "the previous test did not run first"
    assert _probe["wrote_at"] is not None, (
        "the previous test's build never wrote its row, so its teardown did "
        "not wait for it")

    # The row was written -- after the previous test returned -- and is gone,
    # because the truncate ran after the write rather than beside it.
    with session_scope() as session:
        assert ContentRepository(session).get(KEY) is None, (
            "the previous test's build wrote into this test's tables")

    with services.activate():
        with analysis.state().lock:
            assert not analysis.state().inflight, "the previous test's build is still in flight"
        assert analysis.state().queue.unfinished_tasks == 0
