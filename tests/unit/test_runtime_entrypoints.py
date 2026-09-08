"""One entrypoint, and what a deployment reaches when it runs it.

There were two. ``python app.py`` was the development server -- Werkzeug, the
reloader, the interactive debugger, loopback -- and for a while it was also the
container's CMD, so production *was* the development server with a flag turned
off and stayed that way for as long as everyone remembered the flag. ``python
-m wsgi`` was the production one: waitress, one process, a bounded pool of
request threads. These tests pinned the difference between them, and the two
things that must not drift: the production entrypoint must never serve with
debug, and both must serve the same application.

Both went with the Flask console (``docs/legacy-removal.md``). ``python -m
asgi`` is the entrypoint: uvicorn, one process, the same bounded pool. There is
no second one to drift from and no debug switch to get wrong -- which is a
stronger version of what this file used to assert, and what it asserts now is
the rest: the settings it reads, the values it refuses, the start-up it runs,
and that the image runs it.

Why one process is in the module docstring of ``asgi.py``; the short version is
that the Viewer packaging worker is one thread over an in-memory queue and the
pipeline cache is a module global, so a second process would duplicate both.
"""

from __future__ import annotations

import os

import pytest

import asgi as entrypoint
from runtime import bootstrap

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(autouse=True)
def preserve_signal_handlers():
    """Installing real handlers is a side effect of some start-up paths;
    none of them may leave pytest's own process changed."""
    import signal

    names = [n for n in ("SIGTERM", "SIGINT") if hasattr(signal, n)]
    saved = {n: signal.getsignal(getattr(signal, n)) for n in names}
    yield
    for name, handler in saved.items():
        try:
            signal.signal(getattr(signal, name), handler)
        except (ValueError, TypeError, OSError):  # pragma: no cover
            pass


@pytest.fixture
def clean_server_env(monkeypatch):
    for name in ("FLASK_HOST", "FLASK_PORT", "WAITRESS_THREADS",
                 "WAITRESS_CHANNEL_TIMEOUT"):
        monkeypatch.delenv(name, raising=False)


# --------------------------------------------------------------- one server
def test_the_entrypoint_serves_the_application_the_process_composed():
    """``application`` is what any ASGI host is handed, and it is built over
    the one container this process made."""
    assert entrypoint.application.state.services is entrypoint.services


def test_there_is_no_second_entrypoint_to_drift_from():
    """The pair this file was written about. A file back in the tree is a
    second way to start the product, and the first thing that happens then is
    that the two disagree about a setting."""
    for gone in ("app.py", "wsgi.py"):
        assert not os.path.exists(os.path.join(REPO, gone)), f"{gone} is back"


def test_the_server_has_no_debug_setting_to_get_wrong(clean_server_env, monkeypatch):
    """Not "debug defaults to off" -- there is no debug switch on this path.

    ``FLASK_DEBUG=true`` was the developer default and it must not reach a
    deployment. Nothing reads it any more, here or anywhere.
    """
    monkeypatch.setenv("FLASK_DEBUG", "true")

    options = entrypoint.server_options()
    assert set(options) == {"host", "port"}
    assert not any("debug" in str(value).lower() for value in options.values())

    for module in ("asgi.py", "runtime/bootstrap.py", "config/runtime.py"):
        source = open(os.path.join(REPO, module), encoding="utf-8").read()
        assert "FLASK_DEBUG" not in source, f"{module} still reads FLASK_DEBUG"


# ------------------------------------------------------------------ defaults
def test_server_defaults(clean_server_env):
    options = entrypoint.server_options()

    assert options["host"] == "0.0.0.0", "a container's port mapping needs every interface"
    assert options["port"] == 5005


def test_server_options_are_configurable(clean_server_env, monkeypatch):
    monkeypatch.setenv("FLASK_HOST", "127.0.0.1")
    monkeypatch.setenv("FLASK_PORT", "9001")
    monkeypatch.setenv("WAITRESS_THREADS", "2")

    from config.runtime import runtime_from_env

    options = entrypoint.server_options()
    assert (options["host"], options["port"]) == ("127.0.0.1", 9001)
    assert runtime_from_env().request_threads == 2


@pytest.mark.parametrize("blank", ["", "   "])
def test_an_empty_thread_count_means_unset(clean_server_env, monkeypatch, blank):
    """A variable set to nothing is a variable that was not set."""
    from config.runtime import runtime_from_env

    monkeypatch.setenv("WAITRESS_THREADS", blank)

    assert runtime_from_env().request_threads == 8


@pytest.mark.parametrize("bad", ["0", "-4", "eight"])
def test_a_nonsense_thread_count_is_refused_by_name(clean_server_env, monkeypatch, bad):
    """It used to fall back to eight in the server and not in the limits.

    That looked safe and was not: ``config.ingest`` and ``config.query`` read
    the same variable *without* the fallback, so ``WAITRESS_THREADS=-4`` gave a
    server with eight threads and upload/query rations sized against minus
    four. One reader owns it now (``config.runtime``) and refuses a value it
    cannot use, by name, which is what every other limit already did.
    """
    from config.runtime import runtime_from_env

    monkeypatch.setenv("WAITRESS_THREADS", bad)

    with pytest.raises(ValueError, match="WAITRESS_THREADS"):
        runtime_from_env()


def test_the_worker_pool_is_sized_from_the_one_reader(clean_server_env, monkeypatch):
    """Every handler on this surface is a synchronous ``def``, so Starlette
    runs it in a worker thread. That pool is the request concurrency the
    ingest and query rations are sized against, and sizing it from anything
    else would leave them describing a thread count that does not exist."""
    import anyio
    import anyio.to_thread

    monkeypatch.setenv("WAITRESS_THREADS", "3")

    async def size():
        entrypoint._size_thread_pool()
        return anyio.to_thread.current_default_thread_limiter().total_tokens

    assert anyio.run(size) == 3


# --------------------------------------------------------- what main() runs
def test_main_starts_uvicorn_over_the_application(monkeypatch, clean_server_env):
    """No socket is bound here; this pins how ``main`` wires the server up."""
    import uvicorn

    recorded = {}

    def run(application, **kwargs):
        recorded["application"] = application
        recorded["kwargs"] = kwargs

    monkeypatch.setattr(uvicorn, "run", run)
    monkeypatch.setattr(bootstrap, "startup_banner",
                        lambda services: recorded.setdefault("banner", True))

    assert entrypoint.main() == 0

    assert recorded["application"] is entrypoint.application
    assert recorded["kwargs"]["host"] == entrypoint.server_options()["host"]
    assert recorded["kwargs"]["port"] == entrypoint.server_options()["port"]
    assert recorded["banner"] is True


def test_the_lifespan_is_what_picks_up_the_previous_process(monkeypatch):
    """A restart has to settle the last process's ingest jobs and resume an
    interrupted Viewer packaging job. It used to be the entrypoint's own
    start-up; it is the application's lifespan now, so it happens under every
    ASGI host rather than only under the one that remembered to call it."""
    seen = []
    monkeypatch.setattr(bootstrap, "require_database", lambda: seen.append("database"))
    monkeypatch.setattr(bootstrap, "resume_background_work",
                        lambda services: seen.append("resumed"))
    monkeypatch.setattr(entrypoint, "_size_thread_pool", lambda: seen.append("threads"))

    entrypoint._on_start(entrypoint.services)

    assert seen == ["threads", "database", "resumed"]


def test_a_stop_drains_the_jobs_before_it_returns_the_pool(monkeypatch):
    """The one write a job must not lose is its final ledger row, so the
    database pool goes after the jobs that were still writing to it."""
    order = []
    monkeypatch.setattr(entrypoint.services.ingest_jobs, "close",
                        lambda timeout=None: order.append("jobs"))
    import storage as database

    monkeypatch.setattr(database, "dispose", lambda: order.append("database"))

    entrypoint._on_stop(entrypoint.services)

    assert order == ["jobs", "database"]


# ------------------------------------------------- start-up on any console
def test_the_banner_survives_a_console_that_cannot_spell_it():
    """Start-up must not depend on who launched the process.

    On Windows a redirected stream falls back to the machine's code page --
    cp1254 on a Turkish install -- and anything outside it raises
    UnicodeEncodeError. The banner is printed before the server binds, so
    ``python -m asgi > server.log`` died at start-up with a traceback rather
    than serving. It never showed here because the demo launcher sets
    PYTHONIOENCODING and the image sets it too: the application was relying on
    being started by something that knew to.

    Not really about the banner. A document title or a knowledge base name with
    a character the code page cannot spell would do the same, anywhere in the
    start-up path -- which is why the fix widens the stream rather than
    flattening the text.
    """
    import io
    import sys

    narrow = io.TextIOWrapper(io.BytesIO(), encoding="cp1254", errors="strict")
    original = sys.stdout
    sys.stdout = narrow
    try:
        bootstrap.enable_console_utf8()
        bootstrap.startup_banner(entrypoint.services)
        sys.stdout.flush()
    finally:
        sys.stdout = original

    narrow.seek(0)
    assert narrow.buffer.getvalue(), "the banner printed nothing at all"


def test_the_widening_is_not_done_on_import():
    """It changes a global, so only an entrypoint may ask for it.

    ``asgi`` is imported by every test in this suite and by anything that
    embeds the application; reconfiguring the process's streams as a side
    effect of an import would be a surprise none of them asked for.
    """
    import inspect

    source = inspect.getsource(entrypoint)
    calls = [line for line in source.splitlines()
             if "enable_console_utf8()" in line and not line.strip().startswith("def ")]
    assert calls, "nothing calls it"
    for line in calls:
        assert line.startswith("    "), (
            f"enable_console_utf8() is called at module level: {line!r}"
        )


def test_the_entrypoint_widens_before_it_prints():
    import inspect

    source = inspect.getsource(entrypoint.main)
    assert "enable_console_utf8()" in source
    assert source.index("enable_console_utf8()") < source.index("startup_banner("), (
        "the banner is printed before the stream can carry it"
    )


# ------------------------------------------------------------- the container
def test_the_image_runs_the_entrypoint():
    """The Dockerfile's CMD, read as the deployment contract it is."""
    dockerfile = open(os.path.join(REPO, "Dockerfile"), encoding="utf-8").read()

    assert 'CMD ["python", "-m", "asgi"]' in dockerfile
    assert 'CMD ["python", "-m", "wsgi"]' not in dockerfile
    assert 'CMD ["python", "app.py"]' not in dockerfile
