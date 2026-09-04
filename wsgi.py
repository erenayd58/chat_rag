"""The production entrypoint: one process, a bounded pool of request threads.

    python -m wsgi

``app.py`` stays the development entrypoint -- Werkzeug's server, the
reloader, the interactive debugger, loopback only. This module is what a
container or a service manager runs. It exposes ``application`` for any WSGI
host and, run directly, serves it on waitress.

Why one process rather than several worker processes
----------------------------------------------------

This application keeps real state in module globals, and that state is not
shareable between processes:

* ``components.viewer.analysis`` packages documents on **one background
  thread** fed by an in-memory ``queue.Queue``, with an in-process ``_inflight``
  set and per-document locks. Two processes would each run their own worker
  over the same directory on disk, and each would run ``resume_incomplete()``
  at start-up -- so a restart with four workers would repackage every
  unfinished document four times, concurrently.
* ``app.pipelines`` caches a built ``RAGPipeline`` per (session, knowledge
  base). Each process would build and hold its own copy of every model and
  index it touches, and a browser's requests would land on a different cache
  each time.
* the vector store is an embedded Chroma/sqlite database opened by the
  process that uses it, not a database server several processes may share.

So the choice here is not "waitress instead of gunicorn"; it is *one process*
instead of many, and waitress is the server that does that well on both of the
platforms this project runs on. Gunicorn cannot run on Windows at all, which is
where this is developed; waitress is pure Python, runs identically on Windows
and Linux, needs no build tools in a slim image, and is a threaded single
process by design rather than by configuration.

Concurrency is therefore bounded by ``WAITRESS_THREADS`` (default 8) request
threads plus the one Viewer packaging thread, all in one address space. A
future provider limit such as ``LLM_MAX_INFLIGHT`` can be a plain
``threading.Semaphore`` and mean exactly what it says -- which it would not if
the runtime were N processes.

Nothing here decides how ingestion is scheduled; that is Phase 2's. It only
fixes the topology that scheduling has to be designed against.
"""

from __future__ import annotations

import os
import signal
import sys

import app as _app

#: The WSGI callable, for `waitress-serve wsgi:application`, gunicorn, uwsgi,
#: or a test that wants the application without starting a server.
application = _app.app


def _int_env(name: str, default: int) -> int:
    raw = (os.getenv(name) or "").strip()
    try:
        value = int(raw) if raw else default
    except ValueError:
        value = default
    return value if value > 0 else default


def _install_shutdown_handlers() -> None:
    """Turn SIGTERM into the exception waitress already shuts down cleanly on.

    ``waitress``'s serving loop catches ``SystemExit`` and ``KeyboardInterrupt``
    and drains its task dispatcher, so Ctrl+C was always clean. SIGTERM -- what
    ``docker stop`` and every service manager send -- has no Python-level
    meaning by default and would kill the process outright, mid-request. This
    makes the two the same thing.
    """

    def _stop(signum, _frame):  # pragma: no cover - exercised by a real signal
        raise SystemExit(0)

    for name in ("SIGTERM", "SIGINT"):
        handler = getattr(signal, name, None)
        if handler is not None:
            try:
                signal.signal(handler, _stop)
            except (ValueError, OSError):
                # Not the main thread, or a platform without it. The server
                # still serves; only the graceful stop is unavailable.
                pass


def server_options() -> dict:
    """Host, port and thread pool for the production server."""
    return {
        # 0.0.0.0 here, unlike the development server: a container's port
        # mapping only reaches a process listening on every interface, and this
        # server has no debugger to expose.
        "host": (os.getenv("FLASK_HOST") or "0.0.0.0").strip(),
        "port": _int_env("FLASK_PORT", 5005),
        # One thread serves one request. Eight is enough for a console with a
        # handful of users, and small enough that eight concurrent ingests
        # cannot exhaust the machine. It is the number Phase 2 should size its
        # own limits against: request concurrency is this, plus the one Viewer
        # packaging thread, in one address space.
        "threads": _int_env("WAITRESS_THREADS", 8),
        # Long enough for an upload that parses a large PDF on the request
        # thread, short enough that a dead connection is not held forever.
        "channel_timeout": _int_env("WAITRESS_CHANNEL_TIMEOUT", 900),
        # The application is behind nothing that would set them; trusting a
        # client's X-Forwarded-* headers would let a browser choose its own
        # apparent address. A reverse-proxy deployment turns this on
        # deliberately, with the proxy named.
        "clear_untrusted_proxy_headers": True,
    }


def main(argv: list[str] | None = None) -> int:
    from waitress import create_server

    options = server_options()

    # Before the first print: a redirected stream on a non-UTF-8 console would
    # otherwise raise on the banner, and the banner comes before the bind.
    _app.enable_console_utf8()
    _app.startup_banner()
    _app.resume_background_work()

    server = create_server(application, **options)

    print()
    print("=" * 80)
    print(f"Production server (waitress, {options['threads']} threads): "
          f"http://{options['host']}:{options['port']}")
    print("=" * 80)
    print()
    sys.stdout.flush()

    _install_shutdown_handlers()
    try:
        server.run()
    except (SystemExit, KeyboardInterrupt):  # pragma: no cover - signal path
        pass
    finally:
        server.close()
    print("Server stopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
