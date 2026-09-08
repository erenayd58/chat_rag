"""The production entrypoint: one process, a bounded pool of request threads.

    python -m wsgi

``app.py`` stays the development entrypoint -- Werkzeug's server, the
reloader, the interactive debugger, loopback only. This module is what a
container or a service manager runs. It exposes ``application`` for any WSGI
host and, run directly, serves it on waitress.

**One process, not several workers.** Three pieces of this application keep
real state in module globals that processes cannot share: the Viewer packager
is one background thread over an in-memory queue with a per-document lock (two
processes would each resume every unfinished document at start-up), the
pipeline cache holds a built ``RAGPipeline`` per session and knowledge base,
and the vector store is an embedded database opened by the process using it.
So the choice is *one process* rather than many, and waitress is what does
that well on both platforms this runs on -- gunicorn does not run on Windows
at all. That is also what lets the provider limit
(``components/ingest/limits.py``) be a plain semaphore that means what it
says. ``docs/limitations.md`` says what scaling out would take instead.

Concurrency is therefore ``WAITRESS_THREADS`` request threads,
``INGEST_WORKERS`` ingest workers and the one packaging thread, in one address
space. An upload is a job (``components/ingest/jobs.py``); a request thread
waits on one only for a synchronous upload, for at most ``INGEST_SYNC_WAIT``
seconds and only ``INGEST_SYNC_WAITERS`` at a time, so uploads can never take
every thread. A question is answered on the thread that received it, under
``QUERY_MAX_ACTIVE`` admission and the ``QUERY_TIMEOUT`` deadline
(``components/query/limits.py``).

A restart is not a clean slate for clients: a job id handed out before it is
still answerable afterwards, because jobs journal their transitions and
start-up settles anything in flight against the ingest ledger. Nothing is
resumed and nothing was committed, which is what makes that settlement
truthful.
"""

from __future__ import annotations

import signal
import sys

import app as _app
import storage as database
from config.runtime import runtime_from_env

#: The WSGI callable, for `waitress-serve wsgi:application`, gunicorn, uwsgi,
#: or a test that wants the application without starting a server.
application = _app.app


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
    """Host, port and thread pool for the production server.

    Every value comes from ``config.runtime``, which is also where the ingest
    and query rations read the thread count. This module used to read the same
    variables itself with a *silently forgiving* parser, so
    ``WAITRESS_THREADS=-4`` produced a server with eight threads and limits
    sized against minus four. There is one reader now, and a value it cannot
    use is refused by name at start-up.
    """
    return runtime_from_env().server_options()


def main(argv: list[str] | None = None) -> int:
    from waitress import create_server

    options = server_options()

    # Before the first print: a redirected stream on a non-UTF-8 console would
    # otherwise raise on the banner, and the banner comes before the bind.
    _app.enable_console_utf8()
    _app.require_database()
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
        # Return every pooled connection before the process exits, so a
        # restart does not leave backends on the server waiting to time out.
        database.dispose()
    print("Server stopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
