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

Concurrency is therefore ``WAITRESS_THREADS`` (default 8) request threads,
``INGEST_WORKERS`` (default 2) ingest workers and the one Viewer packaging
thread, all in one address space. That is what lets the provider limit
(``PROVIDER_MAX_INFLIGHT``, ``components/ingest/limits.py``) be a plain
``threading.Semaphore`` that means exactly what it says -- which it would not
if the runtime were N processes.

Ingestion is scheduled by ``components/ingest/jobs.py``: an upload request
validates, stages the file and queues a job, and the parse, chunking, model
calls and store writes happen on an ingest worker under those limits. A
request thread waits on a job only for a synchronous upload (no ``async=1``),
for at most ``INGEST_SYNC_WAIT`` seconds -- below ``WAITRESS_CHANNEL_TIMEOUT``
on purpose -- and only ``INGEST_SYNC_WAITERS`` threads may do so at once, so
uploads can never occupy every request thread and leave ``/api/health`` and
job polling unanswerable. An upload that finds no waiting slot is still
accepted and still runs; it is answered 202 with its job.

A question is answered on the request thread that received it, so the same
kind of bound applies to chat (``components/query/limits.py``): at most
``QUERY_MAX_ACTIVE`` threads may be inside a query at once -- by default
``WAITRESS_THREADS - INGEST_SYNC_WAITERS - 1``, so questions and synchronous
uploads together can never take every thread -- one more is refused at once
with 503 and a ``Retry-After`` rather than queued, answer-model calls share
``ANSWER_MAX_INFLIGHT`` process-wide, and the whole query runs under
``QUERY_TIMEOUT``.

A restart is not a clean slate for clients: a job id handed out before it is
still answerable afterwards, because jobs journal their transitions and
start-up settles anything in flight against the ingest ledger
(``components/ingest/journal.py``). Nothing is resumed and nothing was
committed, which is what makes that settlement truthful.
"""

from __future__ import annotations

import signal
import sys

import app as _app
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
