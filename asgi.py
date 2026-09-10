"""The entrypoint: `/api/v1`, served by FastAPI, on uvicorn.

    python -m asgi                      # one process
    uvicorn asgi:application            # or any ASGI host

This is the whole server. Until Step 13 there were two -- ``python -m wsgi``
served the Flask console, its screens and the Viewer's relay with this same
FastAPI application mounted inside it -- and the removal of that surface left
this file standing where it always said it would (``docs/legacy-removal.md``).
No route, schema or status code moved on the way: the contract is what it was,
and the operator's ``/api/ops/metrics`` kept its path and its body.

**One process, not several workers.** Three pieces of this application keep
real state in module globals that processes cannot share: the Viewer packager
is one background thread over an in-memory queue with a per-document lock (two
processes would each resume every unfinished document at start-up), the
pipeline cache holds a built pipeline per session and knowledge base, and the
provider budgets (``components/ingest/limits.py``) are plain semaphores that
mean what they say only inside one address space. ``docs/limitations.md`` says
what scaling out would take instead.

**The request threads are the number the limits are sized against.** Every
handler on this surface is a synchronous ``def`` -- it retrieves, it reads a
store, it waits on a provider -- so Starlette runs it in a worker thread
rather than on the event loop. That pool is sized from ``WAITRESS_THREADS``,
the same variable ``QUERY_MAX_ACTIVE`` and ``INGEST_SYNC_WAITERS`` are
rationed against (``config/runtime.py``); sizing it from anything else would
leave those limits describing a thread count that no longer exists. The name
is the deployment's, not this module's: renaming a setting every ``.env`` and
compose file already carries is a change to somebody's deployment, not a
cleanup, so it is left to whoever wants to make it deliberately.

A restart is not a clean slate for clients: a job id handed out before it is
still answerable afterwards, because jobs journal their transitions and
start-up settles anything in flight against the ingest ledger. Nothing is
resumed and nothing was committed, which is what makes that settlement
truthful.
"""

from __future__ import annotations

import logging
import sys

import interfaces.http as http
from chat_rag.application.services import Services, default_services
from chat_rag.config.runtime import runtime_from_env
from chat_rag import storage as database
from interfaces.http import v1
from runtime import bootstrap

logger = logging.getLogger("RAG.asgi")

#: How long a stop waits for jobs that are already running.
SHUTDOWN_TIMEOUT_SECONDS = 30.0


def _on_start(services: Services) -> None:
    """What this process picks up from the last one, and what it sizes itself
    against."""
    _size_thread_pool()
    bootstrap.require_database()
    bootstrap.resume_background_work(services)


def _on_stop(services: Services) -> None:
    """Let running ingest jobs finish, then stop the workers.

    The pipeline cache and the stores are left to the process's own exit: they
    are closed by eviction and by the store's own finaliser, and closing them
    here would race the jobs that are still draining.
    """
    services.ingest_jobs.close(timeout=SHUTDOWN_TIMEOUT_SECONDS)
    # The database pool goes last, after the jobs that were still writing to
    # it have finished. Disposing it first would fail their final ledger write,
    # which is the one write a job must not lose.
    database.dispose()


def _size_thread_pool() -> None:
    """Give the synchronous handlers the thread count the limits assume.

    Called from inside the lifespan, which is the only place there is a
    running loop to hold the limiter.
    """
    try:
        import anyio.to_thread

        threads = runtime_from_env().request_threads
        anyio.to_thread.current_default_thread_limiter().total_tokens = threads
        logger.info(f"request threads: {threads}")
    except Exception as error:  # noqa: BLE001 - never block start-up on this
        logger.warning(f"Could not size the request thread pool: {error}")


#: The application, composed once for this process.
services = default_services()

#: The ASGI callable, for `uvicorn asgi:application` or any other host.
application = http.create_app(services, on_start=_on_start, on_stop=_on_stop)


def server_options() -> dict:
    """Host and port for this server, from the one reader that owns them."""
    limits = runtime_from_env()
    return {"host": limits.host, "port": limits.port}


def main(argv: list[str] | None = None) -> int:
    import uvicorn

    options = server_options()

    # Before the first print: a redirected stream on a non-UTF-8 console would
    # otherwise raise on the banner, and the banner comes before the bind.
    bootstrap.enable_console_utf8()
    bootstrap.startup_banner(services)

    print()
    print("=" * 80)
    print(f"Server (uvicorn): http://{options['host']}:{options['port']}")
    print(f"Contract: http://{options['host']}:{options['port']}{v1.OPENAPI_PATH}")
    print("The console is a Next.js application; run it from frontend/.")
    print("=" * 80)
    print()
    sys.stdout.flush()

    # Uvicorn installs its own SIGTERM/SIGINT handling and runs the lifespan
    # on the way down, so the shutdown hook above is what stops the workers.
    uvicorn.run(application, log_level="info", **options)
    print("Server stopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
