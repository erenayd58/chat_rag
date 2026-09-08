"""The ASGI entrypoint: `/api/v1`, served by FastAPI, with no Flask under it.

    python -m asgi                      # uvicorn, one process
    uvicorn asgi:application            # or any ASGI host

This is where the product is going. ``python -m wsgi`` is still what a
deployment runs today, because the console's screens, its JavaScript and the
Viewer's relay all speak the Flask-era surface and that surface is not gone
yet; that process serves **both**, with this same FastAPI application mounted
inside it (``interfaces/http/coexistence.py``). What this module adds is the
other half of the same application, standing on its own: the moment nothing
needs the legacy surface, this file is the entrypoint and the coexistence
bridge is deleted, and no route, schema or status code moves.

**One process, not several workers**, for the reason ``wsgi.py`` gives at
length: the packaging queue is one background thread over an in-memory queue,
the pipeline cache holds a built pipeline per session and knowledge base, and
the vector store is an embedded database opened by the process using it. Two
worker processes would each resume every unfinished document at start-up, and
the provider budgets would stop meaning what they say.

**The request threads are the same number.** Every handler on this surface is
a synchronous ``def`` -- it retrieves, it reads a store, it waits on a
provider -- so Starlette runs it in a worker thread rather than on the event
loop. That pool is sized from ``WAITRESS_THREADS``, which is the number
``QUERY_MAX_ACTIVE`` and ``INGEST_SYNC_WAITERS`` are already rationed against;
sizing it from anything else would leave those limits describing a thread
count that no longer exists. The variable keeps its name while both
entrypoints exist, because renaming a setting mid-migration is how a
deployment ends up configured twice.
"""

from __future__ import annotations

import logging
import sys

from application.services import Services, default_services
from config.runtime import runtime_from_env
import storage as database
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
        logger.info(f"api v1 request threads: {threads}")
    except Exception as error:  # noqa: BLE001 - never block start-up on this
        logger.warning(f"Could not size the request thread pool: {error}")


#: The application, composed once for this process, exactly as ``app.py``
#: composes it -- ``default_services()`` returns the same container whichever
#: entrypoint asks for it first.
services = default_services()

#: The ASGI callable, for `uvicorn asgi:application` or any other host.
application = v1.create_app(services, on_start=_on_start, on_stop=_on_stop)


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
    print(f"ASGI server (uvicorn): http://{options['host']}:{options['port']}")
    print(f"Contract: http://{options['host']}:{options['port']}{v1.OPENAPI_PATH}")
    print("The console's screens are not served here; run `python -m wsgi` for those.")
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
