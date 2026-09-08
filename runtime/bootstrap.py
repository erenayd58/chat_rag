"""What a process does before it serves, and what it picks up from the last one.

Both entrypoints run this -- the development server in ``app.py`` and the
production server in ``wsgi.py`` -- because the first question about a running
instance is always which models and which state directory it is on, and the
second is whether that is the directory somebody meant.
"""

from __future__ import annotations

import logging
import os
import sys

import storage as database
from components.ingest import sweep_staging
from config import paths

from application import ingest, workspace

logger = logging.getLogger("RAG.bootstrap")


def require_database() -> None:
    """Refuse to serve without the database, and say which one.

    Called before the container is built, so a wrong or unreachable
    ``DATABASE_URL`` is a start-up message naming the host rather than a server
    that binds a port and then fails every request with a stack trace. The
    application's relational state is PostgreSQL now; there is no degraded mode
    in which it serves without one.

    The connection string is reported through ``config.database``, which takes
    the credential out of it: this message goes to a log file.
    """
    try:
        database.require_reachable()
    except (database.DatabaseNotConfigured, database.DatabaseUnavailable) as error:
        print("\n" + "=" * 80)
        print("Cannot start: " + str(error))
        print("Run `alembic upgrade head` against it once it is reachable; "
              "see docs/database.md.")
        print("=" * 80)
        raise SystemExit(2) from error


def enable_console_utf8() -> None:
    """Let this process write any character to stdout without dying.

    Called by the entrypoints, never on import, because it changes a global.

    On Windows an interactive console already handles UTF-8, but a *redirected*
    stream falls back to the machine's code page -- cp1254 on a Turkish
    install, which is what this is developed on. Anything outside it then
    raises UnicodeEncodeError, and the banner below is printed before the
    server binds, so `python -m wsgi > server.log` died at start-up with a
    traceback instead of serving. It was invisible here only because the demo
    launcher sets PYTHONIOENCODING and the container image sets it too: the
    application depended on being launched by something that knew.

    It is not really about the banner. A document title, a knowledge base name
    or a model id with a character the code page cannot spell would do exactly
    the same thing, anywhere in the start-up path.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError, OSError):
            # Not a reconfigurable text stream (a pytest capture, a pipe
            # someone replaced). Nothing to do, and nothing worth failing for.
            pass


def startup_banner(services) -> None:
    """What this process is configured to do, printed once before it serves."""
    print("\n" + "="*80)
    print("🚀 Starting RAG Chat Web Application")
    print("="*80)
    # One source for the whole block: Settings.effective_configuration(), which
    # /api/ops answers from too, so the banner and the endpoint cannot drift
    # and neither can print a credential -- there is none in it by
    # construction (config/settings.py: SECRET_ATTRIBUTES).
    settings = services.settings
    effective = settings.effective_configuration()
    models, runtime = effective['models'], effective['runtime']
    ingest_cfg, query, logs = effective['ingest'], effective['query'], effective['logging']

    print(f"\nAnswer model: {models['answer_provider']} / {models['answer_model']}"
          + (f"  (fallback: {settings.answer_fallback_provider} / {settings.answer_fallback_model or settings.ollama_model})"
             if settings.answer_fallback_provider not in ('', 'none') else ""))
    print(f"Embedding: {models['embedding_provider']} / {models['embedding_model']}")
    print(f"Retrieval profile: {models['retrieval_profile']}")
    print(f"Deep Analysis model: {models['deep_analysis_model'] or '(not configured)'}")
    print(f"Runtime: {runtime['request_threads']} request thread(s), "
          f"channel timeout {runtime['channel_timeout_seconds']}s")
    print(f"Ingest: {ingest_cfg['workers']} worker(s), queue {ingest_cfg['queue_capacity']}, "
          f"job timeout {ingest_cfg['job_timeout_seconds']:.0f}s, "
          f"{ingest_cfg['sync_waiters']} sync waiter(s); "
          f"provider calls in flight <= {ingest_cfg['provider_max_inflight']} "
          f"(per Deep job <= {ingest_cfg['deep_concurrency']}), "
          f"embedding <= {ingest_cfg['embedding_max_inflight']}")
    print(f"Query: {query['max_active']} at once, answer calls <= "
          f"{query['answer_max_inflight']}, timeout {query['timeout_seconds']:.0f}s, "
          f"{query['free_threads']} thread(s) left free")
    print(f"Caches: {ingest_cfg['pipeline_cache_max']} pipeline(s), "
          f"TTL {ingest_cfg['pipeline_cache_ttl_seconds']:.0f}s")
    print(f"Logging: console {logs['console_level']}, "
          f"file {logs['file_level']} -> {logs['directory']}")
    # Where this process keeps its state, and anything refused on the way to
    # deciding that. One data directory now settles every path below it, so
    # naming them here is what makes a wrong one visible at start-up rather
    # than after something has been written to it.
    db = effective['database']
    print(f"Database: {db.get('url') or '(not configured)'}"
          f"  (pool {db.get('pool_size')}+{db.get('max_overflow')})")
    print(f"Data directory: {effective['data_root'] or '(none: paths are relative to ' + os.getcwd() + ')'}")
    print(f"Vector DB: {effective['vector_db']}")
    print(f"Parser cache: {effective['parser_cache']}")
    for line in effective['warnings'] + logs['fallbacks']:
        print(f"  ! {line}")

    stats = services.documents().get_statistics()
    print(f"\n📊 Ingested Documents: {stats['total_documents']}")
    print(f"📦 Total Chunks: {stats['total_chunks']}")

    if stats['total_documents'] == 0:
        print("\n⚠️  Warning: No documents ingested yet!")
        print("   Upload one at /, or POST /api/documents/upload")


def resume_background_work(services) -> None:
    """Pick up whatever a previous process was in the middle of.

    A Viewer packaging job interrupted by a restart is recorded on disk with
    every input it needs; picking it up here is what makes the integration
    survive a stop/start rather than needing the document re-uploaded.

    This is also why the runtime is one process (see ``wsgi.py``): the
    packaging queue lives in memory and its worker is one thread, so a second
    process running this would resume the same documents a second time.
    """
    try:
        resumed = workspace.resume_incomplete()
        if resumed:
            print(f"🔁 Resuming Viewer analysis for {len(resumed)} document(s)")
    except Exception as e:  # noqa: BLE001 - never block start-up on this
        logger.warning(f"Could not resume Viewer analyses: {e}")

    # Ingest jobs are not resumed -- a half-finished parse is not worth
    # restarting and nothing was committed -- but they are *settled*, so a
    # client holding a job_id gets a truthful answer instead of a 404. The
    # ledger decides: a document carrying the job's id means it finished.
    try:
        interrupted = ingest.recover(services)
        if interrupted:
            unfinished = [r for r in interrupted if r.get('status') != 'succeeded']
            recovered = len(interrupted) - len(unfinished)
            if recovered:
                print(f"✅ {recovered} ingest job(s) had finished before the restart")
            if unfinished:
                print(f"⚠️  {len(unfinished)} ingest job(s) were interrupted by the restart; "
                      "their documents were not registered")
    except Exception as e:  # noqa: BLE001 - never block start-up on this
        logger.warning(f"Could not settle the previous process's ingest jobs: {e}")

    # What a previous process left in the staging directory belongs to no job.
    try:
        swept = sweep_staging(paths.upload_staging())
        if swept:
            print(f"🧹 Removed {len(swept)} staged upload(s) left by a previous process")
    except Exception as e:  # noqa: BLE001 - never block start-up on this
        logger.warning(f"Could not sweep the upload staging directory: {e}")


def development_server_options() -> dict:
    """How `python app.py` runs: the development server, and only that.

    Debug stays on by default, because a developer at a keyboard wants the
    reloader and the traceback page and has always had them; FLASK_DEBUG=false
    turns them off, which is what the demo launcher does to stop the reloader
    building the pipeline twice.

    The host default is loopback, not 0.0.0.0. A server with an interactive
    debugger attached should not be reachable from whatever network the laptop
    has joined, and this one is a development server by definition -- a
    deployment runs `python -m wsgi`, which binds every interface because it
    has no debugger to expose. FLASK_HOST still overrides it for anyone who
    wants that deliberately.
    """
    return {
        'host': os.getenv('FLASK_HOST', '127.0.0.1'),
        'port': int(os.getenv('FLASK_PORT', '5005')),
        'debug': os.getenv('FLASK_DEBUG', 'true').strip().lower() not in {
            '0', 'false', 'no', 'off'
        },
    }
