"""Test isolation: the suite must never touch the developer's live state.

Every persistent thing this application writes resolves, by default, relative
to the working directory (``config/paths.py``): the knowledge base records,
the ingest ledger, the gold set, ``chroma_db/``, ``logs/``, the parser cache
and ``artifacts/viewer-live/``. Several tests import ``app``, which at import
time builds the default pipeline against exactly those paths, and the Viewer
packaging worker keeps writing there from a background thread after a test's
own ``monkeypatch.chdir`` has been undone.

The smallest isolation that covers all of that is to move the *process* out of
the checkout before any application module is imported:

* the working directory becomes a fresh temporary directory for the whole
  session, so every cwd-relative default lands there (``.env`` values such as
  ``VECTOR_DB_PATH=./chroma_db`` included);
* the provider keys are blanked *before* ``config.settings`` runs
  ``load_dotenv``, which never overrides a variable that is already set, so no
  test can reach a real model or embedding endpoint by accident;
* the repository root stays importable (``import app`` must keep working
  whichever directory the session runs in).

Since Step 8 there is a third kind, and it is the one that replaced most of
the first. The knowledge bases, the ingest ledger, the gold set, the ingest
journal and the Viewer's analysis records are rows in PostgreSQL, so pointing
a test's store at its own ``tmp_path`` no longer isolates anything. What
isolates a test now is that the tables are empty when it starts: the schema is
built once per session by Alembic -- the same migrations a deployment runs,
never ``create_all`` -- and truncated before every test. A test that used to
get a fresh file by naming one still gets a fresh database, and the stores
still accept the path they are handed (see ``DocumentTracker.__init__``).

The database is named by ``CHAT_RAG_TEST_DATABASE_URL``, defaulting to the
service in ``docker-compose.test.yml``; it is never a developer's own, and the
session refuses to run rather than guessing at one (``docs/testing.md``).

A second kind of isolation is needed for the same reason, one level down. The
application publishes its configuration *into the environment*: ``config.paths``
applies the ``.env`` file by writing the values it accepts into ``os.environ``,
exactly as ``load_dotenv`` always did. That is correct in production and
dangerous in a test, because ``monkeypatch`` can only undo what it made itself.
A test that calls the real loader leaves those variables set for the rest of the
session -- and one of them, ``CHAT_RAG_DATA_DIR``, decides where *every* later
test's ledger, knowledge base records and vector stores go. Left leaking, later
tests silently share one ledger instead of getting a fresh one under their own
``tmp_path``.

So every test runs against a snapshot of the environment and is handed one back
afterwards, whoever changed it and however. Nothing here changes what the code
under test does; it only changes where a default path points, removes the
credentials, and guarantees that a test's environment is its own. Tests that
already isolate themselves with ``tmp_path`` keep doing so.
"""

from __future__ import annotations

import atexit
import os
import shutil
import sys
import tempfile
import time

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# A test session is a process, and this is the one thing a process has to do
# before it imports the application: the thread-pool defaults configure
# numeric libraries that read their environment when they are imported, and
# the first test to touch ``chat_rag.pipeline`` pulls in torch. It used to
# happen on the way through ``chat_rag/application/__init__.py``; the entry
# points own it now, and a test session is one of them.
#
# Log handlers are deliberately *not* installed. ``chat_rag`` attaches a
# NullHandler and nothing else, which is what a library should do, and a suite
# that wants records has ``caplog``. The session no longer writes a log file.
from chat_rag.process import apply_thread_defaults  # noqa: E402

apply_thread_defaults()

#: Files a test run must leave exactly as it found them.
_DEVELOPER_STATE = (
    ".ingested_documents.json",
    ".knowledge_bases.json",
    ".gold_set.json",
    os.path.join("chroma_db", "chroma.sqlite3"),
)

_PREFIX = "chat_rag-tests-"

# Windows keeps the session's log file and Chroma's sqlite handle open until
# the interpreter exits, so the previous session's directory usually survives
# its own cleanup. Sweep those leftovers here, before making this session's --
# but only ones old enough that no live session can own them, because two
# pytest runs at once would otherwise delete each other's state root.
_STALE_AFTER_SECONDS = 3600
for _stale in os.listdir(tempfile.gettempdir()):
    if not _stale.startswith(_PREFIX):
        continue
    _path = os.path.join(tempfile.gettempdir(), _stale)
    try:
        _idle = time.time() - os.path.getmtime(_path)
    except OSError:
        continue
    if _idle > _STALE_AFTER_SECONDS:
        shutil.rmtree(_path, ignore_errors=True)

SESSION_ROOT = tempfile.mkdtemp(prefix=_PREFIX)

# The keys are read by name at request time; a blank value is "not set" to
# every component (Deep Analysis falls back to the deterministic contract,
# the OpenAI-compatible transports refuse to send). Set here, before the
# application's ``load_dotenv`` runs, so the developer's .env cannot fill them.
for _secret in ("OPENROUTER_API_KEY", "AZURE_API_KEY"):
    os.environ[_secret] = ""

os.chdir(SESSION_ROOT)

#: The database every test in this session runs against.
#:
#: Set here, before ``config`` is imported, so it wins under the same rule the
#: real process environment always wins by -- a ``.env`` naming a developer's
#: own database cannot reach the suite. ``docker-compose.test.yml`` starts the
#: default; ``CHAT_RAG_TEST_DATABASE_URL`` overrides it for CI.
TEST_DATABASE_URL = os.environ.get(
    "CHAT_RAG_TEST_DATABASE_URL",
    "postgresql+psycopg://chat_rag:chat_rag@127.0.0.1:55432/chat_rag_test",
)
os.environ["DATABASE_URL"] = TEST_DATABASE_URL


def _fingerprint() -> dict[str, tuple[int, float] | None]:
    found = {}
    for relative in _DEVELOPER_STATE:
        path = os.path.join(REPO_ROOT, relative)
        try:
            stat = os.stat(path)
            found[relative] = (stat.st_size, stat.st_mtime)
        except OSError:
            found[relative] = None
    return found


_BEFORE = _fingerprint()


def _cleanup() -> None:
    # Chroma keeps its sqlite handle open on Windows; whatever cannot be
    # removed is left in the temp directory rather than failing the run.
    try:
        os.chdir(REPO_ROOT)
    except OSError:
        pass
    shutil.rmtree(SESSION_ROOT, ignore_errors=True)


atexit.register(_cleanup)


def pytest_configure(config):
    """Markers this suite defines, so ``-m migration`` needs no other file.

    ``migration`` selects the contract suite in ``tests/migration``: the
    behaviours a framework, persistence or frontend replacement must keep.
    It is registered here rather than in a ``pytest.ini`` so the repository
    keeps one place that configures the test session.
    """
    config.addinivalue_line(
        "markers",
        "migration: a contract the platform migration must preserve "
        "(tests/migration; run with -m migration)",
    )


def pytest_report_header(config):
    return f"chat_rag test state: {SESSION_ROOT} (cwd for the session; developer state untouched)"


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    changed = [name for name, before in _BEFORE.items() if _fingerprint().get(name) != before]
    if changed:
        terminalreporter.write_sep(
            "!", "developer state changed during this session: " + ", ".join(changed)
            + " (a live console instance writes these too; check before blaming a test)",
        )


# --------------------------------------------------------------- the database


def _build_schema() -> None:
    """An empty database, built by the migrations and by nothing else.

    The public schema is dropped and recreated first, so what the suite runs
    against is the result of ``alembic upgrade head`` on an empty database --
    the same thing a fresh install is, checked on every run rather than
    asserted once.
    """
    from alembic import command
    from sqlalchemy import text

    from chat_rag import storage
    from chat_rag.storage.schema import alembic_config

    with storage.engine().connect() as connection:
        connection.execute(text("DROP SCHEMA IF EXISTS public CASCADE"))
        connection.execute(text("CREATE SCHEMA public"))
        connection.commit()

    # The package's own configuration, not ``alembic.ini``: the migrations are
    # found from where ``chat_rag.storage`` is, which is what an installed
    # wheel has and a checkout's ini file does not describe.
    command.upgrade(alembic_config(), "head")


@pytest.fixture(scope="session", autouse=True)
def _database():
    """One prepared database for the session. Refuses rather than guesses."""
    from chat_rag import storage
    from chat_rag.storage.engine import DatabaseUnavailable

    try:
        storage.require_reachable()
    except DatabaseUnavailable as error:
        raise pytest.UsageError(
            f"{error}\n\nThe suite needs PostgreSQL: this application's "
            "relational state lives there.\n"
            "Start it with `docker compose -f docker-compose.test.yml up -d`, "
            "or set CHAT_RAG_TEST_DATABASE_URL.\nSee docs/testing.md."
        ) from error
    _build_schema()
    yield
    storage.dispose()


@pytest.fixture(autouse=True)
def _empty_tables(_database):
    """Give every test empty tables.

    One statement: ``TRUNCATE ... RESTART IDENTITY CASCADE`` over the whole
    schema. Truncating rather than dropping keeps the schema the migrations
    built, so no test can pass against a table that ``create_all`` would have
    made differently.
    """
    from sqlalchemy import text

    from chat_rag import storage
    from chat_rag.storage.models import ALL_TABLES

    with storage.engine().begin() as connection:
        connection.execute(text(
            "TRUNCATE TABLE " + ", ".join(ALL_TABLES) + " RESTART IDENTITY CASCADE"
        ))
    yield


@pytest.fixture
def fresh_database():
    """A database of this test's own, created and dropped around it.

    Made beside the suite's own, on the same server, so a test can prove a
    claim about an *empty* PostgreSQL -- the migrations building a schema, a
    library engine migrating its own database -- rather than about whatever
    the session has already built. Yields the URL; nothing is applied to it.
    """
    import uuid

    from sqlalchemy import create_engine, text

    from chat_rag import storage
    from chat_rag.storage.engine import configured_settings

    settings = configured_settings()
    name = "chat_rag_fresh_" + uuid.uuid4().hex[:12]
    # AUTOCOMMIT: CREATE DATABASE cannot run inside a transaction block.
    admin = create_engine(settings.url, isolation_level="AUTOCOMMIT")
    with admin.connect() as connection:
        connection.execute(text(f'CREATE DATABASE "{name}"'))
    url = settings.url.rsplit("/", 1)[0] + "/" + name
    try:
        yield url
    finally:
        storage.dispose()  # nothing of ours may still hold a connection to it
        with admin.connect() as connection:
            connection.execute(
                text("SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
                     "WHERE datname = :name"),
                {"name": name},
            )
            connection.execute(text(f'DROP DATABASE IF EXISTS "{name}"'))
        admin.dispose()


@pytest.fixture
def db_session():
    """One session inside its own transaction, for a test that drives a
    repository directly rather than through a store."""
    from chat_rag import storage

    with storage.session_scope() as session:
        yield session


@pytest.fixture(autouse=True)
def _isolated_environment():
    """Give every test the environment the session started with.

    ``monkeypatch.setenv``/``delenv`` are not enough on their own. ``delenv``
    with ``raising=False`` records no undo entry when the variable was not set
    to begin with, and neither records a write made by the code under test --
    and the code under test writes to ``os.environ`` by design, because that is
    how ``.env`` reaches the application. A snapshot and a restore cover both.

    This is not belt and braces: it is the fix for a real cross-test leak. A
    unit test exercising the ``.env`` contract set ``CHAT_RAG_DATA_DIR`` through
    the production loader; every integration test that ran afterwards resolved
    its ledger and its stores under that one directory instead of its own
    ``tmp_path``, and saw the documents the previous tests had written. It only
    showed up when the unit tests were asked to run first.
    """
    from chat_rag.config import paths

    before = dict(os.environ)
    # The same argument applies to the loader's own record of what it applied,
    # which ``paths.diagnostics()`` reads.
    applied = dict(paths._from_env_file)
    ignored = dict(paths._ignored_from_env_file)

    yield

    if os.environ != before:
        os.environ.clear()
        os.environ.update(before)
    paths._from_env_file.clear()
    paths._from_env_file.update(applied)
    paths._ignored_from_env_file.clear()
    paths._ignored_from_env_file.update(ignored)


@pytest.fixture(autouse=True)
def _isolated_process_caches():
    """Give every test the process-wide caches in their empty state.

    Phase 3 added two things that live for the length of the process on
    purpose -- the shared Hybrid boundary model and the metrics registry --
    and process-wide state is exactly what one test leaves behind for the
    next. A test that expects the boundary model to be loaded would otherwise
    find one an earlier test had already put there, and counters would carry
    a previous test's jobs. Both are reset here, for the same reason the
    environment is: a test's shared state should be its own.

    Only modules that are already imported are touched, so this costs nothing
    for the tests that never reach them.
    """
    import sys

    yield

    analysis = sys.modules.get("chat_rag.components.viewer.analysis")
    if analysis is not None:
        analysis.release_boundary_model()
    for shared in ("chat_rag.components.embedding.sentence_transformer_embedding",
                   "chat_rag.components.reranker.cross_encoder_reranker"):
        module = sys.modules.get(shared)
        if module is not None:
            module.release_models()
    # The registry is the current runtime's since L3, not a module global.
    # Only reset one that has actually been built, so a test that never
    # measured anything still pays nothing.
    runtime_module = sys.modules.get("chat_rag.runtime")
    if runtime_module is not None:
        current = runtime_module.current()
        if current._metrics is not None:
            current._metrics.reset()
    app_module = sys.modules.get("app")
    if app_module is not None and hasattr(app_module, "pipeline_cache"):
        app_module.pipeline_cache.clear()


@pytest.fixture(scope="session")
def session_state_root() -> str:
    """Where this session's cwd-relative state lives, for tests that want to
    assert something landed there rather than in the checkout."""
    return SESSION_ROOT
