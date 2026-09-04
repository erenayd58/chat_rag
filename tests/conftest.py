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

Nothing here changes what the code under test does; it only changes where a
default path points and removes the credentials. Tests that already isolate
themselves with ``tmp_path`` keep doing so.
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


def pytest_report_header(config):
    return f"chat_rag test state: {SESSION_ROOT} (cwd for the session; developer state untouched)"


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    changed = [name for name, before in _BEFORE.items() if _fingerprint().get(name) != before]
    if changed:
        terminalreporter.write_sep(
            "!", "developer state changed during this session: " + ", ".join(changed)
            + " (a live console instance writes these too; check before blaming a test)",
        )


@pytest.fixture(scope="session")
def session_state_root() -> str:
    """Where this session's cwd-relative state lives, for tests that want to
    assert something landed there rather than in the checkout."""
    return SESSION_ROOT
