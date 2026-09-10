"""What importing the library is allowed to do, which is almost nothing.

A library that changes the process when it is imported has made decisions on
behalf of a program that has not spoken yet. Two of those were real here:

* ``chat_rag/application/__init__.py`` set four OpenMP thread-pool variables,
  so importing a use case reconfigured every numeric library in the process;
* ``chat_rag/utils/logger.py`` built a singleton at module scope, so importing
  anything created a directory, pruned old files, opened a rotating log and
  attached a console handler.

Both were correct behaviour for the *product* and wrong behaviour for a
library, so both moved to a call an entry point makes
(:func:`chat_rag.process.apply_thread_defaults`,
:func:`chat_rag.utils.logger.configure_logging`). This file is what stops them
coming back.

The checks run in a **subprocess**, because by the time this test executes the
suite has already imported half the package: the only honest way to ask what
an import does is to do one from scratch.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from chat_rag.process import THREAD_DEFAULTS

REPO = Path(__file__).resolve().parents[2]

#: Everything a request path loads, deepest first. Importing the whole product
#: surface is the real question -- ``import chat_rag`` alone would pass this
#: even if ``application`` still mutated the environment.
PRODUCT_IMPORTS = (
    "chat_rag",
    "chat_rag.config",
    "chat_rag.utils.logger",
    "chat_rag.application",
    "chat_rag.application.services",
)

_PROBE = """
import json, logging, os, sys

before = dict(os.environ)
for name in {names!r}:
    __import__(name)

added = {{k: v for k, v in os.environ.items() if k not in before}}
handlers = [type(h).__name__ for h in logging.getLogger("RAG").handlers]
print("@@" + json.dumps({{
    "added": added,
    "handlers": handlers,
    "level": logging.getLogger("RAG").level,
    "cwd_entries": sorted(os.listdir(".")),
}}))
"""


def _import_in_a_fresh_process(tmp_path, names=PRODUCT_IMPORTS) -> dict:
    """Import ``names`` in a new interpreter and report what it changed."""
    finished = subprocess.run(
        [sys.executable, "-c", _PROBE.format(names=list(names))],
        cwd=tmp_path, capture_output=True, text=True, timeout=600,
        env={**_clean_env(), "PYTHONPATH": str(REPO)},
    )
    assert finished.returncode == 0, finished.stderr[-3000:]
    line = next(l for l in finished.stdout.splitlines() if l.startswith("@@"))
    return json.loads(line[2:])


def _clean_env() -> dict:
    import os

    env = {k: v for k, v in os.environ.items() if k not in THREAD_DEFAULTS}
    # The suite's own database, so importing ``storage`` finds a URL rather
    # than warning about one. Nothing here connects.
    env.setdefault("DATABASE_URL", os.environ.get("DATABASE_URL", ""))
    return env


@pytest.fixture(scope="module")
def imported(tmp_path_factory) -> dict:
    return _import_in_a_fresh_process(tmp_path_factory.mktemp("import-probe"))


def test_importing_the_library_sets_no_thread_variables(imported):
    """The OpenMP block that used to run on the way into ``application``.

    It is not gone -- ``asgi.py``, ``python -m cli``, the smoke tools and this
    suite's own ``conftest`` all call it -- it simply is not an import any
    more.
    """
    leaked = sorted(name for name in THREAD_DEFAULTS if name in imported["added"])
    assert leaked == [], (
        f"importing chat_rag set {leaked}; that belongs to "
        "chat_rag.process.apply_thread_defaults, which an entry point calls"
    )


def test_importing_the_library_installs_no_log_handlers(imported):
    """A NullHandler, and nothing that writes anywhere."""
    assert imported["handlers"] == ["NullHandler"], imported["handlers"]
    # NOTSET: the level is the calling program's decision until
    # ``configure_logging`` is asked for the product's.
    assert imported["level"] == 0


def test_importing_the_library_creates_no_log_directory(imported):
    """The singleton used to make one, wherever the process happened to be."""
    assert "logs" not in imported["cwd_entries"], imported["cwd_entries"]


def test_the_entry_points_still_apply_the_thread_defaults():
    """The other half: what moved out of the import is really called.

    Checked at the source, because each entry point has to do it *before* it
    imports the application -- which is the part an ordinary call-graph test
    would not notice.
    """
    for entry in ("asgi.py", "cli/__init__.py", "tools/import_smoke.py",
                  "tests/conftest.py"):
        source = (REPO / entry).read_text(encoding="utf-8")
        assert "apply_thread_defaults()" in source, entry


def test_the_server_and_the_cli_install_the_handlers_the_library_does_not():
    """A running product still logs to a file; it just says so."""
    for entry in ("asgi.py", "cli/__init__.py"):
        source = (REPO / entry).read_text(encoding="utf-8")
        assert "configure_logging()" in source, entry


def test_applying_the_thread_defaults_never_overrides_an_operator(monkeypatch):
    """``setdefault``, so a container that sized its own pools keeps them."""
    from chat_rag.process import apply_thread_defaults

    env = {"OMP_NUM_THREADS": "8"}
    applied = apply_thread_defaults(env)

    assert env["OMP_NUM_THREADS"] == "8", "an explicit value is not overridden"
    assert "OMP_NUM_THREADS" not in applied
    assert env["TOKENIZERS_PARALLELISM"] == "false", "and the rest are still set"
    assert set(applied) == set(THREAD_DEFAULTS) - {"OMP_NUM_THREADS"}
