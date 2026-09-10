"""A declared data root owns every state path, and a stale .env cannot move it.

The failure this exists to prevent was real and quiet. ``config/settings.py``
loads the checkout's ``.env`` at import time, whatever the process is; that file
carries a state path from local development; and that path was read straight
from the environment and beat ``CHAT_RAG_DATA_DIR``. So a smoke check, a
container or a test that had said exactly where its state should live still
wrote into the developer's checkout, and nothing said so. The workaround was
for each caller to force the variable itself, which put the isolation in the
callers rather than in the application -- and a caller that forgot had no way
to find out.

The store that failure was originally about -- ``VECTOR_DB_PATH=./chroma_db``,
which opened the developer's real Chroma directory -- is not a path any more:
Step 9 moved the vectors into PostgreSQL, where ``DATABASE_URL`` names them and
this module's rules do not apply. What is left under the rule is the parser's
canonical-unit cache, and the rule is the same rule.

The contract now, highest precedence first:

1. the real process environment -- a container, a compose file, a test, an
   operator's shell;
2. the ``.env`` file, for anything that is not a state path;
3. the ``.env`` file, for state paths, but only where no data root is declared.

Rule 1 keeps every legitimate explicit override working. Rule 3 is what closes
the hole: ``.env`` describes a developer's own layout, and a deployment that has
named its data directory is not that.
"""

from __future__ import annotations

import os

import pytest

from chat_rag.config import paths


@pytest.fixture
def clean_env(monkeypatch):
    """No data root and no path overrides, whatever the session inherited.

    Preconditions only. Putting the environment back afterwards is the session
    conftest's job, and has to be: these tests call the real loader, which
    writes into ``os.environ`` by design, and ``monkeypatch`` can only undo
    what it made itself.
    """
    for name in (paths.DATA_DIR_ENV, paths.PARSER_CACHE_ENV):
        monkeypatch.delenv(name, raising=False)
    paths._from_env_file.clear()
    paths._ignored_from_env_file.clear()


def env_file(tmp_path, body: str) -> str:
    path = tmp_path / ".env"
    path.write_text(body, encoding="utf-8")
    return str(path)


#: What a developer's .env actually looks like: a path from their checkout,
#: next to settings that have nothing to do with paths.
DEVELOPER_ENV = """
LLM_PROVIDER=ollama
VECTOR_DB_COLLECTION=documents
STRUCTURED_PARSER_CACHE=.cache/canonical-units
DEFAULT_TOP_K=5
"""

#: The same file with the setting Step 9 removed still in it, because a
#: developer's .env is not migrated when the code is.
STALE_DEVELOPER_ENV = DEVELOPER_ENV + "VECTOR_DB_PATH=./chroma_db\n"


# ------------------------------------------------------- the isolation itself


def test_a_data_root_refuses_a_parser_cache_left_in_a_dotenv(tmp_path, monkeypatch, clean_env):
    """The exact accident: an isolated root, and the checkout's cache anyway."""
    monkeypatch.setenv(paths.DATA_DIR_ENV, str(tmp_path / "isolated"))

    paths.load_env_file(env_file(tmp_path, DEVELOPER_ENV))

    assert paths.PARSER_CACHE_ENV not in os.environ
    assert paths.canonical_cache() == os.path.join(
        str(tmp_path / "isolated"), "cache", "canonical-units"
    )
    assert ".cache/canonical-units" not in paths.canonical_cache()


def test_the_refusal_is_reported_rather_than_silent(tmp_path, monkeypatch, clean_env):
    """Whatever the contract does, an operator has to be able to see it."""
    monkeypatch.setenv(paths.DATA_DIR_ENV, str(tmp_path / "isolated"))
    paths.load_env_file(env_file(tmp_path, DEVELOPER_ENV))

    reported = " ".join(paths.diagnostics())

    assert "STRUCTURED_PARSER_CACHE=.cache/canonical-units" in reported
    assert paths.DATA_DIR_ENV in reported


def test_a_stale_vector_store_path_can_no_longer_move_anything(
    tmp_path, monkeypatch, clean_env
):
    """The setting this whole contract was written for is gone.

    A developer's ``.env`` still has ``VECTOR_DB_PATH=./chroma_db`` in it and
    always will; nothing migrates that file. It must now be inert -- not
    merely refused when a data root is declared, but unable to name a store
    at all, because there is no store with a path to name.
    """
    monkeypatch.setenv(paths.DATA_DIR_ENV, str(tmp_path / "isolated"))
    paths.load_env_file(env_file(tmp_path, STALE_DEVELOPER_ENV))

    assert not hasattr(paths, "fallback_vector_store")
    assert not hasattr(paths, "vector_store")
    assert "VECTOR_DB_PATH" not in paths.STATE_PATH_ENV


# --------------------------------------------- what must keep working exactly


def test_without_a_data_root_a_dotenv_still_supplies_the_cache(tmp_path, clean_env):
    """A plain local checkout behaves exactly as it always has."""
    paths.load_env_file(env_file(tmp_path, DEVELOPER_ENV))

    assert os.environ[paths.PARSER_CACHE_ENV] == ".cache/canonical-units"
    assert paths.canonical_cache() == ".cache/canonical-units"


def test_an_explicit_environment_override_still_wins(tmp_path, monkeypatch, clean_env):
    """A deliberate override -- a cache on another mount -- is not the bug."""
    monkeypatch.setenv(paths.DATA_DIR_ENV, str(tmp_path / "isolated"))
    monkeypatch.setenv(paths.PARSER_CACHE_ENV, str(tmp_path / "elsewhere"))

    paths.load_env_file(env_file(tmp_path, DEVELOPER_ENV))

    assert paths.canonical_cache() == str(tmp_path / "elsewhere")


def test_an_override_outside_the_data_root_is_called_out(tmp_path, monkeypatch, clean_env):
    """Allowed, because someone may mean it; never silent, because most do not."""
    monkeypatch.setenv(paths.DATA_DIR_ENV, str(tmp_path / "isolated"))
    monkeypatch.setenv(paths.PARSER_CACHE_ENV, str(tmp_path / "elsewhere"))

    reported = " ".join(paths.diagnostics())

    assert "outside" in reported
    assert str(tmp_path / "elsewhere") in reported


def test_an_override_inside_the_data_root_is_not_called_out(tmp_path, monkeypatch, clean_env):
    monkeypatch.setenv(paths.DATA_DIR_ENV, str(tmp_path / "isolated"))
    monkeypatch.setenv(paths.PARSER_CACHE_ENV,
                       str(tmp_path / "isolated" / "cache" / "canonical-units"))

    assert paths.diagnostics() == []


def test_the_real_environment_beats_the_file_for_everything(tmp_path, monkeypatch, clean_env):
    monkeypatch.setenv("DEFAULT_TOP_K", "12")

    paths.load_env_file(env_file(tmp_path, DEVELOPER_ENV))

    assert os.environ["DEFAULT_TOP_K"] == "12"


def test_a_dotenv_may_still_declare_the_data_root_itself(tmp_path, clean_env):
    """And when it does, it applies before the paths it then governs."""
    root = str(tmp_path / "from-the-file")
    paths.load_env_file(
        env_file(tmp_path,
                 f"{paths.DATA_DIR_ENV}={root}\nSTRUCTURED_PARSER_CACHE=.cache/units\n")
    )

    assert paths.data_root() == root
    assert paths.canonical_cache() == os.path.join(root, "cache", "canonical-units")


def test_a_missing_dotenv_is_not_an_error(tmp_path, clean_env):
    assert paths.load_env_file(str(tmp_path / "nothing-here")) == {}


# ------------------------------------------- the parser cache reads it live


def test_the_parser_cache_is_resolved_when_it_is_needed_not_at_import(
    tmp_path, monkeypatch, clean_env
):
    """It used to be a module constant read at import time.

    Which meant a data directory configured after that module happened to load
    -- a container, a test, a smoke check -- was ignored, and the cache stayed
    wherever the process started.
    """
    from chat_rag.components.parsers import canonical_units_store

    monkeypatch.setenv(paths.DATA_DIR_ENV, str(tmp_path / "first"))
    assert canonical_units_store.default_cache_dir() == (
        tmp_path / "first" / "cache" / "canonical-units"
    )

    monkeypatch.setenv(paths.DATA_DIR_ENV, str(tmp_path / "second"))
    assert canonical_units_store.default_cache_dir() == (
        tmp_path / "second" / "cache" / "canonical-units"
    )


# ------------------------------------------- the isolation these tests need

#: A value written straight into the environment, the way the production
#: loader writes one -- not through monkeypatch, which is the whole point.
_LEAK_MARKER = "chat-rag-isolation-probe"


def test_a_test_may_set_the_data_root_directly():
    """This is what the .env loader does, and what a test doing so looks like.

    Deliberately left set at the end. The next test asserts it is gone: pytest
    runs the tests of a file in definition order, so the pair is a real
    regression check on the session's environment isolation rather than a
    statement about it.

    The leak this pins down cost a full validation cycle. A test setting the
    data root through the production loader left ``CHAT_RAG_DATA_DIR`` set for
    the rest of the session, and every integration test after it resolved its
    ledger and its knowledge base stores under that one directory instead of
    its own ``tmp_path`` -- so uploads accumulated across tests and a test
    expecting one document found eleven. ``monkeypatch.delenv(raising=False)``
    on a variable that was never set records no undo entry, so nothing put it
    back; and it only surfaced when the unit tests were asked to run first.
    """
    os.environ[paths.DATA_DIR_ENV] = _LEAK_MARKER
    os.environ["CHAT_RAG_ISOLATION_PROBE"] = _LEAK_MARKER


def test_the_next_test_does_not_inherit_it():
    assert os.environ.get(paths.DATA_DIR_ENV) != _LEAK_MARKER, (
        "a data root leaked out of the previous test; every test after it would "
        "share one ledger and one set of stores"
    )
    assert "CHAT_RAG_ISOLATION_PROBE" not in os.environ

