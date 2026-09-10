"""Where a setting comes from, what wins, and what a bad value does.

The contract these tests hold, stated once:

    the real process environment  >  accepted values from .env  >  the
    application default written on the setting's own dataclass field

with one deliberate exception for state paths, which
``tests/unit/test_state_isolation.py`` owns: a ``.env`` may not move
``STRUCTURED_PARSER_CACHE`` once ``CHAT_RAG_DATA_DIR`` is declared. That asymmetry exists because ``.env`` describes a developer's
local layout and a deployment has already said where its state lives.

What is proved here:

* the precedence, exercised through the real loader rather than asserted
  about constants;
* one authority per default -- ``env.example`` and ``.env.docker`` cannot
  drift from the code without a declared reason;
* the cross-group rules (a synchronous upload that outlives its connection,
  a thread ration that leaves nothing free) fire where they should;
* no credential reaches the diagnostics;
* the three entrypoints resolve the same configuration.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

from chat_rag.config import database as database_config
from chat_rag.config import ingest as ingest_config
from chat_rag.config import paths, query as query_config, runtime as runtime_config

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture
def clean_env(monkeypatch):
    """No configuration from the developer's shell or .env reaches a test."""
    for name in ("FLASK_HOST", "FLASK_PORT", "WAITRESS_THREADS",
                 "WAITRESS_CHANNEL_TIMEOUT", "INGEST_WORKERS", "INGEST_SYNC_WAIT",
                 "INGEST_SYNC_WAITERS", "QUERY_MAX_ACTIVE", "ANSWER_MAX_INFLIGHT",
                 "LOG_LEVEL", "LOG_FILE_LEVEL", paths.DATA_DIR_ENV,
                 paths.PARSER_CACHE_ENV):
        monkeypatch.delenv(name, raising=False)


# ------------------------------------------------------------- precedence


def test_the_process_environment_beats_the_env_file(tmp_path, monkeypatch, clean_env):
    """Rule 1, through the real loader."""
    env_file = tmp_path / ".env"
    env_file.write_text("WAITRESS_THREADS=32\nINGEST_WORKERS=9\n", encoding="utf-8")
    monkeypatch.setenv("WAITRESS_THREADS", "12")

    applied = paths.load_env_file(str(env_file))

    assert os.environ["WAITRESS_THREADS"] == "12", "the shell wins"
    assert os.environ["INGEST_WORKERS"] == "9", "and the file still supplies the rest"
    assert "WAITRESS_THREADS" not in applied
    assert runtime_config.runtime_from_env().request_threads == 12


def test_the_env_file_beats_the_application_default(tmp_path, clean_env):
    env_file = tmp_path / ".env"
    env_file.write_text("WAITRESS_THREADS=16\n", encoding="utf-8")

    paths.load_env_file(str(env_file))

    assert runtime_config.runtime_from_env().request_threads == 16


def test_the_default_applies_when_nothing_overrides_it(clean_env):
    """Rule 3, read off the dataclass field that is the one place it lives."""
    assert runtime_config.runtime_from_env({}) == runtime_config.RuntimeLimits()
    assert runtime_config.RuntimeLimits().request_threads == 8
    assert ingest_config.limits_from_env({}).workers == 2
    assert query_config.query_limits_from_env({}).answer_max_inflight == 4


def test_a_state_path_still_follows_the_stricter_data_root_rule(tmp_path, monkeypatch, clean_env):
    """The Phase 1C contract, unchanged: a declared data root owns state paths."""
    env_file = tmp_path / ".env"
    env_file.write_text("STRUCTURED_PARSER_CACHE=.cache/units\n", encoding="utf-8")
    monkeypatch.setenv(paths.DATA_DIR_ENV, str(tmp_path / "data"))

    paths.load_env_file(str(env_file))

    assert paths.PARSER_CACHE_ENV not in os.environ
    assert paths.canonical_cache().startswith(str(tmp_path / "data"))
    assert any("owns this path" in line for line in paths.diagnostics())


def test_the_env_file_is_applied_by_the_package_before_anything_reads_it():
    """``.env`` used to be applied by ``config.settings``, so a module that
    reached ``config.paths`` or ``utils.logger`` first read an environment the
    file had not been applied to yet. Package init is the one point that
    cannot happen."""
    from chat_rag import config

    source = (REPO / "src" / "chat_rag" / "config" / "__init__.py").read_text(encoding="utf-8")
    assert "load_env_file" in source
    assert config.APPLIED_ENV_FILE is not None
    settings_source = (REPO / "src" / "chat_rag" / "config" / "settings.py").read_text(encoding="utf-8")
    assert "load_env_file" not in settings_source, "one loader, in one place"


def test_only_one_module_reads_a_dotenv_at_all():
    """A second loader with different semantics is the failure this prevents."""
    readers = []
    for path in sorted(REPO.glob("**/*.py")):
        parts = set(path.relative_to(REPO).parts)
        if parts & {"venv", ".venv", "__pycache__", "tests", "build"}:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if "load_dotenv" in text or "dotenv_values" in text:
            readers.append(path.relative_to(REPO).as_posix())
    assert readers == ["src/chat_rag/config/paths.py"], readers


# --------------------------------------------------- one default per setting


#: The request-thread count is what every other ration is sized against, so it
#: is the one most likely to be read in a second place.
def test_the_thread_count_has_exactly_one_reader():
    for module in ("asgi.py", "src/chat_rag/config/ingest.py", "src/chat_rag/config/query.py"):
        source = (REPO / module).read_text(encoding="utf-8")
        assert '"WAITRESS_THREADS"' not in source, (
            f"{module} reads WAITRESS_THREADS itself; config/runtime.py owns it"
        )
    assert runtime_config.runtime_from_env().request_threads >= 1


def test_the_server_and_the_limits_agree_about_the_thread_pool(monkeypatch, clean_env):
    """The bug this replaced: a server with eight threads and limits sized
    against a different number. One reader now, and the server's worker pool
    is sized from it (``asgi._size_thread_pool``)."""
    monkeypatch.setenv("WAITRESS_THREADS", "4")

    assert runtime_config.runtime_from_env().request_threads == 4
    assert query_config.query_limits_from_env().request_threads == 4
    assert ingest_config.limits_from_env().sync_waiters == 2


def test_the_sync_waiter_ration_is_still_half_the_request_threads(clean_env):
    """It reserves request threads for a wait that no route makes any more --
    the synchronous upload went with the Flask-era surface -- and it is kept,
    because it is subtracted from the ``QUERY_MAX_ACTIVE`` default and
    removing it would raise the number of questions a deployment answers at
    once (``docs/legacy-removal.md``)."""
    assert ingest_config.limits_from_env({"WAITRESS_THREADS": "8"}).sync_waiters == 4
    assert ingest_config.limits_from_env({"WAITRESS_THREADS": "2"}).sync_waiters == 1
    assert ingest_config.limits_from_env({"WAITRESS_THREADS": "1"}).sync_waiters == 1
    assert ingest_config.limits_from_env(
        {"INGEST_SYNC_WAITERS": "3", "WAITRESS_THREADS": "8"}).sync_waiters == 3


def test_the_dataclass_field_is_the_only_place_a_default_is_written():
    """Reading a default off the dataclass rather than restating it as a
    string is what stops the two from drifting apart."""
    for module in ("src/chat_rag/config/ingest.py", "src/chat_rag/config/query.py",
                   "src/chat_rag/config/runtime.py"):
        source = (REPO / module).read_text(encoding="utf-8")
        assert "_DEFAULTS" in source, module
    assert ingest_config._DEFAULTS.workers == ingest_config.IngestLimits().workers


# ---------------------------------------------------------------- validation


@pytest.mark.parametrize("value, wording", [
    ("0", "WAITRESS_THREADS must be at least 1"),
    ("-4", "WAITRESS_THREADS must be at least 1"),
    ("eight", "WAITRESS_THREADS='eight' is not a whole number"),
])
def test_a_bad_thread_count_is_refused_by_name(value, wording):
    with pytest.raises(ValueError) as refused:
        runtime_config.runtime_from_env({"WAITRESS_THREADS": value})
    assert wording in str(refused.value)


@pytest.mark.parametrize("variable, value, wording", [
    ("FLASK_PORT", "0", "FLASK_PORT must be a port number"),
    ("FLASK_PORT", "70000", "FLASK_PORT must be a port number"),
    ("WAITRESS_CHANNEL_TIMEOUT", "0", "WAITRESS_CHANNEL_TIMEOUT must be a positive"),
])
def test_a_bad_server_setting_is_refused_by_name(variable, value, wording):
    with pytest.raises(ValueError) as refused:
        runtime_config.runtime_from_env({variable: value})
    assert wording in str(refused.value)


def test_a_synchronous_upload_may_not_outlive_its_connection():
    """The rule three docstrings asserted and nothing enforced."""
    runtime = runtime_config.RuntimeLimits(channel_timeout_seconds=300)
    ingest = ingest_config.IngestLimits(sync_wait_seconds=840)
    query = query_config.QueryLimits()

    with pytest.raises(ValueError) as refused:
        runtime_config.cross_check(runtime, ingest, query)
    assert "INGEST_SYNC_WAIT" in str(refused.value)
    assert "WAITRESS_CHANNEL_TIMEOUT" in str(refused.value)


def test_the_shipped_combination_passes_the_cross_check():
    assert runtime_config.cross_check(
        runtime_config.RuntimeLimits(),
        ingest_config.limits_from_env({}),
        query_config.query_limits_from_env({}),
    ) == []


def test_a_thread_ration_that_leaves_nothing_free_is_reported_not_refused():
    """An explicit QUERY_MAX_ACTIVE has always been honoured and named rather
    than overruled; an operator may have sized it that way on purpose."""
    warnings = runtime_config.cross_check(
        runtime_config.RuntimeLimits(request_threads=8),
        ingest_config.IngestLimits(sync_waiters=4),
        query_config.QueryLimits(max_active=7, request_threads=8, sync_waiters=4),
    )
    assert any("request threads free" in line for line in warnings)


def test_a_per_job_concurrency_above_the_global_cap_is_reported():
    warnings = runtime_config.cross_check(
        runtime_config.RuntimeLimits(),
        ingest_config.IngestLimits(provider_max_inflight=2, deep_concurrency=8),
        query_config.QueryLimits(),
    )
    assert any("PROVIDER_MAX_INFLIGHT" in line for line in warnings)


def test_settings_refuse_to_build_on_an_impossible_combination(monkeypatch):
    from chat_rag.config import Settings

    monkeypatch.setenv("WAITRESS_CHANNEL_TIMEOUT", "60")
    with pytest.raises(ValueError, match="INGEST_SYNC_WAIT"):
        Settings()


# ------------------------------------------------------------------ secrets


def test_no_credential_appears_in_the_configuration_diagnostics(monkeypatch):
    from chat_rag.config import Settings

    monkeypatch.setenv("AZURE_API_KEY", "sk-do-not-log-me")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-nor-me")
    effective = Settings().effective_configuration()

    rendered = repr(effective)
    assert "sk-do-not-log-me" not in rendered
    assert "sk-nor-me" not in rendered
    # Not even the name of the variable a key comes from: /api/ops/metrics is
    # unauthenticated and carries nothing that names a credential. Whether one
    # is configured is the part an operator needs, and that is a boolean.
    assert "OPENROUTER_API_KEY" not in rendered
    assert effective["models"]["answer_key_configured"] is True


def test_the_settings_dump_redacts_credentials(monkeypatch):
    from chat_rag.config import Settings
    from chat_rag.config.settings import REDACTED, SECRET_ATTRIBUTES

    monkeypatch.setenv("AZURE_API_KEY", "sk-do-not-log-me")
    dumped = Settings().to_dict()

    assert dumped["azure_api_key"] == REDACTED
    assert "sk-do-not-log-me" not in repr(dumped)
    for name in SECRET_ATTRIBUTES:
        assert name in dumped, "a redacted name must still exist"


def test_the_ops_endpoint_reports_configuration_without_secrets(monkeypatch):
    from fastapi.testclient import TestClient

    import asgi as entrypoint
    import interfaces.http as http

    with TestClient(http.create_app(entrypoint.services)) as client:
        payload = client.get("/api/ops/metrics").json()

    configuration = payload["configuration"]
    assert set(configuration) >= {"data_root", "runtime", "ingest", "query", "logging"}
    assert configuration["runtime"]["request_threads"] >= 1
    rendered = repr(configuration)
    for banned in ("OPENROUTER_API_KEY", "AZURE_API_KEY", "sk-"):
        assert banned not in rendered, banned



# ------------------------------------------------------- files cannot drift

#: Values in ``env.example`` / ``.env.docker`` that deliberately differ from
#: the application default, each with the reason. Anything not listed here has
#: to match the code, which is what stops a file from quietly becoming the
#: source of truth. Adding a line here is the moment someone states intent.
DELIBERATE_OVERRIDES = {
    "env.example": {
        "ANSWER_PROVIDER": "the demo profile answers through OpenRouter; the code default is the conservative LLM_PROVIDER",
        "ANSWER_MODEL": "the demo's answer model; there is no default model on purpose",
        "ANSWER_FALLBACK_PROVIDER": "the demo keeps a local Ollama fallback; the default is none",
        "ANSWER_FALLBACK_MODEL": "the demo's fallback model",
        "EMBEDDING_PROVIDER": "the demo embeds through the gateway; the default is a local model needing no key",
        "EMBEDDING_MODEL": "the gateway embedding model that goes with it",
        "RETRIEVAL_PROFILE": "the demo runs the final hybrid_rrf chain; the default is the no-provider bm25_only",
        "LLM_PROVIDER": "the demo has no Azure account; the code default is azure",
        "AZURE_ENDPOINT": "a placeholder, not a value",
        "AZURE_API_KEY": "a placeholder, not a value",
        "DATABASE_URL": "there is no application default and cannot be one -- the only fallback would be a credential in a tracked file; this names the local development database docker-compose.test.yml starts",
    },
    ".env.docker": {
        "DATABASE_URL": "the same reason, pointing at the compose service; a real password belongs in .env.docker.local",
        "LLM_PROVIDER": "the container talks to Ollama on the host, not Azure",
        "OLLAMA_BASE_URL": "host.docker.internal reaches the host from inside a container",
        "OLLAMA_MODEL": "the small model the demo image expects",
    },
}


def _declared_values(path: Path) -> dict[str, str]:
    found = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        found[key.strip()] = value.split("#")[0].strip().strip('"').strip("'")
    return found


def _code_default(name: str) -> str | None:
    """What the application would use with nothing set, as text."""
    for limits in (runtime_config.runtime_from_env({}),
                   ingest_config.limits_from_env({}),
                   query_config.query_limits_from_env({})):
        for field, value in limits.to_dict().items():
            if field.upper() == name or f"{field.upper()}_SECONDS" == name:
                return str(value)
    # The database settings are prefixed, and they live in their own module
    # rather than on ``Settings``: without this the drift guard would silently
    # skip every DATABASE_* line in both files.
    for field, value in database_config.database_from_env({}).to_dict().items():
        prefixed = "DATABASE_" + field.upper()
        if prefixed == name or prefixed.replace("_SECONDS", "") == name:
            return "" if value is None else str(value)
    source = (REPO / "src" / "chat_rag" / "config" / "settings.py").read_text(encoding="utf-8")
    match = re.search(rf'getenv\(\s*"{re.escape(name)}"\s*,\s*"([^"]*)"', source)
    if match:
        return match.group(1)
    if re.search(rf'getenv\(\s*"{re.escape(name)}"\s*\)', source):
        return ""
    return None


@pytest.mark.parametrize("filename", ["env.example", ".env.docker"])
def test_an_env_file_value_matches_the_code_or_is_a_declared_override(filename):
    """The drift guard: 'Python says 8, env.example says 4, Docker says 6'."""
    declared = _declared_values(REPO / filename)
    overrides = DELIBERATE_OVERRIDES[filename]
    drifted = []
    for name, value in sorted(declared.items()):
        default = _code_default(name)
        if default is None or name in overrides:
            continue
        if value != default:
            drifted.append(f"{name}={value!r} but the code default is {default!r}")
    assert drifted == [], "\n".join(
        [f"{filename} disagrees with the code and says no reason:"] + drifted
        + ["add each to DELIBERATE_OVERRIDES with a reason, or fix the file"]
    )


@pytest.mark.parametrize("filename", ["env.example", ".env.docker"])
def test_every_declared_override_is_still_a_real_difference(filename):
    """A reason left behind after the difference is gone is noise."""
    declared = _declared_values(REPO / filename)
    stale = [
        name for name, _ in DELIBERATE_OVERRIDES[filename].items()
        if name in declared and _code_default(name) == declared[name]
    ]
    assert stale == [], (
        f"{filename} now matches the code for {stale}; remove them from "
        "DELIBERATE_OVERRIDES"
    )


def test_no_env_file_documents_a_setting_nothing_reads():
    """``env.example`` used to offer circuit breakers, retries, rate limits and
    a cache that no code has ever read.

    This asks the weaker of the two questions -- is the variable read at all.
    ``test_every_setting_read_is_a_setting_applied`` below asks the other one,
    because being read into an attribute nobody looks at is the same silence
    with an extra step.
    """
    sources = "\n".join(
        path.read_text(encoding="utf-8", errors="replace")
        for path in sorted(REPO.glob("**/*.py"))
        if not (set(path.relative_to(REPO).parts) & {"venv", ".venv", "__pycache__", "build"})
    )
    phantom = []
    for filename in ("env.example", ".env.docker"):
        for name in _declared_values(REPO / filename):
            if f'"{name}"' not in sources and f"'{name}'" not in sources:
                phantom.append(f"{filename}: {name}")
    assert phantom == [], (
        "these are documented but read by no code:\n  " + "\n  ".join(phantom)
    )


def test_no_configuration_file_names_a_developers_checkout():
    for filename in ("env.example", ".env.docker", "docker-compose.yml", "Dockerfile"):
        text = (REPO / filename).read_text(encoding="utf-8", errors="replace")
        assert "OneDrive" not in text and "erena" not in text.lower(), filename


# ---------------------------------------------------------------- entrypoints


def test_the_entrypoint_resolves_one_configuration():
    """There is one entrypoint, and what it binds is what the banner reports.

    There were two -- ``python app.py`` and ``python -m wsgi`` -- and the risk
    this test was written for was that they read the settings differently.
    They went with the Flask console; what is left is that ``asgi.py`` and the
    effective configuration cannot disagree."""
    import asgi as entrypoint

    effective = entrypoint.services.settings.effective_configuration()
    assert entrypoint.server_options()["port"] == effective["runtime"]["port"]
    assert entrypoint.server_options()["host"] == effective["runtime"]["host"]
    assert (runtime_config.runtime_from_env().request_threads
            == effective["runtime"]["request_threads"])


def test_the_smoke_tools_declare_their_own_data_root():
    """A smoke run must not open the developer's store; it says where its
    state goes rather than inheriting it."""
    for tool in ("tools/import_smoke.py", "tools/serve_smoke.py"):
        text = (REPO / tool).read_text(encoding="utf-8")
        assert paths.DATA_DIR_ENV in text, tool


def test_a_test_leaves_no_environment_behind(clean_env, monkeypatch):
    """The suite's own isolation, asserted rather than assumed: conftest
    restores os.environ after every test, including writes the code under
    test made itself."""
    monkeypatch.setenv("WAITRESS_THREADS", "31")
    assert os.environ["WAITRESS_THREADS"] == "31"


# ------------------------------------------- read is not the same as applied

#: Attributes of :class:`Settings` that no other module reads, because
#: ``Settings`` itself is their consumer. Naming them keeps "used internally"
#: from becoming a hiding place for "used by nothing": each is here with a
#: reason, and an attribute that stops having one fails the test below.
#:
#: ``runtime_limits``  the thread pool, handed to ``cross_check`` and reported
#:                     by ``effective_configuration``. The one number anything
#:                     else needs is lifted out as ``request_threads``.
#: ``configuration_warnings``  what ``cross_check`` said, for the start-up
#:                     banner and ``/api/ops/metrics`` to print.
READ_BY_SETTINGS_ITSELF = frozenset({"runtime_limits", "configuration_warnings"})


def test_every_setting_read_is_a_setting_applied():
    """A knob that turns nothing is worse than no knob at all.

    Phase 7B found ``LOG_TOKEN_USAGE`` and ``LOG_PARSING_STATS`` read into
    attributes nothing ever looked at. Phase 8 found ten more beside them --
    the PDF backend, OCR language, text encoding, batch sizes, the generation
    temperature -- each documented in ``env.example`` as though setting it did
    something. ``test_no_env_file_documents_a_setting_nothing_reads`` could
    not see any of them, because they *were* read; they were simply never used.

    So: every attribute ``Settings.__init__`` assigns must be read by
    something. Its own methods count -- a value ``Settings`` itself consumes
    is applied, and says so by being listed in ``READ_BY_SETTINGS_ITSELF``. A
    test does not count, because a setting only tests exercise is a setting
    production ignores.
    """
    import ast

    source = (REPO / "src" / "chat_rag" / "config" / "settings.py").read_text(encoding="utf-8")
    assigned = {
        node.attr
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Attribute)
        and isinstance(node.ctx, ast.Store)
        and isinstance(node.value, ast.Name)
        and node.value.id == "self"
    }

    consumers = [
        path
        for path in sorted(REPO.rglob("*.py"))
        if not (
            set(path.relative_to(REPO).parts)
            & {"venv", ".venv", "__pycache__", "tests", "artifacts", "build"}
        )
        and path != REPO / "src" / "chat_rag" / "config" / "settings.py"
    ]
    elsewhere = "\n".join(
        path.read_text(encoding="utf-8", errors="replace") for path in consumers
    )

    inert = []
    for name in sorted(assigned):
        if re.search(rf"\b{re.escape(name)}\b", elsewhere):
            continue
        reads = len(re.findall(rf"self\.{re.escape(name)}\b", source)) - len(
            re.findall(rf"self\.{re.escape(name)}\s*=", source)
        )
        if reads > 0:
            if name not in READ_BY_SETTINGS_ITSELF:
                inert.append(
                    f"{name} -- read only by Settings itself; add it to "
                    "READ_BY_SETTINGS_ITSELF, with the reason, if that is right"
                )
            continue
        inert.append(f"{name} -- read by nothing at all")

    assert inert == [], (
        "these settings are read and then never applied; wire each one, or "
        "remove it from config/settings.py, env.example and the README:\n  "
        + "\n  ".join(inert)
    )


def test_the_config_package_builds_no_settings_at_import():
    """Importing ``config`` applies the ``.env`` file, and does nothing else.

    It used to construct a module-level ``Settings()`` as well, so anything
    wanting only ``config.paths`` -- a smoke tool, a test, ``utils.logger`` --
    paid for a full configuration read, and would have failed at *import time*
    on a bad value, before any process had said it wanted one. Nothing ever
    imported the instance.
    """
    import types

    from chat_rag import config
    from chat_rag.config import settings as settings_module

    assert not hasattr(settings_module, "settings"), (
        "config/settings.py builds a module-level Settings() again"
    )
    assert "settings" not in config.__all__
    # ``config.settings`` is the submodule, and only ever that.
    assert isinstance(config.settings, types.ModuleType)
