"""What an operations surface is allowed to say, and how big it may get.

Two things here, both about the blast radius of being observable.

The first is that an arbitrary exception string is not a safe thing to serve.
Redacting credentials was never the whole problem: a library writes its
messages for a developer standing in the source tree, so an ordinary
``[Errno 2]`` carries the operator's account name, the deployment's directory
layout and the document's own title, none of which passes a credential filter
because none of it looks like a key. ``/api/ops/metrics`` keeps a few example
messages per error category, so every one of those strings goes through
:func:`redact_message` before it is stored -- once, at the point of storage,
so nothing downstream has to remember.

The second is that the application owns a file log. It is not a container's
stdout that someone else rotates; it writes into the data root, and it used to
grow in two directions at once -- each file without limit, and a new file per
restart that nothing removed. Both are bounded now, and both bounds are
asserted rather than described.

The same file is where this system's DEBUG dumps of prompts, retrieved chunks
and answer context go, so its level is the third thing tested here: INFO by
default, so a deployment does not accumulate a copy of its corpus without
anyone choosing that, with DEBUG available to a developer who asks for it.
"""

from __future__ import annotations

import logging
import logging.handlers
import os
from pathlib import Path

import pytest

from chat_rag.components.observability import events
from chat_rag.components.observability import telemetry as T


# --------------------------------------------------------------- messages
def test_an_absolute_path_does_not_survive_into_a_stored_message():
    """The layout of the deployment is not an operational fact."""
    windows = events.redact_message(
        "[Errno 2] No such file or directory: "
        + repr(os.path.join("C:" + os.sep, "Users", "alice", "uploads", "Q3 board minutes.pdf"))
    )
    assert "alice" not in windows
    assert "Q3 board minutes" not in windows
    assert events.PATH_PLACEHOLDER in windows
    # The half that says what went wrong is kept.
    assert "No such file or directory" in windows

    posix = events.redact_message("could not read '/srv/deploy/data/kb/kb-9/records.json'")
    assert "/srv/deploy" not in posix
    assert events.PATH_PLACEHOLDER in posix


def test_a_credential_shaped_value_is_blanked_wherever_it_appears():
    message = events.redact_message("gateway rejected sk-abcdefghijklmnopqrstuvwxyz012345")
    assert "sk-abcdefghijkl" not in message
    assert events.REDACTED in message
    assert "gateway rejected" in message


def test_an_operationally_useful_message_survives_intact():
    """Over-redaction costs an operator the answer, so this is a real limit."""
    for message in ("embedding endpoint returned HTTP 429",
                    "chunking failed at unit 41 of 118",
                    "the job's deadline passed before it finished"):
        assert events.redact_message(message) == message


def test_a_stored_message_is_truncated_however_long_the_exception_was():
    stored = events.redact_message("x" * 10_000)
    assert len(stored) <= events.MAX_VALUE


def test_the_registry_redacts_before_it_stores(monkeypatch):
    """The one place it happens, so the endpoint cannot forget to."""
    registry = T.MetricsRegistry(window=10)
    registry.record_error(
        "storage",
        "failed to open " + repr(os.path.join("C:" + os.sep, "srv", "data", "corpus.pdf"))
        + " with key sk-abcdefghijklmnopqrstuvwxyz012345",
    )
    stored = registry.snapshot()["errors"]["recent_messages"]["storage"]
    assert stored, "the example message should still be kept"
    joined = " ".join(stored)
    assert "corpus.pdf" not in joined
    assert "sk-abcdefghijkl" not in joined
    assert events.PATH_PLACEHOLDER in joined
    assert "failed to open" in joined


def test_the_examples_per_category_stay_bounded():
    """A provider failing in a loop cannot fill memory with its own prose."""
    registry = T.MetricsRegistry(window=10)
    for index in range(200):
        registry.record_error("provider", f"gateway said {index}")
    stored = registry.snapshot()["errors"]["recent_messages"]["provider"]
    assert len(stored) == T.MESSAGES_PER_CATEGORY
    assert registry.snapshot()["errors"]["by_category"]["provider"] == 200


# ------------------------------------------------------------------- logs
#
# Since L2 the handlers are installed by a call rather than by an import, so
# these drive the same two handlers through ``configure_logging`` instead of
# through a singleton's constructor. What is asserted is unchanged: one run is
# bounded, a restart loop is bounded, a failed prune does not stop the service,
# and the file level defaults to INFO.


@pytest.fixture(autouse=True)
def _no_handlers_left_behind():
    """A test that installs handlers takes them off again.

    The suite runs with the library default -- a NullHandler and nothing else
    -- and a test that opened a rotating file must not leave the next one
    writing into its ``tmp_path``.
    """
    from chat_rag.utils import logger as logger_module

    yield
    logger_module.reset_logging()


def configure(tmp_path, **env):
    """The real entry-point call, into ``tmp_path`` and from a given env."""
    from chat_rag.utils import logger as logger_module

    settings = logger_module.logging_from_env({k: str(v) for k, v in env.items()})
    logger_module.configure_logging(settings, directory=str(tmp_path), force=True)
    return logger_module, settings


def written(tmp_path):
    return "\n".join(path.read_text(encoding="utf-8", errors="replace")
                     for path in sorted(tmp_path.glob("rag_*.log*")))


def content_dump(module):
    """The pre-existing DEBUG dumps, called exactly as the pipeline calls them.

    These are not this layer's lines -- they belong to the retrieval and answer
    code and are what its developers read. The point is what reaches the file
    when nobody has asked for them.
    """
    logger = module.get_logger("pipeline")
    module.RAGLogger.log_llm_request(
        logger, [{"role": "user", "content": "gizli sirket belgesinin tam metni"}], 0.2, 512)
    module.RAGLogger.log_chunks_passed_to_llm(
        logger, [object()], "gizli sirket belgesinin tam metni")
    logger.info("event=ingest.job.succeeded job_id=job-1 kb_id=kb-1 chunks=12")


def test_one_run_cannot_fill_the_disk(tmp_path):
    """The file rotates, so a long-lived process has a bounded log."""
    module, settings = configure(tmp_path, LOG_MAX_BYTES=2048, LOG_BACKUPS=2)
    logger = module.get_logger("ops")
    for index in range(400):
        logger.info("event=ingest.job.finished job_id=%s %s", index, "x" * 200)
    module.reset_logging()

    files = sorted(tmp_path.glob("rag_*.log*"))
    assert files, "the run should have written something"
    # One live file plus at most ``backups`` rotations, each within the cap
    # (a single record may overshoot it, never a multiple of it).
    assert len(files) <= settings.backups + 1
    assert max(path.stat().st_size for path in files) < settings.max_bytes * 2
    total = sum(path.stat().st_size for path in files)
    assert total < settings.max_bytes * (settings.backups + 2)


def test_restarts_cannot_fill_the_directory(tmp_path):
    """A restart loop used to leave one more file behind every time."""
    for index in range(25):
        (tmp_path / f"rag_2026010{index // 10}_{index:06d}.log").write_text("old", encoding="utf-8")
    configure(tmp_path, LOG_RUNS_KEPT=5)

    remaining = sorted(tmp_path.glob("rag_*.log*"))
    # The five newest that were there, plus this run's own file.
    assert len(remaining) == 6
    # Newest kept, oldest gone -- pruning by name is pruning by time, because
    # the name is a timestamp.
    assert any("000024" in path.name for path in remaining)
    assert not any("000000" in path.name for path in remaining)


def test_a_log_that_cannot_be_pruned_does_not_stop_the_service(tmp_path, monkeypatch):
    """Housekeeping must never be able to prevent a start-up."""
    from chat_rag.utils import logger as logger_module

    victim = tmp_path / "rag_20260101_000000.log"
    victim.write_text("locked", encoding="utf-8")

    def refuse(self):
        raise PermissionError("another process still has it open")

    monkeypatch.setattr(Path, "unlink", refuse)
    logger_module._prune_old_runs(tmp_path, keep=0)
    assert victim.exists()


def test_the_default_file_level_writes_no_document_content(tmp_path):
    """INFO by default: a deployment does not keep a copy of its corpus.

    The dumps below write prompts, retrieved chunks and the whole answer
    context at DEBUG. A log file is tailed, shipped and pasted into tickets, so
    a default that puts document text there is a decision nobody makes on
    purpose. What must survive the default is the operational half.
    """
    module, settings = configure(tmp_path)
    assert settings.file_level == "INFO"

    content_dump(module)
    module.reset_logging()

    contents = written(tmp_path)
    assert "gizli sirket belgesinin tam metni" not in contents
    assert "LLM REQUEST" not in contents
    assert "CONTEXT PASSED TO LLM" not in contents
    # ... and the operational events are still there, which is the whole point
    # of choosing INFO rather than switching the file off.
    assert "event=ingest.job.succeeded" in contents


def test_a_developer_can_still_opt_in(tmp_path):
    """DEBUG is available, explicit, and says so at start-up."""
    module, settings = configure(tmp_path, LOG_FILE_LEVEL="DEBUG")
    assert settings.file_level == "DEBUG"

    content_dump(module)
    module.reset_logging()

    contents = written(tmp_path)
    assert "gizli sirket belgesinin tam metni" in contents, "opting in must work"
    assert "LOG_FILE_LEVEL=DEBUG" in contents, "and must be announced"


def test_an_unrecognised_level_falls_back_to_info_not_debug():
    """A typo must not be the thing that starts writing document text."""
    from chat_rag.utils import logger as logger_module

    for value in ("TRACE", "verbose", "", "root", "handlers"):
        settings = logger_module.logging_from_env({"LOG_FILE_LEVEL": value})
        assert settings.file_level == "INFO", value
        assert logger_module.LOG_LEVELS[settings.file_level] == logging.INFO


def test_a_fallback_is_reported_rather_than_silent():
    """The fail-safe half only works if somebody can see it happened."""
    from chat_rag.utils import logger as logger_module

    settings = logger_module.logging_from_env(
        {"LOG_FILE_LEVEL": "TRACE", "LOG_BACKUPS": "-2"})
    joined = " ".join(settings.fallbacks)
    assert "LOG_FILE_LEVEL" in joined and "LOG_BACKUPS" in joined
    # And a clean environment reports nothing, so the list means what it says.
    assert logger_module.logging_from_env({}).fallbacks == ()


def test_the_runtime_configuration_agrees_with_the_code():
    """env.example, the container's environment and the code say one thing."""
    from chat_rag.utils import logger as logger_module

    root = Path(__file__).resolve().parents[2]
    default = logger_module.LoggingSettings()
    assert logger_module.LOG_LEVELS[default.file_level] == logging.INFO
    assert "# LOG_FILE_LEVEL=INFO" in (root / "env.example").read_text(encoding="utf-8")
    assert "LOG_FILE_LEVEL=INFO" in (root / ".env.docker").read_text(encoding="utf-8")


def test_configuring_twice_does_not_open_a_second_file(tmp_path):
    """Two entry points in one process must not produce two log files.

    ``python -m cli`` importing a tool that also configures, a test importing
    both: the second call is a no-op, so the handlers stay one file handler and
    one console handler.
    """
    module, _ = configure(tmp_path)
    module.configure_logging()  # no force: the second caller
    module.configure_logging()

    handlers = logging.getLogger(module.ROOT_LOGGER).handlers
    files = [h for h in handlers if isinstance(h, logging.handlers.RotatingFileHandler)]
    assert len(files) == 1
    assert len(sorted(tmp_path.glob("rag_*.log"))) == 1


def test_the_diagnostics_report_the_handlers_that_exist(tmp_path):
    """``logging_configuration`` answers for the process, not for the
    environment, once a process has said what it wants."""
    from chat_rag.utils import logger as logger_module

    module, _ = configure(tmp_path, LOG_FILE_LEVEL="DEBUG")
    assert logger_module.logging_configuration()["file_level"] == "DEBUG"

    module.reset_logging()
    # With no handlers the honest answer is what the environment would give,
    # which is the default here.
    assert logger_module.logging_configuration()["file_level"] == "INFO"
