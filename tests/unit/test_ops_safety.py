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
import os
from pathlib import Path

import pytest

from components.observability import events
from components.observability import telemetry as T


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
def build_logger(tmp_path, monkeypatch, **env):
    """A fresh RAGLogger writing into ``tmp_path``, whatever ran before."""
    import utils.logger as logger_module

    for name, value in env.items():
        monkeypatch.setenv(name, str(value))
    monkeypatch.setattr(logger_module.paths, "logs", lambda: str(tmp_path))
    for name, value in (("LOG_MAX_BYTES", logger_module._positive("LOG_MAX_BYTES", 10 * 1024 * 1024)),
                        ("LOG_BACKUPS", logger_module._positive("LOG_BACKUPS", 3)),
                        ("LOG_RUNS_KEPT", logger_module._positive("LOG_RUNS_KEPT", 10))):
        monkeypatch.setattr(logger_module, name, value)
    monkeypatch.setattr(logger_module.RAGLogger, "_instance", None)
    monkeypatch.setattr(logger_module.RAGLogger, "_initialized", False)
    instance = logger_module.RAGLogger()
    return logger_module, instance


def test_one_run_cannot_fill_the_disk(tmp_path, monkeypatch):
    """The file rotates, so a long-lived process has a bounded log."""
    module, instance = build_logger(tmp_path, monkeypatch, LOG_MAX_BYTES=2048, LOG_BACKUPS=2)
    logger = instance.get_logger("ops")
    try:
        for index in range(400):
            logger.info("event=ingest.job.finished job_id=%s %s", index, "x" * 200)
    finally:
        for handler in list(instance.logger.handlers):
            handler.close()
            instance.logger.removeHandler(handler)

    files = sorted(tmp_path.glob("rag_*.log*"))
    assert files, "the run should have written something"
    # One live file plus at most LOG_BACKUPS rotations, each within the cap
    # (a single record may overshoot it, never a multiple of it).
    assert len(files) <= module.LOG_BACKUPS + 1
    assert max(path.stat().st_size for path in files) < module.LOG_MAX_BYTES * 2
    total = sum(path.stat().st_size for path in files)
    assert total < module.LOG_MAX_BYTES * (module.LOG_BACKUPS + 2)


def test_restarts_cannot_fill_the_directory(tmp_path, monkeypatch):
    """A restart loop used to leave one more file behind every time."""
    for index in range(25):
        (tmp_path / f"rag_2026010{index // 10}_{index:06d}.log").write_text("old", encoding="utf-8")
    module, instance = build_logger(tmp_path, monkeypatch, LOG_RUNS_KEPT=5)
    try:
        remaining = sorted(tmp_path.glob("rag_*.log*"))
        # The five newest that were there, plus this run's own file.
        assert len(remaining) == 6
        # Newest kept, oldest gone -- pruning by name is pruning by time,
        # because the name is a timestamp.
        assert any("000024" in path.name for path in remaining)
        assert not any("000000" in path.name for path in remaining)
    finally:
        for handler in list(instance.logger.handlers):
            handler.close()
            instance.logger.removeHandler(handler)


def test_a_log_that_cannot_be_pruned_does_not_stop_the_service(tmp_path, monkeypatch):
    """Housekeeping must never be able to prevent a start-up."""
    import utils.logger as logger_module

    victim = tmp_path / "rag_20260101_000000.log"
    victim.write_text("locked", encoding="utf-8")

    def refuse(self):
        raise PermissionError("another process still has it open")

    monkeypatch.setattr(Path, "unlink", refuse)
    logger_module._prune_old_runs(tmp_path, keep=0)
    assert victim.exists()


def reload_logger(monkeypatch, level=None):
    """utils.logger with LOG_FILE_LEVEL read fresh from the environment."""
    import importlib

    import utils.logger as logger_module

    if level is None:
        monkeypatch.delenv("LOG_FILE_LEVEL", raising=False)
    else:
        monkeypatch.setenv("LOG_FILE_LEVEL", level)
    return importlib.reload(logger_module)


@pytest.fixture(autouse=True)
def restore_logger_module():
    """Leave utils.logger as the rest of the suite expects to find it."""
    yield
    import importlib

    import utils.logger as logger_module
    importlib.reload(logger_module)


def content_dump(instance):
    """The pre-existing DEBUG dumps, called exactly as the pipeline calls them.

    These are not this layer's lines -- they belong to the retrieval and answer
    code and are what its developers read. The point is what reaches the file
    when nobody has asked for them.
    """
    module = type(instance)
    logger = instance.get_logger("pipeline")
    module.log_llm_request(
        logger, [{"role": "user", "content": "gizli sirket belgesinin tam metni"}], 0.2, 512)
    module.log_chunks_passed_to_llm(logger, [object()], "gizli sirket belgesinin tam metni")
    logger.info("event=ingest.job.succeeded job_id=job-1 kb_id=kb-1 chunks=12")


def close(instance):
    for handler in list(instance.logger.handlers):
        handler.close()
        instance.logger.removeHandler(handler)


def written(tmp_path):
    return "\n".join(path.read_text(encoding="utf-8", errors="replace")
                        for path in sorted(tmp_path.glob("rag_*.log*")))


def test_the_default_file_level_writes_no_document_content(tmp_path, monkeypatch):
    """INFO by default: a deployment does not keep a copy of its corpus.

    The dumps below write prompts, retrieved chunks and the whole answer
    context at DEBUG. A log file is tailed, shipped and pasted into tickets, so
    a default that puts document text there is a decision nobody makes on
    purpose. What must survive the default is the operational half.
    """
    module = reload_logger(monkeypatch)
    assert module.LOG_FILE_LEVEL == "INFO"

    _, instance = build_logger(tmp_path, monkeypatch)
    try:
        content_dump(instance)
    finally:
        close(instance)

    contents = written(tmp_path)
    assert "gizli sirket belgesinin tam metni" not in contents
    assert "LLM REQUEST" not in contents
    assert "CONTEXT PASSED TO LLM" not in contents
    # ... and the operational events are still there, which is the whole point
    # of choosing INFO rather than switching the file off.
    assert "event=ingest.job.succeeded" in contents


def test_a_developer_can_still_opt_in(tmp_path, monkeypatch):
    """DEBUG is available, explicit, and says so at start-up."""
    module = reload_logger(monkeypatch, "DEBUG")
    assert module.LOG_FILE_LEVEL == "DEBUG"

    _, instance = build_logger(tmp_path, monkeypatch)
    try:
        content_dump(instance)
    finally:
        close(instance)

    contents = written(tmp_path)
    assert "gizli sirket belgesinin tam metni" in contents, "opting in must work"
    assert "LOG_FILE_LEVEL=DEBUG" in contents, "and must be announced"


def test_an_unrecognised_level_falls_back_to_info_not_debug(monkeypatch):
    """A typo must not be the thing that starts writing document text."""
    for value in ("TRACE", "verbose", "", "root", "handlers"):
        module = reload_logger(monkeypatch, value)
        assert module.LOG_FILE_LEVEL == "INFO", value
        assert module.LOG_LEVELS[module.LOG_FILE_LEVEL] == logging.INFO


def test_the_runtime_configuration_agrees_with_the_code():
    """env.example, the container's environment and the code say one thing."""
    import utils.logger as logger_module

    root = Path(__file__).resolve().parents[2]
    assert logger_module.LOG_LEVELS[logger_module.LOG_FILE_LEVEL] == logging.INFO
    assert "# LOG_FILE_LEVEL=INFO" in (root / "env.example").read_text(encoding="utf-8")
    assert "LOG_FILE_LEVEL=INFO" in (root / ".env.docker").read_text(encoding="utf-8")
