"""Logging for the RAG system: named loggers here, handlers only when asked.

A library that installs handlers when it is imported has decided, for every
program that imports it, where the output goes. This one does not. Importing
:mod:`chat_rag` -- or anything under it -- attaches a
:class:`logging.NullHandler` to the ``RAG`` logger and does nothing else: no
directory is created, no file is opened, no console handler is added and the
environment is not read.

:func:`configure_logging` is what installs the file and console handlers, and
the *process* calls it: ``asgi.py`` before it serves, ``python -m cli`` before
it runs, the smoke tools before they import the application. That is the whole
change -- the same two handlers, the same rotation, the same pruning and the
same start-up line, moved from import time to a call.

**Fail-safe, not fail-fast.** Unlike every numeric limit in ``config/``, a
value this module does not understand falls back and is *reported* rather than
refusing to start: the risky direction here is DEBUG, which writes prompts,
retrieved chunks and answer context to a file that gets tailed, shipped and
pasted into tickets. A typo must never be the thing that turns that on, and a
logging typo must never take the service down. See ``docs/configuration.md``.
"""

from __future__ import annotations

import logging
import logging.handlers
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

from chat_rag.config import paths

#: An unrecognised name falls back to INFO rather than to DEBUG. A name is
#: looked up in this table rather than on the logging module, so only a level
#: can ever be named.
LOG_LEVELS = {
    "CRITICAL": logging.CRITICAL, "ERROR": logging.ERROR,
    "WARNING": logging.WARNING, "WARN": logging.WARNING,
    "INFO": logging.INFO, "DEBUG": logging.DEBUG,
}

#: The logger every name in this package hangs off.
ROOT_LOGGER = "RAG"

# The one thing that happens at import, and the reason it is safe: a
# NullHandler decides nothing, writes nothing and is what keeps "no handlers
# could be found" off a library user's stderr.
logging.getLogger(ROOT_LOGGER).addHandler(logging.NullHandler())


def _positive(env: Mapping[str, str], name: str, default: int,
              fallbacks: list) -> int:
    """A positive whole number, or the default with the fallback recorded."""
    raw = (env.get(name) or "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        value = 0
    if value > 0:
        return value
    fallbacks.append(f"{name}={raw!r} is not a positive whole number; using {default}")
    return default


def _level(env: Mapping[str, str], name: str, default: str,
           fallbacks: list) -> str:
    """One log level, or the safe default with the fallback recorded."""
    raw = (env.get(name) or "").strip()
    if not raw:
        return default
    chosen = raw.upper()
    if chosen not in LOG_LEVELS:
        fallbacks.append(f"{name}={raw!r} is not a log level; using {default}")
        return default
    return chosen


@dataclass(frozen=True)
class LoggingSettings:
    """How this process logs, as a value.

    Every default is here, once. What bounds the log directory is two small
    limits and no logging platform: a size cap with a few rotations bounds one
    run, and a count of retained runs bounds the directory. Worst case is
    ``max_bytes * (backups + 1) * runs_kept``, which at the defaults is a
    little under half a gigabyte.
    """

    #: The console handler's level. Console output is not a persisted copy of
    #: the corpus, so DEBUG here is a cheaper decision than DEBUG on the file.
    console_level: str = "INFO"
    #: The file handler's level, and INFO because DEBUG is not a safe thing
    #: for a deployment to do by accident: the static dumps further down write
    #: full prompts, full retrieved chunks and the whole answer context, and
    #: every one of those is a copy of the corpus. A developer chasing a
    #: retrieval problem sets it explicitly, which is the point at which
    #: somebody has decided to keep those dumps on disk.
    file_level: str = "INFO"
    max_bytes: int = 10 * 1024 * 1024
    backups: int = 3
    runs_kept: int = 10
    #: Anything this module fell back on rather than honouring, so a typo is
    #: visible at start-up instead of silent.
    fallbacks: Tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "console_level": self.console_level,
            "file_level": self.file_level,
            "max_bytes": self.max_bytes,
            "backups": self.backups,
            "runs_kept": self.runs_kept,
            "fallbacks": list(self.fallbacks),
        }


def logging_from_env(env: Optional[Mapping[str, str]] = None) -> LoggingSettings:
    """Read the logging settings from an environment. The only reader."""
    env = os.environ if env is None else env
    fallbacks: list = []
    defaults = LoggingSettings()
    return LoggingSettings(
        console_level=_level(env, "LOG_LEVEL", defaults.console_level, fallbacks),
        file_level=_level(env, "LOG_FILE_LEVEL", defaults.file_level, fallbacks),
        max_bytes=_positive(env, "LOG_MAX_BYTES", defaults.max_bytes, fallbacks),
        backups=_positive(env, "LOG_BACKUPS", defaults.backups, fallbacks),
        runs_kept=_positive(env, "LOG_RUNS_KEPT", defaults.runs_kept, fallbacks),
        fallbacks=tuple(fallbacks),
    )


#: What :func:`configure_logging` installed, or ``None`` while this process has
#: only the NullHandler. Read by :func:`logging_configuration` so the
#: diagnostics report the handlers that exist rather than the ones the
#: environment would produce.
_active: Optional[LoggingSettings] = None


def logging_configuration() -> dict:
    """How logging is actually set up, for the start-up banner and /api/ops.

    Before :func:`configure_logging` has run there are no handlers, and the
    honest answer is what the environment would produce -- which is what the
    banner is about to print anyway.
    """
    settings = _active if _active is not None else logging_from_env()
    return {**settings.to_dict(), "directory": paths.logs()}


def _prune_old_runs(log_dir: Path, keep: int) -> None:
    """Leave the newest ``keep`` runs' logs; delete the rest.

    Rotation bounds one run's file. This bounds how many runs accumulate,
    which is the half that a restart loop would otherwise win. A file that
    cannot be deleted -- another process on Windows still holding it -- is
    left alone: pruning logs must never be able to stop the service starting.
    """
    runs: dict[str, list[Path]] = {}
    for path in log_dir.glob("rag_*.log*"):
        runs.setdefault(path.name.split(".log")[0], []).append(path)
    for name in sorted(runs, reverse=True)[keep:]:
        for path in runs[name]:
            try:
                path.unlink()
            except OSError:
                pass


def configure_logging(settings: Optional[LoggingSettings] = None, *,
                      directory: Optional[str] = None,
                      force: bool = False) -> LoggingSettings:
    """Install this process's log handlers. Called by an entry point, once.

    Idempotent: a second call is a no-op unless ``force``, so two entry points
    in one process (the CLI importing a tool, a test importing both) do not
    end up with two file handlers writing two files.

    Returns the settings that are now in force, so a caller can report them.
    """
    global _active
    if _active is not None and not force:
        return _active

    settings = settings if settings is not None else logging_from_env()
    log_dir = Path(directory if directory is not None else paths.logs())
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"rag_{timestamp}.log"
    _prune_old_runs(log_dir, settings.runs_kept)

    logger = logging.getLogger(ROOT_LOGGER)
    logger.setLevel(logging.DEBUG)
    # Everything but the NullHandler, which stays: it is what keeps this safe
    # to call and then undo.
    for handler in list(logger.handlers):
        if not isinstance(handler, logging.NullHandler):
            handler.close()
            logger.removeHandler(handler)

    # File handler (detailed logs), rotated so one long run cannot fill the
    # disk.
    file_handler = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=settings.max_bytes, backupCount=settings.backups,
        encoding="utf-8",
    )
    file_handler.setLevel(LOG_LEVELS[settings.file_level])
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))

    console_handler = logging.StreamHandler()
    console_handler.setLevel(LOG_LEVELS[settings.console_level])
    console_handler.setFormatter(logging.Formatter("%(levelname)s - %(message)s"))

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    _active = settings

    logger.info(
        f"Logging initialized. Log file: {log_file} "
        f"(level {settings.file_level}, console {settings.console_level}, "
        f"rotate at {settings.max_bytes} bytes, "
        f"{settings.backups} backups, {settings.runs_kept} runs kept)"
    )
    for fallback in settings.fallbacks:
        logger.warning(fallback)
    if file_handler.level <= logging.DEBUG:
        # Said out loud, because it is the one setting whose consequence is
        # invisible until someone reads the file.
        logger.warning(
            "LOG_FILE_LEVEL=DEBUG: prompts, retrieved chunks and answer "
            "context will be written to the log file"
        )
    return settings


def reset_logging() -> None:
    """Take this process's handlers back off, leaving the NullHandler.

    For a test that configured logging into its own directory, and for
    :func:`configure_logging` to be callable again afterwards.
    """
    global _active
    logger = logging.getLogger(ROOT_LOGGER)
    for handler in list(logger.handlers):
        if not isinstance(handler, logging.NullHandler):
            handler.close()
            logger.removeHandler(handler)
    logger.setLevel(logging.NOTSET)
    _active = None


def get_logger(name: Optional[str] = None):
    """One named logger. Adds no handler and reads no configuration."""
    if name:
        return logging.getLogger(f"{ROOT_LOGGER}.{name}")
    return logging.getLogger(ROOT_LOGGER)


class RAGLogger:
    """The content dumps, which are DEBUG-only and belong to one place.

    A class rather than four functions because that is how the pipeline and
    the transports already call them, and because grouping them is what makes
    "these all write document text" a single fact rather than four.
    """

    @staticmethod
    def get_logger(name: str = None):
        """Kept so an existing caller holding an instance still works."""
        return get_logger(name)

    @staticmethod
    def log_llm_request(logger, messages, temperature, max_tokens):
        """Log LLM request details with full, human-readable messages"""
        logger.debug("=" * 80)
        logger.debug("LLM REQUEST")
        logger.debug(f"Temperature: {temperature}, Max Tokens: {max_tokens}")
        logger.debug("Messages:")
        for i, msg in enumerate(messages, 1):
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            logger.debug(f"[{i}] {role.upper()}\n{content}")
        logger.debug("=" * 80)

    @staticmethod
    def log_llm_response(logger, response, success=True):
        """Log full LLM response text (human-readable)"""
        logger.debug("-" * 80)
        logger.debug("LLM RESPONSE")
        logger.debug(f"Success: {success}")
        if success:
            text = response if response is not None else ""
            logger.debug(f"Response length: {len(text)} chars")
            logger.debug(text)
        else:
            logger.error(f"Error: {response}")
        logger.debug("-" * 80)

    @staticmethod
    def log_retrieval_results(
        logger,
        query,
        results,
        vectordb_name: str = None,
        embedding_model_name: str = None,
    ):
        """Every retrieved chunk, in full. DEBUG only: this is content."""
        logger.debug("=" * 80)
        logger.debug("RETRIEVAL RESULTS")
        if vectordb_name:
            logger.debug(f"Vector DB: {vectordb_name}")
        if embedding_model_name:
            logger.debug(f"Embedding Model: {embedding_model_name}")
        logger.debug("-" * 80)
        logger.debug(f"Query: {query}")
        logger.debug(f"Number of results: {len(results)}")
        for i, result in enumerate(results, 1):
            chunk = result.chunk
            logger.debug("")
            logger.debug(f"[Result {i}]")
            logger.debug(f"  Document: {chunk.doc_title}")
            if chunk.section_title:
                logger.debug(f"  Section: {chunk.section_title}")
            logger.debug(f"  Chunk #: {getattr(chunk, 'chunk_index', 'NA')}")
            logger.debug(f"  Score: {result.score:.4f}")
            logger.debug(f"  Method: {result.retrieval_method}")
            logger.debug("  Content:")
            logger.debug(chunk.content)
        logger.debug("=" * 80)

    @staticmethod
    def log_chunks_passed_to_llm(logger, chunks, context_text):
        """Log full context passed to LLM (human-readable)"""
        logger.debug("=" * 80)
        logger.debug("CONTEXT PASSED TO LLM")
        logger.debug(f"Number of chunks: {len(chunks)}")
        logger.debug(f"Total context length: {len(context_text)} chars")
        logger.debug("Context:")
        logger.debug(context_text)
        logger.debug("=" * 80)
