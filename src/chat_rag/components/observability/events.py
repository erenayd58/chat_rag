"""Operational events, in one shape, with one rule about what may go in them.

An operator following an incident reads a log. What they need is every
important thing that happened to a job, on one line each, with the same
field names every time, and the job id on all of them so a grep finds the
whole story. That is all this is: ``emit("ingest.job.failed", job_id=...,
kb_id=..., error_category=...)`` renders

    event=ingest.job.failed job_id=6f2a1c kb_id=kb-7 error_category=provider

The rule is what makes it safe to turn on. **Only counts, ids, categories,
durations and states may be emitted.** Never a key, never document text,
never a chunk, never a prompt. This is not a convention -- values are run
through :func:`safe_value`, which truncates long strings and refuses ones
that look like credentials, and a test asserts that a document's contents
cannot reach the log through this path. The cost of being careless here is
that a log file, which is copied into tickets and pasted into chats,
quietly becomes a copy of the corpus.

Field names are the ones the metrics use, so the log and
``/api/ops/metrics`` describe the same system in the same words.
"""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger("RAG.ops")

#: Longest a single field value may be. Ids, categories and states are short;
#: anything long is either a message worth truncating or content that should
#: not be here at all.
MAX_VALUE = 200

#: Field names that must never carry a value, whatever is passed. Belt and
#: braces for the rule above: a caller who adds ``text=chunk.content`` gets
#: a redaction marker rather than the chunk.
FORBIDDEN_FIELDS = frozenset({
    "text", "content", "chunk", "chunks", "prompt", "prompts", "answer",
    "document_text", "body", "payload", "key", "api_key", "token", "secret",
    "authorization", "password",
})

#: Values that look like credentials regardless of the field they arrived in.
#:
#: The last one is the interesting case: a long unbroken run of letters and
#: digits is what an API key looks like. It requires *both* a letter and a
#: digit so that ordinary long words -- a stack frame, a repeated character in
#: a truncated message -- are truncated as messages rather than blanked as
#: secrets. Over-redacting a message costs an operator information; under-
#: redacting a key costs a credential, so where they meet this errs high.
_SECRET_SHAPES = (
    re.compile(r"\bsk-[A-Za-z0-9_\-]{8,}"),
    re.compile(r"\bBearer\s+\S+", re.IGNORECASE),
    re.compile(r"\b(?=[A-Za-z0-9_\-]*[A-Za-z])(?=[A-Za-z0-9_\-]*\d)[A-Za-z0-9_\-]{32,}\b"),
)

REDACTED = "<redacted>"

#: Absolute filesystem paths, Windows and POSIX, including UNC shares. An
#: exception message is written for a developer standing in a source tree, not
#: for an endpoint: an [Errno 2] naming an upload under a home directory gives
#: away the operator's account, the deployment's layout and the document's own
#: title in one string. The path is replaced and the rest of the sentence --
#: the part that says what went wrong -- is kept.
#:
#: A quoted path is matched to its closing quote rather than to the first
#: space, because real filenames contain spaces and a pattern that stopped
#: early would replace the directory and leave the document's name behind.
_QUOTED_PATH = re.compile(
    r"""(['"])(?:[A-Za-z]:[\\/]|\\\\|/(?:[A-Za-z0-9._-]+/)+)[^'"]*\1"""
)
_PATH_SHAPES = (
    re.compile(r"""(?:[A-Za-z]:[\\/]|\\\\)[^\s'"]*"""),
    re.compile(r"(?<![\w.])/(?:[A-Za-z0-9._-]+/){2,}[A-Za-z0-9._-]*"),
)

PATH_PLACEHOLDER = "<path>"


def redact_message(message: Any, *, limit: int = MAX_VALUE) -> str:
    """An arbitrary exception string, made safe to show and to store.

    Everything that reaches ``/api/ops/metrics`` as an error example, and
    every ``reason`` on ``/api/health``, comes through here. Those strings are
    whatever a library chose to write, so the endpoint cannot promise it
    carries only aggregates unless something makes that true: credential
    shapes are blanked, absolute paths are replaced by a placeholder, and the
    result is truncated. What survives is the sentence an operator needs --
    "embedding endpoint returned HTTP 429" -- without the deployment's
    filesystem layout or a document's name attached to it.
    """
    text = str(message or "").strip()
    if not text:
        return ""
    for shape in _SECRET_SHAPES:
        text = shape.sub(REDACTED, text)
    text = _QUOTED_PATH.sub(lambda m: m.group(1) + PATH_PLACEHOLDER + m.group(1), text)
    for shape in _PATH_SHAPES:
        text = shape.sub(PATH_PLACEHOLDER, text)
    text = " ".join(text.split())
    if len(text) > limit:
        text = text[: limit - 3] + "..."
    return text


def safe_value(name: str, value: Any) -> str:
    """One field, rendered so it cannot leak. Never raises."""
    if name.lower() in FORBIDDEN_FIELDS:
        return REDACTED
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.4f}".rstrip("0").rstrip(".") or "0"
    if isinstance(value, (int,)):
        return str(value)
    text = str(value)
    for shape in _SECRET_SHAPES:
        if shape.search(text):
            return REDACTED
    if len(text) > MAX_VALUE:
        text = text[:MAX_VALUE - 3] + "..."
    # Newlines and spaces would break one-line-per-event parsing; a value
    # with them is a message, and a message is quoted.
    if any(character in text for character in " \t\n\r="):
        text = '"' + text.replace("\n", " ").replace("\r", " ").replace('"', "'") + '"'
    return text


def render(event: str, **fields: Any) -> str:
    parts = [f"event={event}"]
    for name, value in fields.items():
        parts.append(f"{name}={safe_value(name, value)}")
    return " ".join(parts)


def emit(event: str, level: int = logging.INFO, **fields: Any) -> str:
    """Log one operational event. Returns the rendered line, for tests."""
    line = render(event, **fields)
    logger.log(level, line)
    return line


def warn(event: str, **fields: Any) -> str:
    return emit(event, level=logging.WARNING, **fields)


def error(event: str, **fields: Any) -> str:
    return emit(event, level=logging.ERROR, **fields)
