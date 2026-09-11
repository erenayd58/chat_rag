"""A progress line is never a reason for an ingest to fail.

Found by the clean-install smoke on Windows: the pipeline narrates an ingest
to stdout, and its last line begins with a check mark. A program that
imported the library from a console on a legacy code page -- or with stdout
redirected to a file, which falls back to one -- got ``UnicodeEncodeError``
out of ``print``, inside the ingest's own ``try``, and the document was
reported as failed. The product never saw it because its entrypoints
reconfigure the streams first (``runtime/bootstrap.py``) and the container
sets ``PYTHONIOENCODING``; a library cannot assume either.

The narrow console is a ``print`` replaced on the module rather than a
``sys.stdout`` replaced on the process: pytest re-installs its own capture
around every phase, so a stream put there does not stay there.
"""

from __future__ import annotations

import io

import pytest

from chat_rag.pipeline import rag_pipeline


@pytest.fixture
def narrow_console(monkeypatch):
    """A console that spells cp1252 and nothing else, the way a redirected
    stream on a Windows machine does. Returns what reached it, in bytes."""
    raw = io.BytesIO()
    stream = io.TextIOWrapper(raw, encoding="cp1252", errors="strict",
                              write_through=True, newline="\n")
    monkeypatch.setattr(rag_pipeline, "print", lambda message: stream.write(message + "\n"),
                        raising=False)
    return raw


def test_a_line_the_console_can_spell_is_written_as_it_is(narrow_console):
    rag_pipeline._progress("  - Created 3 chunks")
    assert narrow_console.getvalue() == b"  - Created 3 chunks\n"


def test_a_line_the_console_cannot_spell_is_escaped_rather_than_raised(narrow_console):
    rag_pipeline._progress("\u2713 Document 'Y\u0131ll\u0131k rapor' ingested successfully!")
    written = narrow_console.getvalue().decode("cp1252")
    assert "Document" in written and "ingested successfully" in written
    assert r"\u2713" in written and r"\u0131" in written


def test_a_stream_that_is_gone_is_skipped(monkeypatch):
    def gone(message):
        raise ValueError("I/O operation on closed file.")

    monkeypatch.setattr(rag_pipeline, "print", gone, raising=False)
    rag_pipeline._progress("still fine")  # no exception is the assertion
