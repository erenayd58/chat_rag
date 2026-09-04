"""Doubles the bounded-ingest tests share: providers that block, count and
fail on cue, and a corpus large enough to make the Deep proposer plan calls.

Nothing here sleeps. A provider that has to "take time" waits on an event
the test controls, so a test proves an ordering by holding and releasing
rather than by hoping a delay was long enough.
"""

from __future__ import annotations

import json
import re
import threading
from typing import Any, Callable


def forbidding_answer(prompt: str) -> str:
    """The answer that makes the pipeline use both stages.

    Votes against every marked boundary in a proposer prompt, and calls
    every verifier comparison a tie: the proposal then differs from the
    deterministic baseline, so the verifier has change groups to judge and
    is actually called -- twice per group, once in each order.
    """
    if "DIVISION ONE" in prompt:
        return '{"better": "EQUAL"}'
    labels = sorted(set(re.findall(r"\[(B\d+)\]", prompt)))
    rows = [
        {"id": label, "strength": 0, "before": "introduces_next", "after": "continues_previous"}
        for label in labels
    ]
    return json.dumps({"boundaries": rows})


class GatedProvider:
    """A provider whose calls block until the test opens the gate.

    ``full`` is set the moment ``expect`` calls are in flight at once, so a
    test can wait for "the limit is reached" and only then look at the
    high-water mark and release everything. ``peak`` is the most calls that
    were ever inside ``complete`` at the same time -- across every thread
    and every document that shares this instance.
    """

    model_id = "test:gated@1"

    def __init__(self, expect: int = 1, answer: str | Callable[[str], str] = "{}"):
        self.expect = expect
        self.answer = answer
        self.gate = threading.Event()
        self.full = threading.Event()
        self.lock = threading.Lock()
        self.inflight = 0
        self.peak = 0
        self.calls = 0

    def complete(self, prompt: str) -> str:
        with self.lock:
            self.inflight += 1
            self.calls += 1
            self.peak = max(self.peak, self.inflight)
            if self.inflight >= self.expect:
                self.full.set()
        try:
            self.gate.wait(20)
            return self.answer(prompt) if callable(self.answer) else self.answer
        finally:
            with self.lock:
                self.inflight -= 1

    def release(self) -> None:
        self.gate.set()


class FailingProvider:
    model_id = "test:failing@1"

    def __init__(self):
        self.calls = 0
        self.lock = threading.Lock()

    def complete(self, prompt: str) -> str:
        with self.lock:
            self.calls += 1
        raise ConnectionError("endpoint unreachable")


def _words(count: int, prefix: str) -> str:
    """A paragraph of ``count`` distinct words, punctuated like prose: a
    boundary after a sentence that never ended reads as a smell to the
    quality layer, and a smelly cut is never offered to the proposer."""
    body = " ".join(f"{prefix}{index}" for index in range(1, count + 1))
    return "Bu paragraf " + body + " ile biter."


def deep_corpus(sections: int = 6, paragraphs: int = 8, words: int = 30,
                document_id: str = "load-doc") -> list[dict[str, Any]]:
    """Structured units in the parser's own shape, big enough that every
    section exceeds the chunker's target budget and so gets a proposer call.

    The numbers are the product's (``structural_chunker``: target 700, hard
    cap 1126 tokens): eight paragraphs of about 190 tokens each is roughly
    1,500 tokens, which must be cut, every paragraph fits the cap on its own,
    and the cut points are clean -- so the proposer is asked once per section.
    """
    units: list[dict[str, Any]] = []
    order = 0
    for section in range(1, sections + 1):
        order += 1
        title = f"{section}. BOLUM"
        units.append({
            "document_id": document_id, "unit_id": f"h-{order:04d}", "order": order,
            "text": title, "type": "heading", "heading_level": 1,
            "section_path": [title], "source": {"page": section, "block": order},
        })
        for para in range(paragraphs):
            order += 1
            units.append({
                "document_id": document_id, "unit_id": f"p-{order:04d}", "order": order,
                "text": _words(words, f"s{section}p{para}w"), "type": "paragraph",
                "section_path": [title], "source": {"page": section, "block": order},
            })
    return units


def deep_text(units: list[dict[str, Any]]) -> str:
    return "\n\n".join(unit["text"] for unit in units)
