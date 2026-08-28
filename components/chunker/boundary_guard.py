"""Structural safety filter over the Deep Analysis boundary judge.

The judge may only choose among boundaries the structural walk already
considers admissible. Some of those are structurally poor places to cut even
though they satisfy the size budget: between two items of one list, or
immediately after a heading, which orphans the heading from the content it
introduces. This guard narrows what the judge can *choose*, using only
metadata the canonical stream already carries -- unit type and unit identity.
No text is inspected and no new heuristic is invented.

How the narrowing works. The guard wraps the judge model rather than the
algorithm: for each decision window it reads the candidate markers out of the
prompt amsc built, decides which candidates are structurally unsafe, and
forces those to ``KEEP`` in the answer. Since the walk chooses the latest
SPLIT and falls back to the structural cut when everything is KEEP, a
candidate forced to KEEP can never be selected. When *every* candidate in a
window is unsafe the answer is predetermined, so no provider round-trip is
made at all.

What this deliberately does not do:

* it never widens eligibility, so Deep Analysis can only become more
  conservative, never less;
* it never moves the structural (greedy) cut, so Standard output is
  untouched and the hard token budget still belongs to the walk;
* it leaves the one-call-per-window batching contract intact -- at most one
  provider call per window, never more;
* it does not guess. A prompt it cannot parse, or a model answer it cannot
  parse, is passed through untouched so the walk's own deterministic
  fallback decides.

Not implemented on purpose: separating a lead-in sentence from the list it
introduces. Detecting that needs a "this paragraph introduces the next
block" claim, and the canonical stream carries none -- ``semantic_role`` is
defined for headings only. Recognising it from the sentence's wording would
be exactly the guessy text heuristic this module refuses to add.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any, Iterable, Sequence

# The markers amsc's window prompt writes. Kept strict: an unknown shape
# means the guard steps aside rather than filtering on a guess.
_CANDIDATE_LINE = re.compile(r"^\[CANDIDATE (C\d+) \| cut before (\w+) (\S+)\]$")
_PIECE_LINE = re.compile(r"^\[(\w+) (\S+)\]$")
_KNOWN_KINDS = {"heading", "paragraph", "list", "table", "visual", "unknown"}

#: Why a candidate boundary was refused. Audit labels only -- each maps to
#: one rule below and nothing else steers on them.
RULE_LIST_ITEM_RUN = "list_item_run"
RULE_LIST_FRAGMENTS = "list_item_fragments"
RULE_HEADING_ORPHAN = "heading_orphan"


def base_unit_id(unit_id: str) -> str:
    """Drop the ``#f2`` fragment suffix the chunker appends when it splits
    an oversized unit."""
    return str(unit_id).split("#", 1)[0]


def unsafe_reason(
    previous_unit_id: str,
    next_unit_id: str,
    kind_of: dict[str, str],
) -> str | None:
    """Why cutting between these two pieces is structurally unsafe, or None.

    ``kind_of`` maps a canonical unit id to its type. Only unit type and unit
    identity are consulted; the rules are total and deterministic.
    """
    previous_base = base_unit_id(previous_unit_id)
    next_base = base_unit_id(next_unit_id)
    previous_kind = kind_of.get(previous_base)
    next_kind = kind_of.get(next_base)

    # Two fragments of one list: a cut here lands between items of the same
    # list, which is the case the manual review flagged.
    if previous_base == next_base and previous_kind == "list":
        return RULE_LIST_FRAGMENTS

    # Consecutive list units are a run of items of one list in this corpus.
    if previous_kind == "list" and next_kind == "list":
        return RULE_LIST_ITEM_RUN

    # Cutting straight after a heading leaves it stranded at the end of a
    # chunk, separated from the content it introduces.
    if previous_kind == "heading":
        return RULE_HEADING_ORPHAN

    return None


def parse_window_candidates(prompt: str) -> list[tuple[str, str, str]] | None:
    """``(candidate_id, previous_unit_id, next_unit_id)`` per candidate.

    Returns None when the prompt does not have the expected shape, which is
    the guard's signal to stay out of the way.
    """
    sequence: list[str] = []
    pending: list[tuple[str, str]] = []
    found: list[tuple[str, str, str]] = []
    for line in (prompt or "").splitlines():
        line = line.strip()
        candidate = _CANDIDATE_LINE.match(line)
        if candidate:
            label, kind, unit_id = candidate.groups()
            if kind not in _KNOWN_KINDS:
                return None
            pending.append((label, unit_id))
            continue
        piece = _PIECE_LINE.match(line)
        if piece:
            kind, unit_id = piece.groups()
            if kind not in _KNOWN_KINDS:
                continue  # ordinary text that happens to look bracketed
            if pending:
                if not sequence:
                    return None  # a candidate before any content: unexpected
                label, marked = pending.pop(0)
                if marked != unit_id:
                    return None  # marker and piece disagree: do not guess
                found.append((label, sequence[-1], unit_id))
            sequence.append(unit_id)
    if pending or not found:
        return None
    return found


class StructurallyGuardedJudge:
    """A boundary judge that cannot choose a structurally unsafe cut.

    Wraps another judge model, keeps the same ``complete(prompt) -> str``
    interface, and records what it refused so the ingest report can show it.
    """

    def __init__(self, inner: Any, units: Iterable[Any]) -> None:
        self._inner = inner
        self._kind_of = {
            unit.unit_id: str(getattr(unit.type, "value", unit.type))
            for unit in units
        }
        self.windows_seen = 0
        self.windows_short_circuited = 0
        self.network_calls = 0
        self.candidates_seen = 0
        self.candidates_blocked = 0
        self.unparsed_prompts = 0
        self.blocked_by_rule: Counter[str] = Counter()
        self.blocked_candidates: list[dict[str, str]] = []

    @property
    def model_id(self) -> str | None:
        return getattr(self._inner, "model_id", None)

    def complete(self, prompt: str) -> str:
        self.windows_seen += 1
        candidates = parse_window_candidates(prompt)
        if candidates is None:
            # An unrecognised prompt is not a licence to filter: ask as-is.
            self.unparsed_prompts += 1
            self.network_calls += 1
            return self._inner.complete(prompt)

        self.candidates_seen += len(candidates)
        blocked: dict[str, str] = {}
        for label, previous_unit_id, next_unit_id in candidates:
            reason = unsafe_reason(previous_unit_id, next_unit_id, self._kind_of)
            if reason:
                blocked[label] = reason
                self.blocked_by_rule[reason] += 1
                self.blocked_candidates.append({
                    "candidate_id": label,
                    "cut_after_unit_id": previous_unit_id,
                    "cut_before_unit_id": next_unit_id,
                    "rule": reason,
                })
        self.candidates_blocked += len(blocked)

        if len(blocked) == len(candidates):
            # Every candidate is refused, so the answer cannot change the
            # outcome: the walk will fall back to its structural cut. Spend
            # no provider call on a foregone conclusion.
            self.windows_short_circuited += 1
            return json.dumps([
                {"candidate_id": label, "decision": "KEEP", "reason_code": "OTHER"}
                for label, _, _ in candidates
            ])

        self.network_calls += 1
        raw = self._inner.complete(prompt)
        if not blocked:
            return raw
        return self._force_keep(raw, blocked)

    @staticmethod
    def _force_keep(raw: str, blocked: dict[str, str]) -> str:
        """Rewrite the model's answer so refused candidates read KEEP.

        An answer this cannot parse is returned untouched, so the walk sees
        the model's own response and applies its own parse-error fallback.
        The reason codes stay as the model wrote them; which candidates were
        overridden is recorded separately rather than disguised here.
        """
        try:
            rows = json.loads(raw)
        except (TypeError, ValueError):
            return raw
        if isinstance(rows, dict):
            rows = rows.get("decisions")
        if not isinstance(rows, list):
            return raw
        out: list[Any] = []
        for row in rows:
            if not isinstance(row, dict):
                return raw
            row = dict(row)
            if str(row.get("candidate_id", "")).strip() in blocked:
                row["decision"] = "KEEP"
            out.append(row)
        return json.dumps(out)

    def report(self) -> dict[str, Any]:
        """What the guard did, for the ingest record. No prompts, no keys."""
        return {
            "windows_seen": self.windows_seen,
            "windows_short_circuited": self.windows_short_circuited,
            "provider_network_calls": self.network_calls,
            "candidates_seen": self.candidates_seen,
            "candidates_blocked": self.candidates_blocked,
            "blocked_by_rule": dict(self.blocked_by_rule),
            "unparsed_prompts": self.unparsed_prompts,
        }
