"""The structural guard over the Deep Analysis boundary judge.

The judge may only choose among boundaries the structural walk already
admits; the guard removes the structurally poor ones from that set. What is
pinned here: the two refusal rules fire, an ordinary paragraph boundary is
still offered to the model, Standard output is untouched, the walk's own
budget still holds, and the one-call-per-window contract survives.
"""

from __future__ import annotations

import json
import re

import pytest

from amsc.llm_boundary_judge import (
    JudgeConfig,
    build_window_prompt,
    chunk_units_with_judge,
)
from amsc.structural_chunker import chunk_units
from amsc.tokenization import TiktokenTokenCounter
from components.chunker.boundary_guard import (
    RULE_HEADING_ORPHAN,
    RULE_LIST_FRAGMENTS,
    RULE_LIST_ITEM_RUN,
    StructurallyGuardedJudge,
    parse_window_candidates,
    unsafe_reason,
)
from components.chunker.normalization_adapter import CanonicalUnitAdapter
from components.chunker.structural_chunker import (
    HARD_MAX_TOKENS,
    MIN_TOKENS,
    SOFT_MAX_TOKENS,
    TARGET_TOKENS,
    StructuralChunker,
)

CANDIDATE_MARKER = re.compile(r"\[CANDIDATE (C\d+) \|")

KINDS = {
    "h-1": "heading", "h-2": "heading",
    "p-1": "paragraph", "p-2": "paragraph", "p-3": "paragraph",
    "l-1": "list", "l-2": "list",
    "t-1": "table",
}


# ------------------------------------------------------------------ rules
def test_a_cut_between_two_list_units_is_refused():
    assert unsafe_reason("l-1", "l-2", KINDS) == RULE_LIST_ITEM_RUN


def test_a_cut_between_fragments_of_one_list_is_refused():
    assert unsafe_reason("l-1#f1", "l-1#f2", KINDS) == RULE_LIST_FRAGMENTS


def test_a_cut_straight_after_a_heading_is_refused():
    assert unsafe_reason("h-1", "p-1", KINDS) == RULE_HEADING_ORPHAN
    assert unsafe_reason("h-1", "l-1", KINDS) == RULE_HEADING_ORPHAN


def test_an_ordinary_paragraph_boundary_stays_admissible():
    assert unsafe_reason("p-1", "p-2", KINDS) is None
    assert unsafe_reason("p-1", "h-1", KINDS) is None  # cutting before a heading
    assert unsafe_reason("p-1", "l-1", KINDS) is None  # lead-in: not detectable
    assert unsafe_reason("t-1", "p-1", KINDS) is None


def test_fragments_of_a_non_list_unit_are_not_refused():
    """Table row groups are the chunker's own device; only lists are in scope."""
    assert unsafe_reason("t-1#f1", "t-1#f2", KINDS) is None


# ----------------------------------------------------------- prompt parsing
class Piece:
    def __init__(self, unit_id, text, tokens=200, page=1):
        self.unit_id = unit_id
        self.text = text
        self.tokens = tokens
        self.page = page
        self.section_path = ("S",)
        self.strategy = "whole_unit"
        self.label = False


def window_prompt(unit_ids, admissible):
    pieces = [Piece(uid, f"{uid} metni " * 12) for uid in unit_ids]
    return build_window_prompt(
        heading="Bolum", section_path=("Bolum",), pieces=pieces,
        start=0, admissible=admissible,
    )


def test_the_prompt_is_read_back_into_candidate_boundaries():
    prompt = window_prompt(["p-1", "l-1", "l-2", "p-2"], [1, 2, 3])
    parsed = parse_window_candidates(prompt)
    assert parsed == [
        ("C1", "p-1", "l-1"),
        ("C2", "l-1", "l-2"),
        ("C3", "l-2", "p-2"),
    ]


def test_an_unrecognised_prompt_disables_the_guard():
    assert parse_window_candidates("no markers at all") is None
    assert parse_window_candidates("") is None


# ------------------------------------------------------------------ guard
class Unit:
    def __init__(self, unit_id, kind):
        self.unit_id = unit_id
        self.type = kind


UNITS = [Unit(uid, kind) for uid, kind in KINDS.items()]


class RecordingJudge:
    model_id = "inner-judge"

    def __init__(self, decision="SPLIT"):
        self.calls = 0
        self.decision = decision

    def complete(self, prompt):
        self.calls += 1
        ids = CANDIDATE_MARKER.findall(prompt)
        return json.dumps([
            {"candidate_id": c, "decision": self.decision, "reason_code": "TOPIC_SHIFT"}
            for c in ids
        ])


def decisions_of(raw):
    return {row["candidate_id"]: row["decision"] for row in json.loads(raw)}


def test_refused_candidates_come_back_as_keep():
    inner = RecordingJudge("SPLIT")
    guard = StructurallyGuardedJudge(inner, UNITS)
    # C1 p-1 -> l-1 (safe), C2 l-1 -> l-2 (list run), C3 l-2 -> p-2 (safe)
    answer = guard.complete(window_prompt(["p-1", "l-1", "l-2", "p-2"], [1, 2, 3]))
    assert decisions_of(answer) == {"C1": "SPLIT", "C2": "KEEP", "C3": "SPLIT"}
    assert inner.calls == 1
    assert guard.candidates_blocked == 1
    assert guard.blocked_by_rule == {RULE_LIST_ITEM_RUN: 1}


def test_a_heading_boundary_is_refused_even_when_the_model_splits():
    inner = RecordingJudge("SPLIT")
    guard = StructurallyGuardedJudge(inner, UNITS)
    answer = guard.complete(window_prompt(["h-1", "p-1", "p-2"], [1, 2]))
    assert decisions_of(answer) == {"C1": "KEEP", "C2": "SPLIT"}
    assert guard.blocked_by_rule == {RULE_HEADING_ORPHAN: 1}


def test_a_window_of_only_unsafe_candidates_costs_no_provider_call():
    inner = RecordingJudge("SPLIT")
    guard = StructurallyGuardedJudge(inner, UNITS)
    answer = guard.complete(window_prompt(["h-1", "l-1", "l-2"], [1, 2]))
    assert decisions_of(answer) == {"C1": "KEEP", "C2": "KEEP"}
    assert inner.calls == 0
    assert guard.windows_short_circuited == 1
    assert guard.network_calls == 0


def test_a_window_of_only_safe_candidates_is_passed_through_untouched():
    inner = RecordingJudge("SPLIT")
    guard = StructurallyGuardedJudge(inner, UNITS)
    answer = guard.complete(window_prompt(["p-1", "p-2", "p-3"], [1, 2]))
    assert decisions_of(answer) == {"C1": "SPLIT", "C2": "SPLIT"}
    assert guard.candidates_blocked == 0
    assert inner.calls == 1


def test_at_most_one_provider_call_per_window():
    inner = RecordingJudge("SPLIT")
    guard = StructurallyGuardedJudge(inner, UNITS)
    for unit_ids, admissible in (
        (["p-1", "p-2", "p-3"], [1, 2]),
        (["h-1", "l-1", "l-2"], [1, 2]),
        (["p-1", "l-1", "l-2", "p-2"], [1, 2, 3]),
    ):
        guard.complete(window_prompt(unit_ids, admissible))
    assert guard.windows_seen == 3
    assert inner.calls <= guard.windows_seen
    assert guard.network_calls == inner.calls


class BrokenJudge:
    model_id = "broken"

    def complete(self, prompt):
        return "definitely not json"


def test_an_unparseable_answer_is_passed_through_for_the_walks_fallback():
    guard = StructurallyGuardedJudge(BrokenJudge(), UNITS)
    answer = guard.complete(window_prompt(["p-1", "l-1", "l-2"], [1, 2]))
    assert answer == "definitely not json"


def test_the_guard_report_carries_no_prompt_text():
    inner = RecordingJudge("SPLIT")
    guard = StructurallyGuardedJudge(inner, UNITS)
    guard.complete(window_prompt(["p-1", "l-1", "l-2", "p-2"], [1, 2, 3]))
    blob = json.dumps({**guard.report(), "blocked": guard.blocked_candidates})
    assert "metni" not in blob  # no document text
    assert "CANDIDATE" not in blob  # no prompt fragments


# ------------------------------------------------- end to end through ingest
def unit_row(unit_id, order, text, kind="paragraph", section=("BOLUM",)):
    row = {
        "unit_id": unit_id,
        "order": order,
        "text": text,
        "type": kind,
        "heading_level": 2 if kind == "heading" else None,
        "section_path": list(section),
        "source": {"page": 1},
    }
    return row


def list_heavy_document():
    """A section long enough to force budget cuts, with a run of list units
    and a heading in the middle of the cutting region."""
    words = "veri kalite gosterge donem sonuc analiz kapsam yontem bulgu deger "
    rows = [unit_row("h-00001", 1, "**1. GENEL DEGERLENDIRME**", "heading")]
    order = 2
    for index in range(4):
        rows.append(unit_row(f"p-{order:05d}", order, words * 12))
        order += 1
    for index in range(4):
        rows.append(unit_row(f"l-{order:05d}", order, words * 10, "list"))
        order += 1
    for index in range(4):
        rows.append(unit_row(f"p-{order:05d}", order, words * 12))
        order += 1
    return rows


class AlwaysSplitJudge:
    model_id = "always-split"

    def __init__(self):
        self.prompts = []

    def complete(self, prompt):
        self.prompts.append(prompt)
        ids = CANDIDATE_MARKER.findall(prompt)
        return json.dumps([
            {"candidate_id": c, "decision": "SPLIT", "reason_code": "TOPIC_SHIFT"}
            for c in ids
        ])


def deep_and_standard():
    rows = list_heavy_document()
    chunker = StructuralChunker()
    standard = chunker.chunk_text("", "doc", "rapor.pdf", parsed_units=rows)
    judge = AlwaysSplitJudge()
    deep, report = chunker.chunk_text_deep(
        "", "doc", "rapor.pdf", judge=judge, parsed_units=rows
    )
    return standard, deep, report, judge, rows


def test_standard_output_is_untouched_by_the_guard():
    """The Standard path never constructs a judge, so it cannot be affected."""
    rows = list_heavy_document()
    chunker = StructuralChunker()
    once = chunker.chunk_text("", "doc", "rapor.pdf", parsed_units=rows)
    twice = chunker.chunk_text("", "doc", "rapor.pdf", parsed_units=rows)
    units = CanonicalUnitAdapter().normalize(
        text="", document_id="doc", parsed_units=rows
    )
    reference = chunk_units(
        units, counter=TiktokenTokenCounter("cl100k_base"),
        min_tokens=MIN_TOKENS, target_tokens=TARGET_TOKENS,
        soft_max_tokens=SOFT_MAX_TOKENS, hard_max_tokens=HARD_MAX_TOKENS,
    )
    assert [c.content for c in once] == [c.content for c in twice]
    assert [c.content for c in once] == [r["text"] for r in reference]


def boundaries_of(chunks):
    """``(last unit of chunk N, first unit of chunk N+1)`` for every cut."""
    ids = [json.loads(c.metadata["unit_ids_json"]) for c in chunks]
    return [
        (before[-1], after[0])
        for before, after in zip(ids, ids[1:])
        if before and after
    ]


def test_a_refused_candidate_is_never_reported_as_split():
    standard, deep, report, judge, rows = deep_and_standard()
    kinds = {row["unit_id"]: row["type"] for row in rows}

    guard = report["structural_guard"]
    assert guard["candidates_blocked"] >= 1, "the fixture must exercise a rule"
    assert set(guard["blocked_by_rule"]) <= {
        RULE_LIST_ITEM_RUN, RULE_LIST_FRAGMENTS, RULE_HEADING_ORPHAN
    }

    for entry in report["decisions"]:
        for verdict in entry["decisions"]:
            reason = unsafe_reason(
                verdict["cut_after_unit_id"], verdict["cut_before_unit_id"], kinds
            )
            if reason:
                assert verdict["decision"] == "KEEP", (
                    f"a refused boundary was offered as SPLIT: "
                    f"{verdict['cut_after_unit_id']} -> "
                    f"{verdict['cut_before_unit_id']} ({reason})"
                )


def test_every_cut_the_judge_moved_lands_on_a_safe_boundary():
    """The guarantee the guard actually makes: the model can never *select*
    a refused boundary. Where the walk keeps its own structural cut, that
    cut is Standard's and is deliberately left alone."""
    standard, deep, report, judge, rows = deep_and_standard()
    kinds = {row["unit_id"]: row["type"] for row in rows}
    for entry in report["decisions"]:
        if entry["chosen_equals_greedy"]:
            continue
        chosen = next(
            verdict for verdict in entry["decisions"]
            if verdict["cut_after_unit_id"] == entry["chosen_after_unit_id"]
        )
        assert unsafe_reason(
            chosen["cut_after_unit_id"], chosen["cut_before_unit_id"], kinds
        ) is None


def test_any_surviving_unsafe_cut_is_one_standard_makes_too():
    """The guard narrows what the judge may choose; it never moves the
    structural cut. So a refused boundary can only survive where Standard
    cuts as well -- Deep Analysis is never worse than Standard here."""
    standard, deep, report, judge, rows = deep_and_standard()
    kinds = {row["unit_id"]: row["type"] for row in rows}
    standard_boundaries = set(boundaries_of(standard))
    for previous, following in boundaries_of(deep):
        if unsafe_reason(previous, following, kinds):
            assert (previous, following) in standard_boundaries, (
                f"deep analysis invented an unsafe cut Standard does not make: "
                f"{previous} -> {following}"
            )


def test_the_refused_boundaries_were_genuinely_on_offer():
    """Without the guard the same judge would have taken a refused cut, so
    the rules are doing work rather than describing what already held."""
    rows = list_heavy_document()
    units = CanonicalUnitAdapter().normalize(
        text="", document_id="doc", parsed_units=rows
    )
    kinds = {row["unit_id"]: row["type"] for row in rows}
    unguarded = chunk_units_with_judge(
        units,
        counter=TiktokenTokenCounter("cl100k_base"),
        judge=AlwaysSplitJudge(),
        config=JudgeConfig(
            min_tokens=MIN_TOKENS, target_tokens=TARGET_TOKENS,
            soft_max_tokens=SOFT_MAX_TOKENS, hard_max_tokens=HARD_MAX_TOKENS,
        ),
    )
    offered = [
        (verdict.cut_after_unit_id, verdict.cut_before_unit_id)
        for entry in unguarded.audit
        for verdict in entry.decisions
    ]
    assert any(
        unsafe_reason(before, after, kinds) is not None for before, after in offered
    ), "the fixture must offer at least one unsafe candidate to the model"


def test_the_hard_token_budget_survives_the_guard():
    _, deep, _, _, _ = deep_and_standard()
    for chunk in deep:
        assert chunk.metadata["token_count"] <= HARD_MAX_TOKENS


def test_the_batching_contract_survives_the_guard():
    _, _, report, judge, _ = deep_and_standard()
    guard = report["structural_guard"]
    # One window is still at most one provider call, never more.
    assert len(judge.prompts) == guard["provider_network_calls"]
    assert guard["provider_network_calls"] <= report["consulted_boundary_count"]
    assert guard["windows_seen"] == report["consulted_boundary_count"]


def test_per_candidate_decisions_are_persisted_in_the_report():
    _, _, report, _, _ = deep_and_standard()
    assert report["decisions"], "the audit must reach the ingest report"
    row = report["decisions"][0]
    assert {"step", "candidate_count", "decisions", "chosen_after_unit_id"} <= set(row)
    for verdict in row["decisions"]:
        assert verdict["decision"] in {"SPLIT", "KEEP"}
        assert verdict["reason_code"]
    blob = json.dumps(report)
    assert "CANDIDATE" not in blob  # no prompt was persisted
    assert "Authorization" not in blob and "api_key" not in blob.lower()
