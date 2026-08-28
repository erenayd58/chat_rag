"""Deep Analysis: the LLM boundary judge at ingest, and only at ingest.

What these tests pin:

* Standard stays Standard — an all-KEEP judge (and a failing one) produces
  byte-identical chunk text to the untouched ``chunk_text`` path;
* a SPLIT vote at a real candidate boundary changes the cut, and the hard
  token budget survives every verdict;
* parse and provider errors fall back to the deterministic structural cut,
  and the report says so;
* missing provider configuration is a loud, named refusal — never a silent
  fall back;
* no key material ever reaches the report, and the query path has no idea
  the judge exists.
"""

from __future__ import annotations

import inspect
import json
import re

import pytest

from components.chunker.boundary_judge import (
    boundary_judge_config_error,
    create_boundary_judge,
)
from components.chunker.structural_chunker import (
    HARD_MAX_TOKENS,
    StructuralChunker,
)
from core.exceptions import ConfigurationException


# ---------------------------------------------------------------- fixtures
def unit(unit_id, order, text, unit_type="paragraph", section=("BOLUM",), page=1):
    return {
        "unit_id": unit_id,
        "order": order,
        "text": text,
        "type": unit_type,
        "heading_level": 2 if unit_type == "heading" else None,
        "section_path": list(section),
        "source": {"page": page},
    }


def oversized_document():
    """One section big enough to force a plain budget cut with a genuine
    choice of admissible positions — the judge's single injection point."""
    rows = [unit("h-1", 1, "**1. GENEL DEGERLENDIRME**", "heading")]
    words = (
        "veri kalite gosterge donem sonuc analiz kapsam yontem bulgu deger "
    )
    for index in range(12):
        rows.append(unit(f"p-{index + 1}", index + 2, words * 12))
    return rows


# The v2 window contract: candidate ids are read from the prompt's own
# ``[CANDIDATE Cn | cut before ...]`` markers, never hardcoded, so a drift
# in the marker format fails these tests instead of being papered over.
CANDIDATE_MARKER = re.compile(r"\[CANDIDATE (C\d+) \|")


class ScriptedJudge:
    """A v2 judge double: one JSON array per window, decisions scripted by
    ``decide(window_number, candidate_index, candidate_id)``."""

    model_id = "stub-judge-v1"

    def __init__(self, decide=None):
        self.prompts = []
        self.candidates_seen = 0
        self._decide = decide or (lambda window, index, cid: "KEEP")

    def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        candidate_ids = CANDIDATE_MARKER.findall(prompt)
        assert candidate_ids, "a window prompt must mark its candidates"
        self.candidates_seen += len(candidate_ids)
        return json.dumps([
            {
                "candidate_id": cid,
                "decision": self._decide(len(self.prompts), index, cid),
                "reason_code": "OTHER",
            }
            for index, cid in enumerate(candidate_ids)
        ])


class GarbageJudge:
    model_id = "garbage-judge"

    def complete(self, prompt: str) -> str:
        return "not json at all"


class WrongIdsJudge:
    """Answers a well-formed array about candidates that do not exist —
    the strict v2 parser must refuse the whole window."""

    model_id = "wrong-ids-judge"

    def complete(self, prompt: str) -> str:
        return json.dumps(
            [{"candidate_id": "C99", "decision": "SPLIT", "reason_code": "OTHER"}]
        )


class RaisingJudge:
    model_id = "raising-judge"

    def complete(self, prompt: str) -> str:
        raise RuntimeError("provider unreachable")


def standard_and_deep(judge):
    chunker = StructuralChunker()
    rows = oversized_document()
    standard = chunker.chunk_text("", "doc", "rapor.pdf", parsed_units=rows)
    deep, report = chunker.chunk_text_deep(
        "", "doc", "rapor.pdf", judge=judge, parsed_units=rows
    )
    return standard, deep, report


# ------------------------------------------------------------ equivalence
def test_the_fixture_actually_consults_the_judge():
    judge = ScriptedJudge()
    _, _, report = standard_and_deep(judge)
    assert report["judge_call_count"] > 0
    assert report["consulted_boundary_count"] > 0
    assert judge.prompts, "the synthetic corpus must offer a real choice"


def test_one_window_is_one_provider_call():
    """The v2 batching: every decision window costs exactly one complete()."""
    judge = ScriptedJudge()
    _, _, report = standard_and_deep(judge)
    assert len(judge.prompts) == report["consulted_boundary_count"]
    assert report["judge_call_count"] == len(judge.prompts)
    # A consulted window always offers a genuine choice: >= 2 candidates.
    for prompt in judge.prompts:
        assert len(CANDIDATE_MARKER.findall(prompt)) >= 2
    # Candidate decisions stay counted per candidate, not per call.
    assert judge.candidates_seen > len(judge.prompts)
    assert report["split_votes"] + report["keep_votes"] == judge.candidates_seen


def test_all_keep_matches_the_standard_output_and_the_hard_budget():
    judge = ScriptedJudge()
    standard, deep, report = standard_and_deep(judge)
    assert [c.content for c in deep] == [c.content for c in standard]
    assert report["keep_votes"] == judge.candidates_seen
    assert report["split_votes"] == 0
    for chunk in deep:
        assert chunk.metadata["token_count"] <= HARD_MAX_TOKENS


def test_a_split_vote_changes_the_cut():
    # SPLIT only the first candidate of the first window; the greedy cut
    # sits at the last admissible position, so an early SPLIT must move it.
    judge = ScriptedJudge(
        lambda window, index, cid: "SPLIT" if window == 1 and index == 0 else "KEEP"
    )
    standard, deep, report = standard_and_deep(judge)
    assert report["split_votes"] >= 1
    assert report["changed_from_greedy_count"] >= 1
    assert [c.content for c in deep] != [c.content for c in standard]
    for chunk in deep:
        assert chunk.metadata["token_count"] <= HARD_MAX_TOKENS


def test_deep_chunks_carry_their_mode_and_model_in_metadata():
    _, deep, _ = standard_and_deep(ScriptedJudge())
    for chunk in deep:
        assert chunk.metadata["chunking_mode"] == "deep_analysis"
        assert chunk.metadata["judge_model"] == "stub-judge-v1"


def test_standard_chunks_carry_no_judge_metadata():
    standard, _, _ = standard_and_deep(ScriptedJudge())
    for chunk in standard:
        assert "chunking_mode" not in chunk.metadata
        assert "judge_model" not in chunk.metadata


# --------------------------------------------------------------- fallback
def test_a_parse_error_falls_back_to_the_structural_cut():
    standard, deep, report = standard_and_deep(GarbageJudge())
    assert [c.content for c in deep] == [c.content for c in standard]
    assert report["fallback_count"] >= 1


def test_unknown_candidate_ids_refuse_the_whole_window():
    standard, deep, report = standard_and_deep(WrongIdsJudge())
    assert [c.content for c in deep] == [c.content for c in standard]
    assert report["fallback_count"] >= 1
    assert report["split_votes"] == 0, "a refused window steers nothing"


def test_a_provider_error_falls_back_to_the_structural_cut():
    standard, deep, report = standard_and_deep(RaisingJudge())
    assert [c.content for c in deep] == [c.content for c in standard]
    assert report["fallback_count"] >= 1


def test_the_report_never_contains_key_material(monkeypatch):
    monkeypatch.setenv("FAKE_JUDGE_KEY", "SECRET-SENTINEL-123")
    _, deep, report = standard_and_deep(ScriptedJudge())
    serialized = json.dumps(report) + json.dumps(
        [chunk.metadata for chunk in deep]
    )
    assert "SECRET-SENTINEL-123" not in serialized
    assert "api_key" not in serialized.lower()


# ----------------------------------------------------------- configuration
def _judge_settings(monkeypatch, **values):
    from config.settings import Settings

    for name in (
        "BOUNDARY_JUDGE_MODEL",
        "BOUNDARY_JUDGE_ENDPOINT",
        "BOUNDARY_JUDGE_API_KEY_ENV",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    for name, value in values.items():
        monkeypatch.setenv(name, value)
    return Settings()


def test_missing_configuration_is_named_not_guessed(monkeypatch):
    settings = _judge_settings(monkeypatch)
    error = boundary_judge_config_error(settings)
    assert error is not None
    assert "BOUNDARY_JUDGE_MODEL" in error
    assert "BOUNDARY_JUDGE_ENDPOINT" in error
    assert "OPENROUTER_API_KEY" in error
    with pytest.raises(ConfigurationException):
        create_boundary_judge(settings)


def test_complete_configuration_builds_a_provider(monkeypatch):
    settings = _judge_settings(
        monkeypatch,
        BOUNDARY_JUDGE_MODEL="test/model",
        BOUNDARY_JUDGE_ENDPOINT="http://localhost:9999/v1/chat/completions",
        BOUNDARY_JUDGE_API_KEY_ENV="FAKE_JUDGE_KEY",
        FAKE_JUDGE_KEY="placeholder",
    )
    assert boundary_judge_config_error(settings) is None
    provider = create_boundary_judge(settings)
    assert provider.model_id == "test/model"
    # The provider stores the *name* of the key variable, never the key.
    assert "placeholder" not in json.dumps(provider.__dict__)


# ------------------------------------------------------------- query path
def test_the_query_path_has_no_judge():
    """The judge exists only inside the deep_analysis branch of ingest."""
    from pipeline.rag_pipeline import RAGPipeline

    for method in ("query", "retrieve", "retrieve_simple", "generate_answer"):
        source = inspect.getsource(getattr(RAGPipeline, method))
        assert "judge" not in source.lower()
        assert "deep_analysis" not in source.lower()
