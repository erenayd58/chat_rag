"""Metrics and gold-entry matching for the regression CLI.

The one thing an evaluation must never do is trust the rank a human happened to
see when they confirmed an answer. Every number here is recomputed from the
retrieval this run performed.
"""

from __future__ import annotations

import pytest

from cli import evaluate as ev
from cli.runtime import ChunkView


def chunk(chunk_id, *, text="metin", doc="doc-1", heading=None, pages=(), units=(),
          paths=()):
    return ChunkView(
        chunk_id=chunk_id, text=text, doc_id=doc, heading=heading,
        section_paths=[list(p) for p in paths], pages=list(pages),
        unit_ids=list(units), token_count=len(text.split()),
    )


ENTRY = {
    "entry_id": "e1",
    "question": "soru?",
    "kb_id": "kb-1",
    "document_id": "doc-1",
    "correct_chunk_id": "doc:s-chunk-0172",
    "section": "KOSGEB ISLETME",
    "pages": [36],
    "unit_ids": ["v-00808"],
    "evidence": "FINDEKS RISK RAPORU SORGU ADEDI 11.000.144 satiri burada duruyor",
    # Deliberately wrong: a run must not read this.
    "found_at_rank": 1,
}


def rank_of(entry, chunks):
    return ev.evaluate_entry(entry, chunks, check_sha=False).rank


# --------------------------------------------------------------- the ladder


def test_unit_ids_identify_the_answer_after_rechunking():
    """The chunk id changed but the canonical units did not."""
    chunks = [chunk("other"), chunk("doc:s-chunk-9999", units=["v-00808"])]
    result = ev.evaluate_entry(ENTRY, chunks, check_sha=False)
    assert result.rank == 2
    assert result.rule == "unit_ids"
    assert result.matched_chunk_id == "doc:s-chunk-9999"
    assert result.expected_chunk_id == "doc:s-chunk-0172"


def test_a_fragment_suffix_still_counts_as_the_same_unit():
    chunks = [chunk("c", units=["v-00808#f2"])]
    assert rank_of(ENTRY, chunks) == 1


def test_evidence_identifies_the_answer_when_unit_ids_are_gone():
    entry = {**ENTRY, "unit_ids": []}
    chunks = [chunk("c", text="onceki metin " + ENTRY["evidence"] + " devam")]
    result = ev.evaluate_entry(entry, chunks, check_sha=False)
    assert result.rank == 1 and result.rule == "evidence"


def test_evidence_matching_ignores_whitespace_and_case():
    entry = {**ENTRY, "unit_ids": []}
    chunks = [chunk("c", text=ENTRY["evidence"].upper().replace(" ", "\n"))]
    assert rank_of(entry, chunks) == 1


def test_a_too_short_evidence_snippet_never_matches_on_its_own():
    entry = {**ENTRY, "unit_ids": [], "evidence": "kisa"}
    assert rank_of(entry, [chunk("c", text="kisa bir metin")]) is None


def test_the_chunk_id_is_a_fallback_not_the_primary_key():
    entry = {**ENTRY, "unit_ids": [], "evidence": ""}
    chunks = [chunk("doc:s-chunk-0172")]
    result = ev.evaluate_entry(entry, chunks, check_sha=False)
    assert result.rank == 1 and result.rule == "chunk_id"


def test_section_and_page_identify_the_answer_as_a_last_resort():
    entry = {**ENTRY, "unit_ids": [], "evidence": "", "correct_chunk_id": None}
    chunks = [chunk("c", heading="**KOSGEB ISLETME**", pages=[36])]
    result = ev.evaluate_entry(entry, chunks, check_sha=False)
    assert result.rank == 1 and result.rule == "section+page"


def test_the_right_section_on_the_wrong_page_is_not_a_match():
    entry = {**ENTRY, "unit_ids": [], "evidence": "", "correct_chunk_id": None}
    chunks = [chunk("c", heading="KOSGEB ISLETME", pages=[12])]
    assert rank_of(entry, chunks) is None


def test_a_chunk_from_another_document_is_never_the_answer():
    chunks = [chunk("c", doc="other-doc", units=["v-00808"])]
    assert rank_of(ENTRY, chunks) is None


def test_nothing_matches_when_no_locator_fires():
    assert rank_of(ENTRY, [chunk("a"), chunk("b")]) is None


# ------------------------------------------------------------------ metrics


@pytest.mark.parametrize(
    "rank,hit1,hit3,hit5,rr",
    [
        (1, True, True, True, 1.0),
        (3, False, True, True, 1 / 3),
        (5, False, False, True, 0.2),
        (7, False, False, False, 1 / 7),
        (None, False, False, False, 0.0),
    ],
)
def test_hits_and_reciprocal_rank_follow_the_measured_rank(rank, hit1, hit3, hit5, rr):
    chunks = [chunk(f"c{i}") for i in range(1, 11)]
    if rank:
        chunks[rank - 1] = chunk("target", units=["v-00808"])
    result = ev.evaluate_entry(ENTRY, chunks, check_sha=False)

    assert result.rank == rank
    assert (result.hit(1), result.hit(3), result.hit(5)) == (hit1, hit3, hit5)
    assert result.reciprocal_rank == pytest.approx(rr)


def test_a_match_beyond_the_retrieved_window_is_simply_not_found():
    chunks = [chunk(f"c{i}") for i in range(5)]
    result = ev.evaluate_entry(ENTRY, chunks, check_sha=False)
    assert result.rank is None
    assert result.hit(5) is False
    assert result.reciprocal_rank == 0.0


def test_the_recorded_rank_is_ignored_entirely():
    """found_at_rank says 1; retrieval puts the answer third."""
    entry = {**ENTRY, "found_at_rank": 1}
    chunks = [chunk("a"), chunk("b"), chunk("target", units=["v-00808"])]
    result = ev.evaluate_entry(entry, chunks, check_sha=False)
    assert result.rank == 3
    assert result.as_dict()["hit@1"] is False
    assert "found_at_rank" not in result.as_dict()


def test_the_summary_averages_over_every_question():
    results = [
        ev.evaluate_entry(ENTRY, [chunk("t", units=["v-00808"])], check_sha=False),
        ev.evaluate_entry(ENTRY, [chunk("a"), chunk("b"),
                                  chunk("t", units=["v-00808"])], check_sha=False),
        ev.evaluate_entry(ENTRY, [chunk("a")], check_sha=False),
    ]
    summary = ev.summarize(results)
    assert summary["questions"] == 3
    assert summary["hit@1"] == pytest.approx(1 / 3)
    assert summary["hit@3"] == pytest.approx(2 / 3)
    assert summary["mrr"] == pytest.approx((1 + 1 / 3 + 0) / 3, abs=1e-6)
    assert summary["not_found"] == 1


def test_an_empty_run_summarises_to_zero():
    assert ev.summarize([])["questions"] == 0


def test_the_rendered_summary_reads_like_the_documented_example():
    text = ev.render_summary(
        {"questions": 20, "hit@1": 0.8, "hit@3": 0.95, "hit@5": 1.0, "mrr": 0.88,
         "not_found": 0}
    )
    assert "Questions: 20" in text
    assert "Hit@1:  80.0%" in text
    assert "MRR:       0.88" in text


# ------------------------------------------------------------------ compare


def run(metrics, questions):
    return {"metrics": metrics, "questions": questions}


def test_compare_names_regressions_and_improvements():
    previous = run({"hit@1": 0.80, "hit@3": 0.95, "hit@5": 1.0, "mrr": 0.88}, [])
    current = run({"hit@1": 0.75, "hit@3": 0.95, "hit@5": 1.0, "mrr": 0.92}, [])

    verdicts = {
        row["metric"]: row["verdict"]
        for row in ev.compare_runs(previous, current)["metrics"]
    }
    assert verdicts["hit@1"] == "REGRESSION"
    assert verdicts["hit@3"] == "same"
    assert verdicts["mrr"] == "IMPROVEMENT"


def test_compare_lists_the_questions_whose_rank_moved():
    previous = run({}, [{"question": "a", "rank": 1}, {"question": "b", "rank": 3}])
    current = run({}, [{"question": "a", "rank": 4}, {"question": "b", "rank": 1}])

    moved = {row["question"]: row for row in ev.compare_runs(previous, current)["moved"]}
    assert moved["a"]["before"] == 1 and moved["a"]["after"] == 4
    assert moved["a"]["change"] == "worse"
    assert moved["b"]["change"] == "better"


def test_compare_marks_an_answer_that_dropped_out_of_the_window():
    previous = run({}, [{"question": "a", "rank": 2}])
    current = run({}, [{"question": "a", "rank": None}])
    assert ev.compare_runs(previous, current)["moved"][0]["change"] == "lost"


def test_compare_notices_questions_added_and_removed():
    previous = run({}, [{"question": "old", "rank": 1}])
    current = run({}, [{"question": "new", "rank": 1}])
    comparison = ev.compare_runs(previous, current)
    assert comparison["moved"][0]["change"] == "new"
    assert comparison["dropped"] == ["old"]


def test_a_run_with_no_metric_change_is_not_a_regression():
    same = run({"hit@1": 0.8, "mrr": 0.88}, [])
    assert ev.has_regression(ev.compare_runs(same, same)) is False


def test_has_regression_is_true_when_any_metric_fell():
    previous = run({"hit@1": 0.8, "mrr": 0.88}, [])
    current = run({"hit@1": 0.8, "mrr": 0.80}, [])
    assert ev.has_regression(ev.compare_runs(previous, current)) is True


def test_the_rendered_comparison_shows_both_metrics_and_moves():
    previous = run({"hit@1": 0.80, "mrr": 0.88}, [{"question": "a", "rank": 1}])
    current = run({"hit@1": 0.75, "mrr": 0.84}, [{"question": "a", "rank": 4}])
    text = ev.render_comparison(ev.compare_runs(previous, current))
    assert "hit@1  0.80 -> 0.75  REGRESSION" in text
    assert "mrr    0.88 -> 0.84  REGRESSION" in text
    assert "#1 rank 1 -> rank 4" in text


# ------------------------------------------------------- document identity


def identity(**by_document):
    return ev.DocumentIdentity(dict(by_document))


HASHED = {**ENTRY, "document_sha256": "0058e0af7460"}


def test_the_hash_decides_which_document_a_chunk_belongs_to():
    """A new ingest renames everything except the bytes."""
    chunks = [chunk("newdoc:s-chunk-0009", doc="new-doc", units=["v-00808"])]
    result = ev.evaluate_entry(
        HASHED, chunks, check_sha=False,
        identity=identity(**{"new-doc": "0058e0af7460"}),
    )
    assert result.rank == 1
    assert result.rule == "unit_ids"


def test_a_document_with_other_bytes_is_refused_however_it_is_named():
    chunks = [chunk("c", doc="doc-1", units=["v-00808"])]
    result = ev.evaluate_entry(
        HASHED, chunks, check_sha=False,
        identity=identity(**{"doc-1": "9999999999ff"}),
    )
    assert result.rank is None, "the same id but different bytes must not match"


def test_the_frozen_document_id_no_longer_gates_when_a_hash_is_known():
    """kb_id and document_id are audit metadata once the hash is there."""
    entry = {**HASHED, "document_id": "upload_LONG_GONE_pdf",
             "kb_id": "kb-LONG-GONE"}
    chunks = [chunk("c", doc="upload_new_pdf", units=["v-00808"])]
    result = ev.evaluate_entry(
        entry, chunks, check_sha=False,
        identity=identity(**{"upload_new_pdf": "0058e0af7460"}),
    )
    assert result.rank == 1


def test_a_document_the_tracker_cannot_resolve_falls_back_to_the_id():
    """Nothing to compare hashes against, so the frozen id is all there is."""
    chunks = [chunk("c", doc="doc-1", units=["v-00808"])]
    assert ev.evaluate_entry(
        HASHED, chunks, check_sha=False, identity=identity(**{"other": "x"})
    ).rank == 1
    assert ev.evaluate_entry(
        {**HASHED, "document_id": "somewhere-else"}, chunks,
        check_sha=False, identity=identity(**{"other": "x"}),
    ).rank is None


def test_an_entry_without_a_hash_keeps_the_old_behaviour():
    chunks = [chunk("c", doc="doc-1", units=["v-00808"])]
    assert ev.evaluate_entry(ENTRY, chunks, check_sha=False).rank == 1
    other = [chunk("c", doc="another", units=["v-00808"])]
    assert ev.evaluate_entry(ENTRY, other, check_sha=False).rank is None


# ------------------------------------------------------------ mismatch


def test_a_knowledge_base_without_those_bytes_is_reported():
    chunks = [chunk("c", doc="doc-1", units=["v-00808"])]
    result = ev.evaluate_entry(
        HASHED, chunks, check_sha=True, identity=identity(**{"doc-1": "ffffffffffff"})
    )
    assert result.warnings
    assert "holds no document with the bytes" in result.warnings[0]
    assert "0058e0af7460" in result.warnings[0]


def test_matching_bytes_raise_nothing():
    chunks = [chunk("c", doc="doc-1", units=["v-00808"])]
    result = ev.evaluate_entry(
        HASHED, chunks, check_sha=True, identity=identity(**{"doc-1": "0058e0af7460"})
    )
    assert result.warnings == []


def test_an_unresolvable_knowledge_base_says_so_rather_than_passing():
    result = ev.evaluate_entry(
        HASHED, [chunk("c", units=["v-00808"])], check_sha=True,
        identity=ev.DocumentIdentity({}),
    )
    assert result.warnings and "could not be checked" in result.warnings[0]


# --------------------------------------------------------------- the ladder


def test_chunk_id_is_the_last_resort_not_an_early_one():
    """Section+page is tried before the id, because the id moves and it does not."""
    names = [name for name, _ in ev.RULES]
    assert names == ["unit_ids", "evidence", "section+page", "chunk_id"]


def test_a_stale_chunk_id_never_outranks_a_real_locator():
    entry = {**ENTRY, "correct_chunk_id": "doc:s-chunk-0172"}
    chunks = [chunk("doc:s-chunk-0172"), chunk("other", units=["v-00808"])]
    result = ev.evaluate_entry(entry, chunks, check_sha=False)
    # The id still wins at rank 1 -- it is the earlier chunk -- but the rule
    # that fired is reported so a reader can see how weak the match was.
    assert (result.rank, result.rule) == (1, "chunk_id")
