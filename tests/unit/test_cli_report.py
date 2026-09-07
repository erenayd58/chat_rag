"""The rules a QA report follows: what it counts, what it forgives, what fails.

A report that quietly exported the wrong corpus, or reported a green status
over a known-bad invariant, would be worse than no report. So the interesting
behaviour here is the refusals.
"""

from __future__ import annotations

import csv
import json

import pytest
from amsc.quality.lint import Finding, Report

from cli import report as rp
from cli import runtime
from cli.runtime import ChunkView


def finding(rule="body_heading", confidence="HIGH", target="p-1", page=3):
    return Finding(
        rule=rule, confidence=confidence, target_id=target, page=page,
        reason="a reason", evidence="some evidence",
    )


def chunk(chunk_id="c1", doc="doc-1", units=("p-1",), heading_source="metadata"):
    return ChunkView(
        chunk_id=chunk_id, text="metin", doc_id=doc, heading="BASLIK",
        section_paths=[], pages=[1], unit_ids=list(units), token_count=3,
        heading_source=heading_source,
    )


def manifest(**overrides):
    base = {
        "knowledge_base": {"kb_id": "kb-1", "name": "kb-one"},
        "counts": {"canonical_units": 1, "chunks": 1,
                   "canonical_units_referenced_by_chunks": 1,
                   "pages_with_canonical_units": 1},
        "pipeline": {
            "retriever": "BM25OnlyRetriever", "retrieval_profile": "bm25_only",
            "uses_embeddings": False, "configured_embedding_model": None,
        },
    }
    for key, value in overrides.items():
        base[key] = {**base.get(key, {}), **value} if isinstance(value, dict) else value
    return base


def run(method="bm25", questions=1):
    return {"retrieval": {"method": method, "top_k": 10},
            "metrics": {"questions": questions}}


def gold(entries):
    return {"entries": entries}


ENTRY = {"question": "q", "document_sha256": "abc123def456"}
DOCUMENT = {"doc_id": "doc-1", "file_hash": "abc123def456"}


@pytest.fixture(autouse=True)
def lexical_only(monkeypatch):
    monkeypatch.setattr(runtime, "capabilities_of", lambda kb: {
        "methods": [
            {"name": "hybrid", "available": False},
            {"name": "vector", "available": False},
            {"name": "bm25", "available": True},
        ]
    })


def checks(**overrides):
    arguments = {
        "kb": {"kb_id": "kb-1", "name": "kb-one"},
        "manifest": manifest(),
        "units": [{"unit_id": "p-1"}],
        "views": [chunk()],
        "document": DOCUMENT,
        "run": run(),
        "gold": gold([ENTRY]),
    }
    arguments.update(overrides)
    return {c.name: c for c in rp.consistency_checks(**arguments)}


# ------------------------------------------------------------- destination


def test_a_second_report_in_the_same_second_gets_its_own_directory(tmp_path):
    root = str(tmp_path / "reports")
    first = rp.report_directory("kb-one", root)
    second = rp.report_directory("kb-one", root)
    assert first != second
    assert all(map(lambda p: __import__("os").path.isdir(p), (first, second)))


def test_a_name_a_windows_path_cannot_hold_is_reduced_to_one_it_can(tmp_path):
    directory = rp.report_directory('kkb: "final" <v1>|test', str(tmp_path))
    name = directory.rsplit("\\", 1)[-1].rsplit("/", 1)[-1]
    assert not set(name) & set(':"<>|*?')


def test_a_very_long_name_does_not_become_a_very_long_path(tmp_path):
    directory = rp.report_directory("k" * 300, str(tmp_path))
    name = directory.rsplit("\\", 1)[-1].rsplit("/", 1)[-1]
    assert len(name) < 90


# ------------------------------------------------------------------ counts


def test_the_qa_summary_is_counted_from_the_findings_themselves():
    report = Report(
        findings=[
            finding("unresolved_visual", "HIGH", "v-1"),
            finding("unresolved_visual", "HIGH", "v-2"),
            finding("body_heading", "MEDIUM", "p-9"),
            finding("running_header", "LOW", "h-4"),
        ],
        unit_count=10, chunk_count=4,
    )
    summary = rp.qa_payload(report, {})["summary"]

    assert (summary["HIGH"], summary["MEDIUM"], summary["LOW"]) == (2, 1, 1)
    assert summary["known_limitation_high"] == 2
    assert summary["unexpected_high"] == 0
    assert summary["rules"]["unresolved_visual"]["HIGH"] == 2


def test_a_high_finding_on_any_other_rule_is_counted_as_unexpected():
    report = Report(findings=[finding("body_heading", "HIGH")], unit_count=1,
                    chunk_count=1)
    summary = rp.qa_payload(report, {})["summary"]
    assert summary["unexpected_high"] == 1
    assert summary["known_limitation_high"] == 0


def test_the_context_a_caller_supplies_travels_into_the_payload():
    payload = rp.qa_payload(Report(), {"manifest": "manifest.json"})
    assert payload["manifest"] == "manifest.json"
    assert payload["kind"] == "structural-qa"


# ------------------------------------------------------------------ status


def test_a_clean_corpus_passes():
    assert rp.status_for(
        {"unexpected_high": 0, "known_limitation_high": 0}, []
    ) == rp.PASS


def test_an_accepted_limitation_passes_but_says_so():
    assert rp.status_for(
        {"unexpected_high": 0, "known_limitation_high": 2}, []
    ) == rp.PASS_WITH_KNOWN_LIMITATIONS


def test_a_warning_alone_also_qualifies_the_pass():
    warning = rp.Check("heading", rp.WARNING, "detail")
    assert rp.status_for(
        {"unexpected_high": 0, "known_limitation_high": 0}, [warning]
    ) == rp.PASS_WITH_KNOWN_LIMITATIONS


def test_an_unexpected_high_finding_fails():
    assert rp.status_for(
        {"unexpected_high": 1, "known_limitation_high": 2}, []
    ) == rp.FAIL


def test_a_broken_invariant_fails_even_with_no_findings():
    failure = rp.Check("counts", rp.FAILURE, "detail")
    assert rp.status_for(
        {"unexpected_high": 0, "known_limitation_high": 0}, [failure]
    ) == rp.FAIL


# ------------------------------------------------------------------ checks


def test_counts_that_disagree_with_the_manifest_are_a_failure():
    result = checks(manifest=manifest(counts={"canonical_units": 99}))
    assert result["canonical_count_matches_manifest"].level == rp.FAILURE
    assert "expected 99" in result["canonical_count_matches_manifest"].detail


def test_an_explicit_expectation_is_checked_when_given():
    assert checks(expect_units=1)["canonical_count_matches_expectation"].ok
    assert not checks(expect_units=7)["canonical_count_matches_expectation"].ok
    assert "canonical_count_matches_expectation" not in checks()


def test_a_canonical_stream_missing_a_cited_unit_is_a_failure():
    """The signature of an export taken from a different parser run."""
    result = checks(units=[{"unit_id": "p-9"}])
    assert result["canonical_covers_chunk_units"].level == rp.FAILURE
    assert "p-1" in result["canonical_covers_chunk_units"].detail


def test_a_fragment_is_matched_against_the_unit_it_belongs_to():
    result = checks(views=[chunk(units=["p-1#f2"])])
    assert result["canonical_covers_chunk_units"].ok


def test_a_chunk_from_another_document_is_a_failure():
    result = checks(views=[chunk(), chunk("c2", doc="other-doc")])
    assert result["chunks_belong_to_the_document"].level == rp.FAILURE
    assert "other-doc" in result["chunks_belong_to_the_document"].detail


def test_a_corpus_without_the_chunker_heading_warns_rather_than_passing_quietly():
    result = checks(views=[chunk(heading_source="section_title")])
    assert result["chunk_heading_metadata"].level == rp.WARNING
    assert "over-report" in result["chunk_heading_metadata"].detail


def test_stored_headings_satisfy_the_heading_check():
    assert checks()["chunk_heading_metadata"].ok


def test_a_method_the_profile_cannot_serve_is_a_failure():
    result = checks(run=run(method="vector"))
    assert result["retrieval_method_supported"].level == rp.FAILURE
    assert "bm25" in result["retrieval_method_supported"].detail


def test_dense_retrieval_without_embeddings_is_a_failure():
    result = checks(run=run(method="hybrid"))
    assert result["no_dense_retrieval_without_embeddings"].level == rp.FAILURE


def test_a_configured_but_unused_model_is_stated_not_hidden():
    result = checks(manifest=manifest(
        pipeline={"configured_embedding_model": "minilm", "uses_embeddings": False}
    ))
    assert result["configured_embedding_model_unused"].ok
    assert "not used" in result["configured_embedding_model_unused"].detail


def test_a_gold_set_with_a_different_document_hash_is_a_failure():
    result = checks(gold=gold([{**ENTRY, "document_sha256": "999999999999"}]))
    assert result["gold_document_sha"].level == rp.FAILURE
    assert "999999999999" in result["gold_document_sha"].detail


def test_a_gold_set_with_no_hash_at_all_warns():
    result = checks(gold=gold([{"question": "q"}]))
    assert result["gold_document_sha"].level == rp.WARNING


def test_fewer_questions_evaluated_than_the_gold_set_holds_is_a_failure():
    result = checks(gold=gold([ENTRY, ENTRY]), run=run(questions=1))
    assert result["gold_question_count"].level == rp.FAILURE


def test_without_a_run_no_retrieval_check_is_invented():
    result = checks(run=None, gold=None)
    assert "retrieval_method_supported" not in result
    assert "gold_document_sha" not in result
    assert result["canonical_count_matches_manifest"].ok


# ------------------------------------------------------------------ files


def test_the_findings_csv_opens_as_a_spreadsheet_and_marks_known_limitations(tmp_path):
    path = str(tmp_path / "findings.csv")
    rp.write_findings_csv(path, [
        finding("unresolved_visual", "HIGH", "v-1"),
        finding("body_heading", "MEDIUM", "p-2"),
    ])

    assert open(path, "rb").read(3) == b"\xef\xbb\xbf", "Excel needs the BOM"
    with open(path, encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["known_limitation"] for row in rows] == ["yes", "no"]
    assert rows[0]["target_id"] == "v-1"
    assert list(rows[0]) == list(rp.FINDING_COLUMNS)


def test_an_exported_corpus_is_one_json_object_per_line(tmp_path):
    path = str(tmp_path / "chunks.jsonl")
    rp.write_jsonl(path, [{"chunk_id": "a", "text": "ü"}, {"chunk_id": "b"}])

    raw = open(path, "rb").read().decode("utf-8")
    assert raw.count("\n") == 2
    assert "\r\n" not in raw
    assert "ü" in raw, "Turkish text is written as itself, not escaped"
    assert [json.loads(line)["chunk_id"] for line in raw.splitlines()] == ["a", "b"]


def test_an_exported_chunk_keeps_what_qa_and_a_reader_both_need():
    row = chunk().as_export_row()
    for key in ("chunk_id", "text", "heading", "section_title", "section_paths",
                "pages", "unit_ids", "token_count", "document_id", "metadata"):
        assert key in row, key
