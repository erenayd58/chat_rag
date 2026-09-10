"""The exportable QA package, end to end, with the store and retriever stubbed.

The point of a report is that someone else can check it, so what matters is
that every file in it came from production state and that the numbers in the
summary are the ones the exported files support. The sharpest risk is exporting
a canonical stream from an older parser run -- it would look clean and describe
a document nobody has -- so that case is exercised directly.
"""

from __future__ import annotations

import csv
import json
import os

import pytest

import cli.runtime as runtime
from cli.__main__ import main
from chat_rag.components.provenance import build_snapshot
from chat_rag.components.retriever import BM25OnlyRetriever, NullEmbedding


class FakeChunk:
    def __init__(self, chunk_id, content, metadata, section_title=None, doc_id="doc-1"):
        self.chunk_id = chunk_id
        self.content = content
        self.doc_id = doc_id
        self.doc_title = "rapor.pdf"
        self.section_title = section_title
        self.metadata = metadata


class FakeResult:
    def __init__(self, chunk, score):
        self.chunk = chunk
        self.score = score


class FakeParser:
    NORMALIZATION_VERSION = "v7-test"
    RUNNING_HEADER_MIN_PAGES = 3
    RECONSTRUCT_VISUAL_GRIDS = True
    DEMOTE_LEAD_IN_HEADINGS = True
    PROMOTE_MISSED_HEADINGS = True
    DEMOTE_CAPTION_HEADINGS = True
    REJOIN_SPLIT_HEADINGS = True
    DEMOTE_SENTENCE_HEADINGS = True
    parser_backend = "pymupdf4llm-layout"

    class _Profile:
        reading_order = "column-major-left-to-right"

    _spread_profile = _Profile()

    def get_name(self):
        return "StructuredPDFParser-test"


class FakeFactory:
    def get_parser(self, path):
        return FakeParser()


class FakeVectorDB:
    def __init__(self, chunks):
        self._chunks = chunks

    def get_all_chunks(self):
        return self._chunks


class FakePipeline:
    retrieval_profile = "bm25_only"

    def __init__(self, chunks, order):
        self.vector_db = FakeVectorDB(chunks)
        self.parser_factory = FakeFactory()
        self.hybrid_retriever = BM25OnlyRetriever(
            embedding_model=NullEmbedding(), vector_db=None
        )
        by_id = {c.chunk_id: c for c in chunks}
        self._ordered = [by_id[cid] for cid in order]
        self.hybrid_retriever.keyword_search = self._keyword

    def _keyword(self, query, top_k=5, **kwargs):
        return [
            FakeResult(chunk, 10.0 - index)
            for index, chunk in enumerate(self._ordered[:top_k])
        ]


# The corpus: three canonical units, two chunks that quote them verbatim.
UNIT_TEXTS = {
    "h-0001": "KURUMSAL PROFIL",
    "p-0002": "KKB Kredi Kayit Burosu 1995 yilinda dokuz banka tarafindan kuruldu.",
    "p-0003": "Findeks Risk Raporu sorgu adedi 2024 yilinda 11.000.144 oldu.",
}


def canonical_stream(texts, page_count=3):
    """A canonical stream shaped the way the parser emits one.

    A heading unit carries its own title as the tail of its section path; the
    body below it inherits that path. Getting this wrong is itself a structural
    QA finding, which is how the shape was confirmed.
    """
    rows = []
    for order, (unit_id, text) in enumerate(texts.items(), start=1):
        heading = unit_id.startswith("h-")
        rows.append({
            "document_id": "document",
            "unit_id": unit_id,
            "order": order,
            "type": "heading" if heading else "paragraph",
            "text": text,
            "section_path": [text] if heading else ["KURUMSAL PROFIL"],
            "heading_level": 2 if heading else None,
            "source": {"page": min(order, page_count)},
        })
    return rows


def make_chunk(chunk_id, unit_ids, pages, heading):
    text = "\n\n".join(UNIT_TEXTS[u] for u in unit_ids)
    return FakeChunk(
        chunk_id, text,
        {
            "heading": heading,
            "unit_ids_json": json.dumps(list(unit_ids)),
            "pages_json": json.dumps(list(pages)),
            "section_paths_json": json.dumps([["KURUMSAL PROFIL"]]),
            "token_count": len(text.split()),
            "page_count": 3,
            "chunker_type": "structure_first",
        },
        section_title="KURUMSAL PROFIL",
    )


CHUNKS = [
    make_chunk("doc-1:s-chunk-0001", ["h-0001", "p-0002"], [1, 2], "KURUMSAL PROFIL"),
    make_chunk("doc-1:s-chunk-0002", ["p-0003"], [3], "KURUMSAL PROFIL"),
]

GOLD_ENTRY = {
    "entry_id": "e1",
    "question": "Findeks sorgu adedi kac?",
    "kb_id": "kb-1",
    "document_id": "doc-1",
    "correct_chunk_id": "doc-1:s-chunk-OLD",
    "section": "KURUMSAL PROFIL",
    "pages": [3],
    "unit_ids": ["p-0003"],
    "evidence": UNIT_TEXTS["p-0003"],
    "document_sha256": "abc123",
    "found_at_rank": 1,
}

KB_RECORD = {
    "name": "kb-one",
    "chunker": {"type": "structure_first"},
    "vector_db_provider": "chroma",
    "embedding_model_name": "paraphrase-multilingual-MiniLM-L12-v2",
}


def ingest_snapshot(**overrides):
    """What a successful ingest would have captured, built the same way.

    Not hand-written: the fixture calls the production capture so a change to
    what is recorded shows up here rather than being silently mirrored.
    """
    snapshot = build_snapshot(
        FakePipeline(CHUNKS, [c.chunk_id for c in CHUNKS]), KB_RECORD,
        kb_id="kb-1", vector_collection="kb-1",
    )
    # The tracker stamps the hash of the bytes it recorded.
    snapshot["document_sha256"] = "abc123"
    for key, value in overrides.items():
        snapshot["pipeline"][key] = value
    return snapshot


DOCUMENT = {
    "file_path": "/tmp/upload_x.pdf",
    "file_name": "upload_x.pdf",
    "doc_id": "doc-1",
    "chunk_count": 2,
    "file_size": 4242,
    "ingested_at": "2026-08-25T10:00:00",
    "file_hash": "abc123",
    "kb_id": "kb-1",
    "metadata": {"original_filename": "rapor.pdf"},
}


def write_stream(directory, name, rows):
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, name)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        runtime, "kb_manager",
        type("KB", (), {
            "get": staticmethod(lambda kb_id: {
                "name": "kb-one", "chunker": {"type": "structure_first"},
                "vector_db_provider": "chroma",
                "embedding_model_name": "paraphrase-multilingual-MiniLM-L12-v2",
            } if kb_id == "kb-1" else None),
            "find_by_name": staticmethod(lambda name: "kb-1" if name == "kb-one" else None),
            "list": staticmethod(lambda: [{"kb_id": "kb-1", "name": "kb-one"}]),
            "collection": staticmethod(lambda kb_id: "kb-1"),
        })(),
    )
    monkeypatch.setattr(runtime, "document_sha", lambda doc_id: "abc123")
    monkeypatch.setattr(
        runtime, "documents_for",
        lambda kb: [{**DOCUMENT, "pipeline_snapshot": ingest_snapshot()}],
    )
    monkeypatch.setattr(runtime, "get_pipeline",
                        lambda *a, **k: FakePipeline(CHUNKS, [c.chunk_id for c in CHUNKS]))

    cache = tmp_path / ".cache" / "canonical-units"
    write_stream(str(cache), "current.jsonl", canonical_stream(UNIT_TEXTS))
    return tmp_path


def write_gold(tmp_path, entries=(GOLD_ENTRY,), name="gold.json"):
    path = tmp_path / name
    path.write_text(
        json.dumps({"schema_version": 1, "kind": "retrieval-gold",
                    "frozen_at": "2026-08-25T10:00:00", "entries": list(entries)},
                   ensure_ascii=False),
        encoding="utf-8",
    )
    return str(path)


def build(workspace, monkeypatch, extra=()):
    out = str(workspace / "report")
    code = main(["report", "--kb", "kb-1", "--gold", write_gold(workspace),
                 "--out", out, *extra])
    return code, out


def read_json(directory, name):
    with open(os.path.join(directory, name), encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(directory, name):
    with open(os.path.join(directory, name), encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


# -------------------------------------------------------------- the package


def test_a_report_contains_every_promised_file(workspace, monkeypatch):
    code, out = build(workspace, monkeypatch)
    assert code == 0
    for name in ("manifest.json", "canonical.units.jsonl", "chunks.jsonl",
                 "structural_qa.json", "structural_qa.csv", "retrieval_eval.json",
                 "summary.md"):
        assert os.path.isfile(os.path.join(out, name)), name


def test_the_exported_corpus_is_the_one_the_store_actually_holds(workspace, monkeypatch):
    _, out = build(workspace, monkeypatch)
    chunks = read_jsonl(out, "chunks.jsonl")
    units = read_jsonl(out, "canonical.units.jsonl")

    assert [c["chunk_id"] for c in chunks] == [c.chunk_id for c in CHUNKS]
    assert {u["unit_id"] for u in units} == set(UNIT_TEXTS)
    assert chunks[0]["heading"] == "KURUMSAL PROFIL"
    assert chunks[0]["unit_ids"] == ["h-0001", "p-0002"]
    assert chunks[0]["document_id"] == "doc-1"


def test_the_manifest_agrees_with_what_was_exported(workspace, monkeypatch):
    _, out = build(workspace, monkeypatch)
    manifest = read_json(out, "manifest.json")

    assert manifest["counts"]["canonical_units"] == len(
        read_jsonl(out, "canonical.units.jsonl")
    )
    assert manifest["counts"]["chunks"] == len(read_jsonl(out, "chunks.jsonl"))
    assert manifest["document"]["sha256"] == "abc123"
    assert manifest["document"]["pages"] == 3
    assert manifest["pipeline"]["normalization_version"] == "v7-test"
    assert manifest["features"]["visual_grid"] is True
    assert manifest["status"] in {"PASS", "PASS_WITH_KNOWN_LIMITATIONS"}
    assert all(c["level"] == "ok" for c in manifest["consistency_checks"])


def test_the_manifest_separates_a_configured_model_from_one_in_use(
    workspace, monkeypatch
):
    _, out = build(workspace, monkeypatch)
    pipeline = read_json(out, "manifest.json")["pipeline"]
    assert pipeline["configured_embedding_model"]
    assert pipeline["uses_embeddings"] is False
    assert pipeline["requires_document_embeddings"] is False


# ------------------------------------------------------- the right corpus


def test_a_near_miss_canonical_stream_is_not_exported(workspace, monkeypatch):
    """An older parser run shares almost every unit id and must still lose.

    Ids are positional: inserting one unit shifts every later id onto someone
    else's text. Matching on ids alone would pick either stream.
    """
    shifted = {
        "h-0001": UNIT_TEXTS["h-0001"],
        "p-0002": "Bu birim eski surumde vardi ve sonra kaldirildi.",
        "p-0003": UNIT_TEXTS["p-0002"],
        "p-0004": UNIT_TEXTS["p-0003"],
    }
    write_stream(str(workspace / ".cache" / "canonical-units"), "aaa-older.jsonl",
                 canonical_stream(shifted))

    _, out = build(workspace, monkeypatch)
    units = {u["unit_id"]: u["text"] for u in read_jsonl(out, "canonical.units.jsonl")}

    assert units == UNIT_TEXTS
    assert read_json(out, "manifest.json")["canonical_source"].endswith("current.jsonl")


def test_a_report_refuses_rather_than_exporting_an_unproven_corpus(
    workspace, monkeypatch, capsys
):
    for name in os.listdir(workspace / ".cache" / "canonical-units"):
        os.remove(workspace / ".cache" / "canonical-units" / name)

    code, _ = build(workspace, monkeypatch)

    assert code == 2
    printed = capsys.readouterr().err
    assert "canonical units" in printed and "--units" in printed


def test_a_hand_supplied_canonical_file_is_used_and_recorded(workspace, monkeypatch):
    path = write_stream(str(workspace / "elsewhere"), "units.jsonl",
                        canonical_stream(UNIT_TEXTS))
    code, out = build(workspace, monkeypatch, extra=["--units", path])

    assert code == 0
    manifest = read_json(out, "manifest.json")
    assert manifest["canonical_source"].endswith("units.jsonl")
    assert any("by hand" in w for w in manifest["warnings"])


# ------------------------------------------------------------ consistency


def test_a_count_that_does_not_match_expectation_fails_loudly(
    workspace, monkeypatch, capsys
):
    code, out = build(workspace, monkeypatch, extra=["--expect-units", "1750"])

    assert code == 1
    printed = capsys.readouterr().out
    assert "Status: FAIL" in printed
    assert "canonical_count_matches_expectation" in printed
    # It still writes the package, so the mismatch can be investigated.
    assert read_json(out, "manifest.json")["status"] == "FAIL"


def test_a_gold_set_confirmed_against_another_document_fails(
    workspace, monkeypatch, capsys
):
    gold = write_gold(workspace, [{**GOLD_ENTRY, "document_sha256": "999999"}],
                      name="other.json")
    out = str(workspace / "report")
    code = main(["report", "--kb", "kb-1", "--gold", gold, "--out", out])

    assert code == 1
    assert "gold_document_sha" in capsys.readouterr().out


# ------------------------------------------------------------------- qa


def test_structural_findings_reach_both_machine_and_spreadsheet_forms(
    workspace, monkeypatch
):
    _, out = build(workspace, monkeypatch)
    qa = read_json(out, "structural_qa.json")

    with open(os.path.join(out, "structural_qa.csv"),
              encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == len(qa["findings"])
    assert qa["unit_count"] == len(read_jsonl(out, "canonical.units.jsonl"))
    assert qa["chunk_count"] == len(read_jsonl(out, "chunks.jsonl"))
    summary = qa["summary"]
    assert summary["HIGH"] == summary["known_limitation_high"] + summary["unexpected_high"]


def test_the_summary_reports_the_same_counts_the_files_hold(workspace, monkeypatch):
    _, out = build(workspace, monkeypatch)
    summary = open(os.path.join(out, "summary.md"), encoding="utf-8").read()
    qa = read_json(out, "structural_qa.json")
    manifest = read_json(out, "manifest.json")

    assert f"- Canonical units: {manifest['counts']['canonical_units']}" in summary
    assert f"- Chunks: {manifest['counts']['chunks']}" in summary
    assert f"- MEDIUM: {qa['summary']['MEDIUM']}" in summary
    assert manifest["status"] in summary
    assert "## Consistency checks" in summary


# ------------------------------------------------------------- retrieval


def test_the_report_metric_is_the_one_eval_would_produce(workspace, monkeypatch):
    gold = write_gold(workspace)
    out = str(workspace / "report")
    main(["report", "--kb", "kb-1", "--gold", gold, "--out", out])
    main(["eval", "--kb", "kb-1", "--gold", gold, "--out", str(workspace / "run.json")])

    from_report = read_json(out, "retrieval_eval.json")
    from_eval = json.loads((workspace / "run.json").read_text(encoding="utf-8"))

    assert from_report["metrics"] == from_eval["metrics"]
    assert from_report["questions"][0]["rank"] == from_eval["questions"][0]["rank"]


def test_the_rank_is_measured_again_and_not_taken_from_the_gold_entry(
    workspace, monkeypatch
):
    """The entry claims rank 1; retrieval puts the answer second."""
    _, out = build(workspace, monkeypatch)
    question = read_json(out, "retrieval_eval.json")["questions"][0]

    assert question["rank"] == 2
    assert question["hit@1"] is False
    assert question["matched_by"] == "unit_ids"
    assert question["expected"]["unit_ids"] == ["p-0003"]
    assert question["expected"]["section"] == "KURUMSAL PROFIL"
    assert question["expected"]["pages"] == [3]


def test_a_report_without_a_gold_set_skips_retrieval_and_says_so(
    workspace, monkeypatch
):
    out = str(workspace / "report")
    code = main(["report", "--kb", "kb-1", "--out", out])

    assert code == 0
    assert not os.path.exists(os.path.join(out, "retrieval_eval.json"))
    summary = open(os.path.join(out, "summary.md"), encoding="utf-8").read()
    assert "Not run (no gold set supplied)" in summary


# --------------------------------------------------------------- inspect


def test_inspect_json_prints_the_same_manifest_schema(workspace, monkeypatch, capsys):
    assert main(["inspect", "--kb", "kb-1", "--json"]) == 0
    manifest = json.loads(capsys.readouterr().out)

    assert manifest["schema_version"] == 1
    assert manifest["kind"] == "pipeline-manifest"
    assert manifest["counts"]["chunks"] == len(CHUNKS)
    assert manifest["document"]["document_id"] == "doc-1"


def test_inspect_can_write_the_manifest_to_a_file(workspace, monkeypatch):
    path = str(workspace / "manifest.json")
    assert main(["inspect", "--kb", "kb-1", "--out", path]) == 0
    assert json.load(open(path, encoding="utf-8"))["kind"] == "pipeline-manifest"


def test_inspect_without_json_still_prints_the_table(workspace, monkeypatch, capsys):
    assert main(["inspect", "--kb", "kb-1"]) == 0
    printed = capsys.readouterr().out
    assert "Uses embeddings" in printed
    assert "kb-one" in printed


# ------------------------------------------------------------ provenance


def test_the_manifest_describes_the_configuration_that_ran_at_ingest(
    workspace, monkeypatch
):
    """Not today's: the snapshot the ingest recorded."""
    monkeypatch.setattr(
        runtime, "documents_for",
        lambda kb: [{**DOCUMENT, "pipeline_snapshot": ingest_snapshot(
            normalization_version="v3-as-it-was-then", chunker="structure_first"
        )}],
    )
    _, out = build(workspace, monkeypatch)
    manifest = read_json(out, "manifest.json")

    assert manifest["provenance"]["source"] == "ingest-snapshot"
    assert manifest["provenance"]["captured_at"]
    assert manifest["pipeline"]["normalization_version"] == "v3-as-it-was-then"
    assert not any("current runtime config" in w for w in manifest["warnings"])
    summary = open(os.path.join(out, "summary.md"), encoding="utf-8").read()
    assert "captured at ingest" in summary


def test_a_document_ingested_before_snapshots_says_so_instead_of_pretending(
    workspace, monkeypatch
):
    monkeypatch.setattr(runtime, "documents_for", lambda kb: [DOCUMENT])

    code, out = build(workspace, monkeypatch)
    manifest = read_json(out, "manifest.json")

    assert code == 0
    assert manifest["provenance"]["source"] == "current-runtime"
    assert any("current runtime config" in w for w in manifest["warnings"])
    assert manifest["status"] == "PASS_WITH_KNOWN_LIMITATIONS"
    levels = {c["name"]: c["level"] for c in manifest["consistency_checks"]}
    assert levels["ingest_snapshot_available"] == "warning"
    # The fallback still describes something real, so the report stays usable.
    assert manifest["pipeline"]["normalization_version"] == "v7-test"
    summary = open(os.path.join(out, "summary.md"), encoding="utf-8").read()
    assert "not the one that produced this corpus" in summary


def test_a_snapshot_this_build_cannot_read_is_not_silently_trusted(
    workspace, monkeypatch
):
    stale = {**ingest_snapshot(), "schema_version": 99}
    monkeypatch.setattr(
        runtime, "documents_for",
        lambda kb: [{**DOCUMENT, "pipeline_snapshot": stale}],
    )
    _, out = build(workspace, monkeypatch)
    manifest = read_json(out, "manifest.json")

    assert manifest["provenance"]["source"] == "current-runtime"
    assert any("cannot read it" in w for w in manifest["warnings"])


def test_a_profile_changed_since_ingest_is_reported_not_hidden(
    workspace, monkeypatch
):
    """Changing the profile is allowed; reading the report as if it had not is not."""
    monkeypatch.setattr(
        runtime, "documents_for",
        lambda kb: [{**DOCUMENT, "pipeline_snapshot": ingest_snapshot(
            retrieval_profile="legacy", retriever="HybridRetriever"
        )}],
    )
    _, out = build(workspace, monkeypatch)
    levels = {c["name"]: c for c in read_json(out, "manifest.json")["consistency_checks"]}

    check = levels["retrieval_config_unchanged_since_ingest"]
    assert check["level"] == "warning"
    assert "legacy" in check["detail"] and "bm25_only" in check["detail"]
    # The retrieval that just ran is still judged against today's retriever.
    assert levels["retrieval_method_supported"]["level"] == "ok"


def test_a_snapshot_taken_against_other_bytes_is_flagged(workspace, monkeypatch):
    replaced = {**ingest_snapshot(), "document_sha256": "0000deadbeef"}
    monkeypatch.setattr(
        runtime, "documents_for",
        lambda kb: [{**DOCUMENT, "pipeline_snapshot": replaced}],
    )
    _, out = build(workspace, monkeypatch)
    warnings = read_json(out, "manifest.json")["warnings"]
    assert any("0000deadbeef" in w for w in warnings)
