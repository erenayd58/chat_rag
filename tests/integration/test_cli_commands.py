"""The CLI end to end, with the retriever and stores stubbed.

What is being checked is the wiring: that a run reads a frozen gold set,
recomputes ranks through the production retrieval entry point, refuses a method
the retriever cannot serve, and writes a run record another run can be compared
against.
"""

from __future__ import annotations

import json

import pytest

import cli.runtime as runtime
from cli.__main__ import main
from components.retriever import BM25OnlyRetriever, NullEmbedding


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


def make_chunk(chunk_id, text, units=(), pages=(), heading=None):
    return FakeChunk(
        chunk_id, text,
        {
            "unit_ids_json": json.dumps(list(units)),
            "pages_json": json.dumps(list(pages)),
            "section_paths_json": "[]",
            "token_count": len(text.split()),
            **({"heading": heading} if heading else {}),
        },
        section_title=heading,
    )


class FakeVectorDB:
    def __init__(self, chunks):
        self._chunks = chunks

    def get_all_chunks(self):
        return self._chunks


class FakePipeline:
    """Ordered results, so a test can place the answer at a chosen rank."""

    retrieval_profile = "bm25_only"

    def __init__(self, chunks, order):
        self.vector_db = FakeVectorDB(chunks)
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


CHUNKS = [
    make_chunk("c-first", "alakasiz metin", units=["p-1"], pages=[1], heading="GIRIS"),
    make_chunk("c-second", "baska metin", units=["p-2"], pages=[2], heading="ORTA"),
    make_chunk("c-answer", "dogru kaynak metni", units=["v-808"], pages=[36],
               heading="KOSGEB"),
]

GOLD_ENTRY = {
    "entry_id": "e1",
    "question": "soru?",
    "kb_id": "kb-1",
    "document_id": "doc-1",
    "correct_chunk_id": "c-answer-OLD-ID",
    "section": "KOSGEB",
    "pages": [36],
    "unit_ids": ["v-808"],
    "evidence": "dogru kaynak metni",
    "found_at_rank": 1,
}


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        runtime, "kb_manager",
        type("KB", (), {
            "get": staticmethod(lambda kb_id: {"name": "kb-one", "chunker": {"type": "structure_first"},
                                               "vector_db_provider": "chroma",
                                               "embedding_model_name": None}
                               if kb_id == "kb-1" else None),
            "find_by_name": staticmethod(lambda name: "kb-1" if name == "kb-one" else None),
            "list": staticmethod(lambda: [{"kb_id": "kb-1", "name": "kb-one"}]),
            "storage_path": staticmethod(lambda kb_id, root=".": "/tmp/store"),
        })(),
    )
    monkeypatch.setattr(runtime, "document_sha", lambda doc_id: "abc123")
    # A document is identified by its bytes, so the evaluator resolves the
    # knowledge base's documents to their hashes.
    monkeypatch.setattr(runtime, "documents_for", lambda kb: [
        {"doc_id": "doc-1", "file_hash": "abc123", "file_name": "rapor.pdf",
         "chunk_count": 3}
    ])
    return tmp_path


def write_gold(tmp_path, entries, name="gold.json"):
    path = tmp_path / name
    path.write_text(
        json.dumps({"schema_version": 1, "frozen_at": "2026-01-01T00:00:00",
                    "entries": entries}, ensure_ascii=False),
        encoding="utf-8",
    )
    return str(path)


def use_order(monkeypatch, order):
    pipeline = FakePipeline(CHUNKS, order)
    monkeypatch.setattr(runtime, "get_pipeline", lambda *a, **k: pipeline)
    return pipeline


# ------------------------------------------------------------------- eval


def test_eval_recomputes_the_rank_instead_of_trusting_the_gold_entry(
    workspace, monkeypatch, capsys
):
    """The entry claims rank 1; retrieval actually puts it third."""
    use_order(monkeypatch, ["c-first", "c-second", "c-answer"])
    gold = write_gold(workspace, [GOLD_ENTRY])
    out = str(workspace / "run.json")

    assert main(["eval", "--kb", "kb-1", "--gold", gold, "--out", out]) == 0

    run = json.loads((workspace / "run.json").read_text(encoding="utf-8"))
    question = run["questions"][0]
    assert question["rank"] == 3
    assert question["hit@1"] is False
    assert question["hit@3"] is True
    assert question["reciprocal_rank"] == pytest.approx(1 / 3, abs=1e-6)
    assert run["metrics"]["mrr"] == pytest.approx(1 / 3, abs=1e-6)


def test_eval_matches_a_chunk_whose_id_changed_since_the_answer_was_confirmed(
    workspace, monkeypatch
):
    use_order(monkeypatch, ["c-answer", "c-first"])
    gold = write_gold(workspace, [GOLD_ENTRY])
    out = str(workspace / "run.json")
    main(["eval", "--kb", "kb-1", "--gold", gold, "--out", out])

    question = json.loads((workspace / "run.json").read_text(encoding="utf-8"))["questions"][0]
    assert question["rank"] == 1
    assert question["matched_by"] == "unit_ids"
    assert question["expected_chunk_id"] == "c-answer-OLD-ID"
    assert question["matched_chunk_id"] == "c-answer"


def test_the_run_record_says_what_produced_the_numbers(workspace, monkeypatch):
    use_order(monkeypatch, ["c-answer"])
    gold = write_gold(workspace, [GOLD_ENTRY])
    out = str(workspace / "run.json")
    main(["eval", "--kb", "kb-1", "--gold", gold, "--out", out])

    run = json.loads((workspace / "run.json").read_text(encoding="utf-8"))
    assert run["timestamp"]
    assert run["environment"]["kb_id"] == "kb-1"
    assert run["environment"]["retrieval_profile"] == "bm25_only"
    assert run["environment"]["chunker"] == "structure_first"
    assert run["environment"]["uses_embeddings"] is False
    assert "git_sha" in run["environment"]
    assert run["gold"]["path"].endswith("gold.json")
    assert run["documents"][0]["sha256"] == "abc123"
    assert run["retrieval"]["method"] == "bm25"


def test_a_different_document_is_reported_not_swallowed(workspace, monkeypatch, capsys):
    """The knowledge base holds other bytes than the answer was confirmed on."""
    use_order(monkeypatch, ["c-answer"])
    monkeypatch.setattr(runtime, "documents_for", lambda kb: [
        {"doc_id": "doc-1", "file_hash": "DIFFERENTHASH", "file_name": "other.pdf"}
    ])
    gold = write_gold(workspace, [{**GOLD_ENTRY, "document_sha256": "abc123"}])
    out = str(workspace / "run.json")

    code = main(["eval", "--kb", "kb-1", "--gold", gold, "--out", out, "--strict"])

    printed = capsys.readouterr().out
    assert "WARNING" in printed and "holds no document with the bytes" in printed
    assert code == 1
    run = json.loads((workspace / "run.json").read_text(encoding="utf-8"))
    assert run["document_sha_mismatch"] is True
    assert run["questions"][0]["warnings"]
    assert run["questions"][0]["rank"] is None, "a different corpus must not score"


def test_a_matching_document_hash_raises_no_warning(workspace, monkeypatch, capsys):
    use_order(monkeypatch, ["c-answer"])
    gold = write_gold(workspace, [{**GOLD_ENTRY, "document_sha256": "abc123"}])
    main(["eval", "--kb", "kb-1", "--gold", gold, "--out", str(workspace / "r.json")])
    assert "WARNING" not in capsys.readouterr().out


def test_a_gold_set_still_scores_after_the_corpus_was_re_ingested(
    workspace, monkeypatch
):
    """Same bytes, new ids: the answer is found and the run is unremarkable.

    This is the whole point of hashing the document rather than naming it.
    A re-ingest hands out a new knowledge base id, a new document id and new
    chunk ids while the content is untouched.
    """
    use_order(monkeypatch, ["c-first", "c-answer"])
    monkeypatch.setattr(runtime, "documents_for", lambda kb: [
        {"doc_id": "doc-1", "file_hash": "abc123", "file_name": "rapor.pdf"}
    ])
    stale = {
        **GOLD_ENTRY,
        "document_sha256": "abc123",
        "kb_id": "kb-FROM-AN-OLDER-INGEST",
        "document_id": "upload_FROM_AN_OLDER_INGEST_pdf",
        "correct_chunk_id": "upload_FROM_AN_OLDER_INGEST_pdf:s-chunk-0001",
    }
    gold = write_gold(workspace, [stale])
    out = str(workspace / "run.json")

    assert main(["eval", "--kb", "kb-1", "--gold", gold, "--out", out]) == 0

    run = json.loads((workspace / "run.json").read_text(encoding="utf-8"))
    question = run["questions"][0]
    assert question["rank"] == 2
    assert question["matched_by"] == "unit_ids"
    assert "warnings" not in question, "matching bytes are not a mismatch"
    assert run["document_sha_mismatch"] is False
    # The run records the documents it measured, not the ones the set names.
    assert run["documents"] == [{"document_id": "doc-1", "sha256": "abc123"}]


def test_an_older_gold_set_without_a_hash_still_uses_the_document_id(
    workspace, monkeypatch
):
    """Backward compatibility: sets frozen before hashes were recorded."""
    use_order(monkeypatch, ["c-answer"])
    entry = {k: v for k, v in GOLD_ENTRY.items() if k != "document_sha256"}
    gold = write_gold(workspace, [{**entry, "document_id": "doc-1"}])
    out = str(workspace / "run.json")

    assert main(["eval", "--kb", "kb-1", "--gold", gold, "--out", out]) == 0
    assert json.loads((workspace / "run.json").read_text(encoding="utf-8"))[
        "questions"][0]["rank"] == 1


def test_an_older_bare_list_gold_file_still_loads(workspace, monkeypatch):
    use_order(monkeypatch, ["c-answer"])
    path = workspace / "legacy.json"
    path.write_text(json.dumps([GOLD_ENTRY]), encoding="utf-8")
    out = str(workspace / "run.json")

    assert main(["eval", "--kb", "kb-1", "--gold", str(path), "--out", out]) == 0
    assert json.loads((workspace / "run.json").read_text(encoding="utf-8"))["metrics"]["questions"] == 1


def test_eval_refuses_a_method_the_retriever_cannot_serve(workspace, monkeypatch, capsys):
    use_order(monkeypatch, ["c-answer"])
    gold = write_gold(workspace, [GOLD_ENTRY])

    code = main(["eval", "--kb", "kb-1", "--gold", gold, "--method", "vector"])

    assert code == 2
    assert "not available" in capsys.readouterr().err


def test_an_unknown_knowledge_base_is_a_clean_error(workspace, capsys):
    assert main(["inspect", "--kb", "nope"]) == 2
    assert "No knowledge base" in capsys.readouterr().err


# ---------------------------------------------------------------- compare


def test_compare_reports_a_regression_between_two_runs(workspace, monkeypatch, capsys):
    gold = write_gold(workspace, [GOLD_ENTRY])

    use_order(monkeypatch, ["c-answer", "c-first"])
    main(["eval", "--kb", "kb-1", "--gold", gold, "--out", str(workspace / "before.json")])

    use_order(monkeypatch, ["c-first", "c-second", "c-answer"])
    code = main(["eval", "--kb", "kb-1", "--gold", gold,
                 "--out", str(workspace / "after.json"),
                 "--compare", str(workspace / "before.json"), "--strict"])

    printed = capsys.readouterr().out
    assert "REGRESSION" in printed
    assert "#1 rank 1 -> rank 3" in printed
    assert code == 1


def test_compare_reports_an_improvement_without_failing(workspace, monkeypatch, capsys):
    gold = write_gold(workspace, [GOLD_ENTRY])

    use_order(monkeypatch, ["c-first", "c-second", "c-answer"])
    main(["eval", "--kb", "kb-1", "--gold", gold, "--out", str(workspace / "before.json")])

    use_order(monkeypatch, ["c-answer", "c-first"])
    code = main(["eval", "--kb", "kb-1", "--gold", gold,
                 "--out", str(workspace / "after.json"),
                 "--compare", str(workspace / "before.json"), "--strict"])

    printed = capsys.readouterr().out
    assert "IMPROVEMENT" in printed
    assert "REGRESSION" not in printed
    assert code == 0


# ----------------------------------------------------------- search/inspect


def test_search_shows_rank_score_section_and_page(workspace, monkeypatch, capsys):
    use_order(monkeypatch, ["c-answer", "c-first"])
    assert main(["search", "--kb", "kb-1", "--query", "soru", "--top-k", "2"]) == 0
    printed = capsys.readouterr().out
    assert "[1] score" in printed
    assert "KOSGEB" in printed
    assert "page 36" in printed
    assert "c-answer" in printed


def test_search_refuses_dense_on_a_lexical_only_profile(workspace, monkeypatch, capsys):
    use_order(monkeypatch, ["c-answer"])
    assert main(["search", "--kb", "kb-1", "--query", "q", "--method", "vector"]) == 2
    assert "not available" in capsys.readouterr().err


def test_inspect_summarises_the_configuration(workspace, monkeypatch, capsys):
    use_order(monkeypatch, ["c-answer"])
    monkeypatch.setattr(runtime, "documents_for", lambda kb: [
        {"file_name": "rapor.pdf", "chunk_count": 3, "file_hash": "abc123def456"}
    ])
    monkeypatch.setattr(runtime.gold_manager, "list", lambda kb_id=None: [GOLD_ENTRY])

    assert main(["inspect", "--kb", "kb-one"]) == 0

    printed = capsys.readouterr().out
    assert "kb-one" in printed
    assert "structure_first" in printed
    assert "Uses embeddings            : no" in printed
    assert "Chunks                     : 3" in printed
    assert "rapor.pdf" in printed
