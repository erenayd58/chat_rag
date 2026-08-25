"""The record of how a corpus was produced, captured once and never edited.

A snapshot is worth having only if it is written exactly when an ingest
succeeds, describes the bytes it was taken against, and can never turn a
working ingest into a failed one. Those three are what is held to here.
"""

from __future__ import annotations

import json

import pytest

from components.provenance import snapshot as provenance
from utils import DocumentTracker


class Parser:
    NORMALIZATION_VERSION = "v7-test"
    RUNNING_HEADER_MIN_PAGES = 3
    RECONSTRUCT_VISUAL_GRIDS = True
    DEMOTE_LEAD_IN_HEADINGS = True
    PROMOTE_MISSED_HEADINGS = True
    DEMOTE_CAPTION_HEADINGS = True
    REJOIN_SPLIT_HEADINGS = False
    DEMOTE_SENTENCE_HEADINGS = True
    parser_backend = "pymupdf4llm-layout"

    class _Profile:
        reading_order = "column-major-left-to-right"

    _spread_profile = _Profile()

    def get_name(self):
        return "StructuredPDFParser-test"


class Factory:
    def __init__(self, parser):
        self._parser = parser

    def get_parser(self, path):
        return self._parser


class Retriever:
    requires_document_embeddings = False

    def keyword_search(self, *a, **k):
        return []


class Pipeline:
    retrieval_profile = "bm25_only"

    def __init__(self, parser=None):
        self.hybrid_retriever = Retriever()
        self.parser_factory = Factory(parser or Parser())


KB = {
    "name": "kb-one",
    "chunker": {"type": "structure_first"},
    "vector_db_provider": "chroma",
    "embedding_model_name": "paraphrase-multilingual-MiniLM-L12-v2",
}


def taken():
    return provenance.build_snapshot(
        Pipeline(), KB, kb_id="kb-1", storage_path="/store"
    )


# ------------------------------------------------------------------ shape


def test_a_snapshot_records_everything_needed_to_explain_a_corpus():
    snapshot = taken()

    assert snapshot["schema_version"] == provenance.SCHEMA_VERSION
    assert snapshot["captured_at"]
    assert snapshot["kb_id"] == "kb-1"

    pipeline = snapshot["pipeline"]
    for field in ("parser", "parser_backend", "normalization_version", "chunker",
                  "retrieval_profile", "retriever", "uses_embeddings",
                  "requires_document_embeddings", "configured_embedding_model",
                  "vector_db_provider"):
        assert field in pipeline, field
    assert pipeline["parser"] == "StructuredPDFParser-test"
    assert pipeline["normalization_version"] == "v7-test"
    assert pipeline["chunker"] == "structure_first"
    assert pipeline["retriever"] == "Retriever"

    assert set(snapshot["features"]) == set(provenance.FEATURE_SOURCES)
    assert snapshot["features"]["visual_grid"] is True
    assert snapshot["features"]["split_headings"] is False

    versions = snapshot["versions"]
    assert "chat_rag_git_sha" in versions and "amsc_git_sha" in versions
    assert set(versions["important_dependencies"]) == set(
        provenance.IMPORTANT_DEPENDENCIES
    )


def test_the_configured_model_and_whether_it_is_used_stay_separate():
    pipeline = taken()["pipeline"]
    assert pipeline["configured_embedding_model"]
    assert pipeline["requires_document_embeddings"] is False
    assert pipeline["uses_embeddings"] is False


def test_the_document_hash_is_left_for_the_tracker_to_stamp():
    """Nothing here reads the file; the tracker already hashed it."""
    assert taken()["document_sha256"] is None


def test_a_snapshot_survives_a_round_trip_through_json():
    """It is stored in the tracker file, so it has to be plain JSON."""
    snapshot = taken()
    restored = json.loads(json.dumps(snapshot))

    assert restored == snapshot
    assert provenance.is_usable(restored)


# ------------------------------------------------------------- robustness


def test_capture_never_turns_a_successful_ingest_into_a_failure(monkeypatch):
    def explode(*a, **k):
        raise RuntimeError("no")

    monkeypatch.setattr(provenance, "version_facts", explode)
    assert provenance.capture(Pipeline(), KB, kb_id="kb-1") is None


def test_a_pipeline_without_a_parser_still_yields_a_usable_snapshot():
    class Bare:
        retrieval_profile = "legacy"
        hybrid_retriever = Retriever()

    snapshot = provenance.build_snapshot(Bare(), KB)
    assert provenance.is_usable(snapshot)
    assert snapshot["pipeline"]["parser"] is None
    assert set(snapshot["features"].values()) == {None}


@pytest.mark.parametrize(
    "stored", [None, {}, "text", {"schema_version": 99, "pipeline": {}},
               {"schema_version": provenance.SCHEMA_VERSION}]
)
def test_an_unreadable_snapshot_is_refused_rather_than_half_read(stored):
    assert provenance.is_usable(stored) is False


def test_a_snapshot_this_build_wrote_is_readable():
    assert provenance.is_usable(taken()) is True


# --------------------------------------------------------------- tracker


@pytest.fixture
def tracker(tmp_path):
    return DocumentTracker(str(tmp_path / "tracked.json"))


@pytest.fixture
def document(tmp_path):
    path = tmp_path / "rapor.pdf"
    path.write_bytes(b"%PDF-1.7 pretend")
    return str(path)


def test_the_tracker_stores_the_snapshot_with_the_document(tracker, document):
    tracker.mark_as_ingested(
        file_path=document, doc_id="doc-1", chunk_count=2,
        kb_id="kb-1", pipeline_snapshot=taken(),
    )

    stored = tracker.get_document_by_doc_id("doc-1")["pipeline_snapshot"]
    assert stored["pipeline"]["normalization_version"] == "v7-test"
    assert tracker.get_all_documents()[0]["pipeline_snapshot"] == stored


def test_the_tracker_stamps_the_hash_of_the_bytes_it_recorded(tracker, document):
    tracker.mark_as_ingested(
        file_path=document, doc_id="doc-1", chunk_count=2,
        pipeline_snapshot=taken(),
    )
    record = tracker.get_document_by_doc_id("doc-1")

    assert record["pipeline_snapshot"]["document_sha256"] == record["file_hash"]
    assert len(record["file_hash"]) == 64


def test_stamping_does_not_mutate_the_snapshot_the_caller_passed(tracker, document):
    snapshot = taken()
    tracker.mark_as_ingested(
        file_path=document, doc_id="doc-1", chunk_count=1, pipeline_snapshot=snapshot,
    )
    assert snapshot["document_sha256"] is None


def test_the_snapshot_reaches_disk_and_reads_back(tracker, document, tmp_path):
    tracker.mark_as_ingested(
        file_path=document, doc_id="doc-1", chunk_count=1, pipeline_snapshot=taken(),
    )
    reopened = DocumentTracker(str(tmp_path / "tracked.json"))
    assert provenance.is_usable(
        reopened.get_document_by_doc_id("doc-1")["pipeline_snapshot"]
    )


def test_an_ingest_that_captured_nothing_records_no_snapshot(tracker, document):
    """The legacy shape, and what a caller that skips capture leaves behind."""
    tracker.mark_as_ingested(file_path=document, doc_id="doc-1", chunk_count=1)
    assert tracker.get_document_by_doc_id("doc-1")["pipeline_snapshot"] is None


def test_a_record_written_before_snapshots_existed_still_reads(tracker, tmp_path):
    path = tmp_path / "tracked.json"
    path.write_text(json.dumps({
        "C:/old.pdf": {"doc_id": "old-1", "file_hash": "abc", "chunk_count": 3,
                       "file_size": 1, "ingested_at": "2026-01-01T00:00:00",
                       "kb_id": "kb-1", "metadata": {}}
    }), encoding="utf-8")
    reopened = DocumentTracker(str(path))

    record = reopened.get_document_by_doc_id("old-1")
    assert record["chunk_count"] == 3
    assert record["pipeline_snapshot"] is None
