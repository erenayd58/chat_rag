"""The configuration snapshot every output is read against.

A manifest is only worth writing if it is derived from the live wiring rather
than restated by hand, and if it stays honest when part of the environment
cannot be determined. Both are what these tests hold to.
"""

from __future__ import annotations

import pytest

from components.provenance import snapshot as provenance

from cli import manifest as mf
from cli import runtime
from cli.runtime import ChunkView


class Parser:
    """Stands in for the structured PDF parser and its declared repairs."""

    NORMALIZATION_VERSION = "v7-test"
    RUNNING_HEADER_MIN_PAGES = 3
    RECONSTRUCT_VISUAL_GRIDS = True
    DEMOTE_LEAD_IN_HEADINGS = True
    PROMOTE_MISSED_HEADINGS = True
    DEMOTE_CAPTION_HEADINGS = False
    REJOIN_SPLIT_HEADINGS = True
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


class LexicalRetriever:
    requires_document_embeddings = False

    def keyword_search(self, *a, **k):
        return []


class DenseRetriever:
    requires_document_embeddings = True

    def keyword_search(self, *a, **k):
        return []

    def vector_search(self, *a, **k):
        return []

    def hybrid_search(self, *a, **k):
        return []


class Pipeline:
    retrieval_profile = "bm25_only"

    def __init__(self, retriever=None, parser=Parser()):
        self.hybrid_retriever = retriever or LexicalRetriever()
        self.parser_factory = Factory(parser) if parser else None


def chunk(chunk_id="c1", units=("p-1",), pages=(2,), extra=None):
    return ChunkView(
        chunk_id=chunk_id, text="metin", doc_id="doc-1", heading="BASLIK",
        section_paths=[], pages=list(pages), unit_ids=list(units), token_count=3,
        extra=extra or {},
    )


def unit(unit_id="p-1", page=2, text="metin"):
    return {"unit_id": unit_id, "text": text, "source": {"page": page}}


KB = {
    "kb_id": "kb-1",
    "name": "kb-one",
    "chunker": {"type": "structure_first"},
    "vector_db_provider": "chroma",
    "embedding_model_name": "paraphrase-multilingual-MiniLM-L12-v2",
}

DOCUMENT = {
    "doc_id": "doc-1",
    "file_name": "upload_x.pdf",
    "file_hash": "abc123def456789",
    "file_size": 42,
    "ingested_at": "2026-08-25T10:00:00",
    "metadata": {"original_filename": "rapor.pdf"},
}


@pytest.fixture
def wired(monkeypatch):
    """Point the manifest at a pipeline and store we control."""
    def use(pipeline):
        monkeypatch.setattr(runtime, "pipeline_for", lambda kb: pipeline)
        monkeypatch.setattr(
            runtime, "kb_manager",
            type("KB", (), {"collection": staticmethod(lambda kb_id: "kb-1")})(),
        )
        return pipeline
    return use


# ------------------------------------------------------------------- shape


def test_the_manifest_carries_every_block_a_reader_needs(wired):
    wired(Pipeline())
    result = mf.build_manifest(
        KB, document=DOCUMENT, units=[unit()], views=[chunk()]
    )

    assert result["schema_version"] == mf.SCHEMA_VERSION
    assert result["created_at"]
    for block in ("document", "pipeline", "counts", "features", "versions"):
        assert block in result, block
    assert result["knowledge_base"] == {"kb_id": "kb-1", "name": "kb-one"}
    assert result["document"]["sha256"] == "abc123def456789"
    assert result["document"]["filename"] == "rapor.pdf"
    assert result["pipeline"]["chunker"] == "structure_first"
    assert result["pipeline"]["parser"] == "StructuredPDFParser-test"
    assert result["pipeline"]["parser_backend"] == "pymupdf4llm-layout"
    assert result["pipeline"]["normalization_version"] == "v7-test"
    assert result["versions"]["python"]


def test_the_live_parser_object_never_reaches_the_written_manifest(wired):
    wired(Pipeline())
    result = mf.build_manifest(KB, document=DOCUMENT, units=[unit()], views=[chunk()])
    assert "_parser_object" not in result["pipeline"]


# ------------------------------------------------------------- embeddings


def test_a_configured_model_is_reported_even_when_no_embeddings_are_computed(wired):
    """The lexical profile keeps a model in its record and never uses it."""
    wired(Pipeline(LexicalRetriever()))
    pipeline = mf.build_manifest(KB, views=[chunk()])["pipeline"]

    assert pipeline["configured_embedding_model"] == (
        "paraphrase-multilingual-MiniLM-L12-v2"
    )
    assert pipeline["requires_document_embeddings"] is False
    assert pipeline["uses_embeddings"] is False


def test_embeddings_count_as_used_when_the_retriever_needs_them(wired):
    wired(Pipeline(DenseRetriever()))
    pipeline = mf.build_manifest(KB, views=[chunk()])["pipeline"]
    assert pipeline["requires_document_embeddings"] is True
    assert pipeline["uses_embeddings"] is True


def test_a_dense_retriever_with_no_model_configured_uses_none(wired):
    wired(Pipeline(DenseRetriever()))
    kb = {**KB, "embedding_model_name": None}
    pipeline = mf.build_manifest(kb, views=[chunk()])["pipeline"]
    assert pipeline["requires_document_embeddings"] is True
    assert pipeline["uses_embeddings"] is False


# --------------------------------------------------------------- features


def test_features_are_read_off_the_parser_not_restated(wired):
    wired(Pipeline())
    features = mf.build_manifest(KB, views=[chunk()])["features"]

    assert set(features) == set(mf.FEATURE_SOURCES)
    assert features["visual_grid"] is True
    assert features["column_order"] is True
    assert features["running_headers"] is True
    # The one flag turned off in the stand-in parser is reported off.
    assert features["table_captions"] is False


def test_a_missing_parser_reports_unknown_features_and_says_so(wired):
    wired(Pipeline(parser=None))
    result = mf.build_manifest(KB, views=[chunk()])

    assert set(result["features"].values()) == {None}
    assert any("structured PDF parser" in w for w in result["warnings"])
    assert result["pipeline"]["normalization_version"] is None


# ----------------------------------------------------------------- counts


def test_counts_come_from_what_was_actually_passed(wired):
    wired(Pipeline())
    counts = mf.build_manifest(
        KB, units=[unit("p-1"), unit("p-2")],
        views=[chunk(units=["p-1"]), chunk("c2", units=["p-2"])],
    )["counts"]

    assert counts["canonical_units"] == 2
    assert counts["chunks"] == 2
    assert counts["canonical_units_referenced_by_chunks"] == 2


def test_a_fragment_of_a_unit_is_not_counted_as_another_unit(wired):
    wired(Pipeline())
    counts = mf.build_manifest(
        KB, units=[unit("p-1")], views=[chunk(units=["p-1", "p-1#f2"])],
    )["counts"]
    assert counts["canonical_units_referenced_by_chunks"] == 1


# ------------------------------------------------------------------ pages


def test_the_page_count_is_the_one_the_parser_already_recorded(wired):
    """No file is reopened: the count travelled into the chunk metadata."""
    wired(Pipeline())
    views = [chunk(pages=[2, 3], extra={"page_count": 85})]
    result = mf.build_manifest(KB, document=DOCUMENT, units=[unit()], views=views)

    assert result["document"]["pages"] == 85
    assert result["counts"]["pages_with_canonical_units"] == 1


def test_without_a_recorded_count_the_pages_covered_stand_in(wired):
    wired(Pipeline())
    units = [unit("p-1", page=4), unit("p-2", page=5), unit("p-3", page=5)]
    result = mf.build_manifest(KB, document=DOCUMENT, units=units, views=[chunk()])
    assert result["document"]["pages"] == 2


def test_no_document_means_no_document_block(wired):
    wired(Pipeline())
    assert mf.build_manifest(KB, views=[chunk()])["document"] is None


# --------------------------------------------------------------- versions


def test_a_missing_commit_is_a_warning_not_a_crash(wired, monkeypatch):
    wired(Pipeline())
    monkeypatch.setattr(provenance, "git_sha", lambda path=None: None)

    result = mf.build_manifest(KB, views=[chunk()])

    assert result["versions"]["chat_rag_git_sha"] is None
    assert any("chat_rag git commit" in w for w in result["warnings"])


def test_an_unreadable_dependency_version_is_reported_as_missing(wired, monkeypatch):
    wired(Pipeline())

    def explode(name):
        raise LookupError(name)

    monkeypatch.setattr(provenance.metadata, "version", explode)
    result = mf.build_manifest(KB, views=[chunk()])

    assert set(result["versions"]["important_dependencies"].values()) == {None}
    assert any("version unavailable" in w for w in result["warnings"])


def test_every_important_dependency_has_a_slot(wired):
    wired(Pipeline())
    reported = mf.build_manifest(KB, views=[chunk()])["versions"]
    assert set(reported["important_dependencies"]) == set(mf.IMPORTANT_DEPENDENCIES)


# ---------------------------------------------------------- determinism


def test_two_manifests_of_the_same_state_differ_only_in_their_timestamp(wired):
    wired(Pipeline())
    first = mf.build_manifest(KB, document=DOCUMENT, units=[unit()], views=[chunk()])
    second = mf.build_manifest(KB, document=DOCUMENT, units=[unit()], views=[chunk()])

    first.pop("created_at"), second.pop("created_at")
    assert first == second
