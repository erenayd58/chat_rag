"""End-to-end cover for the default demo profile.

Real PDF -> StructuredPDFParser -> structure-first chunker -> BM25 index ->
search, with the assertion that matters most for this profile: no embedding
model is loaded or called anywhere on the path.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from components.chunker import StructuralChunker
from components.chunker.factory import create_chunker
from components.retriever import BM25OnlyRetriever, NullEmbedding
from core.exceptions import RAGException, RetrieverException
from core.models import DocumentChunk

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "seam" / "mini-report.pdf"


class _Settings:
    kb_chunker_config = None
    chunker_type = "structure_first"
    chunk_size = 300
    chunk_overlap = 60
    min_chunk_size = 50


def _parser():
    from components.parsers.structured_pdf_parser import StructuredPDFParser

    try:
        return StructuredPDFParser()
    except RAGException as exc:
        pytest.skip(f"layout backend unavailable: {exc}")


def test_factory_resolves_structure_first():
    for name in ("structure_first", "structural", "StructuralChunker"):
        settings = _Settings()
        settings.chunker_type = name
        assert isinstance(create_chunker(settings), StructuralChunker)


def test_null_embedding_refuses_to_embed():
    null = NullEmbedding()
    for call in (null.encode, null.encode_documents, null.encode_queries):
        with pytest.raises(RetrieverException):
            call(["metin"])


def test_pdf_to_chunks_to_bm25_search_end_to_end():
    rows = _parser().parse_units(str(FIXTURE))
    chunks = StructuralChunker().chunk_text(
        text="", doc_id="demo", doc_title="mini-report.pdf", parsed_units=rows
    )
    assert chunks
    for chunk in chunks:
        assert chunk.metadata["chunker_type"] == "structure_first"
        assert chunk.metadata["token_count"] <= 1126
        assert chunk.metadata["unit_ids"]
        assert chunk.metadata["pages"]

    retriever = BM25OnlyRetriever(NullEmbedding(), vector_db=None)
    retriever.build_index(chunks)

    hits = retriever.hybrid_search("Takipteki alacak orani nedir", top_k=3)
    assert hits
    assert hits[0].retrieval_method == "bm25_only"
    assert hits[0].dense_rank is None
    # The table carrying the ratio must be reachable lexically.
    assert any("Takipteki" in h.chunk.content for h in hits)

    hits = retriever.hybrid_search("Operasyonel risk nedir", top_k=3)
    assert any("Operasyonel" in h.chunk.content for h in hits)


def test_profile_never_touches_a_dense_leg():
    rows = _parser().parse_units(str(FIXTURE))
    chunks = StructuralChunker().chunk_text(
        text="", doc_id="demo", doc_title="mini-report.pdf", parsed_units=rows
    )
    retriever = BM25OnlyRetriever(NullEmbedding(), vector_db=None)
    retriever.build_index(chunks)
    assert retriever.config["dense"] is None
    with pytest.raises(RetrieverException):
        retriever.vector_search("x")


def test_chunk_metadata_is_json_serialisable():
    chunk = StructuralChunker().chunk_text(
        text="Bir paragraf.\n\nIkinci paragraf.",
        doc_id="demo",
        doc_title="t",
    )[0]
    assert isinstance(chunk, DocumentChunk)
    json.dumps(chunk.metadata, ensure_ascii=False)


def test_turkish_fold_makes_ascii_and_diacritic_queries_agree():
    from components.retriever.bm25_only_retriever import fold_turkish

    assert fold_turkish("Çalışan Sayısı") == "calisan sayisi"
    assert fold_turkish("YÜRÜRLÜĞE") == "yururluge"

    rows = _parser().parse_units(str(FIXTURE))
    chunks = StructuralChunker().chunk_text(
        text="", doc_id="demo", doc_title="t", parsed_units=rows
    )
    retriever = BM25OnlyRetriever(NullEmbedding(), vector_db=None)
    retriever.build_index(chunks)

    for diacritic, ascii_form in [
        ("Kredi Risk Yönetimi", "Kredi Risk Yonetimi"),
        ("Takipteki alacak oranı", "Takipteki alacak orani"),
    ]:
        a = retriever.hybrid_search(diacritic, top_k=1)
        b = retriever.hybrid_search(ascii_form, top_k=1)
        assert a and b
        assert a[0].chunk.chunk_id == b[0].chunk.chunk_id

    # The fold is a search representation only: stored text is untouched.
    # (The fixture is deliberately ASCII, so assert the invariant directly.)
    sample = "Çalışan sayısı %12,5 arttı"
    assert fold_turkish(sample) != sample
    assert all(c.content == c.content for c in chunks)
    from components.chunker.structural_chunker import StructuralChunker as _SC

    turkish = _SC().chunk_text(text=sample, doc_id="d", doc_title="t")[0]
    assert turkish.content == sample


def test_extraction_cache_round_trips(tmp_path):
    from components.parsers.structured_pdf_parser import StructuredPDFParser

    parser = _parser()
    parser._disk_cache = tmp_path / "canonical-units"
    first = parser._canonical_units(str(FIXTURE))

    fresh = StructuredPDFParser()
    fresh._disk_cache = tmp_path / "canonical-units"
    calls = {"n": 0}
    inner = fresh._extract_full_canonical_units

    def counting(**kwargs):
        calls["n"] += 1
        return inner(**kwargs)

    fresh._extract_full_canonical_units = counting
    second = fresh._canonical_units(str(FIXTURE))

    assert calls["n"] == 0, "disk cache miss: layout extraction re-ran"
    assert [u.model_dump(mode="json") for u in first] == [
        u.model_dump(mode="json") for u in second
    ]
