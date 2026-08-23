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
