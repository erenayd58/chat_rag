"""Where a table's searchable rendering is allowed to be read.

Deep Analysis renders a table it carries into a second representation. It is a
retrieval aid and nothing else: BM25 reads it beside the raw markdown, the
dense leg reads it instead, and the answer context and every citation keep
reading the document's own text.
"""

from __future__ import annotations

from types import SimpleNamespace

from components.chunker import deep_analysis as product
from components.chunker.structural_chunker import (
    HARD_MAX_TOKENS,
    MIN_TOKENS,
    SOFT_MAX_TOKENS,
    TARGET_TOKENS,
    StructuralChunker,
)
from components.retriever.hybrid_rrf_retriever import _lexical_text
from core.models import DocumentChunk


TABLE = "\n".join([
    "|**Ogrenim durumu**|**Ogrenim durumu**|",
    "|---|---|",
    "|**Durum**|**Oran (%)**|",
    "|Lisans|77|",
])


def _deterministic_configuration():
    """Deep Analysis with no model: the deterministic contract alone, which is
    all this seam needs -- the rendering is produced without one."""
    settings = SimpleNamespace(
        deep_analysis_model="", deep_analysis_verifier_model="",
        deep_analysis_endpoint="", deep_analysis_api_key_env="DEEP_SEAM_KEY",
        deep_analysis_use_llm=False, deep_analysis_verify=False,
        deep_analysis_timeout=5.0, deep_analysis_concurrency=1,
    )
    return product.build_configuration(settings, product.deep_config(
        min_tokens=MIN_TOKENS, target_tokens=TARGET_TOKENS,
        soft_max_tokens=SOFT_MAX_TOKENS, hard_max_tokens=HARD_MAX_TOKENS,
    ))


def _chunk(content="RAW", search_text=None):
    metadata = {"search_text": search_text} if search_text is not None else {}
    return DocumentChunk(
        chunk_id="c1", content=content, doc_id="d", doc_title="t",
        chunk_index=0, total_chunks=1, metadata=metadata,
    )


def test_the_rendering_is_read_from_metadata_and_is_optional():
    assert _chunk(search_text="Lisans: Oran (%) = 77").search_text == "Lisans: Oran (%) = 77"
    assert _chunk().search_text is None
    # A chunker that never wrote the key, and a row that had none, read alike.
    bare = DocumentChunk(chunk_id="c", content="x", doc_id="d", doc_title="t",
                         chunk_index=0, total_chunks=1)
    assert bare.search_text is None
    assert _chunk(search_text="").search_text is None


def test_bm25_reads_the_rendering_beside_the_markdown_never_instead_of_it():
    """Additive on purpose: a term that matched the raw table still matches."""
    indexed = _lexical_text(_chunk(content="|Lisans|77|", search_text="Lisans: Oran (%) = 77"))
    assert "|Lisans|77|" in indexed, "the raw table is still indexed"
    assert "Oran (%) = 77" in indexed, "and the rendering is indexed with it"
    assert _lexical_text(_chunk(content="|Lisans|77|")) == "|Lisans|77|"


def test_the_deep_chunker_carries_the_rendering_and_leaves_the_text_alone():
    units = [
        {"unit_id": "h-1", "order": 1, "text": "**1. INSAN KAYNAKLARI**", "type": "heading",
         "heading_level": 2, "section_path": ["**1. INSAN KAYNAKLARI**"], "source": {"page": 1}},
        {"unit_id": "p-1", "order": 2, "text": "alfa " * 80, "type": "paragraph",
         "heading_level": None, "section_path": ["**1. INSAN KAYNAKLARI**"], "source": {"page": 1}},
        {"unit_id": "t-1", "order": 3, "text": TABLE, "type": "table",
         "heading_level": None, "section_path": ["**1. INSAN KAYNAKLARI**"], "source": {"page": 1}},
    ]
    chunks, _report = StructuralChunker().chunk_text_deep(
        "", "doc", "rapor.pdf",
        configuration=_deterministic_configuration(),
        provider=None, verifier_provider=None, parsed_units=units,
    )
    carrying = [chunk for chunk in chunks if chunk.search_text]
    assert carrying, "the chunk holding the table carries a rendering"
    for chunk in carrying:
        assert "Oran (%) = 77" in chunk.search_text
        assert chunk.search_text not in chunk.content, "content is the document's own text"
        assert "|" in chunk.content, "the raw markdown is untouched"


def test_the_standard_chunker_writes_no_rendering():
    """Standard is what it was: the key is simply never written."""
    units = [
        {"unit_id": "h-1", "order": 1, "text": "**1. BOLUM**", "type": "heading",
         "heading_level": 2, "section_path": ["**1. BOLUM**"], "source": {"page": 1}},
        {"unit_id": "t-1", "order": 2, "text": TABLE, "type": "table",
         "heading_level": None, "section_path": ["**1. BOLUM**"], "source": {"page": 1}},
    ]
    chunks = StructuralChunker().chunk_text("", "doc", "rapor.pdf", parsed_units=units)
    assert chunks, "the document still chunks"
    assert all(chunk.search_text is None for chunk in chunks)
