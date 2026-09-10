"""Where a table's searchable rendering is allowed to be read.

Deep Analysis renders a table it carries into a second representation. It is a
retrieval aid and nothing else: both legs read it beside the raw markdown --
never instead of it -- and the answer context and every citation keep reading
the document's own text.
"""

from __future__ import annotations

from types import SimpleNamespace

from chat_rag.components.chunker import deep_analysis as product
from chat_rag.components.chunker.structural_chunker import (
    HARD_MAX_TOKENS,
    MIN_TOKENS,
    SOFT_MAX_TOKENS,
    TARGET_TOKENS,
    StructuralChunker,
)
from chat_rag.components.context.assembler import TABLE_VIEW_HEADER, assemble_context
from chat_rag.components.retriever.hybrid_rrf_retriever import _lexical_text
from chat_rag.core.models import DocumentChunk, RetrievalResult


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


def _chunk(content="RAW", search_text=None, table_view=None):
    metadata = {}
    if search_text is not None:
        metadata["search_text"] = search_text
    if table_view is not None:
        metadata["table_view"] = table_view
    return DocumentChunk(
        chunk_id="c1", content=content, doc_id="d", doc_title="t",
        chunk_index=0, total_chunks=1, metadata=metadata,
    )


def _context(chunk):
    result = RetrievalResult(chunk=chunk, score=1.0, retrieval_method="hybrid_rrf", rank=0)
    return assemble_context([result], max_tokens=8000, max_sources=8,
                            neighbor=None, expand_neighbors=False)


def test_the_rendering_is_read_from_metadata_and_is_optional():
    assert _chunk(search_text="Lisans: Oran (%) = 77").search_text == "Lisans: Oran (%) = 77"
    assert _chunk().search_text is None
    # A chunker that never wrote the key, and a row that had none, read alike.
    bare = DocumentChunk(chunk_id="c", content="x", doc_id="d", doc_title="t",
                         chunk_index=0, total_chunks=1)
    assert bare.search_text is None
    assert _chunk(search_text="").search_text is None


def test_both_legs_read_the_rendering_beside_the_markdown_never_instead_of_it():
    """Additive on purpose: a term that matched the raw table still matches,
    and the sentences a table sits under stay in the chunk's own vector."""
    carrying = _chunk(content="|Lisans|77|", search_text="Lisans: Oran (%) = 77")
    for indexed in (_lexical_text(carrying), carrying.retrieval_text):
        assert "|Lisans|77|" in indexed, "the raw table is still indexed"
        assert "Oran (%) = 77" in indexed, "and the rendering is indexed with it"
    # One representation, read by both legs.
    assert _lexical_text(carrying) == carrying.retrieval_text
    # A chunk with no rendering -- every Standard and Markdown chunk -- is
    # indexed as exactly its own content, byte for byte.
    plain = _chunk(content="|Lisans|77|")
    assert _lexical_text(plain) == "|Lisans|77|"
    assert plain.retrieval_text == "|Lisans|77|"


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


def test_the_answer_context_reads_the_table_reading_beside_the_raw_table():
    """The reading is an aid, never the record: the document's own table stays
    above it, it is marked as derived, and the budget pays for it."""
    chunk = _chunk(content="|Lisans|77|", table_view="Oran (%): Lisans = 77")
    bundle = _context(chunk)

    assert "|Lisans|77|" in bundle.text, "the document's own table is still there"
    assert TABLE_VIEW_HEADER in bundle.text, "and the reading says it is derived"
    assert "Oran (%): Lisans = 77" in bundle.text
    assert bundle.text.index("|Lisans|77|") < bundle.text.index(TABLE_VIEW_HEADER)
    assert bundle.token_count > _context(_chunk(content="|Lisans|77|")).token_count


def test_a_chunk_with_no_reading_renders_exactly_what_it_always_did():
    """Every Standard and Markdown chunk, and every table Deep could not read
    with certainty: the context is the chunk's own text and nothing else."""
    bundle = _context(_chunk(content="|Lisans|77|"))
    assert bundle.text.endswith("|Lisans|77|")
    assert TABLE_VIEW_HEADER not in bundle.text


def test_the_reading_is_for_the_answer_only_and_is_never_indexed():
    """Retrieval reads content and the search rendering; the table reading is
    a context aid and stays out of both legs."""
    chunk = _chunk(content="|Lisans|77|", search_text="Lisans: Oran (%) = 77",
                   table_view="Oran (%): Lisans = 77")
    assert "Oran (%): Lisans = 77" not in chunk.retrieval_text
    assert "Oran (%): Lisans = 77" not in _lexical_text(chunk)
    assert _chunk(content="x").table_view is None
