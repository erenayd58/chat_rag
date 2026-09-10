from __future__ import annotations

import json

from chat_rag.components.chunker.structural_chunker import StructuralChunker


def unit(unit_id, order, text, unit_type="paragraph", section=("BOLUM A",),
         heading_level=None, page=1):
    return {
        "unit_id": unit_id,
        "order": order,
        "text": text,
        "type": unit_type,
        "heading_level": heading_level,
        "section_path": list(section),
        "source": {"page": page},
    }


def two_section_document():
    """Two headings, so at least one chunk carries a real heading string."""
    return [
        unit("h-1", 1, "**1. BOLUM A**", "heading", ("**1. BOLUM A**",), 2),
        unit("h-2", 2, "Alt baslik", "heading", ("Alt baslik",), 2),
        unit("p-1", 3, "alfa " * 60, section=("Alt baslik",)),
        unit("h-3", 4, "**2. BOLUM B**", "heading", ("**2. BOLUM B**",), 2, page=2),
        unit("p-2", 5, "beta " * 60, section=("**2. BOLUM B**",), page=2),
    ]


def chunks_for(rows):
    chunker = StructuralChunker()
    return (
        chunker.chunk_canonical(text="", doc_id="doc", parsed_units=rows),
        chunker.chunk_text("", "doc", "rapor.pdf", parsed_units=rows),
    )


def test_the_chunkers_own_heading_reaches_the_metadata():
    rows, produced = chunks_for(two_section_document())
    assert rows and produced

    for row, chunk in zip(rows, produced):
        assert chunk.metadata["heading"] == row["heading"]


def test_the_heading_keeps_every_accumulated_line():
    """section_title collapses this to the last path element; heading must not."""
    rows, produced = chunks_for(two_section_document())

    first = produced[0]
    assert first.metadata["heading"] == "**1. BOLUM A**\n\nAlt baslik"
    assert first.section_title == "Alt baslik"
    assert first.metadata["heading"] != first.section_title


def test_section_title_behaviour_is_unchanged():
    _, produced = chunks_for(two_section_document())
    for chunk in produced:
        paths = json.loads(chunk.metadata["section_paths_json"])
        expected = " > ".join(paths[0]) if paths else None
        assert chunk.section_title == expected


def test_a_chunk_without_a_heading_carries_no_heading_key_value():
    """Chroma rejects None, and an absent key reads like an older record."""
    rows = [unit("p-1", 1, "govde metni", section=())]
    _, produced = chunks_for(rows)
    assert produced
    assert produced[0].metadata["heading"] is None
    assert produced[0].section_title is None


def test_the_heading_survives_the_none_filter_the_vector_db_applies():
    """chroma_vectordb drops None-valued metadata before writing."""
    _, produced = chunks_for(two_section_document())
    written = {
        key: value
        for key, value in produced[0].metadata.items()
        if value is not None
    }
    assert written["heading"] == "**1. BOLUM A**\n\nAlt baslik"

    _, headless = chunks_for([unit("p-1", 1, "govde", section=())])
    written = {
        key: value
        for key, value in headless[0].metadata.items()
        if value is not None
    }
    assert "heading" not in written


def test_no_other_metadata_key_changed():
    _, produced = chunks_for(two_section_document())
    assert set(produced[0].metadata) == {
        "chunker_type",
        "created_at",
        "word_count",
        "heading",
        "unit_ids_json",
        "pages_json",
        "token_count",
        "section_paths_json",
        "split_strategies_json",
    }
