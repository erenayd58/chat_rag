from __future__ import annotations

from amsc.document.models import UnitType

from components.chunker import CanonicalUnitAdapter


def test_plain_parser_text_becomes_ordered_paragraphs_without_invented_metadata():
    text = "First paragraph.\n\n  Second paragraph.  "
    units = CanonicalUnitAdapter().normalize(
        text=text,
        document_id="demo",
        parser_metadata={"parser": "TextParser", "file_name": "demo.txt"},
    )

    assert [unit.text for unit in units] == ["First paragraph.", "Second paragraph."]
    assert [unit.order for unit in units] == [1, 2]
    assert all(unit.type == UnitType.PARAGRAPH for unit in units)
    assert all(unit.heading_level is None for unit in units)
    assert all(unit.section_path == [] for unit in units)
    assert units[0].source.char_start == 0
    assert units[0].source.char_end == len("First paragraph.")
    assert units[1].source.char_start == text.index("Second")
    assert units[1].source.model_dump(exclude_none=True)["parser"] == "TextParser"
    assert "page" not in units[1].source.model_dump(exclude_none=True)


def test_structured_parser_metadata_maps_without_format_chunking_rules():
    rows = [
        {
            "unit_id": "h-1",
            "order": 1,
            "text": "Section",
            "type": "heading",
            "heading_level": 2,
            "section_path": ["Section"],
            "source": {"page": 3, "block": "title"},
        },
        {
            "unit_id": "l-1",
            "order": 2,
            "text": "- item",
            "list": True,
            "section_path": ["Section"],
            "page": 3,
        },
        {
            "unit_id": "t-1",
            "order": 3,
            "text": "A | B",
            "type": "table",
            "section_path": ["Section"],
            "provenance": {"parser_element": "Table"},
        },
        {
            "unit_id": "v-1",
            "order": 4,
            "text": "Chart labels",
            "type": "visual",
            "section_path": ["Section"],
            "source": {"page": 4},
        },
    ]

    units = CanonicalUnitAdapter().normalize(
        text="unused", document_id="demo", parsed_units=rows
    )

    assert [unit.type for unit in units] == [
        UnitType.HEADING,
        UnitType.LIST,
        UnitType.TABLE,
        UnitType.PARAGRAPH,
    ]
    assert units[0].heading_level == 2
    assert units[1].source.page == 3
    assert units[2].source.model_dump(exclude_none=True)["provenance"] == {
        "parser_element": "Table"
    }
    assert units[3].source.model_dump(exclude_none=True)["content_origin"] == "visual"
