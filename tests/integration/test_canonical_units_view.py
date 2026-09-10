"""The parser-output debug view must show canonical units exactly as stored.

It exists to separate a parser reading-order problem from a chunker one, so the
one thing it must never do is reorder, merge or clean anything on the way out.
"""

from __future__ import annotations

import json

import pytest

from chat_rag.components.parsers.canonical_units_store import (
    find_cache_file,
    load_units,
    select_units,
    summarize_source,
)


def _unit(order, unit_id, unit_type="paragraph", page=1, text="metin", **source):
    row = {
        "unit_id": unit_id,
        "order": order,
        "text": text,
        "type": unit_type,
        "section_path": ["Bolum"],
        "source": {"page": page, "block": order, **source},
    }
    if unit_type == "heading":
        row["heading_level"] = 2
    return row


@pytest.fixture
def cache_dir(tmp_path):
    units = []
    order = 1
    for page in range(1, 11):
        for index in range(20):
            units.append(
                _unit(order, f"p-{order:05d}", page=page, text=f"sayfa {page} birim {index}")
            )
            order += 1
    path = tmp_path / "abc123.jsonl"
    path.write_text(
        "".join(json.dumps(u, ensure_ascii=False) + "\n" for u in units),
        encoding="utf-8",
        newline="\n",
    )
    # A second, unrelated document must not be matched.
    other = [_unit(i, f"x-{i:05d}", page=1) for i in range(1, 6)]
    (tmp_path / "def456.jsonl").write_text(
        "".join(json.dumps(u, ensure_ascii=False) + "\n" for u in other),
        encoding="utf-8",
        newline="\n",
    )
    return tmp_path


def test_units_keep_their_stored_order(cache_dir):
    units = load_units(cache_dir / "abc123.jsonl")
    assert [u["order"] for u in units] == list(range(1, 201))

    window, total, _ = select_units(units, offset=0, limit=200)
    assert total == 200
    assert [u["order"] for u in window] == list(range(1, 201))
    assert [u["unit_id"] for u in window] == [u["unit_id"] for u in units]


def test_pagination_windows_without_gaps_or_overlap(cache_dir):
    units = load_units(cache_dir / "abc123.jsonl")
    seen = []
    offset = 0
    while True:
        window, total, _ = select_units(units, offset=offset, limit=50)
        if not window:
            break
        seen.extend(u["order"] for u in window)
        offset += 50
        if offset >= total:
            break
    assert seen == list(range(1, 201))


def test_page_filter_selects_only_those_pages(cache_dir):
    units = load_units(cache_dir / "abc123.jsonl")
    window, total, pages = select_units(units, page_from=4, page_to=6, limit=500)
    assert pages == [4, 5, 6]
    assert total == 60
    assert all(4 <= u["source"]["page"] <= 6 for u in window)
    assert [u["order"] for u in window] == sorted(u["order"] for u in window)


def test_type_filter(cache_dir):
    units = load_units(cache_dir / "abc123.jsonl")
    units.append(_unit(999, "h-00999", unit_type="heading", page=1, text="Baslik"))
    window, total, _ = select_units(units, unit_type="heading", limit=50)
    assert total == 1
    assert window[0]["heading_level"] == 2


def test_limit_is_bounded(cache_dir):
    units = load_units(cache_dir / "abc123.jsonl")
    window, _, _ = select_units(units, offset=0, limit=100000)
    assert len(window) <= 500


def test_cache_file_matched_by_unit_ids(cache_dir):
    match = find_cache_file(["p-00001", "p-00002", "p-00003"], cache_dir)
    assert match is not None
    assert match.name == "abc123.jsonl"


def test_fragment_suffixes_still_match(cache_dir):
    match = find_cache_file(["p-00001#f1", "p-00002#f2"], cache_dir)
    assert match is not None
    assert match.name == "abc123.jsonl"


def test_unknown_units_match_nothing(cache_dir):
    assert find_cache_file(["zzz-00001", "zzz-00002"], cache_dir) is None
    assert find_cache_file([], cache_dir) is None


def test_source_summary_keeps_layout_metadata_and_drops_empties():
    row = _unit(
        1,
        "p-00001",
        page=7,
        logical_page_side="left",
        logical_column="1",
        layout_band=2,
        layout_reading_order_index=5,
        layout_bbox_physical=[1.0, 2.0, 3.0, 4.0],
        raw_layout_class="text",
        content_origin=None,
    )
    summary = summarize_source(row)
    assert summary["page"] == 7
    assert summary["logical_page_side"] == "left"
    assert summary["layout_bbox_physical"] == [1.0, 2.0, 3.0, 4.0]
    assert summary["layout_reading_order_index"] == 5
    assert "content_origin" not in summary
