"""Read-only access to canonical units already produced by the parser.

This exists purely for inspection: it never parses a PDF and never rewrites the
cache. Units are returned in their stored order with their stored fields, so a
reading-order problem can be attributed to the parser or to the chunker by
comparing this view against the chunk view.

The parser caches units keyed by a streaming hash of the PDF bytes, which
cannot be recomputed once the uploaded temp file is gone. A document is instead
matched to its cache entry by the unit ids its chunks carry.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

DEFAULT_CACHE_DIR = Path(os.getenv("STRUCTURED_PARSER_CACHE", ".cache/canonical-units"))

# Matching a document to its cache file walks every cached file once; the result
# is memoised because neither the files nor a document's unit ids change.
_MATCH_CACHE: Dict[Tuple[str, str], Optional[str]] = {}


def _base_unit_id(unit_id: str) -> str:
    """Strip the ``#f2`` fragment suffix the chunker may append."""
    return unit_id.split("#", 1)[0]


def load_units(path: str | Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _unit_ids_of(path: Path) -> set:
    return {row.get("unit_id") for row in load_units(path)}


def find_cache_file(
    wanted_unit_ids: Iterable[str],
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
) -> Optional[Path]:
    """Return the cache file whose units cover ``wanted_unit_ids``."""
    wanted = {_base_unit_id(u) for u in wanted_unit_ids if u}
    if not wanted:
        return None
    directory = Path(cache_dir)
    if not directory.is_dir():
        return None

    key = (str(directory.resolve()), ",".join(sorted(list(wanted)[:8])))
    if key in _MATCH_CACHE:
        cached = _MATCH_CACHE[key]
        return Path(cached) if cached else None

    best: Optional[Path] = None
    best_overlap = 0.0
    probe = set(list(wanted)[:200])
    for candidate in sorted(directory.glob("*.jsonl")):
        try:
            available = _unit_ids_of(candidate)
        except Exception:
            continue
        overlap = len(probe & available) / len(probe) if probe else 0.0
        if overlap > best_overlap:
            best, best_overlap = candidate, overlap

    # A partial match means a different document; require near-complete cover.
    resolved = best if best_overlap >= 0.9 else None
    _MATCH_CACHE[key] = str(resolved) if resolved else None
    return resolved


def _page_of(row: Dict[str, Any]) -> Optional[int]:
    source = row.get("source") or {}
    page = source.get("page")
    return int(page) if isinstance(page, (int, float)) else None


def select_units(
    units: Sequence[Dict[str, Any]],
    *,
    page_from: Optional[int] = None,
    page_to: Optional[int] = None,
    unit_type: Optional[str] = None,
    offset: int = 0,
    limit: int = 100,
) -> Tuple[List[Dict[str, Any]], int, List[int]]:
    """Filter and window units without reordering or altering them.

    Returns ``(window, total_after_filter, pages_present)``.
    """
    selected = list(units)
    if page_from is not None or page_to is not None:
        low = page_from if page_from is not None else -(10**9)
        high = page_to if page_to is not None else 10**9
        selected = [
            row for row in selected
            if (_page_of(row) is not None and low <= _page_of(row) <= high)
        ]
    if unit_type:
        selected = [row for row in selected if row.get("type") == unit_type]

    pages = sorted({p for p in (_page_of(row) for row in selected) if p is not None})
    offset = max(0, int(offset))
    limit = max(1, min(int(limit), 500))
    return selected[offset : offset + limit], len(selected), pages


def summarize_source(row: Dict[str, Any]) -> Dict[str, Any]:
    """Compact, lossless-by-selection view of the stored source metadata."""
    source = row.get("source") or {}
    keys = (
        "page",
        "physical_page",
        "logical_page_side",
        "logical_column",
        "layout_band",
        "layout_reading_order_index",
        "layout_box_index",
        "block",
        "char_start",
        "char_end",
        "raw_layout_class",
        "content_origin",
        "extraction_method",
        "reading_order_policy",
        "layout_bbox_logical",
        "layout_bbox_physical",
        "picture_bbox",
        "parser",
        "file_name",
    )
    return {key: source[key] for key in keys if key in source and source[key] is not None}
