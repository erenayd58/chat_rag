"""Normalize parser output into the canonical AMSC unit model.

The adapter deliberately contains no chunk-boundary logic.  It preserves
structured parser metadata when it exists and otherwise emits ordered
paragraph units from the parser's text blocks.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import re
from typing import Any

from amsc.io import validate_document_units
from amsc.models import RawDocumentUnit, UnitType


class CanonicalUnitAdapter:
    """Map chat_rag parser output to frozen AMSC ``RawDocumentUnit`` values."""

    _KNOWN_TYPES = {
        "heading": UnitType.HEADING,
        "paragraph": UnitType.PARAGRAPH,
        "list": UnitType.LIST,
        "table": UnitType.TABLE,
        # Frozen AMSC represents visual semantic text as a paragraph carrying
        # explicit visual provenance in SourceSpan.
        "visual": UnitType.PARAGRAPH,
    }
    _BLOCK_SEPARATOR = re.compile(r"(?:\r?\n)[ \t]*(?:\r?\n)+")

    def normalize(
        self,
        *,
        text: str,
        document_id: str,
        parsed_units: Sequence[Mapping[str, Any] | RawDocumentUnit] | None = None,
        parser_metadata: Mapping[str, Any] | None = None,
    ) -> list[RawDocumentUnit]:
        if not document_id or not document_id.strip():
            raise ValueError("document_id is required for canonical normalization")

        if parsed_units is not None:
            units = self._from_structured_units(document_id, parsed_units)
        else:
            units = self._from_text_blocks(
                document_id=document_id,
                text=text,
                parser_metadata=parser_metadata or {},
            )

        validate_document_units(units)
        return units

    def _from_structured_units(
        self,
        document_id: str,
        parsed_units: Sequence[Mapping[str, Any] | RawDocumentUnit],
    ) -> list[RawDocumentUnit]:
        if not parsed_units:
            raise ValueError("parsed_units must contain at least one unit")

        normalized: list[RawDocumentUnit] = []
        for index, item in enumerate(parsed_units, start=1):
            if isinstance(item, RawDocumentUnit):
                unit = item
            else:
                row = dict(item)
                unit_type, is_visual = self._unit_type(row)
                source = self._source(row)
                if is_visual:
                    source.setdefault("content_origin", "visual")

                heading_level = row.get("heading_level")
                if unit_type == UnitType.HEADING and heading_level is None:
                    raise ValueError(
                        f"Structured heading unit {index} has no heading_level"
                    )

                unit = RawDocumentUnit(
                    document_id=str(row.get("document_id") or document_id),
                    unit_id=str(row.get("unit_id") or f"{document_id}:unit-{index:05d}"),
                    order=int(row.get("order", index)),
                    text=str(row.get("text") or ""),
                    type=unit_type,
                    heading_level=(int(heading_level) if heading_level is not None else None),
                    section_path=self._section_path(row.get("section_path")),
                    source=source,
                )

            if unit.document_id != document_id:
                raise ValueError(
                    "Structured units must use the requested document_id: "
                    f"{unit.document_id!r} != {document_id!r}"
                )
            normalized.append(unit)
        return normalized

    def _from_text_blocks(
        self,
        *,
        document_id: str,
        text: str,
        parser_metadata: Mapping[str, Any],
    ) -> list[RawDocumentUnit]:
        if not text or not text.strip():
            raise ValueError("Parser output contains no text")

        blocks = self._text_blocks_with_offsets(text)
        units: list[RawDocumentUnit] = []
        for index, (content, start, end) in enumerate(blocks, start=1):
            source: dict[str, Any] = {"char_start": start, "char_end": end}
            # These values describe the parser/file as a whole and are known;
            # page, heading, table, list, and visual metadata are intentionally
            # not inferred from raw text.
            for key in ("parser", "file_name"):
                if parser_metadata.get(key) is not None:
                    source[key] = parser_metadata[key]

            units.append(
                RawDocumentUnit(
                    document_id=document_id,
                    unit_id=f"{document_id}:unit-{index:05d}",
                    order=index,
                    text=content,
                    type=UnitType.PARAGRAPH,
                    section_path=[],
                    source=source,
                )
            )
        return units

    @classmethod
    def _text_blocks_with_offsets(cls, text: str) -> list[tuple[str, int, int]]:
        blocks: list[tuple[str, int, int]] = []
        cursor = 0
        for separator in cls._BLOCK_SEPARATOR.finditer(text):
            cls._append_block(blocks, text, cursor, separator.start())
            cursor = separator.end()
        cls._append_block(blocks, text, cursor, len(text))
        if not blocks:
            raise ValueError("Parser output contains no non-whitespace text blocks")
        return blocks

    @staticmethod
    def _append_block(
        blocks: list[tuple[str, int, int]], text: str, start: int, end: int
    ) -> None:
        raw = text[start:end]
        content = raw.strip()
        if not content:
            return
        leading = len(raw) - len(raw.lstrip())
        trailing = len(raw) - len(raw.rstrip())
        content_start = start + leading
        content_end = end - trailing
        blocks.append((content, content_start, content_end))

    @classmethod
    def _unit_type(cls, row: Mapping[str, Any]) -> tuple[UnitType, bool]:
        declared = row.get("type") or row.get("unit_type")
        if declared is None:
            flags = [
                name
                for name in cls._KNOWN_TYPES
                if row.get(name) is True
            ]
            if len(flags) > 1:
                raise ValueError(f"Conflicting structured unit types: {flags}")
            declared = flags[0] if flags else "paragraph"

        name = str(declared).strip().lower()
        try:
            return cls._KNOWN_TYPES[name], name == "visual"
        except KeyError as exc:
            raise ValueError(f"Unsupported canonical unit type: {declared!r}") from exc

    @staticmethod
    def _source(row: Mapping[str, Any]) -> dict[str, Any]:
        raw_source = row.get("source")
        if raw_source is None:
            source: dict[str, Any] = {}
        elif isinstance(raw_source, Mapping):
            source = dict(raw_source)
        else:
            raise TypeError("source metadata must be a mapping")

        for key in ("page", "block", "char_start", "char_end"):
            if key in row and key not in source:
                source[key] = row[key]
        if "provenance" in row and "provenance" not in source:
            source["provenance"] = row["provenance"]
        return source

    @staticmethod
    def _section_path(value: Any) -> list[str]:
        if value is None:
            return []
        if not isinstance(value, (list, tuple)):
            raise TypeError("section_path must be a list or tuple")
        return [str(item) for item in value]
