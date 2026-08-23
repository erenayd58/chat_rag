"""Structured PDF parser producing canonical AMSC units.

This is a thin adapter over the already-pinned ``amsc.checkpoint_adapter``
layout extraction used by the research benchmark.  It deliberately implements
no parsing logic of its own: the point is that the application and the
benchmark consume the *same* canonical representation, so a chunker measured
offline behaves the same way in production.

Without this parser the pipeline falls back to flat ``page.get_text()`` output
split on blank lines, which carries no heading, list, table, section or page
provenance -- and every structure-aware chunker setting silently becomes a
no-op.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from .base import BaseParser
from core.exceptions import RAGException


class StructuredPDFParser(BaseParser):
    """PDF parser that recovers headings, lists, tables and page provenance."""

    SUPPORTED_EXTENSIONS = (".pdf",)

    def __init__(self, layout_profile_path: Optional[str] = None) -> None:
        """
        Args:
            layout_profile_path: Optional explicit checkpoint layout profile.
                When omitted the frozen extractor's own reading order is used,
                which is correct for ordinary portrait documents.  Supply a
                profile for two-column landscape spreads (for example the KKB
                annual reports).

        Raises:
            RAGException: If the pinned pymupdf4llm layout backend is missing,
                so ``ParserFactory`` skips this parser and keeps the existing
                plain-text ``PDFParser`` behaviour.
        """
        try:
            from amsc.checkpoint_adapter import (
                CheckpointLayoutUnavailableError,
                load_layout_backend,
            )
            from amsc.prepare_full_checkpoint import extract_full_canonical_units
        except ImportError as exc:
            raise RAGException(
                "amsc-poc is required for structured PDF parsing. "
                "Install with: pip install -e '.[checkpoint]' in the chunk repo"
            ) from exc

        try:
            load_layout_backend()
        except CheckpointLayoutUnavailableError as exc:
            raise RAGException(
                f"pymupdf4llm layout backend unavailable: {exc}"
            ) from exc

        self._extract_full_canonical_units = extract_full_canonical_units
        self.layout_profile_path = layout_profile_path
        self.parser_backend = "pymupdf4llm-layout"
        # Layout extraction is the expensive step (minutes on a large report)
        # and the ingestion path asks for text and units back to back, so cache
        # the most recent extraction keyed by file identity + profile.
        self._cache_key: Optional[tuple] = None
        self._cache_units = None

    def supports(self, file_path: str) -> bool:
        return os.path.splitext(file_path)[1].lower() in self.SUPPORTED_EXTENSIONS

    def get_name(self) -> str:
        return "StructuredPDFParser-pymupdf4llm"

    def parse_units(self, file_path: str, **kwargs) -> Optional[List[Dict[str, Any]]]:
        """Extract canonical units as plain dictionaries.

        ``document_id`` is intentionally omitted from every row so that
        ``CanonicalUnitAdapter`` fills in the pipeline's own ``doc_id``.
        """
        units = self._canonical_units(file_path, **kwargs)
        rows: List[Dict[str, Any]] = []
        for unit in units:
            row = unit.model_dump(mode="json", exclude_none=True)
            row.pop("document_id", None)
            rows.append(row)
        return rows

    def parse(self, file_path: str, **kwargs) -> str:
        """Return the same content as flat text, for the legacy chunker path."""
        units = self._canonical_units(file_path, **kwargs)
        return "\n\n".join(unit.text for unit in units if unit.text)

    def get_metadata(self, file_path: str, **kwargs) -> Dict[str, Any]:
        metadata = super().get_metadata(file_path, **kwargs)
        metadata["parser_backend"] = self.parser_backend
        try:
            import fitz

            with fitz.open(file_path) as document:
                metadata["page_count"] = document.page_count
        except Exception:
            # Page count is informational only; never fail ingestion for it.
            pass
        return metadata

    def _canonical_units(self, file_path: str, **kwargs):
        if not os.path.exists(file_path):
            raise RAGException(f"File not found: {file_path}")
        profile = kwargs.get("layout_profile_path", self.layout_profile_path)

        stat = os.stat(file_path)
        cache_key = (
            os.path.abspath(file_path),
            stat.st_mtime_ns,
            stat.st_size,
            profile,
        )
        if self._cache_key == cache_key and self._cache_units is not None:
            return self._cache_units

        try:
            extraction = self._extract_full_canonical_units(
                input_path=file_path,
                layout_profile_path=profile,
                document_id="document",
            )
        except Exception as exc:
            raise RAGException(f"Structured PDF parsing failed: {exc}") from exc
        if not extraction.units:
            raise RAGException("Structured PDF parsing produced no units")

        self._cache_key = cache_key
        self._cache_units = extraction.units
        return extraction.units
