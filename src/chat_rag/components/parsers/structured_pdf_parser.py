"""Structured PDF parser producing canonical AMSC units.

This is a thin adapter over the already-pinned ``amsc.canonical.adapter``
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

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import BaseParser
from chat_rag.core.exceptions import RAGException


class StructuredPDFParser(BaseParser):
    """PDF parser that recovers headings, lists, tables and page provenance."""

    SUPPORTED_EXTENSIONS = (".pdf",)

    #: Eight fixes to the canonical stream are applied here; bump this marker
    #: whenever they change so a cache written by an older, wronger version is
    #: never reused.
    NORMALIZATION_VERSION = (
        "v7-column-order+running-headers+visual-grid+lead-ins"
        "+numbered-headings+table-captions+split-headings+sentence-headings"
    )

    #: A heading whose text leads a logical page on this many distinct physical
    #: pages is running furniture, not a section start. Positional and
    #: text-agnostic.
    RUNNING_HEADER_MIN_PAGES = 3

    #: Rebuild label -> value pairs inside KPI card grids from page geometry.
    #: The extractor otherwise serializes a picture's text by vertical
    #: position, which silently swaps two values whenever the cards of one row
    #: use different font sizes. Production only: the research canonical is
    #: frozen and must stay byte-identical.
    RECONSTRUCT_VISUAL_GRIDS = True

    #: A sentence that introduces the bullets under it ("Uygulamayla;") is set
    #: apart typographically, so the layout model reports it as a section
    #: header and it opens a section named after a sentence fragment. Demoting
    #: it to body text keeps those bullets in the section they belong to.
    #: Production only: the research canonical is frozen.
    DEMOTE_LEAD_IN_HEADINGS = True

    #: The layout model reports a few numbered section titles as body text, so
    #: the section they open never starts and the tables under them are filed
    #: against the previous note. Production only: the research canonical is
    #: frozen.
    PROMOTE_MISSED_HEADINGS = True

    #: A period label printed above its table is reported as a section header,
    #: so a section named after a table caption swallows the notes below it.
    #: Production only: the research canonical is frozen.
    DEMOTE_CAPTION_HEADINGS = True

    #: One printed heading line can arrive as two layout boxes, the section
    #: number split away from its own title.
    REJOIN_SPLIT_HEADINGS = True

    #: A standfirst set in display type is reported as a section header, so a
    #: whole sentence opens a section. Production only: the research canonical
    #: is frozen.
    DEMOTE_SENTENCE_HEADINGS = True

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
            from amsc.canonical.adapter import (
                CheckpointLayoutUnavailableError,
                load_layout_backend,
            )
            from amsc.canonical.layout import CheckpointLayoutProfile
            from amsc.canonical.prepare import extract_full_canonical_units
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
        # The extractor already treats a landscape physical page as a
        # left/right spread. Without an ordering policy the layout model reads
        # each horizontal band left to right, which interleaves the two columns
        # and detaches a heading from the body below it. Applying column-major
        # order to those logical pages follows the spread assumption the
        # extractor already makes; it is geometry, not a per-document rule.
        self._spread_profile = CheckpointLayoutProfile(
            profile_id="landscape-spread-two-column",
            spread_mode="left-right",
            logical_columns=2,
            reading_order="column-major-left-to-right",
        )
        self.parser_backend = "pymupdf4llm-layout"
        # Layout extraction is the expensive step (minutes on a large report)
        # and the ingestion path asks for text and units back to back, so cache
        # the most recent extraction keyed by file identity + profile.
        self._cache_key: Optional[tuple] = None
        self._cache_units = None
        # Layout extraction is ~4.5 s per logical page and is the dominant
        # ingestion cost; profiling showed it is entirely third-party ONNX
        # inference with no algorithmic slack. Re-extracting a document we have
        # already seen is pure waste, so the canonical units are also cached on
        # disk, keyed by the PDF content hash.
        # Resolved through config.paths, so STRUCTURED_PARSER_CACHE still
        # wins where it is set and a deployment's data root supplies it where
        # it is not -- rather than defaulting to a directory relative to
        # whatever the working directory happens to be.
        from chat_rag.config import paths as _paths

        self._disk_cache = Path(_paths.canonical_cache())

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

        disk_path = self._disk_cache_path(file_path, profile)
        cached = self._read_disk_cache(disk_path)
        if cached is not None:
            self._cache_key, self._cache_units = cache_key, cached
            return cached

        try:
            extraction = self._extract_full_canonical_units(
                input_path=file_path,
                layout_profile_path=profile or self._spread_profile,
                document_id="document",
                running_header_min_pages=self.RUNNING_HEADER_MIN_PAGES,
                reconstruct_visual_grids=self.RECONSTRUCT_VISUAL_GRIDS,
                demote_lead_in_headings=self.DEMOTE_LEAD_IN_HEADINGS,
                promote_missed_headings=self.PROMOTE_MISSED_HEADINGS,
                demote_caption_headings=self.DEMOTE_CAPTION_HEADINGS,
                rejoin_split_headings_enabled=self.REJOIN_SPLIT_HEADINGS,
                demote_sentence_headings_enabled=self.DEMOTE_SENTENCE_HEADINGS,
            )
        except Exception as exc:
            raise RAGException(f"Structured PDF parsing failed: {exc}") from exc
        if not extraction.units:
            raise RAGException("Structured PDF parsing produced no units")

        self._cache_key = cache_key
        self._cache_units = extraction.units
        self._write_disk_cache(disk_path, extraction.units)
        return extraction.units

    def _disk_cache_path(self, file_path: str, profile: Optional[str]) -> Path:
        digest = hashlib.sha256()
        with open(file_path, "rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
        digest.update(b"|")
        digest.update((profile or "").encode("utf-8"))
        digest.update(b"|pymupdf4llm-layout-v1|")
        digest.update(self.NORMALIZATION_VERSION.encode("utf-8"))
        return self._disk_cache / f"{digest.hexdigest()}.jsonl"

    @staticmethod
    def _read_disk_cache(path: Path):
        from amsc.document.models import RawDocumentUnit

        if not path.is_file():
            return None
        try:
            units = tuple(
                RawDocumentUnit.model_validate_json(line)
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
        except Exception:
            # A corrupt or stale cache entry must never break ingestion.
            return None
        return units or None

    @staticmethod
    def _write_disk_cache(path: Path, units) -> None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = "".join(
                unit.model_dump_json(exclude_none=True) + chr(10) for unit in units
            )
            temporary = path.with_suffix(".tmp")
            temporary.write_text(payload, encoding="utf-8", newline=chr(10))
            temporary.replace(path)
        except Exception:
            # Caching is an optimisation; failing to write must not fail ingestion.
            pass
