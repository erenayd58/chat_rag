"""Cover the real PDF -> parser -> adapter -> chunker seam.

This is the seam where production chunk quality was being lost: the pipeline
parsed PDFs into flat text blocks, so the canonical units reaching the chunker
carried no heading, list, table, ``section_path`` or page provenance, and every
structure-aware setting silently degraded into token-budget cutting.

``test_frozen_v4_equivalence`` bypasses the parser by injecting canonical rows
directly, and ``test_ingestion_retrieval_smoke`` uses a toy string, so neither
exercised this chain. These tests do.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pytest

from amsc.models import EmbeddingBatch, SemanticEmbeddingProvenance

from components.chunker import FrozenV4Chunker
from components.chunker.normalization_adapter import CanonicalUnitAdapter
from components.parsers.parser_factory import ParserFactory
from components.parsers.pdf_parser import PDFParser
from core.exceptions import RAGException


FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "seam"
FIXTURE_PDF = FIXTURE_DIR / "mini-report.pdf"
MANIFEST = json.loads((FIXTURE_DIR / "manifest.json").read_text(encoding="utf-8"))

# Checkpoint-adapter unit ids look like ``h-00001`` / ``t-00005``. The flat
# text-block fallback emits ``<doc>:unit-00001``. This is an unambiguous
# discriminator for which adapter branch ran.
STRUCTURED_UNIT_ID = re.compile(r"^[hplt]-\d{5}(#fragment-\d+)?$")


def _structured_parser():
    """Return a StructuredPDFParser, or skip when the layout backend is absent."""
    from components.parsers.structured_pdf_parser import StructuredPDFParser

    try:
        return StructuredPDFParser()
    except RAGException as exc:
        pytest.skip(f"pymupdf4llm layout backend unavailable: {exc}")


class DeterministicBoundaryEmbedder:
    model_id = "test:deterministic-boundary@1"
    prefix_policy = "symmetric_query"
    model_input_limit = 512
    cache_namespace = "test-deterministic-boundary"

    @staticmethod
    def _one(text: str) -> np.ndarray:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        vector = np.asarray([digest[0] + 1, digest[1] + 1, digest[2] + 1], dtype=float)
        return vector / np.linalg.norm(vector)

    def embed_units(self, texts):
        vectors = np.vstack([self._one(text) for text in texts])
        provenance = tuple(
            SemanticEmbeddingProvenance(
                model_id=self.model_id,
                prefix_policy=self.prefix_policy,
                prefix="query: ",
                model_input_limit=self.model_input_limit,
                semantic_fragment_count=1,
                semantic_pooling="token_weighted_mean",
            )
            for _ in texts
        )
        return EmbeddingBatch(vectors=vectors, provenance=provenance)


def test_fixture_pdf_is_unmodified():
    digest = hashlib.sha256(FIXTURE_PDF.read_bytes()).hexdigest()
    assert digest == MANIFEST["sha256"], (
        "Seam fixture changed; regenerate manifest.json deliberately via "
        "tests/fixtures/seam/build_mini_report.py"
    )


def test_structured_parser_recovers_document_structure():
    rows = _structured_parser().parse_units(str(FIXTURE_PDF))

    assert rows, "Structured parser produced no units"
    types = {row["type"] for row in rows}
    for required in ("heading", "paragraph", "table", "list"):
        assert required in types, f"Structured parser lost {required!r} units: {types}"

    # Page provenance is the single most diagnostic property production lost.
    assert all(row.get("source", {}).get("page") is not None for row in rows)
    assert any(row.get("section_path") for row in rows)

    # document_id is deliberately omitted so the adapter fills in the caller's
    # doc_id rather than fighting it.
    assert all("document_id" not in row for row in rows)


def test_adapter_takes_the_structured_branch():
    rows = _structured_parser().parse_units(str(FIXTURE_PDF))
    units = CanonicalUnitAdapter().normalize(
        text="", document_id="seam-doc", parsed_units=rows
    )

    assert all(unit.document_id == "seam-doc" for unit in units)
    assert all(STRUCTURED_UNIT_ID.match(unit.unit_id) for unit in units), sorted(
        {unit.unit_id for unit in units}
    )[:5]
    assert all(unit.source.page is not None for unit in units)
    assert any(unit.section_path for unit in units)
    assert {unit.type.value for unit in units} >= {
        "heading",
        "paragraph",
        "table",
        "list",
    }


def test_factory_prefers_the_structured_parser_for_pdf():
    factory = ParserFactory()
    parser = factory.get_parser(str(FIXTURE_PDF))
    assert parser is not None
    if parser.get_name().startswith("StructuredPDFParser"):
        rows = factory.parse_units(str(FIXTURE_PDF))
        assert rows and all("type" in row for row in rows)
    else:  # layout backend missing: must degrade, not crash
        assert factory.parse_units(str(FIXTURE_PDF)) is None


def test_plain_pdf_parser_path_stays_untyped():
    """Pin the known-lossy fallback so the contrast stays documented.

    This is the behaviour that made every structural chunker setting inert in
    production. It is intentionally still available for backends that cannot
    recover structure; it must simply no longer be the default for PDFs.
    """
    text = PDFParser(use_unstructured=False).parse(str(FIXTURE_PDF))
    units = CanonicalUnitAdapter().normalize(text=text, document_id="seam-doc")

    assert units
    assert all(unit.type.value == "paragraph" for unit in units)
    assert all(unit.section_path == [] for unit in units)
    assert all(unit.source.page is None for unit in units)
    assert all(unit.unit_id.startswith("seam-doc:unit-") for unit in units)


def test_structured_units_reach_the_chunker_and_change_its_decisions():
    rows = _structured_parser().parse_units(str(FIXTURE_PDF))
    chunker = FrozenV4Chunker(boundary_embedder=DeterministicBoundaryEmbedder())

    structured = chunker.chunk_canonical(text="", doc_id="seam-doc", parsed_units=rows)
    assert structured.chunks

    # Section provenance survives all the way into the emitted chunks.
    assert any(chunk.section_paths for chunk in structured.chunks)

    # At least one chunk begins on a heading, which is impossible when the
    # chunker only ever sees untyped paragraphs.
    heading_ids = {row["unit_id"] for row in rows if row["type"] == "heading"}
    assert any(
        set(chunk.unit_ids[:1]) & heading_ids for chunk in structured.chunks
    ), "No chunk starts on a heading"


def test_pipeline_forwards_parsed_units_to_the_chunker():
    """The exact wiring that was missing: parse_units -> chunk_text.

    Regression guard for the defect where ``_from_structured_units`` existed,
    was unit-tested, and was never called by production.
    """
    from pipeline.rag_pipeline import RAGPipeline

    recorded = {}

    class RecordingChunker:
        def chunk_text(self, text, doc_id, doc_title, document_summary=None, **kwargs):
            recorded["parsed_units"] = kwargs.get("parsed_units")
            return []

        def get_name(self):
            return "RecordingChunker"

    pipeline = RAGPipeline.__new__(RAGPipeline)
    pipeline.chunker = RecordingChunker()
    pipeline.retrieval_profile = "benchmark_aligned"
    pipeline.embedding_model = None

    pipeline.ingest_document(
        document_text="irrelevant",
        doc_id="seam-doc",
        doc_title="seam",
        additional_metadata={},
        parsed_units=[
            {"unit_id": "p-00001", "order": 1, "text": "x", "type": "paragraph"}
        ],
    )

    assert recorded["parsed_units"] is not None, (
        "ingest_document dropped parsed_units before reaching the chunker"
    )


def test_structured_parser_does_not_extract_twice_per_file(tmp_path):
    """Ingestion asks for text then units; layout extraction must run once.

    Without this cache a large report is extracted twice per upload, which on
    the KKB annual report is several extra minutes of wall clock.
    """
    parser = _structured_parser()
    parser._disk_cache = tmp_path / "canonical-units"

    calls = {"n": 0}
    inner = parser._extract_full_canonical_units

    def counting(**kwargs):
        calls["n"] += 1
        return inner(**kwargs)

    parser._extract_full_canonical_units = counting

    parser.parse(str(FIXTURE_PDF))
    parser.parse_units(str(FIXTURE_PDF))
    parser.parse_units(str(FIXTURE_PDF))

    assert calls["n"] == 1, f"Layout extraction ran {calls['n']} times, expected 1"
