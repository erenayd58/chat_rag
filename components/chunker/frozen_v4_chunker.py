"""chat_rag adapter for the immutable Phase-5 AMSC V4/A4 package."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from pathlib import Path
from typing import Any

from amsc.embedding.cache import FileEmbeddingCache
from amsc.chunking.adaptive.config import V4Config
from amsc.embedding.boundary import (
    CachedSemanticBoundaryEmbedder,
    SemanticBoundaryEmbedder,
    SentenceTransformerBoundaryEmbedder,
)
from amsc.document.models import ChunkingResult, RawDocumentUnit
from amsc.document.tokenization import TiktokenTokenCounter, TokenCounter
from amsc.chunking.adaptive.v4 import V4Chunker

from core.exceptions import ChunkerException, ConfigurationException
from core.models import DocumentChunk

from .base import BaseChunker
from .normalization_adapter import CanonicalUnitAdapter


FROZEN_AMSC_COMMIT = "1e7f7186c13729c739ccb3170da0892f7350cb27"
FROZEN_AMSC_DEPENDENCY = (
    "amsc-poc @ git+https://github.com/erenayd58/chunk.git@"
    + FROZEN_AMSC_COMMIT
)
FROZEN_V4_CONFIG_HASH = "f29f805deee9189c"
DEFAULT_FROZEN_CONFIG = (
    Path(__file__).resolve().parents[2] / "config" / "frozen_v4.yaml"
)


def load_frozen_v4_config(path: str | Path = DEFAULT_FROZEN_CONFIG) -> V4Config:
    config = V4Config.from_yaml(path)
    if config.config_hash != FROZEN_V4_CONFIG_HASH:
        raise ConfigurationException(
            "Frozen V4 config drift detected: "
            f"expected {FROZEN_V4_CONFIG_HASH}, got {config.config_hash}"
        )
    if config.algorithm.version != "v4" or config.ablation.composition != "a4":
        raise ConfigurationException("Frozen integration requires V4 composition A4")
    return config


class FrozenV4Chunker(BaseChunker):
    """Thin representation adapter around the frozen package's ``V4Chunker``."""

    def __init__(
        self,
        *,
        config_path: str | Path = DEFAULT_FROZEN_CONFIG,
        adapter: CanonicalUnitAdapter | None = None,
        token_counter: TokenCounter | None = None,
        boundary_embedder: SemanticBoundaryEmbedder | None = None,
    ) -> None:
        self.config_path = Path(config_path)
        self.config = load_frozen_v4_config(self.config_path)
        self.adapter = adapter or CanonicalUnitAdapter()
        self._token_counter = token_counter
        self._boundary_embedder = boundary_embedder
        self._core: V4Chunker | None = None

    def chunk_canonical(
        self,
        *,
        text: str,
        doc_id: str,
        parsed_units: Sequence[Mapping[str, Any] | RawDocumentUnit] | None = None,
        parser_metadata: Mapping[str, Any] | None = None,
    ) -> ChunkingResult:
        raw_units = self.adapter.normalize(
            text=text,
            document_id=doc_id,
            parsed_units=parsed_units,
            parser_metadata=parser_metadata,
        )
        return self._core_chunker().chunk(raw_units)

    def chunk_text(
        self,
        text: str,
        doc_id: str,
        doc_title: str,
        document_summary: str = None,
        **kwargs: Any,
    ) -> list[DocumentChunk]:
        try:
            result = self.chunk_canonical(
                text=text,
                doc_id=doc_id,
                parsed_units=kwargs.get("parsed_units"),
                parser_metadata=kwargs.get("parser_metadata"),
            )
            return self._to_document_chunks(result, doc_title, document_summary)
        except (ConfigurationException, ValueError, TypeError, AssertionError):
            raise
        except Exception as exc:
            raise ChunkerException(f"Frozen V4 chunking failed: {exc}") from exc

    def get_name(self) -> str:
        return "FrozenV4Chunker"

    def get_config(self) -> dict[str, Any]:
        return {
            "type": "v4",
            "dependency": FROZEN_AMSC_DEPENDENCY,
            "commit": FROZEN_AMSC_COMMIT,
            "config_hash": self.config.config_hash,
            "ablation": self.config.ablation.composition,
            "config": self.config.model_dump(mode="json"),
        }

    def _core_chunker(self) -> V4Chunker:
        if self._core is not None:
            return self._core

        token_counter = self._token_counter or TiktokenTokenCounter(
            self.config.token_counter.encoding
        )
        boundary_embedder = self._boundary_embedder
        if boundary_embedder is None:
            embedding = self.config.boundary_embedding
            delegate = SentenceTransformerBoundaryEmbedder.from_pretrained(
                embedding.model,
                revision=embedding.revision,
                device=embedding.device,
                prefix=embedding.prefix,
                prefix_policy=embedding.prefix_policy,
                max_input_tokens_override=embedding.max_input_tokens_override,
                normalize_embeddings=embedding.normalize_embeddings,
            )
            cache_dir = embedding.cache_dir
            if not cache_dir.is_absolute():
                cache_dir = Path(__file__).resolve().parents[2] / cache_dir
            boundary_embedder = CachedSemanticBoundaryEmbedder(
                delegate, FileEmbeddingCache(cache_dir)
            )

        self._core = V4Chunker(
            config=self.config,
            token_counter=token_counter,
            boundary_embedder=boundary_embedder,
        )
        return self._core

    @staticmethod
    def _to_document_chunks(
        result: ChunkingResult,
        doc_title: str,
        document_summary: str | None,
    ) -> list[DocumentChunk]:
        total = len(result.chunks)
        converted: list[DocumentChunk] = []
        for index, chunk in enumerate(result.chunks):
            section_title = None
            if chunk.section_paths:
                section_title = " > ".join(chunk.section_paths[0]) or None

            metadata = {
                "chunker_type": "v4",
                "amsc_algorithm_version": chunk.algorithm_version,
                "amsc_ablation_id": chunk.ablation_id or "a4",
                "amsc_frozen_commit": FROZEN_AMSC_COMMIT,
                "amsc_config_hash": chunk.config_hash,
                "amsc_token_count": chunk.token_count,
                "amsc_token_counter_id": chunk.token_counter_id,
                "amsc_start_boundary_reason": chunk.start_boundary.reason,
                "amsc_end_boundary_reason": chunk.end_boundary.reason,
                "amsc_start_boundary_index": chunk.start_boundary.boundary_index,
                "amsc_end_boundary_index": chunk.end_boundary.boundary_index,
                "amsc_unit_ids_json": json.dumps(chunk.unit_ids, ensure_ascii=False),
                "amsc_content_unit_ids_json": json.dumps(
                    chunk.content_unit_ids, ensure_ascii=False
                ),
                "amsc_source_spans_json": json.dumps(
                    chunk.source_spans, ensure_ascii=False, sort_keys=True
                ),
                "amsc_start_boundary_json": chunk.start_boundary.model_dump_json(
                    exclude_none=True
                ),
                "amsc_end_boundary_json": chunk.end_boundary.model_dump_json(
                    exclude_none=True
                ),
                "word_count": len(chunk.text.split()),
            }
            converted.append(
                DocumentChunk(
                    chunk_id=chunk.chunk_id,
                    content=chunk.text,
                    doc_id=chunk.document_id,
                    doc_title=doc_title,
                    chunk_index=index,
                    total_chunks=total,
                    section_title=section_title,
                    previous_context=(
                        result.chunks[index - 1].text[:200] if index > 0 else None
                    ),
                    next_context=(
                        result.chunks[index + 1].text[:200]
                        if index + 1 < total
                        else None
                    ),
                    document_summary=document_summary,
                    metadata=metadata,
                )
            )
        return converted
