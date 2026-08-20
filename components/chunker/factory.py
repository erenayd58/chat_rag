"""Chunker selection while preserving the existing legacy implementation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from core.exceptions import ConfigurationException

from .base import BaseChunker
from .frozen_v4_chunker import FrozenV4Chunker
from .semantic_chunker import SemanticChunker


_LEGACY_NAMES = {"legacy", "semanticchunker", "semantic_chunker"}
_V4_NAMES = {"v4", "frozenv4chunker", "frozen_v4_chunker"}


def create_chunker(
    settings: Any,
    chunker_config: Mapping[str, Any] | None = None,
) -> BaseChunker:
    config = chunker_config
    if config is None:
        config = getattr(settings, "kb_chunker_config", None)

    if config:
        chunker_type = str(config.get("type", "legacy"))
        params = dict(config.get("params") or {})
    else:
        chunker_type = str(getattr(settings, "chunker_type", "legacy"))
        params = {}

    normalized = chunker_type.strip().lower()
    if normalized in _LEGACY_NAMES:
        return SemanticChunker(
            chunk_size=params.get("chunk_size", settings.chunk_size),
            chunk_overlap=params.get("chunk_overlap", settings.chunk_overlap),
            min_chunk_size=params.get("min_chunk_size", settings.min_chunk_size),
            use_semantic_segmentation=params.get("use_semantic_segmentation", True),
            use_embedding_segmentation=params.get("use_embedding_segmentation", False),
            semantic_threshold=params.get("semantic_threshold", 0.6),
            semantic_window=params.get("semantic_window", 1),
        )

    if normalized in _V4_NAMES:
        if params:
            raise ConfigurationException(
                "Frozen V4 accepts no runtime tuning params; use the pinned config"
            )
        return FrozenV4Chunker()

    raise ConfigurationException(
        f"Unsupported chunker type {chunker_type!r}; expected 'legacy' or 'v4'"
    )
