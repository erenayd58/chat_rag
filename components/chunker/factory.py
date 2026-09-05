"""Chunker selection while preserving the existing legacy implementation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from core.exceptions import ConfigurationException

from . import registry
from .base import BaseChunker
from .frozen_v4_chunker import FrozenV4Chunker
from .semantic_chunker import SemanticChunker
from .structural_chunker import StructuralChunker


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

    chunker = registry.resolve(chunker_type)
    if chunker is None:
        raise ConfigurationException(
            f"Unsupported chunker type {chunker_type!r}; expected {registry.expected()}"
        )
    if params and not chunker.accepts_params:
        raise ConfigurationException(
            f"{chunker.params_refusal} tuning params"
            + ("; use the pinned config" if chunker.id == "v4" else "")
        )
    if chunker.id == "legacy":
        return SemanticChunker(
            chunk_size=params.get("chunk_size", settings.chunk_size),
            chunk_overlap=params.get("chunk_overlap", settings.chunk_overlap),
            min_chunk_size=params.get("min_chunk_size", settings.min_chunk_size),
            use_semantic_segmentation=params.get("use_semantic_segmentation", True),
            use_embedding_segmentation=params.get("use_embedding_segmentation", False),
            semantic_threshold=params.get("semantic_threshold", 0.6),
            semantic_window=params.get("semantic_window", 1),
        )
    if chunker.id == "v4":
        return FrozenV4Chunker()
    return StructuralChunker()
