"""Which indexing chunker a knowledge base is built with."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from core.exceptions import ConfigurationException

from . import registry
from .base import BaseChunker
from .frozen_v4_chunker import FrozenV4Chunker
from .structural_chunker import StructuralChunker


def create_chunker(
    settings: Any,
    chunker_config: Mapping[str, Any] | None = None,
) -> BaseChunker:
    config = chunker_config
    if config is None:
        config = getattr(settings, "kb_chunker_config", None)

    if config:
        chunker_type = str(config.get("type", registry.DEFAULT_ID))
        params = dict(config.get("params") or {})
    else:
        chunker_type = str(getattr(settings, "chunker_type", registry.DEFAULT_ID))
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
    if chunker.id == "v4":
        return FrozenV4Chunker()
    return StructuralChunker()
