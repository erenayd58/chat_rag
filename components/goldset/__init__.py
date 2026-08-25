"""Gold-set component: manually confirmed retrieval answers."""

from .manager import (
    EVIDENCE_LIMIT,
    SCHEMA_VERSION,
    GoldSetManager,
    entry_id_for,
    normalize_question,
)

__all__ = [
    "GoldSetManager",
    "SCHEMA_VERSION",
    "EVIDENCE_LIMIT",
    "entry_id_for",
    "normalize_question",
]
