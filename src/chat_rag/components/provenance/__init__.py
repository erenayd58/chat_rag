"""Immutable record of the configuration that produced a stored corpus."""

from .snapshot import (
    FEATURE_SOURCES,
    IMPORTANT_DEPENDENCIES,
    SCHEMA_VERSION,
    build_snapshot,
    capture,
    features_of,
    git_sha,
    is_usable,
    pipeline_facts,
    structured_parser,
    version_facts,
    version_warnings,
)

__all__ = [
    "FEATURE_SOURCES",
    "IMPORTANT_DEPENDENCIES",
    "SCHEMA_VERSION",
    "build_snapshot",
    "capture",
    "features_of",
    "git_sha",
    "is_usable",
    "pipeline_facts",
    "structured_parser",
    "version_facts",
    "version_warnings",
]
