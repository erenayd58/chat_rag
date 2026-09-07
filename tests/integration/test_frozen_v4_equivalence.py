from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from amsc.document.models import EmbeddingBatch, SemanticEmbeddingProvenance

from components.chunker import FrozenV4Chunker
from components.chunker.frozen_v4_chunker import (
    FROZEN_AMSC_COMMIT,
    FROZEN_V4_CONFIG_HASH,
)


FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "frozen_v4"


class SnapshotBoundaryEmbedder:
    """Exact Phase-5 cached vectors, consolidated into one offline fixture."""

    model_id = "intfloat/multilingual-e5-base@default"
    prefix_policy = "symmetric_query"
    model_input_limit = 512
    cache_namespace = "phase5-frozen-boundary-embedding-snapshot"

    def __init__(self, path: Path) -> None:
        with np.load(path, allow_pickle=False) as data:
            hashes = [str(value) for value in data["text_sha256"]]
            vectors = np.asarray(data["vectors"], dtype=np.float32)
            fragment_counts = np.asarray(data["fragment_counts"], dtype=np.int32)
        self._items = {
            key: (vectors[index], int(fragment_counts[index]))
            for index, key in enumerate(hashes)
        }

    def embed_units(self, texts: Sequence[str]) -> EmbeddingBatch:
        vectors: list[np.ndarray] = []
        provenance: list[SemanticEmbeddingProvenance] = []
        for text in texts:
            key = hashlib.sha256(text.encode("utf-8")).hexdigest()
            try:
                vector, fragment_count = self._items[key]
            except KeyError as exc:
                raise AssertionError(
                    "Adapter text is absent from the frozen embedding snapshot"
                ) from exc
            vectors.append(vector)
            provenance.append(
                SemanticEmbeddingProvenance(
                    model_id=self.model_id,
                    prefix_policy=self.prefix_policy,
                    prefix="query: ",
                    model_input_limit=self.model_input_limit,
                    semantic_fragment_count=fragment_count,
                    semantic_pooling="token_weighted_mean",
                    cache_hit=True,
                )
            )
        return EmbeddingBatch(
            vectors=np.vstack(vectors), provenance=tuple(provenance)
        )


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _chunk_boundary_signature(chunk: dict) -> tuple:
    def boundary(value: dict) -> tuple:
        return (
            value.get("reason"),
            value.get("boundary_index"),
            value.get("left_unit_id"),
            value.get("right_unit_id"),
            value.get("selection_strategy"),
        )

    return boundary(chunk["start_boundary"]), boundary(chunk["end_boundary"])


def _boundary_provenance_signature(boundary: dict) -> tuple:
    structural = boundary.get("structural") or {}
    decisions = boundary.get("merge_decisions") or []
    return (
        boundary["boundary_index"],
        boundary["left_unit_id"],
        boundary["right_unit_id"],
        boundary.get("selected_reason"),
        boundary.get("selection_strategy"),
        tuple(structural.get("evidence_types") or []),
        structural.get("structural_assisted_candidate"),
        tuple(
            (
                item.get("proposal_id"),
                item.get("accepted"),
                item.get("rejection_reason"),
                item.get("removed_boundary"),
            )
            for item in decisions
        ),
    )


def test_adapter_and_pinned_v4_are_strictly_equivalent_to_authoritative_a4():
    manifest = json.loads((FIXTURES / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["source_commit"] == FROZEN_AMSC_COMMIT
    assert manifest["config_hash"] == FROZEN_V4_CONFIG_HASH
    for name, expected_sha in manifest["files"].items():
        assert _sha256(FIXTURES / name) == expected_sha

    canonical_rows = _jsonl(FIXTURES / "kkb-2024.units.jsonl")
    expected_chunks = _jsonl(FIXTURES / "authoritative-a4-chunks.jsonl")
    expected_boundaries = _jsonl(FIXTURES / "authoritative-a4-boundaries.jsonl")

    integrated = FrozenV4Chunker(
        boundary_embedder=SnapshotBoundaryEmbedder(
            FIXTURES / "boundary-embeddings.npz"
        )
    ).chunk_canonical(
        text="",
        doc_id="kkb-2024",
        parsed_units=canonical_rows,
    )
    actual_chunks = [item.model_dump(exclude_none=True) for item in integrated.chunks]
    actual_boundaries = [
        item.model_dump(exclude_none=True) for item in integrated.boundaries
    ]

    membership_mismatch = sum(
        (
            actual.get("unit_ids") != expected.get("unit_ids")
            or actual.get("content_unit_ids") != expected.get("content_unit_ids")
        )
        for actual, expected in zip(actual_chunks, expected_chunks, strict=True)
    )
    token_mismatch = sum(
        actual["token_count"] != expected["token_count"]
        for actual, expected in zip(actual_chunks, expected_chunks, strict=True)
    )
    boundary_mismatch = sum(
        _chunk_boundary_signature(actual) != _chunk_boundary_signature(expected)
        for actual, expected in zip(actual_chunks, expected_chunks, strict=True)
    )
    provenance_mismatch = sum(
        _boundary_provenance_signature(actual)
        != _boundary_provenance_signature(expected)
        for actual, expected in zip(
            actual_boundaries, expected_boundaries, strict=True
        )
    )
    text_mismatch = sum(
        actual["text"] != expected["text"]
        for actual, expected in zip(actual_chunks, expected_chunks, strict=True)
    )

    print(
        json.dumps(
            {
                "chunk_count": len(actual_chunks),
                "membership_mismatch": membership_mismatch,
                "token_mismatch": token_mismatch,
                "boundary_mismatch": boundary_mismatch,
                "boundary_provenance_mismatch": provenance_mismatch,
                "text_mismatch": text_mismatch,
            },
            sort_keys=True,
        )
    )

    assert len(actual_chunks) == len(expected_chunks) == 244
    assert len(actual_boundaries) == len(expected_boundaries) == 1328
    assert membership_mismatch == 0
    assert token_mismatch == 0
    assert boundary_mismatch == 0
    assert provenance_mismatch == 0
    assert text_mismatch == 0
