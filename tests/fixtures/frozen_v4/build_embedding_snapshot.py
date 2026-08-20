"""Build the compact boundary-embedding fixture from a frozen chunk checkout.

This is a fixture maintenance utility, not an alternate embedding or chunking
implementation. It only consolidates exact vectors already produced by the
frozen AMSC cache so the equivalence test is deterministic and offline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from amsc.config import V4Config
from amsc.embeddings import SemanticFragmentPooler
from amsc.io import load_jsonl_units
from amsc.tokenization import TiktokenTokenCounter
from amsc.units import HeadingAttachmentBuilder, RenderedTokenBudgeter


FROZEN_CANONICAL_SHA256 = (
    "2776742d5bddad7dcf2a03320dca36e6b384e2ba042ab99ccdecce61612720d5"
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    root = args.checkpoint_root.resolve()
    canonical_path = root / "data" / "kkb-2024.units.jsonl"
    if hashlib.sha256(canonical_path.read_bytes()).hexdigest() != FROZEN_CANONICAL_SHA256:
        raise ValueError("Canonical input does not match the frozen Phase-5 SHA")

    config = V4Config.from_yaml(root / "configs" / "v4.yaml")
    counter = TiktokenTokenCounter(config.token_counter.encoding)
    prepared = HeadingAttachmentBuilder(
        RenderedTokenBudgeter(counter, config.tokens.hard_max_tokens)
    ).build(load_jsonl_units(canonical_path))

    embedding = config.boundary_embedding
    model_id = f"{embedding.model}@{embedding.revision or 'default'}"
    namespace = "|".join(
        [
            "semantic-boundary",
            model_id,
            embedding.prefix_policy,
            embedding.prefix,
            str(embedding.max_input_tokens_override or 512),
            SemanticFragmentPooler.POLICY_VERSION,
        ]
    )
    cache_dir = root / embedding.cache_dir

    hashes: list[str] = []
    vectors: list[np.ndarray] = []
    fragment_counts: list[int] = []
    seen: set[str] = set()
    for unit in prepared:
        text = unit.text_for_embedding
        if text is None:
            continue
        text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if text_hash in seen:
            continue
        seen.add(text_hash)
        cache_key = hashlib.sha256(f"{namespace}\0{text}".encode("utf-8")).hexdigest()
        cache_path = cache_dir / f"{cache_key}.npz"
        if not cache_path.exists():
            raise FileNotFoundError(f"Missing frozen cache entry: {cache_path}")
        with np.load(cache_path, allow_pickle=False) as data:
            vector = np.asarray(data["vector"], dtype=np.float32)
            metadata = json.loads(str(data["metadata"].item()))
        hashes.append(text_hash)
        vectors.append(vector)
        fragment_counts.append(int(metadata["semantic_fragment_count"]))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        text_sha256=np.asarray(hashes),
        vectors=np.vstack(vectors),
        fragment_counts=np.asarray(fragment_counts, dtype=np.int32),
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "unique_texts": len(hashes),
                "dimensions": int(vectors[0].shape[0]),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
