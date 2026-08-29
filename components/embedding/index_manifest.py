"""Which embedding space a vector store holds, and whether it is the current one.

A vector store is only meaningful together with the model that wrote it.
This manifest sits beside the store (``embedding_index.json``) and records
provider, model, dimension and fingerprint at the last write, so the
retriever can tell a compatible store from one that needs re-indexing --
and never compares a Qwen3-Embedding query vector against vectors another
model produced, or against the placeholder vectors a lexical-only profile
stores.

States reported by :func:`index_status`:

* ``empty`` -- no chunks stored yet; anything may be written;
* ``compatible`` -- the manifest fingerprint and the stored width match the
  current embedding;
* ``no_dense_index`` -- chunks exist but were stored without real vectors
  (a lexical-only profile's placeholders) or without a manifest;
* ``reindex_required`` -- chunks exist and were embedded by a different
  model or at a different width.

Only counts, ids and names are stored; nothing here can carry a key.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Dict, Optional

MANIFEST_NAME = "embedding_index.json"

STATE_EMPTY = "empty"
STATE_COMPATIBLE = "compatible"
STATE_NO_DENSE_INDEX = "no_dense_index"
STATE_REINDEX_REQUIRED = "reindex_required"

#: Width of the placeholder vectors a lexical-only ingestion stores.
PLACEHOLDER_DIMENSION = 1


def manifest_path(store_path: str) -> str:
    return os.path.join(store_path, MANIFEST_NAME)


def read_manifest(store_path: Optional[str]) -> Optional[Dict[str, Any]]:
    if not store_path:
        return None
    path = manifest_path(store_path)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def write_manifest(
    store_path: str,
    identity: Dict[str, Any],
    *,
    dimension: int,
    chunk_count: int,
) -> Dict[str, Any]:
    """Record the space the store now holds. ``identity`` is an embedding's
    ``describe()``; the key variable name is kept, the key never exists here."""
    os.makedirs(store_path, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "embedding_provider": identity.get("provider"),
        "embedding_model": identity.get("model"),
        "embedding_endpoint": identity.get("endpoint"),
        "embedding_dimension": int(dimension),
        "embedding_fingerprint": identity.get("fingerprint"),
        "chunk_count": int(chunk_count),
        "written_at": datetime.now().isoformat(timespec="seconds"),
    }
    tmp = manifest_path(store_path) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
    os.replace(tmp, manifest_path(store_path))
    return manifest


def index_status(
    *,
    manifest: Optional[Dict[str, Any]],
    identity: Dict[str, Any],
    stored_dimension: Optional[int],
    stored_count: int,
) -> Dict[str, Any]:
    """Compare what the store holds with the embedding in hand."""
    current = {
        "provider": identity.get("provider"),
        "model": identity.get("model"),
        "dimension": identity.get("dimension"),
        "fingerprint": identity.get("fingerprint"),
    }
    stored = {
        "provider": (manifest or {}).get("embedding_provider"),
        "model": (manifest or {}).get("embedding_model"),
        "dimension": (manifest or {}).get("embedding_dimension", stored_dimension),
        "fingerprint": (manifest or {}).get("embedding_fingerprint"),
        "chunk_count": stored_count,
        "written_at": (manifest or {}).get("written_at"),
    }

    def result(state: str, reason: str) -> Dict[str, Any]:
        return {
            "state": state,
            "compatible": state in (STATE_EMPTY, STATE_COMPATIBLE),
            "dense_available": state == STATE_COMPATIBLE,
            "reason": reason,
            "stored": stored,
            "current": current,
        }

    if stored_count <= 0:
        return result(STATE_EMPTY, "No chunks are stored yet.")
    if stored_dimension is not None and stored_dimension <= PLACEHOLDER_DIMENSION:
        return result(
            STATE_NO_DENSE_INDEX,
            "The stored chunks carry placeholder vectors (indexed by a lexical-only "
            "profile); re-index them with the current embedding model to enable "
            "dense retrieval.",
        )
    if manifest is None:
        return result(
            STATE_NO_DENSE_INDEX,
            "The store has vectors but no embedding manifest, so the model that "
            "wrote them is unknown; re-index with the current embedding model.",
        )
    if stored["fingerprint"] != current["fingerprint"]:
        return result(
            STATE_REINDEX_REQUIRED,
            f"The store was indexed with {stored['provider']}:{stored['model']} "
            f"({stored['dimension']} dims); the current embedding is "
            f"{current['provider']}:{current['model']}. Re-index to use dense retrieval.",
        )
    if (
        current["dimension"] is not None
        and stored["dimension"] is not None
        and int(current["dimension"]) != int(stored["dimension"])
    ):
        return result(
            STATE_REINDEX_REQUIRED,
            f"Vector width changed ({stored['dimension']} stored, {current['dimension']} "
            "current); re-index to use dense retrieval.",
        )
    return result(STATE_COMPATIBLE, "The store matches the current embedding model.")
