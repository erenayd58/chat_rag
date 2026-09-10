"""The smallest thing that satisfies the document-store contract.

The contract next door is run against every store the product ships. Until
Step 3 that was two -- Chroma and FAISS -- and running it against two
independent implementations is what kept it a *contract* rather than a
description of Chroma. FAISS was removed because no knowledge base used it and
it implemented seven of the twelve methods the routes call; with it gone the
contract had one implementation left, and a Chroma-shaped assumption could
have settled into it unnoticed. That is the protection this file restores.

It is not a product store and is never configured as one. It is:

* **the second implementation**, so a contract test that only passes because
  of how Chroma behaves fails here immediately;
* **the checklist for pgvector**, in runnable form -- everything a new store
  has to do, with nothing else in the way.

Cosine distance in the range Chroma uses (0 identical, 2 opposite), because
the retrievers and the RRF fusion assume lower-is-better and the console turns
a distance into a similarity with ``1 - distance / 2``.
"""

from __future__ import annotations

import json
import math
import os
from typing import Any, Optional, Sequence

from chat_rag.components.vectordb import BaseVectorDB
from chat_rag.core.models import DocumentChunk

#: Metadata keys that carry a field of the record itself rather than extra
#: information about it. They are written flat so a row is one dictionary, and
#: read back out on the way in -- which is exactly the round trip the contract
#: pins.
_RECORD_FIELDS = ("doc_id", "doc_title", "chunk_index", "total_chunks", "section_title")


def _cosine_distance(left: Sequence[float], right: Sequence[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right))
    norm = math.sqrt(sum(a * a for a in left)) * math.sqrt(sum(b * b for b in right))
    if norm == 0:
        return 1.0
    return 1.0 - dot / norm


class ReferenceVectorDB(BaseVectorDB):
    """A store with no dependencies, written to the contract and nothing else."""

    def __init__(self, path: str, collection_name: str = "documents"):
        self.path = str(path)
        self.collection_name = collection_name
        self._rows: dict[str, dict[str, Any]] = {}
        self._load()

    # ----------------------------------------------------------- persistence
    @property
    def _file(self) -> str:
        return os.path.join(self.path, self.collection_name + ".json")

    def _load(self) -> None:
        try:
            with open(self._file, "r", encoding="utf-8") as handle:
                self._rows = json.load(handle)
        except (FileNotFoundError, NotADirectoryError, ValueError):
            self._rows = {}

    def _save(self) -> None:
        os.makedirs(self.path, exist_ok=True)
        with open(self._file, "w", encoding="utf-8") as handle:
            json.dump(self._rows, handle)

    def close(self) -> None:
        self._save()

    # --------------------------------------------------------------- writing
    def _row(self, chunk: DocumentChunk, embedding: Sequence[float]) -> dict[str, Any]:
        metadata = dict(chunk.metadata or {})
        for field in _RECORD_FIELDS:
            value = getattr(chunk, field, None)
            if value is not None:
                metadata[field] = value
        return {
            "chunk_id": chunk.chunk_id,
            "content": chunk.content,
            "metadata": metadata,
            "embedding": [float(v) for v in (embedding or [])],
        }

    def add_chunks(self, chunks: Sequence[DocumentChunk],
                   embeddings: Sequence[Sequence[float]]) -> None:
        for chunk, embedding in zip(chunks, embeddings):
            self._rows[chunk.chunk_id] = self._row(chunk, embedding)
        self._save()

    def add_single_chunk(self, chunk: DocumentChunk, embedding: Sequence[float]) -> None:
        self.add_chunks([chunk], [embedding])

    def update_chunk(self, chunk_id: str, content: Optional[str] = None,
                     metadata: Optional[dict] = None,
                     embedding: Optional[Sequence[float]] = None) -> None:
        row = self._rows.get(chunk_id)
        if row is None:
            return
        if content is not None:
            row["content"] = content
        if metadata is not None:
            row["metadata"] = {**row["metadata"], **metadata}
        if embedding is not None:
            row["embedding"] = [float(v) for v in embedding]
        self._save()

    def delete_chunk(self, chunk_id: str) -> None:
        self._rows.pop(chunk_id, None)
        self._save()

    def delete_by_doc_id(self, doc_id: str) -> None:
        for chunk_id in [key for key, row in self._rows.items()
                         if row["metadata"].get("doc_id") == doc_id]:
            del self._rows[chunk_id]
        self._save()

    # --------------------------------------------------------------- reading
    @staticmethod
    def _matches(row: dict[str, Any], filter_dict: Optional[dict]) -> bool:
        return all(row["metadata"].get(key) == value
                   for key, value in (filter_dict or {}).items())

    def _selected(self, filter_dict: Optional[dict] = None) -> list[dict[str, Any]]:
        return [row for row in self._rows.values() if self._matches(row, filter_dict)]

    def query(self, query_embedding: Sequence[float], top_k: int = 5,
              filter_dict: Optional[dict] = None) -> list[dict[str, Any]]:
        scored = [
            {
                "chunk_id": row["chunk_id"],
                "content": row["content"],
                "metadata": dict(row["metadata"]),
                "distance": _cosine_distance(query_embedding, row["embedding"]),
            }
            for row in self._selected(filter_dict)
        ]
        # The chunk id breaks a tie, so a page of equally distant rows is at
        # least stable between calls.
        scored.sort(key=lambda row: (row["distance"], row["chunk_id"]))
        return scored[:top_k]

    def _as_chunk(self, row: dict[str, Any]) -> DocumentChunk:
        metadata = row["metadata"]
        return DocumentChunk(
            chunk_id=row["chunk_id"],
            content=row["content"],
            doc_id=metadata.get("doc_id") or "",
            doc_title=metadata.get("doc_title") or "",
            chunk_index=metadata.get("chunk_index") or 0,
            total_chunks=metadata.get("total_chunks") or 0,
            section_title=metadata.get("section_title"),
            # Metadata goes back whole, which is what lets the two derived
            # renderings reach their properties on the way out.
            metadata=dict(metadata),
        )

    def get_all_chunks(self) -> list[DocumentChunk]:
        return [self._as_chunk(row) for row in self._rows.values()]

    def get_chunk_by_id(self, chunk_id: str) -> Optional[dict[str, Any]]:
        row = self._rows.get(chunk_id)
        if row is None:
            return None
        return {"chunk_id": row["chunk_id"], "content": row["content"],
                "metadata": dict(row["metadata"]), "embedding": list(row["embedding"])}

    @staticmethod
    def _page(rows: list[dict[str, Any]], offset: int, limit: int) -> dict[str, Any]:
        window = rows[offset:offset + limit]
        return {
            "chunks": [{"chunk_id": row["chunk_id"], "content": row["content"],
                        "metadata": dict(row["metadata"])} for row in window],
            "total": len(rows),
            "offset": offset,
            "limit": limit,
        }

    def get_chunks_paginated(self, offset: int = 0, limit: int = 20,
                             filter_dict: Optional[dict] = None) -> dict[str, Any]:
        return self._page(self._selected(filter_dict), offset, limit)

    def search_chunks_by_text(self, search_text: str, offset: int = 0,
                              limit: int = 20) -> dict[str, Any]:
        needle = (search_text or "").casefold()
        found = [row for row in self._rows.values() if needle in row["content"].casefold()]
        return self._page(found, offset, limit)

    def count(self) -> int:
        return len(self._rows)

    def get_name(self) -> str:
        return "ReferenceVectorDB"
