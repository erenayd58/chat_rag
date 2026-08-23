# /Users/murseltasgin/projects/chat_rag/components/vectordb/faiss_vectordb.py
"""
FAISS vector database implementation with on-disk persistence and optional BM25/keyword search.
"""
from typing import List, Dict, Any, Optional
import os
import json
import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize

from .base import BaseVectorDB
from core.models import DocumentChunk
from core.exceptions import VectorDBException


class FaissVectorDB(BaseVectorDB):
    """FAISS-based vector store with persistence and auxiliary text search."""

    def __init__(
        self,
        path: str = "./faiss_db",
        index_file: str = "index.faiss",
        meta_file: str = "meta.json",
        use_cosine: bool = True,
        rebuild_bm25_on_load: bool = True,
    ):
        self.path = path
        self.index_path = os.path.join(path, index_file)
        self.meta_path = os.path.join(path, meta_file)
        self.use_cosine = use_cosine

        os.makedirs(self.path, exist_ok=True)

        # Runtime stores
        self.index = None  # faiss index (with IDMap)
        self.meta: Dict[str, Any] = {"next_int_id": 1, "items": {}}  # int_id -> {chunk_id, content, metadata}
        self.chunkid_to_int: Dict[str, int] = {}

        # BM25
        self._bm25 = None
        self._bm25_tokens: List[List[str]] = []

        # Load persistent state if present
        self._load()
        if rebuild_bm25_on_load:
            self._rebuild_bm25()

    def _save_meta(self) -> None:
        try:
            with open(self.meta_path, "w", encoding="utf-8") as f:
                json.dump(self.meta, f, ensure_ascii=False)
        except Exception as e:
            raise VectorDBException(f"Failed to save meta: {e}")

    def _load(self) -> None:
        # Load index
        if os.path.exists(self.index_path):
            try:
                self.index = faiss.read_index(self.index_path)
            except Exception as e:
                raise VectorDBException(f"Failed to load FAISS index: {e}")
        else:
            # Create empty index; dimension will be set on first add
            self.index = None

        # Load meta
        if os.path.exists(self.meta_path):
            try:
                with open(self.meta_path, "r", encoding="utf-8") as f:
                    self.meta = json.load(f)
                # Build reverse map
                self.chunkid_to_int = {
                    v["chunk_id"]: int(k) for k, v in self.meta.get("items", {}).items()
                }
            except Exception as e:
                raise VectorDBException(f"Failed to load meta: {e}")

    def _ensure_index(self, dim: int) -> None:
        if self.index is not None:
            return
        # Cosine via inner product over L2-normalized vectors
        base = faiss.IndexFlatIP(dim)
        self.index = faiss.IndexIDMap2(base)

    def _normalize(self, arr: np.ndarray) -> np.ndarray:
        if not self.use_cosine:
            return arr.astype(np.float32, copy=False)
        x = arr.astype(np.float32, copy=False)
        norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
        return x / norms

    def _assign_int_ids(self, chunk_ids: List[str]) -> List[int]:
        out = []
        for cid in chunk_ids:
            if cid in self.chunkid_to_int:
                out.append(self.chunkid_to_int[cid])
            else:
                int_id = int(self.meta.get("next_int_id", 1))
                self.meta["next_int_id"] = int_id + 1
                self.chunkid_to_int[cid] = int_id
                out.append(int_id)
        return out

    def _persist_index(self) -> None:
        try:
            if self.index is not None:
                faiss.write_index(self.index, self.index_path)
        except Exception as e:
            raise VectorDBException(f"Failed to write FAISS index: {e}")

    def _rebuild_bm25(self) -> None:
        try:
            items = self.meta.get("items", {})
            corpus = [items[k]["content"] for k in items]
            self._bm25_tokens = [word_tokenize((doc or "").lower()) for doc in corpus]
            if self._bm25_tokens:
                self._bm25 = BM25Okapi(self._bm25_tokens)
            else:
                self._bm25 = None
        except Exception:
            self._bm25 = None

    def add_chunks(
        self,
        chunks: List[DocumentChunk],
        embeddings: List[List[float]],
        **kwargs
    ) -> None:
        try:
            if not chunks:
                return
            if not embeddings:
                raise VectorDBException(
                    "FAISS stores vectors only; a lexical-only retrieval profile "
                    "needs a document store such as chroma"
                )
            # Prepare data
            chunk_ids = [c.chunk_id for c in chunks]
            int_ids = self._assign_int_ids(chunk_ids)
            emb = np.array(embeddings, dtype=np.float32)
            dim = emb.shape[1]
            self._ensure_index(dim)
            if self.use_cosine:
                emb = self._normalize(emb)

            # Upsert into index (remove existing ids first to avoid duplicates)
            to_remove = [iid for iid in int_ids if self._has_int_id(iid)]
            if to_remove:
                idsel = faiss.IDSelectorBatch(np.array(to_remove, dtype=np.int64))
                self.index.remove_ids(idsel)

            self.index.add_with_ids(emb, np.array(int_ids, dtype=np.int64))

            # Update meta
            for c, iid in zip(chunks, int_ids):
                self.meta["items"][str(iid)] = {
                    "chunk_id": c.chunk_id,
                    "content": c.content,
                    "metadata": {
                        "doc_id": c.doc_id,
                        "doc_title": c.doc_title,
                        "chunk_index": c.chunk_index,
                        "total_chunks": c.total_chunks,
                        "section_title": c.section_title,
                        "document_summary": c.document_summary,
                        **(c.metadata or {})
                    }
                }

            # Persist
            self._persist_index()
            self._save_meta()
            # Refresh BM25 index
            self._rebuild_bm25()
        except Exception as e:
            raise VectorDBException(f"Failed to add chunks to FAISS: {e}")

    def _has_int_id(self, int_id: int) -> bool:
        try:
            return str(int_id) in self.meta.get("items", {})
        except Exception:
            return False

    def query(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        filter_dict: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        try:
            if self.index is None or self.count() == 0:
                return []
            q = np.array([query_embedding], dtype=np.float32)
            if self.use_cosine:
                q = self._normalize(q)
            scores, ids = self.index.search(q, top_k)
            scores = scores[0]
            ids = ids[0]

            items = self.meta.get("items", {})
            results = []
            for s, iid in zip(scores, ids):
                if iid == -1:
                    continue
                item = items.get(str(int(iid)))
                if not item:
                    continue
                md = item["metadata"]
                # Apply optional filter on metadata keys
                if filter_dict:
                    if not all(md.get(k) == v for k, v in filter_dict.items()):
                        continue
                # Convert IP similarity to a distance compatible with existing code
                similarity = float(s)
                similarity = max(min(similarity, 1.0), -1.0)
                distance = 1.0 - similarity  # lower is better
                results.append({
                    "chunk_id": item["chunk_id"],
                    "content": item["content"],
                    "distance": distance,
                    "metadata": md
                })
            return results
        except Exception as e:
            raise VectorDBException(f"FAISS query failed: {e}")

    def get_all_chunks(self) -> List[DocumentChunk]:
        try:
            chunks: List[DocumentChunk] = []
            for item in self.meta.get("items", {}).values():
                md = item["metadata"]
                chunks.append(DocumentChunk(
                    chunk_id=item["chunk_id"],
                    content=item["content"],
                    doc_id=md.get("doc_id", ""),
                    doc_title=md.get("doc_title", ""),
                    chunk_index=md.get("chunk_index", 0),
                    total_chunks=md.get("total_chunks", 0),
                    section_title=md.get("section_title"),
                    document_summary=md.get("document_summary"),
                    metadata=md
                ))
            return chunks
        except Exception as e:
            raise VectorDBException(f"Failed to retrieve chunks: {e}")

    def delete_by_doc_id(self, doc_id: str) -> None:
        try:
            to_remove_ints = []
            items = self.meta.get("items", {})
            for k, v in list(items.items()):
                md = v.get("metadata", {})
                if md.get("doc_id") == doc_id:
                    to_remove_ints.append(int(k))
            if to_remove_ints and self.index is not None:
                idsel = faiss.IDSelectorBatch(np.array(to_remove_ints, dtype=np.int64))
                self.index.remove_ids(idsel)
            # Purge meta and reverse map
            for iid in to_remove_ints:
                chunk_id = items[str(iid)]["chunk_id"]
                self.chunkid_to_int.pop(chunk_id, None)
                items.pop(str(iid), None)
            self._persist_index()
            self._save_meta()
            self._rebuild_bm25()
        except Exception as e:
            raise VectorDBException(f"Failed to delete by doc_id: {e}")

    def get_chunks_paginated(
        self,
        offset: int = 0,
        limit: int = 20,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Get paginated chunks with optional filtering"""
        try:
            items = self.meta.get("items", {})
            all_chunks = []
            
            # Filter chunks if filter_dict is provided
            for item_data in items.values():
                metadata = item_data.get("metadata", {})
                
                # Apply filter if provided
                if filter_dict:
                    # Check if all filter conditions match
                    if not all(metadata.get(k) == v for k, v in filter_dict.items()):
                        continue
                
                # Build chunk dict
                all_chunks.append({
                    "chunk_id": item_data["chunk_id"],
                    "content": item_data["content"],
                    "metadata": metadata
                })
            
            # Apply pagination
            total = len(all_chunks)
            start = offset
            end = min(offset + limit, total)
            paginated_chunks = all_chunks[start:end]
            
            return {
                "chunks": paginated_chunks,
                "total": total,
                "offset": offset,
                "limit": limit
            }
        except Exception as e:
            raise VectorDBException(f"Failed to get paginated chunks: {e}")

    def get_name(self) -> str:
        return "FAISS"

    def count(self) -> int:
        try:
            if self.index is None:
                return 0
            return int(self.index.ntotal)
        except Exception as e:
            raise VectorDBException(f"Failed to count chunks: {e}")

    # --- Auxiliary searches for the UI parity ---
    def search_chunks_by_text(
        self,
        search_text: str,
        offset: int = 0,
        limit: int = 20
    ) -> Dict[str, Any]:
        """Naive substring search over stored documents (case-insensitive)."""
        try:
            items = self.meta.get("items", {})
            search_lower = (search_text or "").lower()
            matches = []
            for v in items.values():
                if search_lower in (v["content"] or "").lower():
                    matches.append({
                        "chunk_id": v["chunk_id"],
                        "content": v["content"],
                        "metadata": v["metadata"],
                    })
            total = len(matches)
            start = offset
            end = min(offset + limit, total)
            return {"chunks": matches[start:end], "total": total, "offset": offset, "limit": limit}
        except Exception as e:
            raise VectorDBException(f"Failed to search chunks: {e}")

    def bm25_search(self, query_text: str, top_k: int = 20) -> List[Dict[str, Any]]:
        """BM25 keyword search using internal corpus tokens; returns best candidates even if scores are 0."""
        if not query_text:
            return []
        try:
            if self._bm25 is None:
                self._rebuild_bm25()
            if self._bm25 is None:
                return []
            tokens = word_tokenize(query_text.lower())
            scores = self._bm25.get_scores(tokens)
            idxs = np.argsort(scores)[-top_k:][::-1]
            items = list(self.meta.get("items", {}).values())
            results = []
            for rank, i in enumerate(idxs):
                if i < 0 or i >= len(items):
                    continue
                v = items[i]
                results.append({
                    "chunk_id": v["chunk_id"],
                    "content": v["content"],
                    "metadata": v["metadata"],
                    "score": float(scores[i]),
                    "retrieval_method": "bm25",
                    "rank": rank
                })
            return results
        except Exception as e:
            raise VectorDBException(f"BM25 search failed: {e}")

    def add_single_chunk(
        self,
        chunk_id: str,
        content: str,
        embedding: List[float],
        metadata: Dict[str, Any]
    ) -> None:
        try:
            # Wrap to reuse add_chunks
            dummy_chunk = DocumentChunk(
                chunk_id=chunk_id,
                content=content,
                doc_id=metadata.get("doc_id", ""),
                doc_title=metadata.get("doc_title", ""),
                chunk_index=metadata.get("chunk_index", 0),
                total_chunks=metadata.get("total_chunks", 0),
                section_title=metadata.get("section_title"),
                document_summary=metadata.get("document_summary"),
                metadata=metadata
            )
            self.add_chunks([dummy_chunk], [embedding])
        except Exception as e:
            raise VectorDBException(f"Failed to add single chunk: {e}")


