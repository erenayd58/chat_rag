"""Persistent store for manually confirmed retrieval answers.

The retrieval review screen lets a human mark which returned source actually
answers the question. Those judgements were kept in the browser's local storage
and evaporated with it. They are the only production-side ground truth this
project has, and the regression CLI will be built on them, so they belong in a
file.

One entry per (knowledge base, question): re-marking the same question updates
the entry instead of appending a second one, so the set stays a set.

**An entry is not addressed by chunk id.** Chunk ids move whenever the parser
or the chunker changes -- this project has re-chunked the same report a dozen
times. The id is recorded, but so are the unit ids, the page span, the section
and a verbatim evidence snippet, so an entry can be matched again against a
freshly ingested corpus.
"""

from __future__ import annotations

from datetime import datetime
import hashlib
import json
import os
from typing import Any, Callable, Dict, List, Optional

from config import paths

#: How much of the confirmed chunk to keep as re-matchable evidence.
EVIDENCE_LIMIT = 600

SCHEMA_VERSION = 1


def normalize_question(question: str) -> str:
    """Identity of a question: case and spacing are not part of it."""
    return " ".join(str(question or "").split()).casefold()


def entry_id_for(
    kb_id: str, question: str, document_sha256: Optional[str] = None
) -> str:
    """Identity of one confirmed answer.

    Scoped to the document's bytes when they are known, so the same question
    about the same document keeps its id after the corpus is re-ingested --
    which is what a frozen set needs to stay readable across knowledge bases.
    Without a hash it falls back to the knowledge base id, which is how the
    runtime store has always keyed marks and how every entry written before
    this one is keyed.
    """
    # The knowledge-base scope is the bare id, unprefixed: every entry already
    # in a runtime store was keyed that way and must keep the same id.
    scope = f"sha256:{document_sha256}" if document_sha256 else str(kb_id or "")
    digest = hashlib.sha256()
    digest.update(scope.encode("utf-8"))
    digest.update(b"\n")
    digest.update(normalize_question(question).encode("utf-8"))
    return digest.hexdigest()[:16]


def _ensure_parent(path: str) -> None:
    """Create the directory a store file lives in, if it is not there yet.

    Historically these files sat in the working directory, which always
    exists. A configured data directory does not, until something makes it.
    """
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


class GoldSetManager:
    """JSON-backed upsert store, mirroring KnowledgeBaseManager's persistence."""

    def __init__(
        self,
        store_path: Optional[str] = None,
        *,
        now: Optional[Callable[[], str]] = None,
    ) -> None:
        # Defaults to the historical file unless a data directory is set.
        self.store_path = store_path or paths.gold_set()
        self._now = now or (lambda: datetime.now().isoformat())
        self.entries: Dict[str, Dict[str, Any]] = {}
        self._load()

    # ---------------------------------------------------------------- io
    def _load(self) -> None:
        if not os.path.exists(self.store_path):
            return
        try:
            with open(self.store_path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
        except Exception:
            # A corrupt store must not take the app down; it is rebuilt by
            # marking again, and the unreadable file is left untouched.
            self.entries = {}
            return
        raw = data.get("entries") if isinstance(data, dict) else data
        if isinstance(raw, list):
            self.entries = {
                item["entry_id"]: item
                for item in raw
                if isinstance(item, dict) and item.get("entry_id")
            }
        elif isinstance(raw, dict):
            self.entries = {k: v for k, v in raw.items() if isinstance(v, dict)}

    def _save(self) -> None:
        payload = {
            "schema_version": SCHEMA_VERSION,
            "entries": [self.entries[key] for key in sorted(self.entries)],
        }
        _ensure_parent(self.store_path)
        tmp = self.store_path + ".tmp"
        with open(tmp, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(tmp, self.store_path)

    # ------------------------------------------------------------ queries
    def list(self, kb_id: Optional[str] = None) -> List[Dict[str, Any]]:
        items = [self.entries[key] for key in sorted(self.entries)]
        if kb_id:
            items = [item for item in items if item.get("kb_id") == kb_id]
        return items

    def get(self, kb_id: str, question: str) -> Optional[Dict[str, Any]]:
        return self.entries.get(entry_id_for(kb_id, question))

    # ------------------------------------------------------------ mutation
    def upsert(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Record a confirmed answer, replacing any earlier one for the same
        question in the same knowledge base."""
        if not isinstance(data, dict):
            raise ValueError("Gold-set entry must be an object")

        question = " ".join(str(data.get("question") or "").split())
        if not question:
            raise ValueError("question is required")
        kb_id = str(data.get("kb_id") or "").strip()
        if not kb_id:
            raise ValueError("kb_id is required")

        chunk_id = data.get("correct_chunk_id") or None
        unit_ids = list(data.get("unit_ids") or [])
        evidence = str(data.get("evidence") or "").strip()
        if not (chunk_id or unit_ids or evidence):
            raise ValueError(
                "an entry needs at least one locator: correct_chunk_id, "
                "unit_ids or evidence"
            )

        entry_id = entry_id_for(kb_id, question)
        existing = self.entries.get(entry_id)
        stamp = self._now()

        entry = {
            "entry_id": entry_id,
            "schema_version": SCHEMA_VERSION,
            "question": question,
            "kb_id": kb_id,
            "document_id": data.get("document_id"),
            "document_title": data.get("document_title"),
            "document_sha256": data.get("document_sha256"),
            "correct_chunk_id": chunk_id,
            "section": data.get("section"),
            "pages": list(data.get("pages") or []),
            "unit_ids": unit_ids,
            "evidence": evidence[:EVIDENCE_LIMIT],
            "retrieval_method": data.get("retrieval_method"),
            "found_at_rank": data.get("found_at_rank"),
            "created_at": (existing or {}).get("created_at") or stamp,
            "updated_at": stamp,
        }
        self.entries[entry_id] = entry
        self._save()
        return entry

    def delete(self, entry_id: str) -> bool:
        if entry_id in self.entries:
            self.entries.pop(entry_id)
            self._save()
            return True
        return False

    def delete_for_question(self, kb_id: str, question: str) -> bool:
        return self.delete(entry_id_for(kb_id, question))
