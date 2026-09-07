"""The sources a human confirmed answer a question.

One entry per (knowledge base, question): marking the same question again
replaces the entry rather than appending another. The entries are what
``python -m cli evaluate`` scores a retrieval change against, which is why an
entry records the bytes it was confirmed over and not just the document id.
"""

from __future__ import annotations

import logging
from typing import Optional

from .errors import InvalidRequest, NotFound

logger = logging.getLogger("RAG.goldsets")


def list_all(services, kb_id: Optional[str] = None) -> list[dict]:
    return services.gold_manager.list(kb_id or None)


def upsert(services, payload: dict) -> dict:
    entry = dict(payload or {})
    # The ingest ledger already recorded the file's sha256; carry it over
    # rather than hashing the document again. An entry that knows which bytes
    # it was confirmed against can warn when the corpus is replaced.
    if not entry.get('document_sha256') and entry.get('document_id'):
        tracked = services.documents().get_document_by_doc_id(entry['document_id'])
        if tracked and tracked.get('file_hash'):
            entry['document_sha256'] = tracked['file_hash']
    try:
        return services.gold_manager.upsert(entry)
    except ValueError as error:
        raise InvalidRequest(str(error)) from error


def delete(services, entry_id: str) -> None:
    if not services.gold_manager.delete(entry_id):
        raise NotFound('Entry not found')
