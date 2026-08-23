"""Structure-first chunker adapter.

Wraps ``amsc.structural_chunker``: document structure decides the boundaries,
token limits only constrain them, and oversized units are split at table row /
list item / sentence seams. No embeddings are involved, so chunking runs at
parser speed with zero model cost.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Sequence

from amsc.structural_chunker import chunk_units
from amsc.tokenization import TiktokenTokenCounter

from core.exceptions import ChunkerException
from core.models import DocumentChunk

from .base import BaseChunker
from .normalization_adapter import CanonicalUnitAdapter

CHUNKER_ID = "structure_first"
TOKEN_ENCODING = "cl100k_base"
MIN_TOKENS = 160
TARGET_TOKENS = 700
SOFT_MAX_TOKENS = 900
HARD_MAX_TOKENS = 1126


class StructuralChunker(BaseChunker):
    """Structure-first chunking over canonical units."""

    def __init__(self) -> None:
        self._adapter = CanonicalUnitAdapter()
        self._counter = TiktokenTokenCounter(TOKEN_ENCODING)

    def get_name(self) -> str:
        return "StructuralChunker"

    def get_config(self) -> dict[str, Any]:
        return {
            "type": CHUNKER_ID,
            "token_counter": f"tiktoken:{TOKEN_ENCODING}",
            "min_tokens": MIN_TOKENS,
            "target_tokens": TARGET_TOKENS,
            "soft_max_tokens": SOFT_MAX_TOKENS,
            "hard_max_tokens": HARD_MAX_TOKENS,
            "uses_embeddings": False,
        }

    def chunk_canonical(
        self,
        *,
        text: str,
        doc_id: str,
        parsed_units: Sequence[Any] | None = None,
        parser_metadata: Any | None = None,
    ) -> list[dict]:
        units = self._adapter.normalize(
            text=text,
            document_id=doc_id,
            parsed_units=parsed_units,
            parser_metadata=parser_metadata,
        )
        return chunk_units(
            units,
            counter=self._counter,
            min_tokens=MIN_TOKENS,
            target_tokens=TARGET_TOKENS,
            soft_max_tokens=SOFT_MAX_TOKENS,
            hard_max_tokens=HARD_MAX_TOKENS,
        )

    def chunk_text(
        self,
        text: str,
        doc_id: str,
        doc_title: str,
        document_summary: str = None,
        **kwargs: Any,
    ) -> list[DocumentChunk]:
        try:
            rows = self.chunk_canonical(
                text=text,
                doc_id=doc_id,
                parsed_units=kwargs.get("parsed_units"),
                parser_metadata=kwargs.get("parser_metadata"),
            )
        except (ValueError, TypeError, AssertionError):
            raise
        except Exception as exc:
            raise ChunkerException(f"Structure-first chunking failed: {exc}") from exc

        created_at = datetime.now().isoformat()
        chunks: list[DocumentChunk] = []
        for index, row in enumerate(rows):
            section_paths = row.get("section_paths") or []
            section_title = " > ".join(section_paths[0]) if section_paths else None
            chunks.append(
                DocumentChunk(
                    chunk_id=row["chunk_id"],
                    doc_id=doc_id,
                    content=row["text"],
                    chunk_index=index,
                    total_chunks=len(rows),
                    doc_title=doc_title,
                    section_title=section_title,
                    document_summary=document_summary,
                    metadata={
                        "chunker_type": CHUNKER_ID,
                        "created_at": created_at,
                        "word_count": len(row["text"].split()),
                        "unit_ids": list(row["unit_ids"]),
                        "pages": list(row.get("pages") or []),
                        "token_count": int(row["token_count"]),
                        "section_paths": section_paths,
                        "split_strategies": row.get("split_strategies") or [],
                    },
                )
            )
        return chunks
