"""Structure-first chunker adapter.

Wraps ``amsc.structural_chunker``: document structure decides the boundaries,
token limits only constrain them, and oversized units are split at table row /
list item / sentence seams. No embeddings are involved, so chunking runs at
parser speed with zero model cost.
"""

from __future__ import annotations

import json
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

        return self._rows_to_chunks(
            rows,
            doc_id=doc_id,
            doc_title=doc_title,
            document_summary=document_summary,
        )

    def chunk_text_deep(
        self,
        text: str,
        doc_id: str,
        doc_title: str,
        document_summary: str = None,
        *,
        judge: Any,
        **kwargs: Any,
    ) -> tuple[list[DocumentChunk], dict[str, Any]]:
        """Deep Analysis: the same structural walk, with the LLM boundary
        judge consulted at plain budget cuts that offer a real choice.

        The judge is wrapped in :class:`StructurallyGuardedJudge`, which
        refuses structurally unsafe candidates (inside a list, straight after
        a heading) so the model can only choose among the remaining ones, and
        refuses a whole window whose answer pairs a decision with a reason
        code that contradicts it. The guard narrows choices and never widens
        them, so the walk's structural cut and its token budget are untouched.

        Returns the chunks plus a document-level report (judge model, call
        and decision counts, per-step fallbacks, what the guard refused, and
        the per-candidate decisions the model actually returned).
        ``chunk_text`` above stays the untouched Standard path; with an
        all-KEEP judge or none at all the amsc walk is pinned byte-identical
        to ``chunk_units``.
        """
        from amsc.llm_boundary_judge import (
            JudgeConfig,
            ProductChunkingMode,
            audit_rows,
            chunk_with_product_mode,
        )

        from .boundary_guard import StructurallyGuardedJudge

        units = self._adapter.normalize(
            text=text,
            document_id=doc_id,
            parsed_units=kwargs.get("parsed_units"),
            parser_metadata=kwargs.get("parser_metadata"),
        )
        guarded = StructurallyGuardedJudge(judge, units)
        try:
            result = chunk_with_product_mode(
                units,
                counter=self._counter,
                mode=ProductChunkingMode.DEEP_ANALYSIS,
                judge=guarded,
                config=JudgeConfig(
                    min_tokens=MIN_TOKENS,
                    target_tokens=TARGET_TOKENS,
                    soft_max_tokens=SOFT_MAX_TOKENS,
                    hard_max_tokens=HARD_MAX_TOKENS,
                ),
            )
        except (ValueError, TypeError, AssertionError):
            raise
        except Exception as exc:
            raise ChunkerException(f"Deep Analysis chunking failed: {exc}") from exc

        diagnostics = result.diagnostics
        report = {
            "mode": "deep_analysis",
            "boundary_judge_model": getattr(judge, "model_id", None),
            "consulted_boundary_count": diagnostics.get("consulted_boundary_count"),
            "judge_call_count": diagnostics.get("llm_call_count"),
            "split_votes": diagnostics.get("split_votes"),
            "keep_votes": diagnostics.get("keep_votes"),
            "fallback_count": diagnostics.get("fallback_count"),
            "changed_from_greedy_count": diagnostics.get("changed_from_greedy_count"),
            "tuning_status": diagnostics.get("tuning_status"),
            # What the structural guard refused, and which candidates it
            # refused. Counts and unit ids only.
            "structural_guard": {
                **guarded.report(),
                "blocked_candidates": guarded.blocked_candidates,
                "contract_violations": guarded.contract_violations,
            },
            # The decisions the model actually returned, per candidate, as
            # amsc recorded them, with the fallback reason restated on the
            # windows the guard refused for a decision/reason_code conflict.
            # Prompts are never included, so nothing here can carry document
            # text or credentials.
            "decisions": guarded.relabel_audit_rows(audit_rows(result)),
        }
        chunks = self._rows_to_chunks(
            result.chunks,
            doc_id=doc_id,
            doc_title=doc_title,
            document_summary=document_summary,
            extra_metadata={
                "chunking_mode": "deep_analysis",
                "judge_model": getattr(judge, "model_id", None),
            },
        )
        return chunks, report

    def _rows_to_chunks(
        self,
        rows: list[dict],
        *,
        doc_id: str,
        doc_title: str,
        document_summary: str | None,
        extra_metadata: dict[str, Any] | None = None,
    ) -> list[DocumentChunk]:
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
                        # The chunker's own heading, verbatim. ``section_title``
                        # above is a display string derived from the section
                        # path and loses the accumulated heading lines, so
                        # anything that needs the real heading -- structural QA,
                        # the retrieval review screen -- had to rebuild it from
                        # the canonical units. Persist it instead. Omitted when
                        # the chunk has no heading: Chroma rejects None and a
                        # missing key reads the same as an older record.
                        "heading": row.get("heading"),
                        # Chroma keeps scalar metadata only and silently drops
                        # list values, so provenance is serialised the same way
                        # FrozenV4Chunker serialises its own.
                        "unit_ids_json": json.dumps(list(row["unit_ids"])),
                        "pages_json": json.dumps(list(row.get("pages") or [])),
                        "token_count": int(row["token_count"]),
                        # Chroma metadata values must be scalars or flat lists,
                        # and section_paths is a list of paths. Serialised the
                        # same way FrozenV4Chunker serialises its structures.
                        "section_paths_json": json.dumps(
                            section_paths, ensure_ascii=False
                        ),
                        "split_strategies_json": json.dumps(
                            row.get("split_strategies") or []
                        ),
                        **(extra_metadata or {}),
                    },
                )
            )
        return chunks
