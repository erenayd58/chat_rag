"""Structure-first chunker adapter.

Wraps ``amsc.structural_chunker``: document structure decides the boundaries,
token limits only constrain them, and oversized units are split at table row /
list item / sentence seams. No embeddings are involved, so chunking runs at
parser speed with zero model cost. That is the Standard path; Deep Analysis
(``chunk_text_deep``) sends the same canonical units through
``amsc.deep_pipeline`` and indexes its rows the same way.
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
        configuration: Any,
        provider: Any = None,
        verifier_provider: Any = None,
        **kwargs: Any,
    ) -> tuple[list[DocumentChunk], dict[str, Any]]:
        """Deep Analysis: ``amsc.deep_pipeline.chunk_document`` in deep mode.

        The same canonical units as ``chunk_text`` go through the final
        pipeline -- the frozen structural walk, the LLM proposer, the
        deterministic quality selector, the double-order verifier and the
        quality measurement -- and come back as rows in the structural
        chunker's row schema, so indexing is identical in both modes.

        ``configuration`` is a :class:`DeepAnalysisConfiguration`. When it
        lacks what a model run needs, the deterministic contract runs alone
        and the result is labelled ``fallback_no_provider``; a provider that
        fails is the pipeline's own ``fallback_provider_error`` /
        ``degraded``. Provider problems never raise; a structural error
        (bad canonical, hard-cap breach) still does, because that is a bug.
        ``provider`` / ``verifier_provider`` exist so tests can inject a
        double; production builds them from ``configuration`` inside amsc,
        which is the only place the key's variable name is ever used.

        Returns the chunks plus a JSON-serialisable report: the pipeline's
        own summary (status, model ids, chunk and smell counts before and
        after, regression counts, proposer/verifier usage, timings) and the
        product's structural checks. Never prompt text, never a key.
        ``chunk_text`` above stays the untouched Standard path.
        """
        from amsc.deep_pipeline import (
            MODE_DEEP,
            STATUS_FALLBACK_NO_PROVIDER,
            chunk_document,
        )

        from .deep_analysis import CHUNKING_MODE

        units = self._adapter.normalize(
            text=text,
            document_id=doc_id,
            parsed_units=kwargs.get("parsed_units"),
            parser_metadata=kwargs.get("parser_metadata"),
        )
        try:
            result = chunk_document(
                units,
                mode=MODE_DEEP,
                settings=configuration.settings,
                counter=self._counter,
                provider=provider,
                verifier_provider=verifier_provider,
            )
        except (ValueError, TypeError, AssertionError):
            raise
        except Exception as exc:
            raise ChunkerException(f"Deep Analysis chunking failed: {exc}") from exc

        status = result.status
        report: dict[str, Any] = dict(result.report)
        if configuration.missing:
            # The model run was never attempted: the contract ran alone.
            # Say so with the product's status, not the pipeline's
            # "deterministic" (which means the LLM was not *requested*).
            status = STATUS_FALLBACK_NO_PROVIDER
            report["fallback_reason"] = configuration.fallback_reason
        report["pipeline_mode"] = report.get("mode")
        report["mode"] = CHUNKING_MODE
        report["status"] = status

        # The product's own structural checks, computed on the rows that
        # will be indexed: nothing over the hard cap, and every canonical
        # unit exactly once in Standard's order.
        standard_rows = result.deep.standard_rows if result.deep is not None else []
        max_tokens = max((int(row["token_count"]) for row in result.rows), default=0)
        report["checks"] = {
            "hard_max_tokens": HARD_MAX_TOKENS,
            "max_token_count": max_tokens,
            "hard_cap_ok": max_tokens <= HARD_MAX_TOKENS,
            "coverage_ok": (
                [uid for row in result.rows for uid in row["unit_ids"]]
                == [uid for row in standard_rows for uid in row["unit_ids"]]
            ) if standard_rows else True,
        }
        llm = configuration.llm_available
        report["configuration"] = {
            "use_llm": configuration.settings.use_llm,
            "verify": configuration.settings.verify,
            "proposer_model": configuration.settings.proposer_model if llm else None,
            "verifier_model": (
                configuration.settings.effective_verifier_model
                if llm and configuration.settings.verify else None
            ),
            "endpoint": configuration.settings.endpoint if llm else None,
            "api_key_env": configuration.settings.api_key_env,
        }

        chunks = self._rows_to_chunks(
            result.rows,
            doc_id=doc_id,
            doc_title=doc_title,
            document_summary=document_summary,
            extra_metadata={
                "chunking_mode": CHUNKING_MODE,
                "deep_analysis_status": status,
                "proposer_model": report.get("model_id"),
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
                        **{k: v for k, v in (extra_metadata or {}).items() if v is not None},
                    },
                )
            )
        return chunks
