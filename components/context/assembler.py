"""Context assembly: from ranked chunks to what the answer model reads.

Retrieval returns hits; the answer model needs a bounded, de-duplicated,
traceable context. The rules, in order:

1. Hits are taken in fused rank order. A chunk that is already in the
   context -- same id, or the same text under another id -- is dropped.
2. **Every ranked hit is offered the budget before any expansion is.**
   Retrieval ranked these chunks against the question; a neighbour was
   pulled in only because it sits beside one. Spending the budget on the
   first hit's neighbours while a lower-ranked hit is still waiting loses
   the evidence retrieval had already found -- measured on this corpus,
   that is how a hit at rank 5 never reached the answer model.
3. Only then may a hit bring its structural neighbour: the adjacent chunk
   of the same document under the *same heading* (a section the token
   budget cut in two). Never a different section, never more than one
   chunk each side, and only while the budget allows. The neighbour is
   labelled as a source of its own and marked as an expansion of the hit
   that pulled it in.
4. Within each pass the token budget is spent in rank order; a chunk that
   does not fit is dropped and counted, never truncated mid-text. Ranked
   hits dropped this way are counted separately from neighbours that could
   not be afforded: the first is a loss, the second is the budget working.
5. Reading order is admission order: every hit, in rank order, then the
   expansions. Laying each hit beside its own neighbours would undo rule 2
   one layer up -- a neighbour of rank 1 would read ahead of the hit at
   rank 5, and the more budget expansion wins the deeper the lowest-ranked
   evidence is buried. An expansion names the hit it continues instead.
6. Every chunk appears once with a stable label ``S1..Sn`` in context
   order, so a citation in the answer maps back to exactly one chunk, one
   heading and one page list.

Deterministic, bounded, and free of any model call.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

from core.models import DocumentChunk, RetrievalResult

_counter = None


def estimate_tokens(text: str) -> int:
    """cl100k token count (the chunker's own tokenizer); words as a fallback."""
    global _counter
    if _counter is None:
        try:
            from amsc.tokenization import TiktokenTokenCounter

            _counter = TiktokenTokenCounter("cl100k_base")
        except Exception:  # pragma: no cover - tiktoken is a hard dependency in practice
            _counter = False
    if _counter:
        return int(_counter.count(text))
    return len(text.split())


def _text_key(text: str) -> str:
    normalised = re.sub(r"\s+", " ", (text or "")).strip().lower()
    return hashlib.sha256(normalised.encode("utf-8")).hexdigest()


def _pages(chunk: DocumentChunk) -> List[Any]:
    metadata = chunk.metadata or {}
    try:
        pages = json.loads(metadata.get("pages_json") or "[]")
        if pages:
            return list(pages)
    except (TypeError, ValueError):
        pass
    for key in ("page", "page_number", "page_start"):
        value = metadata.get(key)
        if value not in (None, ""):
            return [value]
    return []


def _heading(chunk: DocumentChunk) -> Optional[str]:
    metadata = chunk.metadata or {}
    heading = metadata.get("heading")
    if heading:
        return str(heading)
    return chunk.section_title or None


@dataclass
class ContextSource:
    label: str
    chunk: DocumentChunk
    rank: Optional[int]
    score: Optional[float]
    dense_rank: Optional[int]
    bm25_rank: Optional[int]
    tokens: int
    expanded_from: Optional[str] = None

    @property
    def legs(self) -> List[str]:
        legs = []
        if self.dense_rank is not None:
            legs.append("dense")
        if self.bm25_rank is not None:
            legs.append("lexical")
        return legs

    def as_dict(self) -> Dict[str, Any]:
        chunk = self.chunk
        metadata = chunk.metadata or {}
        return {
            "label": self.label,
            "chunk_id": chunk.chunk_id,
            "doc_id": chunk.doc_id,
            "document": chunk.doc_title,
            "heading": _heading(chunk),
            "section": _heading(chunk),
            "pages": _pages(chunk) or None,
            "chunking_mode": metadata.get("chunking_mode") or "standard",
            "rank": self.rank,
            "score": self.score,
            "dense_rank": self.dense_rank,
            "bm25_rank": self.bm25_rank,
            "legs": self.legs,
            "tokens": self.tokens,
            "expanded_from": self.expanded_from,
        }


@dataclass
class ContextBundle:
    sources: List[ContextSource]
    text: str
    token_count: int
    dropped_over_budget: int = 0
    #: How many of those drops were *ranked hits* rather than neighbours. A
    #: neighbour that did not fit is the budget doing its job; a hit that did
    #: not fit is evidence retrieval found and the context lost.
    dropped_ranked_hits: int = 0
    deduplicated: int = 0
    expanded: int = 0
    max_tokens: int = 0
    labels: Dict[str, ContextSource] = field(default_factory=dict)


def _render(source: ContextSource, seed_label: Optional[str] = None) -> str:
    chunk = source.chunk
    header = [f"Belge: {chunk.doc_title}"]
    heading = _heading(chunk)
    if heading:
        header.append(f"Bölüm: {heading}")
    pages = _pages(chunk)
    if pages:
        header.append("Sayfa: " + ", ".join(str(page) for page in pages))
    if seed_label:
        # An expansion no longer sits beside the hit that pulled it in, so it
        # says which one it continues. The link survives; the hits keep the top.
        header.append(f"Devam: {seed_label}")
    return f"[{source.label}] " + " | ".join(header) + "\n" + (chunk.content or "").strip()


def assemble_context(
    results: Sequence[RetrievalResult],
    *,
    max_tokens: int = 3200,
    max_sources: int = 8,
    neighbor: Optional[Callable[[DocumentChunk, int], Optional[DocumentChunk]]] = None,
    expand_neighbors: bool = True,
) -> ContextBundle:
    """Build the labelled context the answer model reads."""
    seen_ids: set = set()
    seen_text: set = set()
    chosen: List[ContextSource] = []
    dropped = 0
    dropped_hits = 0
    deduplicated = 0
    expanded = 0
    spent = 0

    def try_add(chunk: DocumentChunk, result: Optional[RetrievalResult], expanded_from: Optional[str]) -> Optional[ContextSource]:
        nonlocal spent, dropped, dropped_hits, deduplicated
        if chunk.chunk_id in seen_ids:
            deduplicated += 1
            return None
        key = _text_key(chunk.content)
        if key in seen_text:
            deduplicated += 1
            seen_ids.add(chunk.chunk_id)
            return None
        if len(chosen) >= max_sources:
            dropped += 1
            dropped_hits += 1 if result is not None else 0
            return None
        tokens = estimate_tokens(chunk.content or "")
        if spent + tokens > max_tokens and chosen:
            dropped += 1
            dropped_hits += 1 if result is not None else 0
            return None
        if spent + tokens > max_tokens and not chosen:
            # The single best hit is larger than the whole budget: it still
            # goes in whole rather than truncated; the budget is a target.
            pass
        source = ContextSource(
            label="",
            chunk=chunk,
            rank=(result.rank + 1) if result is not None else None,
            score=(float(result.score) if result is not None else None),
            dense_rank=getattr(result, "dense_rank", None) if result is not None else None,
            bm25_rank=getattr(result, "bm25_rank", None) if result is not None else None,
            tokens=tokens,
            expanded_from=expanded_from,
        )
        chosen.append(source)
        seen_ids.add(chunk.chunk_id)
        seen_text.add(key)
        spent += tokens
        return source

    # Pass 1 -- the ranked hits, in rank order, all of them.
    seeded: List[RetrievalResult] = []
    for result in results:
        if try_add(result.chunk, result, None) is not None:
            seeded.append(result)

    # Pass 2 -- same-heading neighbours, on whatever budget the hits left.
    if expand_neighbors and neighbor is not None:
        for result in seeded:
            heading = _heading(result.chunk)
            if not heading:
                continue
            for offset in (-1, 1):
                candidate = neighbor(result.chunk, offset)
                if candidate is None or _heading(candidate) != heading:
                    continue
                if candidate.chunk_id in seen_ids:
                    continue
                added = try_add(candidate, None, result.chunk.chunk_id)
                if added is not None:
                    expanded += 1

    # Reading order is admission order. ``chosen`` already holds every ranked
    # hit in rank order followed by the neighbours, because that is the order
    # the two passes filled it in. Interleaving each hit with its own
    # neighbours would undo pass 1 one layer up: a neighbour of rank 1 would
    # read ahead of the hit at rank 5, and every expansion admitted above it
    # would push that hit further down -- so the more budget expansion is
    # given, the deeper the lowest-ranked evidence is buried. The hits keep
    # the top of the context; an expansion names the hit it continues instead
    # of sitting next to it.
    ordered = chosen
    for index, source in enumerate(ordered, start=1):
        source.label = f"S{index}"
    seed_labels = {source.chunk.chunk_id: source.label for source in ordered}
    text = "\n\n".join(
        _render(source, seed_labels.get(source.expanded_from))
        for source in ordered
    )
    return ContextBundle(
        sources=ordered,
        text=text,
        token_count=spent,
        dropped_over_budget=dropped,
        dropped_ranked_hits=dropped_hits,
        deduplicated=deduplicated,
        expanded=expanded,
        max_tokens=max_tokens,
        labels={source.label: source for source in ordered},
    )
