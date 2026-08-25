"""Retrieval regression: run every gold question and recompute its rank.

The rank recorded when a human marked an answer is **not** used. It was true of
one corpus at one moment; the whole point of a regression run is to find out
whether it is still true, so every rank here is measured again.

Matching a retrieved chunk against a gold entry is deliberately a short ladder
of exact checks rather than a similarity score. Chunk ids move whenever the
parser or the chunker changes -- this corpus has been re-chunked repeatedly --
so the id is the weakest signal, not the strongest, and every match reports
which rule fired.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import runtime
from .runtime import ChunkView

#: How much of the confirmed evidence has to reappear verbatim in a chunk.
#: Long enough to be specific to one passage, short enough to survive a
#: different chunk boundary.
EVIDENCE_PROBE = 120

HIT_LEVELS = (1, 3, 5)


def _collapse(text: Any) -> str:
    return " ".join(str(text or "").split())


def _normalize(text: Any) -> str:
    return _collapse(text).casefold()


def _strip_emphasis(text: str) -> str:
    return text.strip().strip("*_").strip()


@dataclass
class Match:
    rank: int
    rule: str
    chunk_id: str


@dataclass
class QuestionResult:
    question: str
    entry_id: Optional[str]
    rank: Optional[int]
    rule: Optional[str]
    matched_chunk_id: Optional[str]
    expected_chunk_id: Optional[str]
    top_chunk_id: Optional[str]
    retrieved: int
    warnings: List[str] = field(default_factory=list)
    #: The locators the gold entry offered, so a miss can be read without
    #: opening the gold file next to the run.
    expected: Dict[str, Any] = field(default_factory=dict)

    @property
    def reciprocal_rank(self) -> float:
        return 1.0 / self.rank if self.rank else 0.0

    def hit(self, level: int) -> bool:
        return self.rank is not None and self.rank <= level

    def as_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "entry_id": self.entry_id,
            "rank": self.rank,
            "matched_by": self.rule,
            "matched_chunk_id": self.matched_chunk_id,
            "expected_chunk_id": self.expected_chunk_id,
            "top_chunk_id": self.top_chunk_id,
            "retrieved": self.retrieved,
            "reciprocal_rank": round(self.reciprocal_rank, 6),
            **{f"hit@{level}": self.hit(level) for level in HIT_LEVELS},
            "expected": self.expected,
            **({"warnings": self.warnings} if self.warnings else {}),
        }


# --------------------------------------------------------------- the locator


def _same_document(entry: Dict[str, Any], chunk: ChunkView) -> bool:
    """A chunk from another document can never be the confirmed answer."""
    wanted = entry.get("document_id")
    return not wanted or not chunk.doc_id or chunk.doc_id == wanted


def _unit_overlap(entry: Dict[str, Any], chunk: ChunkView) -> bool:
    """Canonical unit ids survive re-chunking; chunk ids do not."""
    wanted = {str(u).split("#", 1)[0] for u in entry.get("unit_ids") or []}
    if not wanted:
        return False
    have = {str(u).split("#", 1)[0] for u in chunk.unit_ids}
    return bool(wanted & have)


def _evidence_present(entry: Dict[str, Any], chunk: ChunkView) -> bool:
    probe = _normalize(entry.get("evidence"))[:EVIDENCE_PROBE]
    return len(probe) >= 20 and probe in _normalize(chunk.text)


def _section_and_page(entry: Dict[str, Any], chunk: ChunkView) -> bool:
    section = _strip_emphasis(_normalize(entry.get("section")))
    if not section:
        return False
    candidates = {_strip_emphasis(_normalize(chunk.heading))}
    for path in chunk.section_paths:
        candidates.add(_strip_emphasis(_normalize(" > ".join(path))))
        for part in path:
            candidates.add(_strip_emphasis(_normalize(part)))
    if section not in candidates:
        return False
    pages = {str(p) for p in entry.get("pages") or []}
    return not pages or bool(pages & {str(p) for p in chunk.pages})


#: Tried in order; the first that fires decides, and its name is reported.
#: ``chunk_id`` sits below the locators that survive re-chunking on purpose.
RULES = (
    ("unit_ids", _unit_overlap),
    ("evidence", _evidence_present),
    ("chunk_id", lambda entry, chunk: bool(
        entry.get("correct_chunk_id")) and chunk.chunk_id == entry["correct_chunk_id"]),
    ("section+page", _section_and_page),
)


def match_entry(entry: Dict[str, Any], chunks: Sequence[ChunkView]) -> Optional[Match]:
    """First retrieved chunk that satisfies any locator rule."""
    for position, chunk in enumerate(chunks, start=1):
        if not _same_document(entry, chunk):
            continue
        for name, rule in RULES:
            if rule(entry, chunk):
                return Match(rank=position, rule=name, chunk_id=chunk.chunk_id)
    return None


# ------------------------------------------------------------------- the run


def evaluate_entry(
    entry: Dict[str, Any],
    chunks: Sequence[ChunkView],
    *,
    check_sha: bool = True,
) -> QuestionResult:
    warnings: List[str] = []
    if check_sha and entry.get("document_sha256") and entry.get("document_id"):
        current = runtime.document_sha(entry["document_id"])
        if current and current != entry["document_sha256"]:
            warnings.append(
                "document changed since this answer was confirmed: gold "
                f"{entry['document_sha256'][:12]} != ingested {current[:12]}"
            )
        elif not current:
            warnings.append(
                f"document {entry['document_id']} is not in the ingest tracker; "
                "its hash could not be checked"
            )

    match = match_entry(entry, chunks)
    return QuestionResult(
        question=entry.get("question", ""),
        entry_id=entry.get("entry_id"),
        rank=match.rank if match else None,
        rule=match.rule if match else None,
        matched_chunk_id=match.chunk_id if match else None,
        expected_chunk_id=entry.get("correct_chunk_id"),
        top_chunk_id=chunks[0].chunk_id if chunks else None,
        retrieved=len(chunks),
        warnings=warnings,
        expected={
            "document_id": entry.get("document_id"),
            "section": entry.get("section"),
            "pages": entry.get("pages") or [],
            "unit_ids": entry.get("unit_ids") or [],
            "evidence": _collapse(entry.get("evidence"))[:EVIDENCE_PROBE],
        },
    )


def summarize(results: Sequence[QuestionResult]) -> Dict[str, Any]:
    total = len(results)
    if not total:
        return {"questions": 0, **{f"hit@{n}": 0.0 for n in HIT_LEVELS}, "mrr": 0.0,
                "not_found": 0}
    return {
        "questions": total,
        **{
            f"hit@{level}": round(
                sum(1 for r in results if r.hit(level)) / total, 6
            )
            for level in HIT_LEVELS
        },
        "mrr": round(sum(r.reciprocal_rank for r in results) / total, 6),
        "not_found": sum(1 for r in results if r.rank is None),
    }


def execute(
    kb: Dict[str, Any],
    gold_path: str,
    *,
    top_k: int = 10,
    method: Optional[str] = None,
    check_sha: bool = True,
) -> Tuple[Dict[str, Any], List[QuestionResult]]:
    """Run every gold question through production retrieval, once.

    Returns the run record and the per-question results. Both the ``eval``
    command and the report build on this, so a metric can never differ
    depending on which command asked for it.
    """
    gold = runtime.load_gold(gold_path)
    entries = gold.get("entries") or []
    if not entries:
        raise runtime.CliError(f"{gold_path} has no entries")

    method = method or runtime.default_method(kb)
    results: List[QuestionResult] = []
    for entry in entries:
        hits = runtime.search(kb, entry["question"], top_k, method)
        chunks = [ChunkView.of(hit.chunk) for hit in hits]
        results.append(evaluate_entry(entry, chunks, check_sha=check_sha))

    run = {
        "kind": "retrieval-eval",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "gold": {
            "path": os.path.abspath(gold_path),
            "schema_version": gold.get("schema_version"),
            "frozen_at": gold.get("frozen_at"),
            "questions": len(entries),
        },
        "retrieval": {"method": method, "top_k": top_k},
        "environment": runtime.run_environment(kb),
        "documents": [
            {"document_id": doc_id, "sha256": runtime.document_sha(doc_id)}
            for doc_id in sorted({
                e["document_id"] for e in entries if e.get("document_id")
            })
        ],
        "metrics": summarize(results),
        "questions": [result.as_dict() for result in results],
        "document_sha_mismatch": any(r.warnings for r in results),
    }
    return run, results


def render_summary(summary: Dict[str, Any]) -> str:
    lines = [f"Questions: {summary['questions']}"]
    for level in HIT_LEVELS:
        lines.append(f"Hit@{level}: {summary[f'hit@{level}'] * 100:5.1f}%")
    lines.append(f"MRR:   {summary['mrr']:8.2f}")
    if summary.get("not_found"):
        lines.append(f"Not found: {summary['not_found']}")
    return "\n".join(lines)


# -------------------------------------------------------------------- compare


REGRESSION_EPSILON = 1e-9


def compare_runs(previous: Dict[str, Any], current: Dict[str, Any]) -> Dict[str, Any]:
    """Metric deltas and the questions whose rank moved."""
    old_metrics = previous.get("metrics", {})
    new_metrics = current.get("metrics", {})
    metrics = []
    for name in [f"hit@{n}" for n in HIT_LEVELS] + ["mrr"]:
        before, after = old_metrics.get(name), new_metrics.get(name)
        if before is None or after is None:
            continue
        delta = after - before
        if delta > REGRESSION_EPSILON:
            verdict = "IMPROVEMENT"
        elif delta < -REGRESSION_EPSILON:
            verdict = "REGRESSION"
        else:
            verdict = "same"
        metrics.append(
            {"metric": name, "before": before, "after": after,
             "delta": round(delta, 6), "verdict": verdict}
        )

    def by_question(run):
        return {_normalize(q["question"]): q for q in run.get("questions", [])}

    old_q, new_q = by_question(previous), by_question(current)
    moved = []
    for index, question in enumerate(current.get("questions", []), start=1):
        before = old_q.get(_normalize(question["question"]))
        if before is None:
            moved.append({"index": index, "question": question["question"],
                          "before": None, "after": question["rank"], "change": "new"})
            continue
        if before["rank"] != question["rank"]:
            moved.append({
                "index": index,
                "question": question["question"],
                "before": before["rank"],
                "after": question["rank"],
                "change": _rank_change(before["rank"], question["rank"]),
            })
    dropped = [q for key, q in old_q.items() if key not in new_q]
    return {"metrics": metrics, "moved": moved,
            "dropped": [q["question"] for q in dropped]}


def _rank_change(before: Optional[int], after: Optional[int]) -> str:
    if after is None:
        return "lost"
    if before is None:
        return "found"
    return "better" if after < before else "worse"


def render_comparison(comparison: Dict[str, Any]) -> str:
    lines = []
    for row in comparison["metrics"]:
        mark = "" if row["verdict"] == "same" else f"  {row['verdict']}"
        lines.append(
            f"{row['metric']:<6} {row['before']:.2f} -> {row['after']:.2f}{mark}"
        )
    if comparison["moved"]:
        lines.append("")
        for row in comparison["moved"]:
            before = "not found" if row["before"] is None else f"rank {row['before']}"
            after = "not found" if row["after"] is None else f"rank {row['after']}"
            lines.append(f"#{row['index']} {before} -> {after}")
    if comparison["dropped"]:
        lines.append("")
        lines.append(f"{len(comparison['dropped'])} question(s) not in this run")
    return "\n".join(lines)


def has_regression(comparison: Dict[str, Any]) -> bool:
    return any(row["verdict"] == "REGRESSION" for row in comparison["metrics"])
