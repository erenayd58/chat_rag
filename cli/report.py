"""A shareable quality package for one knowledge base and document.

The report answers a question a metric alone cannot: *can someone else look at
this corpus and check it?* So it exports what was actually stored -- the
canonical units the parser produced, the chunks the store holds -- next to the
structural findings and the retrieval numbers measured from them, and a
manifest saying what produced all of it.

Nothing here re-parses, re-chunks or re-scores anything. Every artefact comes
from production state, every finding from ``amsc.quality.lint``, every metric
from the same evaluator ``eval`` uses. The one thing this module adds is
consistency checking: a report that quietly exported the wrong corpus would be
worse than no report at all, so mismatches are named and they change the
status.
"""

from __future__ import annotations

import csv
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence

from . import evaluate as ev
from . import runtime
from .manifest import build_manifest
from .manifest import pipeline_facts as current_pipeline_facts
from .runtime import ChunkView, CliError

REPORTS_DIR = os.path.join("artifacts", "reports")

#: Rules whose HIGH findings are accepted, documented limitations rather than
#: regressions. They are still listed in full -- accepted is not hidden.
KNOWN_LIMITATION_RULES = frozenset({"unresolved_visual"})

PASS = "PASS"
PASS_WITH_KNOWN_LIMITATIONS = "PASS_WITH_KNOWN_LIMITATIONS"
FAIL = "FAIL"

OK, WARNING, FAILURE = "ok", "warning", "failure"

_UNSAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")


@dataclass(frozen=True)
class Check:
    name: str
    level: str
    detail: str

    @property
    def ok(self) -> bool:
        return self.level == OK

    def as_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "level": self.level, "detail": self.detail}


# --------------------------------------------------------------- destination


def report_directory(kb_name: str, root: str = REPORTS_DIR) -> str:
    """A fresh directory per report, never overwriting an earlier one.

    Two reports can be produced inside the same second, and on Windows a
    knowledge base name may contain characters a path cannot, so the name is
    reduced to safe characters and a suffix is added until the name is unused.
    """
    safe = _UNSAFE_NAME.sub("-", kb_name or "kb").strip("-") or "kb"
    base = os.path.join(root, f"{safe[:60]}-{datetime.now():%Y%m%dT%H%M%S}")
    candidate, index = base, 1
    while os.path.exists(candidate):
        index += 1
        candidate = f"{base}-{index}"
    os.makedirs(candidate)
    return candidate


# ------------------------------------------------------------------ exports


def write_json(path: str, payload: Any) -> str:
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    return path


def write_jsonl(path: str, rows: Sequence[Dict[str, Any]]) -> str:
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return path


FINDING_COLUMNS = (
    "rule", "confidence", "target_id", "page", "known_limitation", "reason", "evidence",
)


def write_findings_csv(path: str, findings: Sequence[Any]) -> str:
    """The same findings as the JSON, in a form a spreadsheet opens.

    Written with a BOM because the evidence is Turkish and Excel otherwise
    reads UTF-8 as the local code page.
    """
    with open(path, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(FINDING_COLUMNS)
        for finding in findings:
            writer.writerow([
                finding.rule,
                finding.confidence,
                finding.target_id,
                "" if finding.page is None else finding.page,
                "yes" if finding.rule in KNOWN_LIMITATION_RULES else "no",
                finding.reason,
                finding.evidence,
            ])
    return path


def qa_payload(report: Any, context: Dict[str, Any]) -> Dict[str, Any]:
    """Machine-readable structural QA, counted from the findings themselves.

    ``context`` says where the findings came from -- a pointer to the
    manifest beside it in a report directory, or the run environment when
    the QA file stands alone.
    """
    grouped = report.by_confidence()
    rules = {
        rule: dict(counts) for rule, counts in sorted(report.rule_counts().items())
    }
    high = grouped.get("HIGH") or []
    return {
        "kind": "structural-qa",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        **context,
        "unit_count": report.unit_count,
        "chunk_count": report.chunk_count,
        "summary": {
            "HIGH": len(high),
            "MEDIUM": len(grouped.get("MEDIUM") or []),
            "LOW": len(grouped.get("LOW") or []),
            "rules": rules,
            "known_limitation_high": sum(
                1 for f in high if f.rule in KNOWN_LIMITATION_RULES
            ),
            "unexpected_high": sum(
                1 for f in high if f.rule not in KNOWN_LIMITATION_RULES
            ),
        },
        "known_limitation_rules": sorted(KNOWN_LIMITATION_RULES),
        "findings": [
            {**finding.__dict__,
             "known_limitation": finding.rule in KNOWN_LIMITATION_RULES}
            for finding in report.findings
        ],
    }


# ----------------------------------------------------------------- checks


def consistency_checks(
    *,
    kb: Dict[str, Any],
    manifest: Dict[str, Any],
    units: Sequence[Dict[str, Any]],
    views: Sequence[ChunkView],
    document: Optional[Dict[str, Any]],
    run: Optional[Dict[str, Any]],
    gold: Optional[Dict[str, Any]],
    current: Optional[Dict[str, Any]] = None,
    expect_units: Optional[int] = None,
    expect_chunks: Optional[int] = None,
) -> List[Check]:
    """Everything that has to agree, checked out loud.

    A silent mismatch is the failure mode this exists to prevent, so each one
    is named and carries a level the status rules act on.

    ``manifest["pipeline"]`` describes the corpus -- ingest-time when that was
    captured -- while ``current`` describes the wiring the retrieval in this
    run actually used. Corpus questions are asked of the first, retrieval
    questions of the second, and a difference between them is itself reported.
    """
    checks: List[Check] = []
    counts = manifest["counts"]
    pipeline = manifest["pipeline"]
    current = current or pipeline
    provenance = manifest.get("provenance") or {}

    checks.append(_equal_count(
        "canonical_count_matches_manifest", len(units), counts["canonical_units"],
        "exported canonical units",
    ))
    checks.append(_equal_count(
        "chunk_count_matches_manifest", len(views), counts["chunks"],
        "exported chunks",
    ))

    if expect_units is not None:
        checks.append(_equal_count(
            "canonical_count_matches_expectation", len(units), expect_units,
            "canonical units (--expect-units)",
        ))
    if expect_chunks is not None:
        checks.append(_equal_count(
            "chunk_count_matches_expectation", len(views), expect_chunks,
            "chunks (--expect-chunks)",
        ))

    # Which configuration this report is describing. A fallback is legitimate
    # for a corpus older than snapshots, but it must never read as history.
    if provenance.get("source") == "ingest-snapshot":
        checks.append(Check(
            "ingest_snapshot_available", OK,
            "pipeline configuration captured at ingest on "
            f"{provenance.get('captured_at')}",
        ))
    else:
        checks.append(Check(
            "ingest_snapshot_available", WARNING,
            "no configuration was captured when this document was ingested; "
            "the pipeline block describes the current runtime instead",
        ))

    # A knowledge base can be re-pointed at another profile after ingest. That
    # is allowed, but then the corpus and the retrieval in this report were
    # produced by two different configurations.
    if provenance.get("source") == "ingest-snapshot":
        drifted = [
            field for field in ("retrieval_profile", "retriever")
            if pipeline.get(field) != current.get(field)
        ]
        checks.append(Check(
            "retrieval_config_unchanged_since_ingest",
            OK if not drifted else WARNING,
            "retrieval is configured as it was at ingest" if not drifted else
            "; ".join(
                f"{field}: {pipeline.get(field)!r} at ingest, "
                f"{current.get(field)!r} now"
                for field in drifted
            ),
        ))

    # Every unit a chunk cites has to exist in the exported stream, or the
    # export is not the stream those chunks were built from.
    known = {row.get("unit_id") for row in units}
    missing = sorted({
        u.split("#", 1)[0] for view in views for u in view.unit_ids
        if u.split("#", 1)[0] not in known
    })
    checks.append(Check(
        "canonical_covers_chunk_units",
        OK if not missing else FAILURE,
        "every unit cited by a chunk is in the exported canonical stream"
        if not missing else
        f"{len(missing)} cited unit id(s) are absent, e.g. {', '.join(missing[:5])}",
    ))

    # Chunks belonging to another document mean the export mixes corpora.
    document_id = (document or {}).get("doc_id")
    foreign = sorted({v.doc_id for v in views if document_id and v.doc_id != document_id})
    checks.append(Check(
        "chunks_belong_to_the_document",
        OK if not foreign else FAILURE,
        f"all {len(views)} chunks belong to {document_id}" if not foreign else
        f"chunks from other document(s): {', '.join(foreign[:5])}",
    ))

    # Heading metadata: a knowledge base ingested before the chunker's heading
    # was persisted cannot be QA'd on headings, and reporting its findings as
    # if it could is the misleading outcome.
    with_heading = sum(1 for v in views if v.heading_source == "metadata")
    if not views:
        checks.append(Check("chunk_heading_metadata", FAILURE, "no chunks stored"))
    elif with_heading:
        checks.append(Check(
            "chunk_heading_metadata", OK,
            f"{with_heading}/{len(views)} chunks carry the chunker's own heading",
        ))
    else:
        checks.append(Check(
            "chunk_heading_metadata", WARNING,
            "no chunk carries the chunker's heading; this corpus predates that "
            "metadata, so heading rules over-report. Re-ingest for accurate ones",
        ))

    # Retrieval configuration: what ran has to be something this profile can
    # serve, and a profile that computes no embeddings must not have run dense.
    if run:
        method = run["retrieval"]["method"]
        available = [
            entry["name"]
            for entry in runtime.capabilities_of(kb)["methods"]
            if entry["available"]
        ]
        checks.append(Check(
            "retrieval_method_supported",
            OK if method in available else FAILURE,
            f"{method!r} is served by {current['retriever']}" if method in available
            else f"{method!r} is not available on the {current['retrieval_profile']} "
                 f"profile (available: {', '.join(available) or 'none'})",
        ))
        dense_ran = method in {"vector", "hybrid"}
        checks.append(Check(
            "no_dense_retrieval_without_embeddings",
            OK if not (dense_ran and not current["uses_embeddings"]) else FAILURE,
            f"uses_embeddings={current['uses_embeddings']}, method={method!r}",
        ))

        if current["configured_embedding_model"] and not current["uses_embeddings"]:
            checks.append(Check(
                "configured_embedding_model_unused", OK,
                f"{current['configured_embedding_model']!r} is configured but not "
                f"used: {current['retriever']} needs no document embeddings",
            ))

        if gold is not None:
            expected_questions = len(gold.get("entries") or [])
            checks.append(_equal_count(
                "gold_question_count", run["metrics"]["questions"], expected_questions,
                "questions evaluated",
            ))

        stored_sha = (document or {}).get("file_hash")
        gold_shas = sorted({
            e["document_sha256"] for e in (gold or {}).get("entries") or []
            if e.get("document_sha256")
        })
        if not gold_shas:
            checks.append(Check(
                "gold_document_sha", WARNING,
                "no gold entry records a document hash, so a replaced corpus "
                "cannot be told from a real regression",
            ))
        elif stored_sha and gold_shas == [stored_sha]:
            checks.append(Check(
                "gold_document_sha", OK,
                f"gold was confirmed against the ingested document ({stored_sha[:12]})",
            ))
        else:
            checks.append(Check(
                "gold_document_sha", FAILURE,
                f"gold references {', '.join(s[:12] for s in gold_shas)} but the "
                f"ingested document is {(stored_sha or 'unknown')[:12]}",
            ))

    return checks


def _equal_count(name: str, actual: int, expected: int, what: str) -> Check:
    return Check(
        name,
        OK if actual == expected else FAILURE,
        f"{actual} {what}" if actual == expected
        else f"{actual} {what}, expected {expected}",
    )


def status_for(qa_summary: Dict[str, Any], checks: Sequence[Check]) -> str:
    """Deterministic, and never quiet about what it forgave.

    An unexpected HIGH finding or a broken invariant fails. A HIGH finding on
    a rule that is an accepted limitation, or a check that only warns, passes
    but says so in the status itself.
    """
    if any(check.level == FAILURE for check in checks):
        return FAIL
    if qa_summary["unexpected_high"]:
        return FAIL
    if qa_summary["known_limitation_high"] or any(
        check.level == WARNING for check in checks
    ):
        return PASS_WITH_KNOWN_LIMITATIONS
    return PASS


# ----------------------------------------------------------------- summary


def render_summary(
    *,
    manifest: Dict[str, Any],
    qa: Dict[str, Any],
    run: Optional[Dict[str, Any]],
    checks: Sequence[Check],
    status: str,
    known_high: Sequence[Any],
) -> str:
    document = manifest.get("document") or {}
    pipeline = manifest["pipeline"]
    counts = manifest["counts"]
    name = document.get("filename") or manifest["knowledge_base"]["name"]

    lines: List[str] = [f"# {name} Pipeline QA Report", ""]

    lines += ["## Pipeline", ""]
    embeddings = (
        "Embeddings computed"
        if pipeline["uses_embeddings"]
        else "No embeddings"
        + (
            f" ({pipeline['configured_embedding_model']} configured, unused)"
            if pipeline["configured_embedding_model"]
            else ""
        )
    )
    # Fenced so the arrows stack: as plain lines markdown would reflow the
    # whole pipeline into one paragraph.
    lines += [
        "```",
        f"{pipeline['parser'] or 'unknown parser'} "
        f"({pipeline['parser_backend'] or 'no backend'})",
        f"→ Canonical Document Model ({pipeline['normalization_version'] or 'n/a'})",
        f"→ {pipeline['chunker']} chunking",
        f"→ {pipeline['retriever']} ({pipeline['retrieval_profile']})",
        f"→ {embeddings}",
        "→ No LLM in this path",
        "```",
        "",
    ]

    provenance = manifest.get("provenance") or {}
    if provenance.get("source") == "ingest-snapshot":
        lines += [
            f"Configuration captured at ingest ({provenance.get('captured_at')}).",
            "",
        ]
    else:
        lines += [
            "**Configuration below is the current runtime, not the one that "
            "produced this corpus** -- nothing was captured when this document "
            "was ingested.",
            "",
        ]

    lines += ["## Document", ""]
    lines += [
        f"- Document: {document.get('document_id') or '-'}",
        f"- SHA256: {document.get('sha256') or '-'}",
        f"- Pages: {document.get('pages') or '-'}",
        f"- Canonical units: {counts['canonical_units']}",
        f"- Chunks: {counts['chunks']}",
        f"- Canonical units referenced by chunks: "
        f"{counts['canonical_units_referenced_by_chunks']}",
        f"- Pages with canonical units: {counts['pages_with_canonical_units']}",
        "",
    ]

    summary = qa["summary"]
    lines += ["## Structural QA", ""]
    lines += [
        f"- HIGH: {summary['HIGH']}  "
        f"(unexpected: {summary['unexpected_high']}, "
        f"known limitation: {summary['known_limitation_high']})",
        f"- MEDIUM: {summary['MEDIUM']}",
        f"- LOW: {summary['LOW']}",
        "",
        "| rule | HIGH | MEDIUM | LOW |",
        "| --- | ---: | ---: | ---: |",
    ]
    for rule, counts_by_level in qa["summary"]["rules"].items():
        lines.append(
            f"| {rule} | {counts_by_level.get('HIGH', 0)} | "
            f"{counts_by_level.get('MEDIUM', 0)} | {counts_by_level.get('LOW', 0)} |"
        )
    lines.append("")
    lines.append("Known limitations (accepted, still reported):")
    if known_high:
        for finding in known_high:
            lines.append(f"- `{finding.target_id}` [{finding.rule}] {finding.reason}")
    else:
        lines.append("- none")
    lines.append("")

    lines += ["## Retrieval QA", ""]
    if run:
        metrics = run["metrics"]
        lines += [
            f"- Gold: {os.path.basename(run['gold']['path'])} "
            f"(frozen {run['gold']['frozen_at'] or 'n/a'})",
            f"- Method: {run['retrieval']['method']}, top-k {run['retrieval']['top_k']}",
            f"- Questions: {metrics['questions']}",
            f"- Hit@1: {metrics['hit@1'] * 100:.1f}%",
            f"- Hit@3: {metrics['hit@3'] * 100:.1f}%",
            f"- Hit@5: {metrics['hit@5'] * 100:.1f}%",
            f"- MRR: {metrics['mrr']:.2f}",
            f"- Not found: {metrics['not_found']}",
            "",
            "Reported as a baseline; no threshold gates the status.",
            "",
        ]
    else:
        lines += ["- Not run (no gold set supplied).", ""]

    lines += ["## Consistency checks", ""]
    for check in checks:
        mark = {OK: "OK", WARNING: "WARNING", FAILURE: "FAILURE"}[check.level]
        lines.append(f"- **{mark}** {check.name}: {check.detail}")
    lines.append("")

    lines += ["## Status", "", status, ""]
    lines += [
        "Rules: an unexpected HIGH finding or a failed consistency check is FAIL; "
        "a HIGH finding on an accepted-limitation rule, or a warning, is "
        "PASS_WITH_KNOWN_LIMITATIONS; otherwise PASS. Retrieval metrics are "
        "reported, not gated.",
        "",
    ]

    lines += ["## Versions", ""]
    versions = manifest["versions"]
    lines += [
        f"- chat_rag: {versions['chat_rag_git_sha'] or 'unknown'}",
        f"- amsc: {versions['amsc_git_sha'] or 'unknown'} "
        f"({versions.get('amsc_version') or 'n/a'})",
        f"- Python {versions['python']} on {versions['platform']}",
    ]
    for package, version in versions["important_dependencies"].items():
        lines.append(f"- {package}: {version or 'not installed'}")
    lines.append("")

    if manifest["warnings"]:
        lines += ["## Manifest warnings", ""]
        lines += [f"- {warning}" for warning in manifest["warnings"]]
        lines.append("")
    return "\n".join(lines)


# ------------------------------------------------------------------ build


def build(
    kb: Dict[str, Any],
    *,
    gold_path: Optional[str] = None,
    document: Optional[Dict[str, Any]] = None,
    units_path: Optional[str] = None,
    out_dir: Optional[str] = None,
    top_k: int = 10,
    method: Optional[str] = None,
    expect_units: Optional[int] = None,
    expect_chunks: Optional[int] = None,
) -> Dict[str, Any]:
    """Produce the report directory and return what it contains."""
    from amsc.quality.lint import lint

    views = runtime.all_chunks(kb)
    if not views:
        raise CliError(f"{kb['name']} has no chunks to report on")
    if document:
        views = [v for v in views if v.doc_id == document.get("doc_id")] or views

    units, units_source, warnings = _canonical_units(views, units_path)

    manifest = build_manifest(
        kb, document=document, units=units, views=views, extra_warnings=warnings,
    )
    manifest["canonical_source"] = units_source

    run: Optional[Dict[str, Any]] = None
    gold: Optional[Dict[str, Any]] = None
    if gold_path:
        gold = runtime.load_gold(gold_path)
        run, _ = ev.execute(kb, gold_path, top_k=top_k, method=method)

    qa_report = lint(units, [view.as_qa_row() for view in views])
    qa = qa_payload(qa_report, {"manifest": "manifest.json"})
    known_high = [
        finding for finding in qa_report.by_confidence().get("HIGH") or []
        if finding.rule in KNOWN_LIMITATION_RULES
    ]

    checks = consistency_checks(
        kb=kb, manifest=manifest, units=units, views=views, document=document,
        run=run, gold=gold, current=current_pipeline_facts(kb),
        expect_units=expect_units, expect_chunks=expect_chunks,
    )
    status = status_for(qa["summary"], checks)
    manifest["status"] = status
    manifest["consistency_checks"] = [check.as_dict() for check in checks]

    directory = out_dir or report_directory(kb.get("name") or kb["kb_id"])
    os.makedirs(directory, exist_ok=True)

    write_json(os.path.join(directory, "manifest.json"), manifest)
    write_jsonl(os.path.join(directory, "canonical.units.jsonl"), units)
    write_jsonl(
        os.path.join(directory, "chunks.jsonl"),
        [view.as_export_row() for view in views],
    )
    write_json(os.path.join(directory, "structural_qa.json"), qa)
    write_findings_csv(
        os.path.join(directory, "structural_qa.csv"), qa_report.findings
    )
    if run:
        write_json(os.path.join(directory, "retrieval_eval.json"), run)

    summary = render_summary(
        manifest=manifest, qa=qa, run=run, checks=checks, status=status,
        known_high=known_high,
    )
    with open(os.path.join(directory, "summary.md"), "w",
              encoding="utf-8", newline="\n") as handle:
        handle.write(summary)

    return {
        "directory": directory,
        "manifest": manifest,
        "qa": qa,
        "run": run,
        "checks": checks,
        "status": status,
        "units": len(units),
        "chunks": len(views),
    }


def _canonical_units(
    views: Sequence[ChunkView], units_path: Optional[str],
) -> tuple:
    """The canonical stream these chunks came from, or a refusal.

    A report whose canonical export came from an older parser run would look
    clean and mean nothing, so an unproven stream is never substituted.
    """
    if units_path:
        from chat_rag.components.parsers.canonical_units_store import load_units

        if not os.path.isfile(units_path):
            raise CliError(f"Canonical units file not found: {units_path}")
        return load_units(units_path), os.path.abspath(units_path), [
            f"canonical units supplied by hand from {units_path}"
        ]

    match = runtime.resolve_canonical_units(views)
    if match is None:
        raise CliError(
            "Could not identify the canonical units these chunks were built "
            "from. The parser cache holds no stream that contains every unit "
            "the chunks cite and reproduces their text, so exporting one would "
            "risk shipping an older document's units. Re-ingest the document, "
            "or pass --units <path> if you know which stream is correct."
        )
    return match.units, os.path.abspath(match.path), []
