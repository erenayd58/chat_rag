"""Minimal CLI for the retrieval proof of concept.

Its reason to exist is `eval`: run a frozen gold set through production
retrieval and say whether the numbers moved. The other commands are the small
amount of context needed to act on that answer -- look at one query, check the
corpus is structurally sound, see how a knowledge base is configured.

Everything runs through the web app's own wiring, so a number here is a number
the app would produce. Argparse only; no CLI framework is added for four
commands.

    python -m cli eval    --kb kkb-final --gold artifacts/gold/kkb2024-v1.json
    python -m cli search  --kb kkb-final --query "..." --top-k 5
    python -m cli qa      --kb kkb-final
    python -m cli inspect --kb kkb-final [--json]
    python -m cli report  --kb kkb-final --gold artifacts/gold/kkb2024-v1.json
    python -m cli gold export --kb kkb-final --out artifacts/gold/kkb2024-v1.json --all
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional

from . import evaluate as ev
from . import report as reporting
from . import runtime
from components.goldset.manager import entry_id_for as gold_entry_id

from .manifest import build_manifest
from .runtime import (
    ChunkView,
    CliError,
    all_chunks,
    default_method,
    gold_manager,
    resolve_kb,
    run_environment,
    search,
)

RUNS_DIR = os.path.join("artifacts", "runs")


def _write_json(path: str, payload: Dict[str, Any]) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    return path


# ------------------------------------------------------------------ eval


def _print_sha_warnings(results: List[ev.QuestionResult]) -> List[ev.QuestionResult]:
    warned = [r for r in results if r.warnings]
    if warned:
        print()
        print(f"WARNING: {len(warned)} question(s) reference a document that has "
              f"changed or is untracked:")
        for result in warned[:10]:
            for warning in result.warnings:
                print(f"  - {result.question[:60]}: {warning}")
    return warned


def cmd_eval(args: argparse.Namespace) -> int:
    kb = resolve_kb(args.kb)
    run, results = ev.execute(
        kb, args.gold, top_k=args.top_k, method=args.method,
        check_sha=not args.no_sha_check,
    )
    print(ev.render_summary(run["metrics"]))
    warned = _print_sha_warnings(results)

    out = args.out or os.path.join(
        RUNS_DIR, f"{datetime.now():%Y%m%dT%H%M%S}-retrieval.json"
    )
    _write_json(out, run)
    print(f"\nRun written to {out}")

    exit_code = 0
    if args.compare:
        with open(args.compare, "r", encoding="utf-8") as handle:
            previous = json.load(handle)
        comparison = ev.compare_runs(previous, run)
        print()
        print(f"Compared with {args.compare}")
        print(ev.render_comparison(comparison))
        if ev.has_regression(comparison) and args.strict:
            exit_code = 1
    if warned and args.strict:
        exit_code = 1
    return exit_code


# ---------------------------------------------------------------- search


def cmd_search(args: argparse.Namespace) -> int:
    kb = resolve_kb(args.kb)
    method = args.method or default_method(kb)
    hits = search(kb, args.query, args.top_k, method)
    if not hits:
        print("No results.")
        return 0
    print(f"{args.query!r} • {method} • top {args.top_k}\n")
    for rank, hit in enumerate(hits, start=1):
        view = ChunkView.of(hit.chunk)
        pages = ", ".join(str(p) for p in view.pages) or "-"
        preview = " ".join((view.text or "").split())[:160]
        print(f"[{rank}] score {hit.score:.4f}  page {pages}")
        print(f"    {view.heading or '(no heading)'}")
        print(f"    {preview}{'…' if len(view.text or '') > 160 else ''}")
        print(f"    {view.chunk_id}")
        print()
    return 0


# -------------------------------------------------------------------- qa


def cmd_qa(args: argparse.Namespace) -> int:
    """Wrapper around amsc.structural_qa; no QA logic lives here."""
    try:
        from amsc.structural_qa import lint, render
    except ImportError as exc:  # pragma: no cover - dependency error path
        raise CliError(f"amsc.structural_qa is unavailable: {exc}")

    kb = resolve_kb(args.kb)
    views = all_chunks(kb)
    if not views:
        raise CliError(f"{kb['name']} has no chunks to check")
    chunks = [view.as_qa_row() for view in views]

    # Heading-dependent rules compare a chunk's heading against the titles in
    # its body. On a knowledge base ingested before the chunker's heading was
    # persisted, only the derived section_title survives -- it drops the
    # accumulated heading lines, so those rules fire on chunks that are fine.
    if not any(view.heading_source == "metadata" for view in views):
        print("NOTE: this knowledge base was ingested before the chunk heading")
        print("      was stored, so only the derived section_title is available.")
        print("      chunk_heading_mismatch will over-report here. Re-ingest the")
        print("      document to get accurate heading rules.")
        print()

    units: List[Dict[str, Any]] = []
    if args.units:
        with open(args.units, "r", encoding="utf-8") as handle:
            units = [json.loads(line) for line in handle if line.strip()]
    else:
        units = _canonical_units_for(views)

    report = lint(units, chunks)
    print(render(report, limit=args.limit))
    if args.out:
        _write_json(args.out, reporting.qa_payload(
            report, {"environment": run_environment(kb)}
        ))
        print(f"Report written to {args.out}")
    return 0


def _canonical_units_for(views: List[ChunkView]) -> List[Dict[str, Any]]:
    """The canonical stream these chunks were built from, or none at all.

    Unit-level rules are skipped rather than run against a stream that only
    resembles this one: a near-match comes from an older parser version, and
    its findings would be about a document nobody has.
    """
    match = runtime.resolve_canonical_units(views)
    if match is None:
        print("NOTE: no canonical-unit cache provably matches this knowledge "
              "base; unit-level rules are skipped. Pass --units to point at "
              "one.\n")
        return []
    return match.units


# --------------------------------------------------------------- inspect


def _select_document(
    kb: Dict[str, Any], reference: Optional[str], *, required: bool = True
) -> Optional[Dict[str, Any]]:
    """The one document a report describes, named explicitly when ambiguous."""
    documents = runtime.documents_for(kb)
    if not documents:
        if required:
            raise CliError(f"{kb['name']} has no ingested documents")
        return None
    if reference:
        for document in documents:
            names = {
                document.get("doc_id"),
                document.get("file_name"),
                (document.get("metadata") or {}).get("original_filename"),
            }
            if reference in names:
                return document
        known = ", ".join(sorted(d.get("doc_id", "?") for d in documents))
        raise CliError(f"No document {reference!r} in {kb['name']}. Known: {known}")
    if len(documents) == 1:
        return documents[0]
    if not required:
        return None
    known = ", ".join(sorted(d.get("doc_id", "?") for d in documents))
    raise CliError(
        f"{kb['name']} holds {len(documents)} documents; pass --document <id>. "
        f"Known: {known}"
    )


def cmd_inspect(args: argparse.Namespace) -> int:
    kb = resolve_kb(args.kb)
    documents = runtime.documents_for(kb)
    views = all_chunks(kb)
    unit_ids = {u for view in views for u in view.unit_ids}
    gold_entries = gold_manager.list(kb_id=kb["kb_id"])

    if args.json or args.out:
        document = _select_document(kb, args.document, required=False)
        extra = []
        if document is None and len(documents) > 1:
            extra.append(
                f"{len(documents)} documents in this knowledge base; pass "
                "--document to describe one of them"
            )
        match = runtime.resolve_canonical_units(views)
        if match is None:
            extra.append("no canonical-unit stream provably matches these chunks")
        manifest = build_manifest(
            kb, document=document, units=match.units if match else [],
            views=views, extra_warnings=extra,
        )
        if match:
            manifest["canonical_source"] = os.path.abspath(match.path)
        if args.out:
            _write_json(args.out, manifest)
            print(f"Manifest written to {args.out}")
        else:
            print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    environment = run_environment(kb)
    rows = [
        ("Knowledge base", f"{environment['kb_name']}  ({environment['kb_id']})"),
        ("Chunker", environment["chunker"]),
        ("Retrieval profile", environment["retrieval_profile"]),
        ("Retriever", environment["retriever"]),
        ("Uses embeddings", "yes" if environment["uses_embeddings"] else "no"),
        ("Embedding model", environment["embedding_model"] or "-"),
        ("Vector store", environment["storage_path"]),
        ("Documents", str(len(documents))),
        ("Chunks", str(len(views))),
        ("Canonical units referenced", str(len(unit_ids))),
        ("Runtime gold questions", str(len(gold_entries))),
    ]
    width = max(len(label) for label, _ in rows)
    for label, value in rows:
        print(f"{label:<{width}} : {value}")

    if documents:
        print("\nDocuments")
        for doc in documents:
            sha = (doc.get("file_hash") or "")[:12] or "-"
            print(f"  {doc['file_name']}  chunks={doc['chunk_count']}  sha={sha}")
    return 0


# ---------------------------------------------------------------- report


def cmd_report(args: argparse.Namespace) -> int:
    """The whole quality package for one document, in one directory."""
    kb = resolve_kb(args.kb)
    document = _select_document(kb, args.document)
    built = reporting.build(
        kb,
        gold_path=args.gold,
        document=document,
        units_path=args.units,
        out_dir=args.out,
        top_k=args.top_k,
        method=args.method,
        expect_units=args.expect_units,
        expect_chunks=args.expect_chunks,
    )

    summary = built["qa"]["summary"]
    print(f"Report: {built['directory']}")
    print(f"Status: {built['status']}")
    print()
    print(f"Canonical units: {built['units']}")
    print(f"Chunks: {built['chunks']}")
    print()
    print("Structural QA:")
    print(f"HIGH: {summary['HIGH']}")
    print(f"MEDIUM: {summary['MEDIUM']}")
    print(f"LOW: {summary['LOW']}")
    print(f"Known visual HIGH: {summary['known_limitation_high']}")
    if summary["unexpected_high"]:
        print(f"Unexpected HIGH: {summary['unexpected_high']}")

    run = built["run"]
    if run:
        print()
        print("Retrieval:")
        print(ev.render_summary(run["metrics"]))

    problems = [c for c in built["checks"] if not c.ok]
    if problems:
        print()
        print("Consistency:")
        for check in problems:
            print(f"  {check.level.upper():<7} {check.name}: {check.detail}")

    if built["status"] == reporting.FAIL:
        return 1
    if args.strict and problems:
        return 1
    return 0


# ------------------------------------------------------------------ gold


def cmd_gold_export(args: argparse.Namespace) -> int:
    """Freeze reviewed runtime marks into a regression set.

    Nothing is treated as verified on its own: without --all or --entry the
    export refuses, and what it writes is printed for the human to check.
    """
    kb = resolve_kb(args.kb)
    entries = gold_manager.list(kb_id=kb["kb_id"])
    if not entries:
        raise CliError(f"No runtime gold entries for {kb['name']}")

    if args.entry:
        wanted = set(args.entry)
        chosen = [e for e in entries if e["entry_id"] in wanted]
        missing = wanted - {e["entry_id"] for e in chosen}
        if missing:
            raise CliError("Unknown entry id(s): " + ", ".join(sorted(missing)))
    elif args.all:
        chosen = entries
    else:
        print(f"{len(entries)} runtime entr(ies) for {kb['name']}:")
        for entry in entries:
            print(f"  {entry['entry_id']}  {entry['question']}")
        raise CliError(
            "Refusing to freeze anything implicitly. Review the list above, "
            "then re-run with --all or with --entry <id> for the ones you trust."
        )

    # found_at_rank is where the answer happened to sit when it was marked. It
    # is not ground truth and every run recomputes it, so it is not frozen.
    frozen = []
    backfilled = 0
    for entry in chosen:
        row = {k: v for k, v in entry.items() if k != "found_at_rank"}
        # A frozen set has to say which bytes it was confirmed against, or a
        # later run cannot tell a real regression from a replaced corpus.
        # Entries marked before the hash was carried over still lack it; the
        # ingest tracker already knows it, so nothing is re-read.
        if not row.get("document_sha256") and row.get("document_id"):
            found = runtime.document_sha(row["document_id"])
            if found:
                row["document_sha256"] = found
                backfilled += 1
        # A frozen set is about a document, not about the knowledge base that
        # happened to hold it, so its entries are keyed by the document's
        # bytes. The same question about the same document then keeps its id
        # after a re-ingest. Runtime marks stay keyed by knowledge base.
        if row.get("document_sha256"):
            row["entry_id"] = gold_entry_id(
                kb["kb_id"], row["question"], row["document_sha256"]
            )
        frozen.append(row)

    payload = {
        "schema_version": 1,
        "kind": "retrieval-gold",
        "kb_id": kb["kb_id"],
        "kb_name": kb.get("name"),
        "frozen_at": datetime.now().isoformat(timespec="seconds"),
        "entries": sorted(frozen, key=lambda e: e["entry_id"]),
    }
    _write_json(args.out, payload)
    print(f"Froze {len(frozen)} question(s) into {args.out}")
    if backfilled:
        print(f"  ({backfilled} document hash(es) filled in from the ingest tracker)")
    for entry in payload["entries"]:
        sha = (entry.get("document_sha256") or "")[:12] or "no-sha"
        print(f"  {entry['entry_id']}  [{sha}]  {entry['question']}")
    print("\nReview this file before trusting it as a regression set.")
    return 0


# ------------------------------------------------------------------ main


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m cli",
        description="Retrieval regression and inspection for the PoC.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("eval", help="run a frozen gold set through retrieval")
    run.add_argument("--kb", required=True, help="knowledge base id or name")
    run.add_argument("--gold", required=True, help="frozen gold set path")
    run.add_argument("--top-k", type=int, default=10)
    run.add_argument("--method", default=None, help="defaults to what the retriever serves")
    run.add_argument("--out", default=None, help="run JSON path")
    run.add_argument("--compare", default=None, help="previous run JSON to diff against")
    run.add_argument("--strict", action="store_true",
                     help="exit non-zero on a regression or a document mismatch")
    run.add_argument("--no-sha-check", action="store_true",
                     help="skip the document hash check")
    run.set_defaults(handler=cmd_eval)

    look = sub.add_parser("search", help="run one query and show the sources")
    look.add_argument("--kb", required=True)
    look.add_argument("--query", required=True)
    look.add_argument("--top-k", type=int, default=5)
    look.add_argument("--method", default=None)
    look.set_defaults(handler=cmd_search)

    qa = sub.add_parser("qa", help="structural QA over the stored corpus")
    qa.add_argument("--kb", required=True)
    qa.add_argument("--units", default=None, help="canonical units JSONL override")
    qa.add_argument("--limit", type=int, default=None, help="findings shown per level")
    qa.add_argument("--out", default=None)
    qa.set_defaults(handler=cmd_qa)

    info = sub.add_parser("inspect", help="how a knowledge base is configured")
    info.add_argument("--kb", required=True)
    info.add_argument("--document", default=None,
                      help="which document the manifest describes")
    info.add_argument("--json", action="store_true",
                      help="print the configuration manifest instead of a table")
    info.add_argument("--out", default=None, help="write the manifest to a file")
    info.set_defaults(handler=cmd_inspect)

    package = sub.add_parser(
        "report", help="exportable QA package for one knowledge base and document"
    )
    package.add_argument("--kb", required=True)
    package.add_argument("--gold", default=None,
                         help="frozen gold set; retrieval QA is skipped without one")
    package.add_argument("--document", default=None)
    package.add_argument("--units", default=None,
                         help="canonical units JSONL, when it cannot be identified")
    package.add_argument("--out", default=None,
                         help="report directory (default artifacts/reports/<kb>-<ts>)")
    package.add_argument("--top-k", type=int, default=10)
    package.add_argument("--method", default=None)
    package.add_argument("--expect-units", type=int, default=None,
                         help="fail unless the canonical stream has this many units")
    package.add_argument("--expect-chunks", type=int, default=None,
                         help="fail unless the store holds this many chunks")
    package.add_argument("--strict", action="store_true",
                         help="exit non-zero on a warning as well as a failure")
    package.set_defaults(handler=cmd_report)

    gold = sub.add_parser("gold", help="freeze reviewed marks into a gold set")
    gold_sub = gold.add_subparsers(dest="gold_command", required=True)
    export = gold_sub.add_parser("export")
    export.add_argument("--kb", required=True)
    export.add_argument("--out", required=True)
    export.add_argument("--all", action="store_true", help="freeze every entry")
    export.add_argument("--entry", action="append", default=[],
                        help="freeze one entry id (repeatable)")
    export.set_defaults(handler=cmd_gold_export)

    return parser


def main(argv: List[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.handler(args)
    except CliError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
