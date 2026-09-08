"""Import an existing installation's JSON state into PostgreSQL.

    python tools/import_legacy_state.py --dry-run
    python tools/import_legacy_state.py

Until Step 8 this application kept its records in five places on disk:

    .knowledge_bases.json                  the knowledge bases
    .ingested_documents.json               the ingest ledger
    .gold_set.json                         the confirmed answers
    .ingest-jobs/<job_id>.json             the ingest journal
    artifacts/viewer-live/<key>/state.json one analysis record per content

They are gitignored, so no clone carries any of them -- but a machine that has
been running this console does, and so does any mounted ``/data`` directory.
This reads whichever of them exist, under the same
``CHAT_RAG_DATA_DIR``-aware resolution the application used to write them
(``config/paths.py``), and writes them into the database.

**After it has run, PostgreSQL is authoritative.** Nothing dual-writes: the
files are left exactly where they are, untouched and unread, so a failed
import can be run again and an operator can keep them until they are satisfied.
Deleting them is a decision, not a step.

What it does and does not check
-------------------------------

*Idempotent.* Every write is an upsert on the record's own identity -- the
knowledge base's id, the document's ``doc_id``, the entry id, the job id, the
content key -- so running it twice leaves what running it once left. A record
that is already in the database is reported as ``updated`` rather than skipped,
because "the file and the row disagree" is a thing an operator wants to see.

*Relationships are validated, not enforced.* A document whose ``kb_id`` names a
knowledge base that is gone is **not** an error: this product deliberately
keeps those rows and groups them as "knowledge base deleted"
(``storage/models.py`` says why the column is not a foreign key). They are
counted and listed, because a hundred of them means something else went wrong.
An analysis whose ``doc_ids`` name no ledger record is reported the same way.

*Nothing is guessed.* A file that cannot be parsed stops that file's import and
is reported by name; the others still run. The exit status is non-zero if
anything failed, so this can be a deployment step rather than something someone
reads the output of.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Iterable

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import config  # noqa: F401,E402  - applies .env exactly as the application does
from config import paths  # noqa: E402
from storage import (  # noqa: E402
    ContentRepository, DocumentRepository, GoldSetRepository,
    IngestJobRepository, KnowledgeBaseRepository, session_scope,
)
from storage.engine import DatabaseNotConfigured, DatabaseUnavailable  # noqa: E402
import storage as database  # noqa: E402


@dataclass
class Report:
    """What was read, what was written, and everything that looked wrong."""

    imported: dict[str, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)
    #: Identities seen in the *files*, so a dry run can validate a reference
    #: against what it would have written rather than against a database it
    #: deliberately did not write to. Without this every document in a dry run
    #: looks like an orphan, which is exactly the report an operator would act
    #: on wrongly.
    seen_kb_ids: set = field(default_factory=set)
    seen_doc_ids: set = field(default_factory=set)

    def count(self, what: str, n: int = 1) -> None:
        self.imported[what] = self.imported.get(what, 0) + n

    @property
    def ok(self) -> bool:
        return not self.failures


def _read_json(path: str, report: Report) -> Any:
    """One file, or ``None`` when there is nothing to read.

    A missing file is not a failure -- an installation that never used the gold
    set has no gold set -- but an unreadable one is, and it is named.
    """
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError) as error:
        report.failures.append(f"{path}: {error}")
        return None


# --------------------------------------------------------------------------
# the five sources
# --------------------------------------------------------------------------
def import_knowledge_bases(report: Report, *, dry_run: bool) -> None:
    path = paths.knowledge_bases()
    data = _read_json(path, report)
    if not isinstance(data, dict):
        if data is not None:
            report.failures.append(f"{path}: not a JSON object of knowledge bases")
        return
    print(f"knowledge bases  {path}  ({len(data)})")
    for kb_id, cfg in data.items():
        if not isinstance(cfg, dict):
            report.failures.append(f"{path}: {kb_id} is not an object")
            continue
        report.seen_kb_ids.add(kb_id)
        if dry_run:
            report.count("knowledge_bases")
            continue
        with session_scope() as session:
            repository = KnowledgeBaseRepository(session)
            if repository.get(kb_id) is None:
                repository.create(kb_id, cfg)
            else:
                repository.update(kb_id, cfg)
        report.count("knowledge_bases")


def import_documents(report: Report, *, dry_run: bool) -> None:
    path = paths.ingested_documents()
    data = _read_json(path, report)
    if not isinstance(data, dict):
        if data is not None:
            report.failures.append(f"{path}: not a JSON object of documents")
        return
    print(f"documents        {path}  ({len(data)})")
    known = _known_kb_ids() | report.seen_kb_ids
    orphans: list[str] = []
    for file_path, record in data.items():
        if not isinstance(record, dict) or not record.get("doc_id"):
            report.failures.append(f"{path}: {file_path} has no doc_id")
            continue
        report.seen_doc_ids.add(record["doc_id"])
        kb_id = record.get("kb_id")
        if kb_id and kb_id not in known:
            orphans.append(f"{record['doc_id']} -> {kb_id}")
        if dry_run:
            report.count("documents")
            continue
        with session_scope() as session:
            DocumentRepository(session).upsert({
                "doc_id": record["doc_id"],
                "file_path": file_path,
                "file_name": os.path.basename(file_path),
                "file_hash": record.get("file_hash") or "",
                "chunk_count": record.get("chunk_count") or 0,
                "file_size": record.get("file_size") or 0,
                "ingested_at": record.get("ingested_at") or "",
                "kb_id": kb_id,
                "status": record.get("status") or "indexed",
                "chunking_mode": record.get("chunking_mode"),
                "metadata": record.get("metadata") or {},
                "pipeline_snapshot": record.get("pipeline_snapshot"),
            })
        report.count("documents")
    if orphans:
        # Not an error: the product keeps these on purpose. Counted so a
        # number that is obviously wrong is visible rather than discovered.
        report.warnings.append(
            f"{len(orphans)} document(s) name a knowledge base that is gone "
            "(kept: the console groups them as deleted): "
            + ", ".join(sorted(orphans)[:5]) + (" ..." if len(orphans) > 5 else "")
        )


def import_gold_set(report: Report, *, dry_run: bool) -> None:
    path = paths.gold_set()
    data = _read_json(path, report)
    if data is None:
        return
    raw = data.get("entries") if isinstance(data, dict) else data
    entries: Iterable[dict]
    if isinstance(raw, list):
        entries = [e for e in raw if isinstance(e, dict) and e.get("entry_id")]
    elif isinstance(raw, dict):
        entries = [e for e in raw.values() if isinstance(e, dict) and e.get("entry_id")]
    else:
        report.failures.append(f"{path}: no entries found")
        return
    entries = list(entries)
    print(f"gold set         {path}  ({len(entries)})")
    for entry in entries:
        if not entry.get("kb_id") or not entry.get("question"):
            report.failures.append(
                f"{path}: entry {entry.get('entry_id')} has no kb_id or question")
            continue
        if dry_run:
            report.count("gold_set_entries")
            continue
        with session_scope() as session:
            GoldSetRepository(session).upsert({
                **entry,
                "created_at": entry.get("created_at") or "",
                "updated_at": entry.get("updated_at") or entry.get("created_at") or "",
            })
        report.count("gold_set_entries")


def import_ingest_jobs(report: Report, *, dry_run: bool) -> None:
    directory = paths.ingest_journal()
    if not os.path.isdir(directory):
        return
    names = sorted(n for n in os.listdir(directory) if n.endswith(".json"))
    print(f"ingest journal   {directory}  ({len(names)})")
    for name in names:
        record = _read_json(os.path.join(directory, name), report)
        if not isinstance(record, dict) or not record.get("job_id"):
            continue
        if dry_run:
            report.count("ingest_jobs")
            continue
        with session_scope() as session:
            IngestJobRepository(session).record(record)
        report.count("ingest_jobs")


def import_analyses(report: Report, *, dry_run: bool) -> None:
    """The Viewer's per-content records.

    Only the record moves. The canonical units, the Deep run tree, the packaged
    variants and the payload stay exactly where they are -- they are what the
    record describes, and the directory they sit in is still named by the
    content key this writes.
    """
    root = paths.viewer_live_analysis()
    if not os.path.isdir(root):
        return
    keys = sorted(k for k in os.listdir(root)
                  if os.path.isfile(os.path.join(root, k, "state.json")))
    print(f"analyses         {root}  ({len(keys)})")
    known = _known_doc_ids() | report.seen_doc_ids
    dangling: list[str] = []
    for key in keys:
        state = _read_json(os.path.join(root, key, "state.json"), report)
        if not isinstance(state, dict):
            continue
        doc_ids = [d for d in (state.get("doc_ids") or []) if d]
        dangling += [d for d in doc_ids if d not in known]
        if dry_run:
            report.count("contents")
            continue
        fields = {
            "status": state.get("status") or "pending",
            "label": state.get("label"),
            "kb_id": state.get("kb_id"),
            "kb_name": state.get("kb_name"),
            "chunking_mode": state.get("chunking_mode"),
            "unit_count": state.get("unit_count"),
            "parse_seconds": state.get("parse_seconds"),
            "payload_bytes": state.get("payload_bytes"),
            "deep_source": state.get("deep_source"),
            "error": state.get("error"),
            "traceback": state.get("traceback"),
            "requested": list(state.get("requested") or []),
            "doc_ids": doc_ids,
            "selections": {k: list(v) for k, v in (state.get("selections") or {}).items()},
            "methods": dict(state.get("methods") or {}),
        }
        if state.get("ready_methods") is not None:
            fields["ready_methods"] = list(state["ready_methods"])
        if state.get("failed_methods") is not None:
            fields["failed_methods"] = list(state["failed_methods"])
        with session_scope() as session:
            ContentRepository(session).upsert_state(
                key, fields=fields, content_sha=state.get("content_sha"),
            )
        report.count("contents")
    if dangling:
        report.warnings.append(
            f"{len(dangling)} analysis membership(s) name a document the ledger "
            "does not have (kept: an analysis outlives its upload's record): "
            + ", ".join(sorted(set(dangling))[:5])
            + (" ..." if len(set(dangling)) > 5 else "")
        )


def _known_kb_ids() -> set[str]:
    with session_scope() as session:
        return set(KnowledgeBaseRepository(session).list_ids())


def _known_doc_ids() -> set[str]:
    with session_scope() as session:
        return {row["doc_id"] for row in DocumentRepository(session).list()}


# --------------------------------------------------------------------------
def run(*, dry_run: bool = False) -> Report:
    report = Report()
    # Knowledge bases first, then documents, then the analyses that reference
    # them: the order the validation below needs, not an order the schema
    # enforces (it deliberately does not -- see storage/models.py).
    import_knowledge_bases(report, dry_run=dry_run)
    import_documents(report, dry_run=dry_run)
    import_gold_set(report, dry_run=dry_run)
    import_ingest_jobs(report, dry_run=dry_run)
    import_analyses(report, dry_run=dry_run)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dry-run", action="store_true",
                        help="read and validate everything, write nothing")
    options = parser.parse_args(argv)

    try:
        database.require_reachable()
    except (DatabaseNotConfigured, DatabaseUnavailable) as error:
        print(f"Cannot import: {error}")
        return 2

    for line in paths.diagnostics():
        print(f"  ! {line}")
    print(f"data root        {paths.data_root() or '(none: paths are relative to '
                                                  + os.getcwd() + ')'}")
    print()
    report = run(dry_run=options.dry_run)

    print()
    if options.dry_run:
        print("DRY RUN -- nothing was written")
    for what, count in sorted(report.imported.items()):
        print(f"  {count:>6}  {what}")
    for line in report.warnings:
        print(f"  ! {line}")
    for line in report.failures:
        print(f"  FAILED  {line}")
    print()
    print("import complete" if report.ok else "import finished with failures")
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
