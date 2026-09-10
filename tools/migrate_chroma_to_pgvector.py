"""Move an existing Chroma store into PostgreSQL/pgvector, once.

Step 9 replaced the vector store. A deployment that has been running has real
corpora in ``chroma_db/<kb_id>/`` -- chunk text, chunk metadata and vectors
that cost real provider calls to produce -- and re-ingesting to get them back
would mean re-parsing every document and paying for every embedding again. So
they are moved rather than rebuilt.

    chroma_db/<kb_id>/          ->   vector_collections  (the manifest row)
      chunk ids, documents,          chunk_vectors       (one row per chunk)
      metadatas, embeddings
      embedding_index.json

**Vectors are exported, not recomputed.** Chroma returns the stored embedding
verbatim, and this writes exactly those floats: a migrated corpus answers the
same query with the same ranking it did before, and no provider is called at
all. (If a store *cannot* give its embeddings back -- a corrupt index, a build
of Chroma that refuses the include -- the tool says so and stops rather than
guessing; the controlled rebuild path is then
``POST /api/v1/knowledge-bases/<kb_id>/embedding-index/rebuild``, which
re-embeds the migrated text with the current model. Migrate the text first,
then rebuild.)

**Idempotent.** A chunk already in PostgreSQL with the same text, metadata and
vector is skipped, so an interrupted run is finished by running it again. A
chunk already there with *different* content is a conflict: it is reported and
left alone, because two different texts under one id is a fact somebody has to
look at, not one this tool should decide. ``--overwrite`` decides it the other
way, deliberately.

**Scoped to one knowledge base.** A collection is named by a knowledge base's
id, and the tool refuses an id that is not in the ``knowledge_bases`` table --
a collection with no owner is a corpus nothing can delete.

Usage::

    # what would happen, touching nothing
    python -m tools.migrate_chroma_to_pgvector --store ./chroma_db/7f3a1b2c \\
        --kb 7f3a1b2c --dry-run

    # do it
    python -m tools.migrate_chroma_to_pgvector --store ./chroma_db/7f3a1b2c \\
        --kb 7f3a1b2c

    # every knowledge base under one root, each directory named by its kb id
    python -m tools.migrate_chroma_to_pgvector --root ./chroma_db

Exit status is 0 when everything was migrated or already present, and 1 when
anything failed or conflicted.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Iterator, Optional

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

#: The manifest Chroma-era stores kept beside them. Its fields are the
#: ``vector_collections`` manifest columns, under the names the file used.
MANIFEST_NAME = "embedding_index.json"

#: How many chunk rows are written per statement. The whole store still goes
#: in one transaction.
BATCH = 500


@dataclass
class Outcome:
    """What happened to one collection."""

    collection: str
    kb_id: Optional[str]
    read: int = 0
    migrated: int = 0
    skipped: int = 0
    failed: int = 0
    conflicts: list[str] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)
    manifest: bool = False

    @property
    def ok(self) -> bool:
        return not self.failed and not self.problems

    def line(self) -> str:
        return (f"{self.collection}: read {self.read}, migrated {self.migrated}, "
                f"skipped {self.skipped}, failed {self.failed}"
                + (", manifest carried" if self.manifest else ""))


# ------------------------------------------------------------------ reading
def read_chroma(store_path: str) -> tuple[list[dict[str, Any]], Optional[dict]]:
    """Every row in a Chroma store, plus the manifest file beside it.

    ``chromadb`` is imported here and nowhere else. It is no longer a
    dependency of this application -- that is the point of Step 9 -- so the
    one tool that still needs it says so by name instead of failing on an
    import at the top of a module the product loads.
    """
    try:
        import chromadb
        from chromadb.config import Settings
    except ImportError as error:  # pragma: no cover - depends on the machine
        raise SystemExit(
            "chromadb is not installed. It is deliberately not a dependency of "
            "this application any more; install it just to run this migration:\n"
            "    pip install chromadb"
        ) from error

    client = chromadb.PersistentClient(path=store_path,
                                       settings=Settings(anonymized_telemetry=False))
    collection = client.get_or_create_collection(name="documents",
                                                 embedding_function=None)
    found = collection.get(include=["documents", "metadatas", "embeddings"])
    ids = list(found.get("ids") or [])
    documents = list(found.get("documents") or [])
    metadatas = list(found.get("metadatas") or [])
    embeddings = found.get("embeddings")
    embeddings = [] if embeddings is None else list(embeddings)

    rows = []
    for index, chunk_id in enumerate(ids):
        vector = embeddings[index] if index < len(embeddings) else None
        rows.append({
            "chunk_id": chunk_id,
            "content": documents[index] if index < len(documents) else "",
            "metadata": dict(metadatas[index] or {}) if index < len(metadatas) else {},
            "embedding": None if vector is None else [float(v) for v in vector],
        })

    manifest = None
    path = os.path.join(store_path, MANIFEST_NAME)
    if os.path.isfile(path):
        try:
            with open(path, encoding="utf-8") as handle:
                loaded = json.load(handle)
            manifest = loaded if isinstance(loaded, dict) else None
        except (OSError, ValueError):
            manifest = None
    return rows, manifest


# --------------------------------------------------------------- validation
def validate(rows: list[dict[str, Any]]) -> list[str]:
    """What must be true of a store before any of it is written.

    Checked over the whole store rather than row by row, because the two
    things that actually go wrong -- an id used twice, and two embedding
    widths in one collection -- are only visible across rows, and either one
    makes a collection that cannot be queried.
    """
    problems: list[str] = []
    seen: set[str] = set()
    duplicates: set[str] = set()
    widths: set[int] = set()
    for row in rows:
        chunk_id = row["chunk_id"]
        if not isinstance(chunk_id, str) or not chunk_id:
            problems.append("a row has no chunk id")
            continue
        if chunk_id in seen:
            duplicates.add(chunk_id)
        seen.add(chunk_id)
        if not isinstance(row["content"], str):
            problems.append(f"{chunk_id}: content is not text")
        if row["embedding"] is not None:
            widths.add(len(row["embedding"]))
    if duplicates:
        problems.append("the store repeats these chunk ids: "
                        + ", ".join(sorted(duplicates)[:5]))
    if len(widths) > 1:
        problems.append("the store holds vectors of several widths "
                        f"({sorted(widths)}); a collection holds one embedding space")
    return problems


# ------------------------------------------------------------------ writing
def _existing(store) -> dict[str, dict[str, Any]]:
    """What the target collection already holds, by chunk id."""
    return {row["chunk_id"]: store.get_chunk_by_id(row["chunk_id"])
            for row in store.get_chunks_paginated(offset=0, limit=10 ** 9)["chunks"]}


def _same(source: dict[str, Any], stored: dict[str, Any]) -> bool:
    """Is this row already in the target, unchanged?

    Text and vector are compared exactly. They were copied, not recomputed, so
    any difference means the target was written by something else -- which is
    exactly the case a second run must not paper over.

    Metadata is compared by *containment*: every key the export carries must
    be present in the target with the same value. A store fills in the fields
    it derives (a missing ``word_count`` becomes ``0``, a null is omitted), so
    demanding an equal dictionary would call every second run a conflict --
    while a key whose value actually changed is still caught.
    """
    if source["content"] != stored["content"]:
        return False
    if list(source["embedding"] or []) != list(stored["embedding"] or []):
        return False
    target = stored["metadata"] or {}
    return all(target.get(key) == value
               for key, value in (source["metadata"] or {}).items()
               if value is not None)


def migrate_store(store_path: str, *, kb_id: Optional[str], collection: str,
                  dry_run: bool = False, overwrite: bool = False) -> Outcome:
    """Move one Chroma directory into one pgvector collection."""
    from chat_rag.components.vectordb import PgVectorStore
    from chat_rag.core.models import DocumentChunk

    outcome = Outcome(collection=collection, kb_id=kb_id)
    rows, manifest = read_chroma(store_path)
    outcome.read = len(rows)
    outcome.problems = validate(rows)
    if outcome.problems:
        outcome.failed = len(rows)
        return outcome

    store = PgVectorStore(collection=collection, kb_id=kb_id)
    # Read even on a dry run: "what would happen" has to account for what is
    # already there, or a second run reports every row as new.
    already = _existing(store)

    to_write: list[dict[str, Any]] = []
    for row in rows:
        found = already.get(row["chunk_id"])
        if found is None:
            to_write.append(row)
        elif _same(row, found):
            outcome.skipped += 1
        elif overwrite:
            to_write.append(row)
        else:
            outcome.conflicts.append(row["chunk_id"])
            outcome.failed += 1

    if dry_run:
        outcome.migrated = len(to_write)
        outcome.manifest = manifest is not None
        return outcome

    for start in range(0, len(to_write), BATCH):
        batch = to_write[start:start + BATCH]
        chunks = [
            DocumentChunk(
                chunk_id=row["chunk_id"],
                content=row["content"],
                doc_id=row["metadata"].get("doc_id", ""),
                doc_title=row["metadata"].get("doc_title", ""),
                chunk_index=row["metadata"].get("chunk_index", 0),
                total_chunks=row["metadata"].get("total_chunks", 0),
                section_title=row["metadata"].get("section_title"),
                document_summary=row["metadata"].get("document_summary"),
                metadata=dict(row["metadata"]),
            )
            for row in batch
        ]
        vectors = [row["embedding"] for row in batch]
        if any(vector is None for vector in vectors):
            # Chroma always stores a vector; a missing one means the export
            # could not read the index. Never invent one -- say so, and let
            # the rebuild endpoint re-embed the text that did migrate.
            outcome.problems.append(
                "some rows came back without their embedding; migrate the text "
                "and then rebuild the index with the current model"
            )
            outcome.failed += len(batch)
            continue
        store.add_chunks(chunks, vectors)
        outcome.migrated += len(batch)

    if manifest:
        store.write_manifest(manifest)
        outcome.manifest = True
    return outcome


def _known_knowledge_bases() -> dict[str, str]:
    from chat_rag.storage import KnowledgeBaseRepository, session_scope

    with session_scope() as session:
        return {kb_id: record.get("name") or kb_id
                for kb_id, record in KnowledgeBaseRepository(session).all().items()}


def _stores_under(root: str) -> Iterator[tuple[str, str]]:
    """``(kb_id, path)`` for each per-knowledge-base directory under a root."""
    for name in sorted(os.listdir(root)):
        path = os.path.join(root, name)
        if os.path.isdir(path) and os.path.isfile(os.path.join(path, "chroma.sqlite3")):
            yield name, path


# ---------------------------------------------------------------------- cli
def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Move a Chroma vector store into PostgreSQL/pgvector.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--store", help="one Chroma directory")
    source.add_argument("--root", help="a directory of them, one per knowledge base")
    parser.add_argument("--kb", help="the knowledge base that owns --store")
    parser.add_argument("--collection",
                        help="the target collection (default: the knowledge base id, "
                             "or VECTOR_DB_COLLECTION when there is none)")
    parser.add_argument("--dry-run", action="store_true",
                        help="report what would be written and write nothing")
    parser.add_argument("--overwrite", action="store_true",
                        help="replace a chunk already present with different content")
    args = parser.parse_args(argv)

    known = _known_knowledge_bases()
    jobs: list[tuple[Optional[str], str, str]] = []
    if args.root:
        for kb_id, path in _stores_under(args.root):
            jobs.append((kb_id, path, kb_id))
    else:
        kb_id = args.kb
        collection = args.collection or kb_id or os.getenv("VECTOR_DB_COLLECTION",
                                                           "documents")
        jobs.append((kb_id, args.store, collection))

    if not jobs:
        print("nothing to migrate: no Chroma store found")
        return 0

    failures = 0
    for kb_id, path, collection in jobs:
        if kb_id is not None and kb_id not in known:
            print(f"{collection}: SKIPPED -- no knowledge base {kb_id!r} in the "
                  "database; import the records first (tools/import_legacy_state.py)")
            failures += 1
            continue
        outcome = migrate_store(path, kb_id=kb_id, collection=collection,
                                dry_run=args.dry_run, overwrite=args.overwrite)
        prefix = "would migrate -- " if args.dry_run else ""
        print(prefix + outcome.line())
        for problem in outcome.problems:
            print(f"  ! {problem}")
        if outcome.conflicts:
            print(f"  ! {len(outcome.conflicts)} chunk(s) already present with "
                  "different content, left alone: "
                  + ", ".join(outcome.conflicts[:5])
                  + ("..." if len(outcome.conflicts) > 5 else "")
                  + "  (re-run with --overwrite to replace them)")
        if not outcome.ok:
            failures += 1

    if args.dry_run:
        print("\nnothing was written (--dry-run)")
    return 1 if failures else 0


if __name__ == "__main__":  # pragma: no cover - the entry point
    raise SystemExit(main())
