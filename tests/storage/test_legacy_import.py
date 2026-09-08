"""Importing an existing installation's JSON state into PostgreSQL.

No clone carries any of these files -- they are gitignored, which is why this
migration ships an import tool rather than a data migration inside Alembic. A
machine that has been running this console does carry them, and so does any
mounted ``/data`` directory, so the tool has to be right about state nobody can
put in a fixture: five files written by five different versions of this
product over a year.

What is asserted here is what an operator is entitled to assume before running
it against the only copy of their records:

* everything that is there is imported, with its identities intact -- an
  upload's ``doc_id``, a content's key, an entry id, a job id;
* running it twice is running it once;
* the relationships the schema deliberately does not enforce are *reported*,
  so a hundred orphans are visible rather than discovered later;
* a file it cannot read stops that file and nothing else, and is named.

The tool never writes to the files, so every test here can check that they are
still exactly as they were: an import that destroys its own input cannot be
run again.
"""

from __future__ import annotations

import json
import os

import pytest

from storage import (
    ContentRepository, DocumentRepository, GoldSetRepository,
    IngestJobRepository, KnowledgeBaseRepository, session_scope,
)
from tools import import_legacy_state


@pytest.fixture
def installation(tmp_path, monkeypatch):
    """A data directory holding one of everything, as the file era wrote it."""
    monkeypatch.setenv("CHAT_RAG_DATA_DIR", str(tmp_path))
    state = tmp_path / "state"
    state.mkdir(parents=True)

    (state / "knowledge_bases.json").write_text(json.dumps({
        "kb-1": {
            "name": "Yillik raporlar",
            "chunker": {"type": "structure_first", "params": {}},
            "embedding_model_name": "paraphrase-multilingual-MiniLM-L12-v2",
            "embedding_provider": "sentence_transformers",
            "vector_db_provider": "chroma", "vector_db_path": None,
            "retrieval_method": "hybrid", "extra": {},
        },
    }), encoding="utf-8")

    (state / "ingested_documents.json").write_text(json.dumps({
        "C:/uploads/rapor.pdf": {
            "doc_id": "doc-1", "file_hash": "a" * 64, "chunk_count": 42,
            "file_size": 1024, "ingested_at": "2026-01-01T10:00:00",
            "kb_id": "kb-1", "status": "indexed", "chunking_mode": "deep_analysis",
            "metadata": {"original_filename": "Rapor.pdf", "ingest_job_id": "job-7"},
            "pipeline_snapshot": {"chunker": "structure_first"},
        },
        "C:/uploads/eski.pdf": {
            "doc_id": "doc-2", "file_hash": "b" * 64, "chunk_count": 3,
            "file_size": 99, "ingested_at": "2025-06-01T10:00:00",
            "kb_id": "kb-gone", "metadata": {},
        },
    }), encoding="utf-8")

    (state / "gold_set.json").write_text(json.dumps({
        "schema_version": 1,
        "entries": [{
            "entry_id": "e-1", "schema_version": 1, "kb_id": "kb-1",
            "question": "Takipteki alacaklar ne kadar?",
            "correct_chunk_id": "doc:s-0001", "pages": [36],
            "unit_ids": ["v-00808"], "evidence": "...",
            "created_at": "2026-01-02T00:00:00", "updated_at": "2026-01-02T00:00:00",
        }],
    }), encoding="utf-8")

    jobs = state / "ingest-jobs"
    jobs.mkdir()
    (jobs / "job-7.json").write_text(json.dumps({
        "job_id": "job-7", "status": "succeeded", "kb_id": "kb-1",
        "filename": "Rapor.pdf", "doc_id": "doc-1", "journalled_at": 1_700_000_000.0,
        "result": {"success": True, "chunks_created": 42},
    }), encoding="utf-8")

    analysis = tmp_path / "viewer-live" / "doc-aaaaaaaaaaaaaaaaaaaaaaa"
    analysis.mkdir(parents=True)
    (analysis / "state.json").write_text(json.dumps({
        "key": "doc-aaaaaaaaaaaaaaaaaaaaaaa", "status": "ready",
        "label": "Rapor.pdf", "kb_id": "kb-1", "kb_name": "Yillik raporlar",
        "content_sha": "a" * 64, "unit_count": 120, "parse_seconds": 3.5,
        "doc_ids": ["doc-1"], "requested": ["structure-only", "deep"],
        "ready_methods": ["structure-only", "deep"],
        "selections": {"doc-1": ["structure-only"]},
        "methods": {"structure-only": {"status": "ready", "chunk_count": 42},
                    "deep": {"status": "ready", "source": "ingest_deep_run"}},
        "deep_source": "ingest_deep_run",
    }), encoding="utf-8")
    # The artifacts stay files; only the record moves.
    (analysis / "units.jsonl").write_text("", encoding="utf-8")
    return tmp_path


def _stored():
    with session_scope() as session:
        return {
            "knowledge_bases": KnowledgeBaseRepository(session).all(),
            "documents": {r["doc_id"]: r for r in DocumentRepository(session).list()},
            "gold": GoldSetRepository(session).all(),
            "jobs": {r["job_id"]: r for r in IngestJobRepository(session).snapshots()},
            "contents": ContentRepository(session).all(),
        }


def test_every_record_is_imported_with_its_identity_intact(installation):
    report = import_legacy_state.run()

    assert report.failures == []
    assert report.imported == {
        "knowledge_bases": 1, "documents": 2, "gold_set_entries": 1,
        "ingest_jobs": 1, "contents": 1,
    }

    stored = _stored()
    kb = stored["knowledge_bases"]["kb-1"]
    assert kb["name"] == "Yillik raporlar"
    assert kb["chunker"] == {"type": "structure_first", "params": {}}
    # A field with no column of its own is not lost to the schema.
    assert kb["embedding_provider"] == "sentence_transformers"

    document = stored["documents"]["doc-1"]
    assert document["chunk_count"] == 42 and document["kb_id"] == "kb-1"
    assert document["ingested_at"] == "2026-01-01T10:00:00"
    assert document["metadata"]["original_filename"] == "Rapor.pdf"
    assert document["pipeline_snapshot"] == {"chunker": "structure_first"}
    assert document["file_name"] == "rapor.pdf", "the name the console shows"

    assert stored["gold"]["e-1"]["question"] == "Takipteki alacaklar ne kadar?"
    assert stored["jobs"]["job-7"]["result"]["chunks_created"] == 42


def test_the_two_identities_and_the_choice_between_them_survive(installation):
    """The part of the import that is easy to get wrong: an analysis is a
    content, the uploads that point at it are memberships, and what each of
    them selected belongs to the membership."""
    import_legacy_state.run()

    state = _stored()["contents"]["doc-aaaaaaaaaaaaaaaaaaaaaaa"]
    assert state["status"] == "ready"
    assert state["doc_ids"] == ["doc-1"]
    assert state["selections"] == {"doc-1": ["structure-only"]}
    assert state["requested"] == ["structure-only", "deep"]
    assert state["ready_methods"] == ["structure-only", "deep"]
    assert state["content_sha"] == "a" * 64
    assert state["unit_count"] == 120 and state["parse_seconds"] == 3.5
    assert state["methods"]["deep"] == {"status": "ready", "source": "ingest_deep_run"}


def test_the_ingest_job_can_still_be_settled_against_the_ledger(installation):
    """The one cross-record lookup a restart makes. It is a column projected
    out of the document's metadata, so an import that did not project it would
    leave every recovered job answering "your upload never happened"."""
    import_legacy_state.run()

    with session_scope() as session:
        assert DocumentRepository(session).get_by_ingest_job("job-7")["doc_id"] == "doc-1"


def test_running_it_twice_is_running_it_once(installation):
    import_legacy_state.run()
    before = _stored()

    second = import_legacy_state.run()

    assert second.failures == []
    after = _stored()
    for table in ("knowledge_bases", "documents", "gold", "jobs", "contents"):
        assert set(before[table]) == set(after[table]), table
    assert after["documents"]["doc-1"]["chunk_count"] == 42
    assert after["contents"]["doc-aaaaaaaaaaaaaaaaaaaaaaa"]["doc_ids"] == ["doc-1"]


def test_a_document_naming_a_knowledge_base_that_is_gone_is_kept_and_reported(installation):
    """Not an error: this product deliberately keeps those rows and groups them
    as "knowledge base deleted". Counted, because a hundred of them means
    something else went wrong."""
    report = import_legacy_state.run()

    assert _stored()["documents"]["doc-2"]["kb_id"] == "kb-gone"
    assert any("knowledge base that is gone" in line for line in report.warnings)
    assert report.failures == []


def test_a_file_that_cannot_be_read_stops_that_file_and_is_named(installation):
    (installation / "state" / "gold_set.json").write_text("{ not json", encoding="utf-8")

    report = import_legacy_state.run()

    assert any("gold_set.json" in line for line in report.failures)
    assert report.ok is False
    # The others still ran.
    assert report.imported["knowledge_bases"] == 1
    assert report.imported["documents"] == 2


def test_a_dry_run_writes_nothing_and_still_counts_everything(installation):
    report = import_legacy_state.run(dry_run=True)

    assert report.imported["documents"] == 2
    assert report.failures == []
    stored = _stored()
    assert stored["knowledge_bases"] == {} and stored["documents"] == {}
    assert stored["contents"] == {} and stored["jobs"] == {}


def test_a_dry_run_validates_against_the_files_not_the_empty_database(installation):
    """The report an operator reads *before* running it for real, so it has to
    be about the same records the real run would write.

    Validating a document's knowledge base against the database in a dry run
    would report every document as an orphan -- the knowledge bases have not
    been written -- and that is the one report someone would act on wrongly.
    """
    report = import_legacy_state.run(dry_run=True)

    orphan_warnings = [w for w in report.warnings if "knowledge base that is gone" in w]
    assert len(orphan_warnings) == 1
    assert "doc-2 -> kb-gone" in orphan_warnings[0]
    assert "doc-1" not in orphan_warnings[0], "doc-1's knowledge base is in the file"
    assert not any("the ledger does not have" in w for w in report.warnings)


def test_the_files_are_left_exactly_as_they_were(installation):
    """An import that destroyed its own input could not be run again, and an
    operator could not keep the files until they were satisfied."""
    before = {
        path: os.stat(path).st_mtime_ns
        for path in (installation / "state").rglob("*")
        if path.is_file()
    }

    import_legacy_state.run()

    assert {path: os.stat(path).st_mtime_ns
            for path in (installation / "state").rglob("*")
            if path.is_file()} == before
    assert (installation / "viewer-live" / "doc-aaaaaaaaaaaaaaaaaaaaaaa"
            / "units.jsonl").is_file(), "the artifacts stay where they are"


def test_an_installation_with_nothing_to_import_is_not_a_failure(tmp_path, monkeypatch):
    """A fresh install has none of these files, and running the tool against
    one must be a clean no-op rather than five errors."""
    monkeypatch.setenv("CHAT_RAG_DATA_DIR", str(tmp_path))

    report = import_legacy_state.run()

    assert report.imported == {} and report.failures == [] and report.ok
