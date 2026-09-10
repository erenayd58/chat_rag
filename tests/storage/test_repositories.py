"""What the PostgreSQL repositories do, asked of them directly.

The rest of the suite drives this storage through the product: a route, a use
case, a record store. That is the right level for "does the console still
behave", and it is the wrong level for "does the schema hold". A cascade that
deletes one row too many is invisible from a route that never looks at the
other row; a unique constraint that is missing is invisible until two writers
arrive at once.

So this module talks to the repositories, and states the things the schema is
responsible for:

* the CRUD each record store rests on;
* the two identities -- an upload and a content -- and the membership between
  them, which is the edge the whole Viewer product model hangs off;
* what a deletion reaches and what it deliberately does not;
* that a failure part way through a unit of work leaves nothing behind.

``tests/migration/test_domain_relations.py`` states the same relations from
outside, through the HTTP surface, and does not know there is a database. Both
are wanted: that one would still pass against a wrong schema that happened to
be papered over in Python, and this one would still pass against a schema
nobody had wired up.
"""

from __future__ import annotations

import pytest
from sqlalchemy.exc import IntegrityError

from chat_rag.storage import (
    ContentRepository, DocumentRepository, GoldSetRepository,
    IngestJobRepository, KnowledgeBaseRepository, session_scope,
)
from chat_rag.storage.models import Content, ContentDocument, ContentVariant


@pytest.fixture
def knowledge_bases(db_session):
    return KnowledgeBaseRepository(db_session)


@pytest.fixture
def documents(db_session):
    return DocumentRepository(db_session)


@pytest.fixture
def contents(db_session):
    return ContentRepository(db_session)


def _config(name="Yillik raporlar", **overrides):
    return {
        "name": name,
        "chunker": {"type": "structure_first", "params": {}},
        "embedding_model_name": None,
        "vector_db_provider": "chroma",
        "vector_db_path": None,
        "retrieval_method": "hybrid",
        "extra": {},
        **overrides,
    }


def _document(doc_id="doc-1", **overrides):
    return {
        "doc_id": doc_id,
        "file_path": f"/staging/{doc_id}.pdf",
        "file_name": f"{doc_id}.pdf",
        "file_hash": "a" * 64,
        "chunk_count": 3,
        "file_size": 1024,
        "ingested_at": "2026-01-01T00:00:00",
        "kb_id": None,
        "status": "indexed",
        "chunking_mode": "standard",
        "metadata": {},
        "pipeline_snapshot": None,
        **overrides,
    }


# ==========================================================================
# knowledge bases
# ==========================================================================
def test_a_knowledge_base_is_created_read_updated_and_deleted(knowledge_bases):
    knowledge_bases.create("kb-1", _config())
    assert knowledge_bases.get("kb-1")["name"] == "Yillik raporlar"

    knowledge_bases.update("kb-1", {"name": "Raporlar", "extra": {"note": "x"}})
    reread = knowledge_bases.get("kb-1")
    assert reread["name"] == "Raporlar" and reread["extra"] == {"note": "x"}
    # The fields the update did not name are untouched, which is what makes
    # PATCH a patch rather than a replacement.
    assert reread["chunker"] == {"type": "structure_first", "params": {}}

    assert knowledge_bases.delete("kb-1") is True
    assert knowledge_bases.get("kb-1") is None
    assert knowledge_bases.delete("kb-1") is False


def test_two_knowledge_bases_cannot_share_a_name(knowledge_bases, db_session):
    """Case and inner spacing are not part of a name's identity, and the index
    is on the normalised form -- so this is the database refusing, not a read
    that happened to notice."""
    knowledge_bases.create("kb-1", _config(name="Yillik raporlar"))
    with pytest.raises(IntegrityError):
        knowledge_bases.create("kb-2", _config(name="  YILLIK   Raporlar "))
    db_session.rollback()


def test_a_blank_name_is_refused_by_the_schema(knowledge_bases, db_session):
    with pytest.raises(IntegrityError):
        knowledge_bases.create("kb-1", _config(name=""))
    db_session.rollback()


def test_a_field_with_no_column_of_its_own_still_reads_back(knowledge_bases):
    """Records written by an older console carry keys this schema does not
    model -- ``embedding_provider`` is the one still in live stores. Losing one
    to the migration would silently change which model a knowledge base is
    searched with."""
    knowledge_bases.create("kb-1", _config(
        embedding_provider="sentence_transformers", legacy_flag="kept"))
    record = knowledge_bases.get("kb-1")
    assert record["embedding_provider"] == "sentence_transformers"
    assert record["legacy_flag"] == "kept"


def test_knowledge_bases_are_listed_in_a_stable_total_order(knowledge_bases):
    """What pagination rests on. Without a total order two pages can repeat a
    row and skip another, and the client is never told."""
    for index in range(5):
        knowledge_bases.create(f"kb-{index}", _config(name=f"KB {index}"))
    first = list(knowledge_bases.all())
    assert first == list(knowledge_bases.all())
    assert len(set(first)) == 5
    # Paged the way ``/api/v1`` pages it: every row once, in one order.
    paged = first[0:2] + first[2:4] + first[4:6]
    assert paged == first


# ==========================================================================
# documents
# ==========================================================================
def test_a_document_is_created_read_and_deleted_by_its_own_identity(documents):
    documents.upsert(_document())
    assert documents.get_by_doc_id("doc-1")["chunk_count"] == 3

    assert documents.delete_by_doc_id("doc-1") is True
    assert documents.get_by_doc_id("doc-1") is None
    assert documents.delete_by_doc_id("doc-1") is False


def test_recording_the_same_document_twice_updates_one_row(documents):
    documents.upsert(_document(chunk_count=3))
    documents.upsert(_document(chunk_count=9, status="reindexed"))
    rows = documents.list()
    assert len(rows) == 1
    assert rows[0]["chunk_count"] == 9 and rows[0]["status"] == "reindexed"


def test_documents_are_scoped_and_counted_by_knowledge_base(documents):
    documents.upsert(_document("doc-a", kb_id="kb-1", chunk_count=2, file_size=10))
    documents.upsert(_document("doc-b", kb_id="kb-1", chunk_count=3, file_size=20))
    documents.upsert(_document("doc-c", kb_id="kb-2", chunk_count=5, file_size=30))

    assert {r["doc_id"] for r in documents.list("kb-1")} == {"doc-a", "doc-b"}
    assert documents.statistics("kb-1") == {
        "total_documents": 2, "total_chunks": 5, "total_size_bytes": 30,
        "oldest_ingestion": "2026-01-01T00:00:00",
        "latest_ingestion": "2026-01-01T00:00:00",
    }
    assert documents.statistics()["total_documents"] == 3
    assert documents.statistics("kb-none") == {
        "total_documents": 0, "total_chunks": 0, "total_size_bytes": 0}


def test_a_document_is_found_by_the_ingest_job_that_wrote_it(documents):
    """The lookup a restart settles a ``job_id`` with. It is a column rather
    than a scan of every record's metadata, so it has to be projected out on
    the way in."""
    documents.upsert(_document("doc-a", metadata={"ingest_job_id": "job-7"}))
    documents.upsert(_document("doc-b", metadata={}))

    assert documents.get_by_ingest_job("job-7")["doc_id"] == "doc-a"
    assert documents.get_by_ingest_job("job-none") is None


def test_documents_are_listed_newest_first_in_a_stable_total_order(documents):
    for index in range(4):
        documents.upsert(_document(f"doc-{index}",
                                   ingested_at=f"2026-01-0{index + 1}T00:00:00"))
    ids = [row["doc_id"] for row in documents.list()]
    assert ids == ["doc-3", "doc-2", "doc-1", "doc-0"]
    assert ids == [row["doc_id"] for row in documents.list()]


def test_a_documents_path_is_not_its_identity(documents):
    """Two uploads of the same file, from the same staging name, are two
    documents. The path used to be the ledger's key, which made this
    impossible to express."""
    documents.upsert(_document("doc-a", file_path="/staging/upload.pdf"))
    documents.upsert(_document("doc-b", file_path="/staging/upload.pdf"))
    assert len(documents.list()) == 2


# ==========================================================================
# contents: identity, membership, variants
# ==========================================================================
def test_a_content_carries_its_analysis_state_and_its_variants(contents):
    contents.upsert_state("doc-abc", fields={
        "status": "ready", "label": "Rapor.pdf", "requested": ["structure-only"],
        "ready_methods": ["structure-only"], "unit_count": 12,
        "methods": {"structure-only": {"status": "ready", "chunk_count": 7,
                                       "seconds": 1.5, "run_mode": "deep"}},
    }, content_sha="abc")

    state = contents.get("doc-abc")
    assert state["status"] == "ready" and state["unit_count"] == 12
    assert state["ready_methods"] == ["structure-only"]
    # A variant's own record round-trips whole: the columns are a projection of
    # it, and the fields with no column of their own are not dropped.
    assert state["methods"]["structure-only"] == {
        "status": "ready", "chunk_count": 7, "seconds": 1.5, "run_mode": "deep"}


def test_a_content_that_has_never_been_built_has_no_ready_methods_key(contents):
    """Absent is not empty. "No build has run" and "a build ran and produced
    nothing" are two different answers, and the Step 6 contract reads the
    difference."""
    contents.upsert_state("doc-abc", fields={"status": "pending",
                                             "requested": ["structure-only"]})
    state = contents.get("doc-abc")
    assert "ready_methods" not in state
    assert state["requested"] == ["structure-only"]


def test_the_same_content_key_is_one_row_however_often_it_is_written(contents,
                                                                    db_session):
    """Content dedup, at the level the schema is responsible for. The second
    upload of the same bytes joins the analysis that is there; it does not make
    a second one."""
    contents.upsert_state("doc-abc", fields={"doc_ids": ["upload-1"]},
                          content_sha="abc")
    contents.upsert_state("doc-abc", merge=lambda state: {
        "doc_ids": sorted(set(state["doc_ids"]) | {"upload-2"})})

    assert db_session.query(Content).count() == 1
    assert contents.get("doc-abc")["doc_ids"] == ["upload-1", "upload-2"]


def test_two_uploads_of_one_content_keep_their_own_selections(contents):
    """The whole product model in one assertion: one content, two uploads, two
    choices, one set of variants."""
    contents.upsert_state("doc-abc", fields={
        "doc_ids": ["upload-1", "upload-2"],
        "selections": {"upload-1": ["structure-only"], "upload-2": ["markdown"]},
        "requested": ["structure-only", "markdown"],
    })
    state = contents.get("doc-abc")
    assert state["selections"] == {"upload-1": ["structure-only"],
                                   "upload-2": ["markdown"]}
    assert state["requested"] == ["structure-only", "markdown"]


def test_an_upload_with_no_recorded_selection_is_absent_rather_than_empty(contents):
    """Null and ``[]`` mean different things here: no recorded choice falls
    back to the content's request, an empty choice would show nothing."""
    contents.upsert_state("doc-abc", fields={
        "doc_ids": ["old-upload", "new-upload"],
        "selections": {"new-upload": ["markdown"]},
    })
    assert contents.get("doc-abc")["selections"] == {"new-upload": ["markdown"]}


def test_one_upload_belongs_to_exactly_one_content():
    """Attaching an upload to a second content is refused by the unique
    constraint on ``doc_id`` alone -- not by the composite primary key, which
    would happily allow it.

    Committed first, so what survives the refusal can be read back: the
    membership that was already there.
    """
    with session_scope() as session:
        ContentRepository(session).upsert_state("doc-aaa", fields={"doc_ids": ["upload-1"]})
        ContentRepository(session).upsert_state("doc-bbb", fields={"doc_ids": []})

    with pytest.raises(IntegrityError):
        with session_scope() as session:
            other = session.query(Content).filter_by(content_key="doc-bbb").one()
            session.add(ContentDocument(content_id=other.id, doc_id="upload-1"))
            session.flush()

    with session_scope() as session:
        repository = ContentRepository(session)
        assert repository.key_for_document("upload-1") == "doc-aaa"
        assert repository.get("doc-bbb")["doc_ids"] == []


def test_a_membership_cannot_name_a_content_that_does_not_exist(db_session):
    """The referential edge that *is* a foreign key."""
    db_session.add(ContentDocument(content_id="no-such-content", doc_id="upload-1"))
    with pytest.raises(IntegrityError):
        db_session.flush()
    db_session.rollback()


def test_a_variant_cannot_name_a_content_that_does_not_exist(db_session):
    db_session.add(ContentVariant(content_id="no-such-content", method="markdown",
                                  status="ready"))
    with pytest.raises(IntegrityError):
        db_session.flush()
    db_session.rollback()


def test_detaching_one_upload_leaves_the_content_and_the_other_upload(contents,
                                                                     db_session):
    """The edge that is deliberately *not* owned. Deleting one of two uploads
    of a PDF must take its membership and its choice, and nothing of the
    content's -- a cascade the other way would delete a live document's
    analysis from under it."""
    contents.upsert_state("doc-abc", fields={
        "doc_ids": ["upload-1", "upload-2"],
        "selections": {"upload-1": ["structure-only"], "upload-2": ["markdown"]},
        "requested": ["structure-only", "markdown"],
        "ready_methods": ["structure-only", "markdown"],
    })

    assert contents.detach_document("doc-abc", "upload-1") == 1

    state = contents.get("doc-abc")
    assert state["doc_ids"] == ["upload-2"]
    assert state["selections"] == {"upload-2": ["markdown"]}
    # The content keeps every variant either upload ever asked for.
    assert state["requested"] == ["structure-only", "markdown"]
    assert state["ready_methods"] == ["structure-only", "markdown"]


def test_deleting_a_content_takes_its_memberships_and_variants_with_it(contents,
                                                                      db_session):
    """The cascade that *is* wanted: nothing points at the content any more, so
    keeping its children would be a leak with no way back to them."""
    contents.upsert_state("doc-abc", fields={
        "doc_ids": ["upload-1"],
        "selections": {"upload-1": ["markdown"]},
        "methods": {"markdown": {"status": "ready"}},
    })
    assert db_session.query(ContentDocument).count() == 1
    assert db_session.query(ContentVariant).count() == 1

    assert contents.delete("doc-abc") is True

    assert db_session.query(Content).count() == 0
    assert db_session.query(ContentDocument).count() == 0
    assert db_session.query(ContentVariant).count() == 0


def test_deleting_a_content_leaves_another_contents_variants_alone(contents,
                                                                  db_session):
    contents.upsert_state("doc-aaa", fields={
        "doc_ids": ["upload-1"], "methods": {"markdown": {"status": "ready"}}})
    contents.upsert_state("doc-bbb", fields={
        "doc_ids": ["upload-2"], "methods": {"markdown": {"status": "ready"}}})

    contents.delete("doc-aaa")

    assert db_session.query(ContentVariant).count() == 1
    assert contents.get("doc-bbb")["methods"] == {"markdown": {"status": "ready"}}


def test_the_content_of_an_upload_is_found_by_the_upload(contents):
    contents.upsert_state("doc-abc", fields={"doc_ids": ["upload-1", "upload-2"]})
    assert contents.key_for_document("upload-2") == "doc-abc"
    assert contents.key_for_document("upload-none") is None


def test_a_content_that_is_still_being_built_is_found_by_its_status(contents):
    """What a restart resumes from."""
    contents.upsert_state("doc-aaa", fields={"status": "running"})
    contents.upsert_state("doc-bbb", fields={"status": "ready"})
    contents.upsert_state("doc-ccc", fields={"status": "pending"})

    assert contents.keys_with_status(["pending", "running"]) == ["doc-aaa", "doc-ccc"]


def test_a_variant_that_is_no_longer_recorded_is_removed(contents, db_session):
    contents.upsert_state("doc-abc", fields={
        "methods": {"markdown": {"status": "ready"}, "deep": {"status": "failed"}}})
    contents.upsert_state("doc-abc", fields={"methods": {"markdown": {"status": "ready"}}})
    assert list(contents.get("doc-abc")["methods"]) == ["markdown"]
    assert db_session.query(ContentVariant).count() == 1


# ==========================================================================
# ingest jobs
# ==========================================================================
def test_a_job_record_is_written_replaced_forgotten_and_pruned(db_session):
    jobs = IngestJobRepository(db_session)
    jobs.record({"job_id": "job-1", "status": "queued", "journalled_at": 100.0,
                 "filename": "a.pdf"})
    jobs.record({"job_id": "job-1", "status": "succeeded", "journalled_at": 200.0,
                 "filename": "a.pdf", "result": {"chunks_created": 3}})

    (record,) = jobs.snapshots()
    assert record["status"] == "succeeded" and record["result"]["chunks_created"] == 3

    jobs.record({"job_id": "job-old", "status": "succeeded", "journalled_at": 1.0})
    assert jobs.prune(50.0) == 1
    assert [r["job_id"] for r in jobs.snapshots()] == ["job-1"]

    jobs.forget("job-1")
    assert jobs.snapshots() == []


def test_a_job_record_with_no_id_is_not_written(db_session):
    jobs = IngestJobRepository(db_session)
    jobs.record({"status": "queued"})
    assert jobs.snapshots() == []


# ==========================================================================
# the gold set
# ==========================================================================
def _entry(entry_id="e-1", **overrides):
    return {
        "entry_id": entry_id, "schema_version": 1, "kb_id": "kb-1",
        "question": "Takipteki alacaklar ne kadar?", "document_id": None,
        "document_title": None, "document_sha256": None,
        "correct_chunk_id": "doc:s-0001", "section": None, "pages": [36],
        "unit_ids": ["v-00808"], "evidence": "...", "retrieval_method": None,
        "found_at_rank": 1, "created_at": "2026-01-01T00:00:00",
        "updated_at": "2026-01-01T00:00:00", **overrides,
    }


def test_re_marking_a_question_replaces_the_answer_and_keeps_the_first_date(db_session):
    gold = GoldSetRepository(db_session)
    gold.upsert(_entry())
    gold.upsert(_entry(correct_chunk_id="doc:s-0002", updated_at="2026-02-02T00:00:00"))

    (entry,) = gold.all().values()
    assert entry["correct_chunk_id"] == "doc:s-0002"
    assert entry["created_at"] == "2026-01-01T00:00:00", "the day it was first confirmed"
    assert entry["updated_at"] == "2026-02-02T00:00:00"

    assert gold.delete("e-1") is True
    assert gold.all() == {}
    assert gold.delete("e-1") is False


# ==========================================================================
# transactions
# ==========================================================================
def test_a_failure_part_way_through_a_unit_of_work_writes_nothing():
    """The multi-write case, and the reason the transaction boundary is the
    caller's: this writes a knowledge base, a document and a content
    membership, and fails on the last one. Any of the three surviving would be
    a half-created domain graph -- a knowledge base with a document nobody can
    reach through it."""
    with pytest.raises(RuntimeError):
        with session_scope() as session:
            KnowledgeBaseRepository(session).create("kb-1", _config())
            DocumentRepository(session).upsert(_document("doc-1", kb_id="kb-1"))
            ContentRepository(session).upsert_state(
                "doc-abc", fields={"doc_ids": ["doc-1"]})
            raise RuntimeError("the last write failed")

    with session_scope() as session:
        assert KnowledgeBaseRepository(session).get("kb-1") is None
        assert DocumentRepository(session).get_by_doc_id("doc-1") is None
        assert ContentRepository(session).get("doc-abc") is None


def test_a_constraint_violation_rolls_the_whole_unit_of_work_back():
    """The same rule when the database is the one refusing. The knowledge base
    written first must not survive the duplicate name written second."""
    with session_scope() as session:
        KnowledgeBaseRepository(session).create("kb-1", _config(name="Taken"))

    with pytest.raises(IntegrityError):
        with session_scope() as session:
            repository = KnowledgeBaseRepository(session)
            repository.create("kb-2", _config(name="Fine"))
            repository.create("kb-3", _config(name="taken"))

    with session_scope() as session:
        assert sorted(KnowledgeBaseRepository(session).all()) == ["kb-1"]


def test_a_committed_unit_of_work_is_visible_to_the_next_one():
    with session_scope() as session:
        KnowledgeBaseRepository(session).create("kb-1", _config())
    with session_scope() as session:
        assert KnowledgeBaseRepository(session).get("kb-1")["name"] == "Yillik raporlar"
