from __future__ import annotations

import json

import pytest

from components.knowledgebase.manager import KnowledgeBaseManager
from components.vectordb import PgVectorStore
from core.models import DocumentChunk


def manager(tmp_path):
    return KnowledgeBaseManager(str(tmp_path / "knowledge-bases.json"))


def fill(kb_id, *chunk_ids):
    """Put some vectors in a knowledge base's collection, and return the store."""
    store = PgVectorStore(collection=kb_id, kb_id=kb_id)
    store.add_chunks(
        [DocumentChunk(chunk_id=cid, content="metin " + cid, doc_id="doc-1",
                       doc_title="rapor.pdf", chunk_index=i, total_chunks=len(chunk_ids),
                       metadata={"word_count": 2})
         for i, cid in enumerate(chunk_ids)],
        [[1.0, 0.0, 0.0] for _ in chunk_ids],
    )
    return store


# ------------------------------------------------------------ duplicates


def test_a_second_knowledge_base_cannot_take_an_existing_name(tmp_path):
    """Eight knowledge bases named kkb-kkb once shared one empty store."""
    kb = manager(tmp_path)
    kb.create(name="kkb-final")
    with pytest.raises(ValueError, match="already exists"):
        kb.create(name="kkb-final")
    assert len(kb.list()) == 1


def test_name_matching_ignores_case_and_spacing(tmp_path):
    kb = manager(tmp_path)
    kb.create(name="KKB Final")
    with pytest.raises(ValueError, match="already exists"):
        kb.create(name="  kkb   final ")


def test_a_name_freed_by_deletion_can_be_reused(tmp_path):
    kb = manager(tmp_path)
    created = kb.create(name="kkb-final")
    kb.delete(created["kb_id"])
    assert kb.create(name="kkb-final")["name"] == "kkb-final"


def test_an_empty_name_is_refused(tmp_path):
    with pytest.raises(ValueError, match="name is required"):
        manager(tmp_path).create(name="   ")


def test_an_unknown_provider_is_refused(tmp_path):
    with pytest.raises(ValueError, match="vector_db_provider"):
        manager(tmp_path).create(name="kb", vector_db_provider="sqlite")


# --------------------------------------------------------- half records


def test_a_rejected_payload_leaves_no_record_behind(tmp_path):
    kb = manager(tmp_path)
    with pytest.raises(ValueError):
        kb.create_from_payload({"name": "bad", "chunker": {"type": "nope"}})
    assert kb.list() == []
    # And nothing was written under another name either: a rejected payload
    # leaves the store exactly as it found it.
    assert manager(tmp_path).list() == []


def test_a_failed_save_leaves_no_record_behind(tmp_path, monkeypatch):
    """A creation that fails after the row is written leaves nothing.

    This used to be about an in-memory record surviving a failed file write.
    It is now the transaction: the insert has happened inside the unit of
    work when the failure lands, so only a rollback can make this assertion
    true -- a repository that committed as it went would leave the row.
    """
    from storage.repositories import KnowledgeBaseRepository

    kb = manager(tmp_path)
    real = KnowledgeBaseRepository.create

    def insert_then_fail(self, kb_id, config):
        real(self, kb_id, config)
        raise OSError("disk full")

    monkeypatch.setattr(KnowledgeBaseRepository, "create", insert_then_fail)
    with pytest.raises(OSError):
        kb.create(name="kb")
    assert kb.list() == []


# ------------------------------------------------------------- storage
#
# A knowledge base's vectors were a directory until Step 9 and are rows in a
# collection now. Everything below states the same rule the directory version
# stated -- the record must not outlive its corpus, nor the corpus its record
# -- against the thing that actually enforces it: one transaction, and a
# foreign key that cascades.


def test_the_collection_is_the_knowledge_base_id(tmp_path):
    """Nothing to resolve and nothing to configure. The pipeline builder and
    the deletion path read the same primary key, so they cannot disagree the
    way two path resolvers could."""
    kb = manager(tmp_path)
    created = kb.create(name="kkb-final")
    assert kb.collection(created["kb_id"]) == created["kb_id"]


def test_an_unknown_knowledge_base_has_no_collection(tmp_path):
    assert manager(tmp_path).collection("nope") is None


def test_delete_removes_the_record_and_its_vectors(tmp_path):
    kb = manager(tmp_path)
    created = kb.create(name="kb")
    store = fill(created["kb_id"], "c-1", "c-2")
    assert store.count() == 2

    result = kb.delete_with_storage(created["kb_id"], str(tmp_path))

    assert result["deleted"] and result["vectors_removed"] == 2
    assert result["vector_collection"] == created["kb_id"]
    assert store.count() == 0
    assert kb.list() == []


def test_deleting_one_knowledge_base_leaves_anothers_vectors_alone(tmp_path):
    """The isolation a directory per knowledge base used to give, as a key."""
    kb = manager(tmp_path)
    first = kb.create(name="one")
    second = kb.create(name="two")
    fill(first["kb_id"], "a-1")
    kept = fill(second["kb_id"], "b-1", "b-2")

    kb.delete_with_storage(first["kb_id"], str(tmp_path))

    assert kept.count() == 2
    assert len(kb.list()) == 1


def test_deleting_a_knowledge_base_that_never_stored_anything_is_not_an_error(tmp_path):
    kb = manager(tmp_path)
    created = kb.create(name="kb")
    result = kb.delete_with_storage(created["kb_id"], str(tmp_path))
    assert result["deleted"] and result["vectors_removed"] == 0


def test_deleting_an_unknown_knowledge_base_reports_not_found(tmp_path):
    assert manager(tmp_path).delete_with_storage("nope")["deleted"] is False


def test_deleting_twice_is_not_found_the_second_time(tmp_path):
    kb = manager(tmp_path)
    created = kb.create(name="kb")
    fill(created["kb_id"], "c-1")
    assert kb.delete_with_storage(created["kb_id"], str(tmp_path))["deleted"] is True
    second = kb.delete_with_storage(created["kb_id"], str(tmp_path))
    assert second["deleted"] is False and second["reason"] == "not found"


def test_the_record_is_gone_from_the_store_after_deletion(tmp_path):
    """Not merely gone from this manager's view: a second manager, reading the
    store fresh, does not find it either."""
    kb = KnowledgeBaseManager(str(tmp_path / "knowledge-bases.json"))
    created = kb.create(name="kb")
    kb.delete_with_storage(created["kb_id"], str(tmp_path))
    assert KnowledgeBaseManager(str(tmp_path / "knowledge-bases.json")).list() == []


def test_no_vector_row_survives_its_knowledge_base(tmp_path):
    """Stated against the database rather than against the store object: an
    orphaned row is one a later knowledge base could be matched against."""
    from sqlalchemy import func, select

    from storage import session_scope
    from storage.models import ChunkVector, VectorCollection

    kb = manager(tmp_path)
    created = kb.create(name="kb")
    fill(created["kb_id"], "c-1", "c-2", "c-3")
    kb.delete_with_storage(created["kb_id"], str(tmp_path))

    with session_scope() as session:
        assert session.scalar(select(func.count()).select_from(ChunkVector)) == 0
        assert session.scalar(select(func.count()).select_from(VectorCollection)) == 0


def test_a_provider_this_deployment_cannot_serve_is_refused(tmp_path):
    with pytest.raises(ValueError, match="pgvector"):
        manager(tmp_path).create(name="kb", vector_db_provider="chroma")
