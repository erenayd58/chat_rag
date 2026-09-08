from __future__ import annotations

import json
import pathlib

import pytest

from components.knowledgebase.manager import KnowledgeBaseManager


def manager(tmp_path):
    return KnowledgeBaseManager(str(tmp_path / "knowledge-bases.json"))


def store_dir(tmp_path, *parts):
    path = tmp_path.joinpath(*parts)
    path.mkdir(parents=True, exist_ok=True)
    (path / "chroma.sqlite3").write_text("x", encoding="utf-8")
    return path


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


def test_storage_path_follows_an_explicit_configuration(tmp_path):
    kb = manager(tmp_path)
    created = kb.create(name="kb", vector_db_path="./chroma_db/kkb_final")
    assert kb.storage_path(created["kb_id"], str(tmp_path)).endswith("kkb_final")


def test_storage_path_falls_back_to_a_directory_named_after_the_kb(tmp_path):
    kb = manager(tmp_path)
    chroma = kb.create(name="c", vector_db_provider="chroma")
    second = kb.create(name="f")

    chroma_path = pathlib.Path(kb.storage_path(chroma["kb_id"], str(tmp_path)))
    second_path = pathlib.Path(kb.storage_path(second["kb_id"], str(tmp_path)))

    assert chroma_path.name == chroma["kb_id"]
    assert chroma_path.parent.name == "chroma_db"
    assert second_path.name == second["kb_id"]
    assert second_path.parent.name == "chroma_db"


def test_delete_removes_the_record_and_its_store(tmp_path):
    kb = manager(tmp_path)
    created = kb.create(name="kb", vector_db_path="./chroma_db/kb_one")
    store = store_dir(tmp_path, "chroma_db", "kb_one")

    result = kb.delete_with_storage(created["kb_id"], str(tmp_path))

    assert result["deleted"] and result["storage_removed"]
    assert not store.exists()
    assert kb.list() == []


def test_a_store_another_knowledge_base_still_uses_is_kept(tmp_path):
    kb = manager(tmp_path)
    first = kb.create(name="one", vector_db_path="./chroma_db/shared")
    kb.create(name="two", vector_db_path="./chroma_db/shared")
    store = store_dir(tmp_path, "chroma_db", "shared")

    result = kb.delete_with_storage(first["kb_id"], str(tmp_path))

    assert result["deleted"] and not result["storage_removed"]
    assert "still used by" in result["storage_note"]
    assert store.exists()


def test_the_shared_default_store_is_never_removed(tmp_path):
    """chroma_db/ itself is the fallback store when no KB is selected."""
    kb = manager(tmp_path)
    created = kb.create(name="kb", vector_db_path="./chroma_db")
    store = store_dir(tmp_path, "chroma_db")

    result = kb.delete_with_storage(created["kb_id"], str(tmp_path))

    assert result["deleted"] and not result["storage_removed"]
    assert "shared/default" in result["storage_note"]
    assert store.exists()


def test_deleting_a_knowledge_base_with_no_store_is_not_an_error(tmp_path):
    kb = manager(tmp_path)
    created = kb.create(name="kb", vector_db_path="./chroma_db/never_created")
    result = kb.delete_with_storage(created["kb_id"], str(tmp_path))
    assert result["deleted"] and not result["storage_removed"]
    assert result["storage_note"] == "already absent"


def test_deleting_an_unknown_knowledge_base_reports_not_found(tmp_path):
    assert manager(tmp_path).delete_with_storage("nope")["deleted"] is False


def test_the_record_is_gone_from_the_store_after_deletion(tmp_path):
    """Not merely gone from this manager's view: a second manager, reading the
    store fresh, does not find it either."""
    kb = KnowledgeBaseManager(str(tmp_path / "knowledge-bases.json"))
    created = kb.create(name="kb")
    kb.delete_with_storage(created["kb_id"], str(tmp_path))
    assert KnowledgeBaseManager(str(tmp_path / "knowledge-bases.json")).list() == []


def test_a_store_that_cannot_be_removed_keeps_its_record(tmp_path, monkeypatch):
    """Dropping the record first would leave the orphan this method prevents.

    On Windows an open ChromaDB handle is enough to make the directory
    undeletable, and the route used to 500 after the record was already gone.
    """
    import shutil

    kb = manager(tmp_path)
    created = kb.create(name="kb", vector_db_path="./chroma_db/locked")
    store = store_dir(tmp_path, "chroma_db", "locked")

    def refuse(path):
        raise OSError("file is in use by another process")

    monkeypatch.setattr(shutil, "rmtree", refuse)
    result = kb.delete_with_storage(created["kb_id"], str(tmp_path))

    assert result["deleted"] is False
    assert "in use" in result["reason"]
    assert store.exists()
    assert kb.get(created["kb_id"]) is not None, "record must survive with its store"
    assert len(KnowledgeBaseManager(str(tmp_path / "knowledge-bases.json")).list()) == 1
