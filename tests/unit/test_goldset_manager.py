from __future__ import annotations

import json

import pytest

from components.goldset import GoldSetManager, entry_id_for, normalize_question


def clock():
    ticks = iter([f"2026-01-0{n}T00:00:00" for n in range(1, 10)])
    return lambda: next(ticks)


def manager(tmp_path, **kwargs):
    return GoldSetManager(str(tmp_path / "gold.json"), now=clock(), **kwargs)


ENTRY = {
    "question": "2024 Findeks Risk Raporu sorgu adedi kac?",
    "kb_id": "kb-1",
    "document_id": "upload_abc_pdf",
    "document_title": "rapor.pdf",
    "correct_chunk_id": "doc:s-chunk-0172",
    "section": "KOSGEB ISLETME DEGERLENDIRME RAPORU",
    "pages": [36],
    "unit_ids": ["p-00805", "v-00808"],
    "evidence": "FINDEKS RISK RAPORU SORGU ADEDI | 11.000.144",
    "retrieval_method": "bm25_only",
    "found_at_rank": 1,
}


def test_an_entry_round_trips_through_the_file(tmp_path):
    store = tmp_path / "gold.json"
    first = GoldSetManager(str(store), now=clock())
    saved = first.upsert(ENTRY)

    reloaded = GoldSetManager(str(store))
    assert [e["entry_id"] for e in reloaded.list()] == [saved["entry_id"]]
    assert reloaded.get("kb-1", ENTRY["question"])["evidence"] == ENTRY["evidence"]


def test_marking_the_same_question_updates_instead_of_appending(tmp_path):
    gold = manager(tmp_path)
    first = gold.upsert(ENTRY)
    second = gold.upsert({**ENTRY, "correct_chunk_id": "doc:s-chunk-0999", "found_at_rank": 3})

    assert len(gold.list()) == 1
    assert first["entry_id"] == second["entry_id"]
    assert second["correct_chunk_id"] == "doc:s-chunk-0999"
    assert second["created_at"] == first["created_at"]
    assert second["updated_at"] != first["updated_at"]


def test_question_identity_ignores_case_and_spacing(tmp_path):
    gold = manager(tmp_path)
    gold.upsert(ENTRY)
    gold.upsert({**ENTRY, "question": "  2024 FINDEKS Risk   Raporu sorgu adedi kac?  "})
    assert len(gold.list()) == 1


def test_the_same_question_in_another_knowledge_base_is_a_separate_entry(tmp_path):
    gold = manager(tmp_path)
    gold.upsert(ENTRY)
    gold.upsert({**ENTRY, "kb_id": "kb-2"})
    assert len(gold.list()) == 2
    assert len(gold.list(kb_id="kb-1")) == 1


def test_an_entry_carries_locators_beyond_the_chunk_id(tmp_path):
    """Chunk ids move whenever the parser or chunker changes."""
    entry = manager(tmp_path).upsert(ENTRY)
    assert entry["unit_ids"] == ["p-00805", "v-00808"]
    assert entry["pages"] == [36]
    assert entry["section"]
    assert entry["evidence"]
    assert entry["document_id"] == "upload_abc_pdf"


def test_evidence_alone_is_enough_to_record_an_answer(tmp_path):
    entry = manager(tmp_path).upsert({
        "question": "soru", "kb_id": "kb-1", "evidence": "kanit metni",
    })
    assert entry["correct_chunk_id"] is None
    assert entry["evidence"] == "kanit metni"


def test_an_entry_without_any_locator_is_refused(tmp_path):
    with pytest.raises(ValueError, match="locator"):
        manager(tmp_path).upsert({"question": "soru", "kb_id": "kb-1"})


@pytest.mark.parametrize("missing", ["question", "kb_id"])
def test_question_and_kb_are_required(tmp_path, missing):
    payload = {**ENTRY}
    payload[missing] = "   "
    with pytest.raises(ValueError, match=missing):
        manager(tmp_path).upsert(payload)


def test_evidence_is_truncated_so_the_file_stays_readable(tmp_path):
    entry = manager(tmp_path).upsert({**ENTRY, "evidence": "x" * 5000})
    assert len(entry["evidence"]) == 600


def test_delete_removes_the_entry(tmp_path):
    gold = manager(tmp_path)
    entry = gold.upsert(ENTRY)
    assert gold.delete(entry["entry_id"]) is True
    assert gold.list() == []
    assert gold.delete(entry["entry_id"]) is False


def test_delete_for_question_finds_the_entry_by_its_identity(tmp_path):
    gold = manager(tmp_path)
    gold.upsert(ENTRY)
    assert gold.delete_for_question("kb-1", "  2024 findeks RISK raporu sorgu adedi kac? ") is True
    assert gold.list() == []


def test_the_listing_is_deterministic_and_schema_versioned(tmp_path):
    """Two properties the regression CLI depends on: every entry carries the
    schema version it was written under, and the order is the entry id's, so
    two runs over one store produce the same list."""
    store = tmp_path / "gold.json"
    gold = GoldSetManager(str(store), now=clock())
    gold.upsert(ENTRY)
    gold.upsert({**ENTRY, "question": "ikinci soru", "kb_id": "kb-1"})

    entries = GoldSetManager(str(store)).list()
    assert {e["schema_version"] for e in entries} == {1}
    ids = [e["entry_id"] for e in entries]
    assert ids == sorted(ids)


def test_a_store_that_cannot_be_read_does_not_take_the_app_down(tmp_path, monkeypatch):
    """Reading the gold set is a screen's read, not a critical path: a store
    that cannot be reached must not be able to take the console with it.

    This used to be a corrupt JSON file; it is now an unreachable table, which
    is the same question about the same guarantee.
    """
    from storage.repositories import GoldSetRepository

    gold = GoldSetManager(str(tmp_path / "gold.json"))
    gold.upsert(ENTRY)

    def refuse(self):
        raise ConnectionError("the connection was closed")

    monkeypatch.setattr(GoldSetRepository, "all", refuse)
    with pytest.raises(ConnectionError):
        gold.list()
    # And it recovers the moment the store does: nothing was lost or rewritten.
    monkeypatch.undo()
    assert len(gold.list()) == 1


def test_entry_id_is_stable_for_the_same_question(tmp_path):
    assert entry_id_for("kb-1", "Bir Soru") == entry_id_for("kb-1", "  bir   soru ")
    assert entry_id_for("kb-1", "a") != entry_id_for("kb-2", "a")
    assert normalize_question("  A  B ") == "a b"


def test_question_identity_uses_invariant_lowercasing():
    """The browser mirrors this rule to find a saved mark again.

    A Turkish-locale lowercase would fold 'I' to dotless 'ı' and the two sides
    would disagree about the same question.
    """
    assert normalize_question("BIR SORU") == "bir soru"
    assert entry_id_for("kb", "BIR") == entry_id_for("kb", "bir")
