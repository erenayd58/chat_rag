"""Moving an existing Chroma corpus into pgvector, without a Chroma install.

The tool's one irreplaceable property is that it *copies* -- text byte for
byte, metadata whole, vectors as the floats they already were -- because the
alternative is re-embedding a corpus that cost real provider calls. Everything
below is about that, and about the two things a migration is judged on when it
goes wrong halfway: running it again finishes the job, and it never quietly
decides what to do about a conflict.

``read_chroma`` is the only part that needs ``chromadb``, and it is stubbed
here for the same reason it is a separate function: this repository does not
depend on Chroma any more, so its test suite must not either.
"""

from __future__ import annotations

import pytest

from components.vectordb import PgVectorStore
from tools import migrate_chroma_to_pgvector as tool

COLLECTION = "kb-migrated"

#: A Chroma export, in the shape ``read_chroma`` returns.
ROWS = [
    {"chunk_id": "doc-a:c-0", "content": "Takipteki alacaklar 2024 yilinda azaldi.",
     "metadata": {"doc_id": "doc-a", "doc_title": "rapor.pdf", "chunk_index": 0,
                  "total_chunks": 2, "section_title": "Bolum 0",
                  "chunker_type": "structure_first", "word_count": 5,
                  "search_text": "takip alacak 2024"},
     "embedding": [1.0, 0.0, 0.0, 0.0]},
    {"chunk_id": "doc-a:c-1", "content": "Ayni bolumun ikinci paragrafi.",
     "metadata": {"doc_id": "doc-a", "doc_title": "rapor.pdf", "chunk_index": 1,
                  "total_chunks": 2, "chunker_type": "structure_first"},
     "embedding": [0.9, 0.1, 0.0, 0.0]},
]

MANIFEST = {
    "schema_version": 1,
    "embedding_provider": "openai_compatible",
    "embedding_model": "qwen/qwen3-embedding-8b",
    "embedding_endpoint": "https://gw/v1/embeddings",
    "embedding_dimension": 4,
    "embedding_fingerprint": "fp-migrated",
    "chunk_count": 2,
    "written_at": "2026-09-01T10:00:00",
}


@pytest.fixture
def chroma(monkeypatch):
    """A stubbed Chroma export the tool reads instead of a real store."""
    state = {"rows": [dict(row, metadata=dict(row["metadata"])) for row in ROWS],
             "manifest": dict(MANIFEST)}
    monkeypatch.setattr(tool, "read_chroma",
                        lambda path: ([dict(r, metadata=dict(r["metadata"]))
                                       for r in state["rows"]], state["manifest"]))
    return state


def run(**kwargs):
    return tool.migrate_store("./chroma_db/kb", kb_id=None, collection=COLLECTION,
                              **kwargs)


# ------------------------------------------------------------- the copy
def test_a_store_is_copied_text_metadata_and_vectors(chroma):
    outcome = run()

    assert (outcome.read, outcome.migrated, outcome.skipped, outcome.failed) == (2, 2, 0, 0)
    store = PgVectorStore(collection=COLLECTION)
    assert store.count() == 2

    first = store.get_chunk_by_id("doc-a:c-0")
    assert first["content"] == ROWS[0]["content"], "content is not byte for byte"
    assert first["metadata"] == ROWS[0]["metadata"], "metadata did not survive whole"
    assert first["embedding"] == ROWS[0]["embedding"], "the vector was not copied"


def test_no_embedding_is_recomputed(chroma):
    """The floats in the target are the floats Chroma held, exactly."""
    run()
    store = PgVectorStore(collection=COLLECTION)
    for row in ROWS:
        assert store.get_chunk_by_id(row["chunk_id"])["embedding"] == row["embedding"]


def test_the_manifest_moves_with_the_vectors(chroma):
    """A corpus without its manifest is a corpus the retriever will not search
    densely -- the file beside the store has to become the row beside the
    vectors, or a migration silently disables the dense leg."""
    outcome = run()

    assert outcome.manifest
    manifest = PgVectorStore(collection=COLLECTION).read_manifest()
    assert manifest["embedding_fingerprint"] == "fp-migrated"
    assert manifest["embedding_model"] == "qwen/qwen3-embedding-8b"
    assert manifest["embedding_dimension"] == 4


def test_a_migrated_corpus_answers_the_query_it_answered_before(chroma):
    run()
    hits = PgVectorStore(collection=COLLECTION).query([1.0, 0.0, 0.0, 0.0], top_k=2)
    assert [hit["chunk_id"] for hit in hits] == ["doc-a:c-0", "doc-a:c-1"]
    assert hits[0]["distance"] == pytest.approx(0.0, abs=1e-6)


# ------------------------------------------------------------ dry running
def test_a_dry_run_writes_nothing_and_still_reports_what_it_would(chroma):
    outcome = run(dry_run=True)

    assert outcome.migrated == 2 and outcome.read == 2
    assert PgVectorStore(collection=COLLECTION).count() == 0


def test_a_dry_run_after_a_real_one_reports_nothing_left_to_do(chroma):
    run()
    outcome = run(dry_run=True)
    assert outcome.migrated == 0 and outcome.skipped == 2


# ------------------------------------------------------------ idempotence
def test_running_it_twice_migrates_nothing_the_second_time(chroma):
    run()
    outcome = run()

    assert (outcome.migrated, outcome.skipped, outcome.failed) == (0, 2, 0)
    assert PgVectorStore(collection=COLLECTION).count() == 2


def test_an_interrupted_run_is_finished_by_running_it_again(chroma):
    """Half the store already there, the other half not."""
    chroma["rows"] = chroma["rows"][:1]
    run()
    chroma["rows"] = [dict(row, metadata=dict(row["metadata"])) for row in ROWS]

    outcome = run()

    assert (outcome.migrated, outcome.skipped) == (1, 1)
    assert PgVectorStore(collection=COLLECTION).count() == 2


# -------------------------------------------------------------- conflicts
def test_a_chunk_already_there_with_different_content_is_a_conflict(chroma):
    run()
    chroma["rows"][0]["content"] = "Bambaska bir metin."

    outcome = run()

    assert outcome.conflicts == ["doc-a:c-0"] and outcome.failed == 1
    assert not outcome.ok
    kept = PgVectorStore(collection=COLLECTION).get_chunk_by_id("doc-a:c-0")
    assert kept["content"] == ROWS[0]["content"], "a conflict must not be resolved silently"


def test_overwrite_resolves_a_conflict_deliberately(chroma):
    run()
    chroma["rows"][0]["content"] = "Bambaska bir metin."

    outcome = run(overwrite=True)

    assert outcome.migrated == 1 and outcome.failed == 0
    assert PgVectorStore(collection=COLLECTION).get_chunk_by_id(
        "doc-a:c-0")["content"] == "Bambaska bir metin."


# ------------------------------------------------------------- validation
def test_a_store_that_repeats_a_chunk_id_is_refused_whole(chroma):
    chroma["rows"].append(dict(ROWS[0], content="another text under the same id"))

    outcome = run()

    assert outcome.migrated == 0 and outcome.failed == 3
    assert any("repeats these chunk ids" in problem for problem in outcome.problems)
    assert PgVectorStore(collection=COLLECTION).count() == 0, "nothing partial was written"


def test_a_store_holding_two_embedding_widths_is_refused_whole(chroma):
    chroma["rows"][1]["embedding"] = [0.1] * 8

    outcome = run()

    assert outcome.migrated == 0
    assert any("several widths" in problem for problem in outcome.problems)
    assert PgVectorStore(collection=COLLECTION).count() == 0


def test_a_row_with_no_chunk_id_is_refused(chroma):
    chroma["rows"].append({"chunk_id": "", "content": "x", "metadata": {},
                           "embedding": [0.0, 0.0, 0.0, 1.0]})
    outcome = run()
    assert any("no chunk id" in problem for problem in outcome.problems)


def test_a_row_whose_embedding_could_not_be_read_is_reported_not_invented(chroma):
    """The one case where the tool cannot copy: it says so and leaves that
    batch alone, and the documented answer is a rebuild with the current
    model -- never a vector made up here."""
    chroma["rows"][1]["embedding"] = None

    outcome = run()

    assert outcome.failed == 2 and outcome.migrated == 0
    assert any("rebuild" in problem for problem in outcome.problems)


# ------------------------------------------------------------ kb scoping
def test_the_collection_is_owned_by_the_knowledge_base_it_was_migrated_for(chroma):
    """A collection with no owner is a corpus nothing can delete, so the row
    carries the foreign key from the moment it is written."""
    from sqlalchemy import select

    from storage import session_scope
    from storage.models import KnowledgeBase, VectorCollection

    with session_scope() as session:
        session.add(KnowledgeBase(id="kb-9", name="migrated", name_key="migrated",
                                  chunker_type="structure_first"))

    tool.migrate_store("./chroma_db/kb", kb_id="kb-9", collection="kb-9")

    with session_scope() as session:
        row = session.scalar(select(VectorCollection)
                             .where(VectorCollection.collection == "kb-9"))
        assert row.kb_id == "kb-9"
