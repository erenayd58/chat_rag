"""What the vector store does when a write fails or two callers overlap.

The store used to be a directory of files opened by one process. It is a table
several threads reach through one pooled engine now, and that changes which
mistakes are possible: a half-written ingest, a search reading a corpus while
a re-index replaces it, a deletion under a query, two workers writing the same
chunk. Each of those is a transaction question, and each is asked here.

The rule underneath all of them: **an ingest is one unit of work.** Either
every chunk of a document is searchable or none of it is. A corpus that is
half in the old embedding space and half in the new one, or half a document's
chunks in the index, is worse than an ingest that failed cleanly -- because
nothing reports it and the answers just get quietly worse.

Nothing here sleeps to make an ordering happen; threads are released by events
the test holds.
"""

from __future__ import annotations

import threading

import pytest
from sqlalchemy import func, select

from chat_rag.components.vectordb import PgVectorStore
from chat_rag.core.exceptions import VectorDBException
from chat_rag.core.models import DocumentChunk
from chat_rag.storage import session_scope
from chat_rag.storage.models import ChunkVector, KnowledgeBase, VectorCollection
from chat_rag.storage.repositories import ChunkVectorRepository


def chunk(index, doc="doc-a", text=None):
    return DocumentChunk(
        chunk_id=f"{doc}:c-{index:04d}",
        content=text or f"parca {index} hakkinda metin",
        doc_id=doc, doc_title=doc + ".pdf", chunk_index=index, total_chunks=1200,
        section_title="Bolum", metadata={"word_count": 4},
    )


def corpus(count, doc="doc-a"):
    chunks = [chunk(i, doc) for i in range(count)]
    return chunks, [[1.0, float(i) / 1000.0, 0.0] for i in range(count)]


def rows_in(collection):
    with session_scope() as session:
        return int(session.scalar(
            select(func.count()).select_from(ChunkVector)
            .where(ChunkVector.collection == collection)) or 0)


# ------------------------------------------------------------- atomicity
def test_a_failure_part_way_through_an_ingest_leaves_nothing_searchable(monkeypatch):
    """An ingest larger than one statement still commits once.

    ``add_chunks`` writes in batches so a single statement stays a sane size;
    the batches are inside one transaction, and this is what says so. Before,
    a failure in the middle left the first batches visible to search and the
    document's ledger row absent -- chunks nothing could delete.
    """
    store = PgVectorStore(collection="atomic")
    chunks, vectors = corpus(1200)

    real = ChunkVectorRepository.upsert
    calls = {"n": 0}

    def fail_on_the_third_batch(self, collection, rows):
        calls["n"] += 1
        if calls["n"] == 3:
            raise RuntimeError("connection reset half way through the ingest")
        return real(self, collection, rows)

    monkeypatch.setattr(ChunkVectorRepository, "upsert", fail_on_the_third_batch)

    with pytest.raises(VectorDBException):
        store.add_chunks(chunks, vectors)

    assert calls["n"] == 3, "the write really did fail after earlier batches"
    assert rows_in("atomic") == 0, "two batches stayed searchable after a failed ingest"
    assert store.count() == 0


def test_a_failed_reindex_leaves_the_corpus_as_it_was(monkeypatch):
    """``replace_all`` empties and refills. If the refill fails, the emptying
    must go too -- otherwise a re-index that broke half way is a knowledge
    base with no vectors at all."""
    store = PgVectorStore(collection="reindex")
    chunks, vectors = corpus(10)
    store.add_chunks(chunks, vectors)

    real = ChunkVectorRepository.upsert
    monkeypatch.setattr(
        ChunkVectorRepository, "upsert",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("the gateway went away")))

    with pytest.raises(VectorDBException):
        store.replace_all(chunks, [[0.0, 1.0, 0.0] for _ in chunks])

    monkeypatch.setattr(ChunkVectorRepository, "upsert", real)
    assert store.count() == 10
    assert store.query([1.0, 0.0, 0.0], top_k=1)[0]["chunk_id"] == chunks[0].chunk_id


def test_a_refused_write_is_refused_before_anything_is_written():
    """The width check is part of the same transaction as the write."""
    store = PgVectorStore(collection="widths")
    store.add_chunks([chunk(0)], [[1.0, 0.0, 0.0]])

    with pytest.raises(VectorDBException):
        store.add_chunks([chunk(1), chunk(2)], [[1.0] * 8, [1.0] * 8])

    assert store.count() == 1


# ------------------------------------------------------------- duplicates
def test_writing_the_same_chunk_twice_leaves_one_row():
    """A retry, a duplicate job, an ingest of a document already there."""
    store = PgVectorStore(collection="dupes")
    store.add_chunks([chunk(0)], [[1.0, 0.0, 0.0]])
    store.add_chunks([chunk(0)], [[1.0, 0.0, 0.0]])

    assert store.count() == 1
    assert rows_in("dupes") == 1


def test_re_ingesting_a_document_replaces_its_chunks_rather_than_doubling_them():
    store = PgVectorStore(collection="reingest")
    chunks, vectors = corpus(5)
    store.add_chunks(chunks, vectors)
    # The same document, chunked again: same ids, new text.
    again = [chunk(i, text=f"yeniden yazilmis parca {i}") for i in range(5)]
    store.add_chunks(again, vectors)

    assert store.count() == 5
    stored = {c.chunk_id: c for c in store.get_all_chunks()}
    assert stored[again[0].chunk_id].content == "yeniden yazilmis parca 0"


def test_two_threads_writing_the_same_chunk_both_finish():
    """Two ingest workers racing for one chunk id: one row, no error.

    The upsert is what makes this true; a plain insert would make one of them
    fail on the primary key, and an ingest that failed because another ingest
    succeeded is not a failure anybody can act on.
    """
    store = PgVectorStore(collection="race")
    start = threading.Event()
    errors: list[BaseException] = []

    def write(text):
        start.wait(10)
        try:
            store.add_chunks([chunk(0, text=text)], [[1.0, 0.0, 0.0]])
        except BaseException as error:  # noqa: BLE001
            errors.append(error)

    threads = [threading.Thread(target=write, args=(f"metin {i}",)) for i in range(2)]
    for thread in threads:
        thread.start()
    start.set()
    for thread in threads:
        thread.join(30)

    assert errors == []
    assert store.count() == 1


def test_two_threads_creating_one_collection_do_not_race_for_it():
    """Two ingests into a knowledge base that has never been written to."""
    errors: list[BaseException] = []
    start = threading.Event()

    def write(index):
        start.wait(10)
        try:
            PgVectorStore(collection="fresh").add_chunks(
                [chunk(index)], [[1.0, 0.0, 0.0]])
        except BaseException as error:  # noqa: BLE001
            errors.append(error)

    threads = [threading.Thread(target=write, args=(i,)) for i in range(4)]
    for thread in threads:
        thread.start()
    start.set()
    for thread in threads:
        thread.join(30)

    assert errors == []
    assert PgVectorStore(collection="fresh").count() == 4
    with session_scope() as session:
        assert session.scalar(
            select(func.count()).select_from(VectorCollection)
            .where(VectorCollection.collection == "fresh")) == 1


# ------------------------------------------------------------- concurrency
def _hammer(work, seconds_of_work=40):
    """Run ``work`` in a thread and give back anything it raised."""
    failures: list[BaseException] = []

    def run():
        try:
            work()
        except BaseException as error:  # noqa: BLE001
            failures.append(error)

    thread = threading.Thread(target=run)
    thread.start()
    return thread, failures


def test_a_search_during_a_reindex_sees_one_corpus_or_the_other_never_half():
    """A re-index replaces every vector. A query overlapping it must not see
    a collection that is empty in the middle -- which is what a delete
    followed by a separate insert would give it."""
    store = PgVectorStore(collection="busy")
    chunks, vectors = corpus(200)
    store.add_chunks(chunks, vectors)
    flipped = [[0.0, 1.0, 0.0] for _ in chunks]

    counts: list[int] = []
    stop = threading.Event()

    def reindex():
        for _ in range(3):
            store.replace_all(chunks, flipped)
            store.replace_all(chunks, vectors)
        stop.set()

    thread, failures = _hammer(reindex)
    while not stop.is_set():
        counts.append(store.count())
    thread.join(60)

    assert failures == []
    assert counts, "the search thread never got to look"
    assert set(counts) == {200}, f"a search saw a partly rewritten corpus: {sorted(set(counts))}"


def test_a_search_during_a_document_deletion_never_sees_half_a_document():
    store = PgVectorStore(collection="deleting")
    keep, keep_vectors = corpus(50, doc="doc-keep")
    store.add_chunks(keep, keep_vectors)

    seen: list[int] = []
    stop = threading.Event()

    def churn():
        for _ in range(5):
            going, going_vectors = corpus(50, doc="doc-going")
            store.add_chunks(going, going_vectors)
            store.delete_by_doc_id("doc-going")
        stop.set()

    thread, failures = _hammer(churn)
    while not stop.is_set():
        seen.append(len(store.get_chunks_paginated(
            offset=0, limit=200, filter_dict={"doc_id": "doc-going"})["chunks"]))
    thread.join(60)

    assert failures == []
    assert set(seen) <= {0, 50}, f"a reader saw part of a document: {sorted(set(seen))}"
    assert store.count() == 50, "the untouched document lost or gained rows"


# ---------------------------------------------------------------- cascades
def _knowledge_base(kb_id, name):
    with session_scope() as session:
        session.add(KnowledgeBase(id=kb_id, name=name, name_key=name,
                                  chunker_type="structure_first"))


def test_deleting_a_knowledge_base_row_takes_its_vectors_with_it():
    """By the database, not by a caller remembering to. A vector that outlived
    its knowledge base is an orphan a later knowledge base could match."""
    _knowledge_base("kb-a", "a")
    _knowledge_base("kb-b", "b")
    PgVectorStore(collection="kb-a", kb_id="kb-a").add_chunks(*corpus(4))
    PgVectorStore(collection="kb-b", kb_id="kb-b").add_chunks(*corpus(3, doc="doc-b"))

    with session_scope() as session:
        session.execute(KnowledgeBase.__table__.delete()
                        .where(KnowledgeBase.id == "kb-a"))

    assert rows_in("kb-a") == 0
    assert rows_in("kb-b") == 3
    with session_scope() as session:
        left = set(session.scalars(select(VectorCollection.collection)))
    assert left == {"kb-b"}


def test_deleting_a_collection_leaves_no_orphan_chunk_rows():
    PgVectorStore(collection="gone").add_chunks(*corpus(6))
    with session_scope() as session:
        removed = ChunkVectorRepository(session).delete_collection("gone")

    assert removed == 6
    with session_scope() as session:
        assert session.scalar(select(func.count()).select_from(ChunkVector)) == 0


def test_deleting_a_collection_twice_is_not_an_error():
    PgVectorStore(collection="twice").add_chunks(*corpus(2))
    with session_scope() as session:
        assert ChunkVectorRepository(session).delete_collection("twice") == 2
    with session_scope() as session:
        assert ChunkVectorRepository(session).delete_collection("twice") == 0


def test_deleting_a_document_twice_is_not_an_error():
    store = PgVectorStore(collection="doc-twice")
    store.add_chunks(*corpus(3))
    store.delete_by_doc_id("doc-a")
    store.delete_by_doc_id("doc-a")
    assert store.count() == 0


def test_one_content_ingested_into_two_knowledge_bases_is_two_corpora():
    """The same PDF uploaded twice is two documents and one content, and each
    upload's chunks belong to its own knowledge base. Deleting one upload must
    not empty the other's index."""
    _knowledge_base("kb-1", "one")
    _knowledge_base("kb-2", "two")
    chunks, vectors = corpus(4, doc="doc-shared")
    PgVectorStore(collection="kb-1", kb_id="kb-1").add_chunks(chunks, vectors)
    PgVectorStore(collection="kb-2", kb_id="kb-2").add_chunks(chunks, vectors)

    PgVectorStore(collection="kb-1").delete_by_doc_id("doc-shared")

    assert PgVectorStore(collection="kb-1").count() == 0
    assert PgVectorStore(collection="kb-2").count() == 4
