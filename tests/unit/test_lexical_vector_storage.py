"""What a lexical-only profile actually puts in the vector store.

The claim this profile makes is that no document embeddings are computed:
`uses_embeddings` is false, the retriever declares it needs no document
vectors, and the pipeline prints that it is skipping them. The store it used
to write to quietly broke that -- Chroma's `embedding_function` parameter
defaults to an ONNX all-MiniLM-L6-v2 instance rather than to None, and any
record added without a vector got one computed: an 80 MB model download and a
384-dimensional index nothing on this profile would ever read.

pgvector cannot make that mistake -- a column has no model attached to it --
but the behaviour on this side of the store is the one that has to be kept,
because the manifest reads it: a chunk stored without an embedding carries a
constant placeholder, one dimension wide, and ``index_status`` recognises
exactly that width as "no dense index" and refuses the dense leg. So these
tests hold the two things that matter about it: nothing is derived from the
text, and a collection that already holds real vectors keeps working.
"""

from __future__ import annotations

import pytest

from chat_rag.components.vectordb import PgVectorStore
from chat_rag.core.models import DocumentChunk


def chunk(index, text=None):
    return DocumentChunk(
        chunk_id=f"c-{index}",
        doc_id="doc-1",
        content=text or f"operasyonel risk hakkinda metin {index}",
        chunk_index=index,
        total_chunks=3,
        doc_title="rapor.pdf",
        section_title="Operasyonel Risk",
        metadata={"word_count": 5, "chunker_type": "structure_first"},
    )


@pytest.fixture
def store():
    return PgVectorStore(collection="documents")


def vectors_in(store):
    """The stored vectors, read back through the store's own accessor."""
    return [store.get_chunk_by_id(c.chunk_id)["embedding"]
            for c in store.get_all_chunks()]


# ------------------------------------------------ nothing derived from text


def test_a_lexical_add_stores_nothing_derived_from_the_document(store):
    store.add_chunks([chunk(0), chunk(1)], [])

    stored = vectors_in(store)
    assert len(stored) == 2
    assert stored[0] == stored[1], "two different texts produced the same vector"
    assert stored[0] == [1.0], "the placeholder is not the constant it should be"


def test_two_completely_different_texts_get_the_identical_placeholder(store):
    store.add_chunks(
        [chunk(0, "kredi riski olcumu"), chunk(1, "bambaska bir konu hakkinda")], []
    )
    first, second = vectors_in(store)
    assert first == second


def test_the_placeholder_has_a_length_cosine_distance_can_use(store):
    """A zero vector has no direction; cosine distance on it is undefined."""
    store.add_chunks([chunk(0)], [])
    vector = vectors_in(store)[0]
    assert sum(value * value for value in vector) > 0


def test_the_text_and_metadata_are_still_what_gets_stored(store):
    store.add_chunks([chunk(0)], [])

    found = store.get_chunk_by_id("c-0")
    assert found["content"] == "operasyonel risk hakkinda metin 0"
    assert found["metadata"]["chunker_type"] == "structure_first"
    assert found["metadata"]["doc_id"] == "doc-1"


def test_the_chunks_read_back_through_the_normal_accessor(store):
    store.add_chunks([chunk(0), chunk(1), chunk(2)], [])
    read = store.get_all_chunks()

    assert len(read) == 3
    assert {c.chunk_id for c in read} == {"c-0", "c-1", "c-2"}
    assert read[0].content


# --------------------------------------------------- dense profiles intact


def test_a_dense_add_stores_exactly_the_vectors_it_was_given(store):
    given = [[0.5] * 384, [-0.25] * 384]
    store.add_chunks([chunk(0), chunk(1)], given)

    stored = vectors_in(store)
    assert [len(v) for v in stored] == [384, 384]
    assert stored[0][0] == pytest.approx(0.5)
    assert stored[1][0] == pytest.approx(-0.25)


def test_dense_vectors_are_not_replaced_by_the_placeholder(store):
    store.add_chunks([chunk(0)], [[0.125] * 384])
    assert vectors_in(store)[0] != [1.0]


# ------------------------------------------------------------ compatibility


def test_a_lexical_add_into_an_existing_dense_store_keeps_its_width(store):
    """A collection holds one width, whatever wrote it first.

    A store written before this fix holds 384-wide vectors from Chroma's own
    model, and a migrated one holds them still; adding to it at a different
    width would make the collection unqueryable.
    """
    store.add_chunks([chunk(0)], [[0.5] * 384])
    store.add_chunks([chunk(1)], [])

    stored = vectors_in(store)
    assert {len(v) for v in stored} == {384}
    assert store.count() == 2


def test_the_placeholder_written_into_a_wide_store_is_still_constant(store):
    store.add_chunks([chunk(0)], [[0.5] * 768])
    store.add_chunks([chunk(1), chunk(2)], [])

    wide = [v for v in vectors_in(store) if v[0] != pytest.approx(0.5)]
    assert len(wide) == 2
    assert wide[0] == wide[1]
    assert len(wide[0]) == 768


def test_an_empty_collection_reports_no_stored_width(store):
    assert store._stored_dimension() is None


def test_a_populated_collection_reports_its_width(store):
    store.add_chunks([chunk(0)], [[0.1] * 384])
    assert store._stored_dimension() == 384


def test_reopening_a_store_written_earlier_still_reads():
    """A restart is a second instance naming the same collection."""
    first = PgVectorStore(collection="documents")
    first.add_chunks([chunk(0), chunk(1)], [])

    reopened = PgVectorStore(collection="documents")
    assert reopened.count() == 2
    assert len(reopened.get_all_chunks()) == 2


# ---------------------------------------------- one collection, one space
def test_a_dense_add_at_a_different_width_is_refused_by_name(store):
    """The store this replaced fixed the width at its first record and
    rejected anything else. A column would take both and the *query* would
    fail instead, on a row nobody could point at -- so the write is refused
    while there is still something to do about it."""
    from chat_rag.core.exceptions import VectorDBException

    store.add_chunks([chunk(0)], [[0.5] * 384])
    with pytest.raises(VectorDBException) as failure:
        store.add_chunks([chunk(1)], [[0.5] * 768])
    assert "384" in str(failure.value) and "768" in str(failure.value)


def test_a_reindex_may_change_the_width_because_it_replaces_everything(store):
    """``replace_all`` is the one call that is allowed to move a collection
    into another embedding space, because it empties it first."""
    store.add_chunks([chunk(0), chunk(1)], [[0.5] * 384, [0.25] * 384])
    store.replace_all([chunk(0), chunk(1)], [[0.5] * 768, [0.25] * 768])

    assert store.count() == 2
    assert store._stored_dimension() == 768
