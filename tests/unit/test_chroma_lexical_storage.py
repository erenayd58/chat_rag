"""What a lexical-only profile actually puts in Chroma.

The claim this profile makes is that no document embeddings are computed:
`uses_embeddings` is false, the retriever declares it needs no document
vectors, and the pipeline prints that it is skipping them. Chroma quietly
broke that. Its `embedding_function` parameter defaults to an ONNX
all-MiniLM-L6-v2 instance, not to None, and any record added without a vector
gets one computed -- an 80 MB model download and a 384-dimensional index that
nothing on this profile would ever read.

Chroma has no storage-only mode, so a constant placeholder is supplied
instead. These tests hold the two things that matter about it: nothing is
derived from the text, and a store that already holds real vectors keeps
working.
"""

from __future__ import annotations

import os

import pytest

from components.vectordb.chroma_vectordb import ChromaVectorDB
from core.models import DocumentChunk


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
def store(tmp_path):
    return ChromaVectorDB(path=str(tmp_path / "store"), collection_name="documents")


def vectors_in(store):
    found = store.collection.get(include=["embeddings"])
    return [list(v) for v in found["embeddings"]]


# ------------------------------------------------------- no model attached


def test_the_collection_carries_no_embedding_function(store):
    """Chroma's default is a model instance, not None; it has to be refused."""
    assert getattr(store.collection, "_embedding_function", "unset") is None


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

    found = store.collection.get(include=["documents", "metadatas"])
    assert found["documents"][0] == "operasyonel risk hakkinda metin 0"
    assert found["metadatas"][0]["chunker_type"] == "structure_first"
    assert found["metadatas"][0]["doc_id"] == "doc-1"


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
    """Chroma fixes a collection's width at its first record.

    Stores written before this fix hold 384-wide vectors from Chroma's own
    model; adding to one at a different width would be rejected outright.
    """
    store.add_chunks([chunk(0)], [[0.5] * 384])
    store.add_chunks([chunk(1)], [])

    stored = vectors_in(store)
    assert {len(v) for v in stored} == {384}
    assert store.collection.count() == 2


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


def test_reopening_a_store_written_earlier_still_reads(tmp_path):
    path = str(tmp_path / "store")
    first = ChromaVectorDB(path=path, collection_name="documents")
    first.add_chunks([chunk(0), chunk(1)], [])

    reopened = ChromaVectorDB(path=path, collection_name="documents")
    assert reopened.collection.count() == 2
    assert len(reopened.get_all_chunks()) == 2
