"""What retrieval must still do after the store underneath it changed.

The document-store contract next door says what *a* store must do. This says
something narrower and, during a store migration, more useful: the store the
product now runs and the reference implementation, given the same corpus and
the same query, must answer with the same ranking.

Behavioural parity, not floating-point parity. Two implementations may compute
a cosine distance to different last bits and neither is wrong; what may not
differ is the *order* they come back in, which document a filtered query is
allowed to see, what a tie does, what an empty result is, and which direction
``distance`` runs -- because everything above the store reads exactly those
(the RRF fusion, the score thresholds, and the console's
``1 - distance / 2``).

Chroma is not one of the implementations here. It was, while both stores
existed: parity was proven against it before it was removed, and the
reference store is what keeps this a comparison rather than a description of
pgvector once Chroma is gone. That is the same argument
``reference_store.py`` was written for.
"""

from __future__ import annotations

import pytest

from chat_rag.components.vectordb import PgVectorStore
from chat_rag.core.models import DocumentChunk

# --------------------------------------------------------------- the corpus
#: Written down rather than computed, so a ranking is a fact about the stores
#: and not about a model. ``t1``/``t2`` are deliberately equidistant from the
#: probe, and ``same-a``/``same-b`` deliberately carry identical text.
VECTORS = {
    "a1": [1.0, 0.0, 0.0, 0.0],
    "a2": [0.9, 0.1, 0.0, 0.0],
    "b1": [0.0, 1.0, 0.0, 0.0],
    "t2": [0.0, 0.0, 1.0, 0.0],
    "t1": [0.0, 0.0, 0.0, 1.0],
    "same-b": [0.5, 0.5, 0.0, 0.0],
    "same-a": [0.5, 0.5, 0.0, 0.0],
}

PROBE = [1.0, 0.0, 0.0, 0.0]
#: Equidistant from ``t1`` and ``t2`` (and from nothing else), so the tie is
#: the tie and not an accident of rounding.
TIE_PROBE = [0.0, 0.0, 1.0, 1.0]


def _chunk(chunk_id, doc_id, content, index, **metadata):
    return DocumentChunk(
        chunk_id=chunk_id, content=content, doc_id=doc_id, doc_title=doc_id + ".pdf",
        chunk_index=index, total_chunks=3, section_title="Bolum " + str(index),
        metadata={"word_count": len(content.split()), **metadata},
    )


CHUNKS = [
    _chunk("a1", "doc-a", "Takipteki alacaklar 2024 yilinda azaldi.", 0,
           search_text="takip alacak 2024", table_view="| Donem | Tutar |"),
    _chunk("a2", "doc-a", "Ayni bolumun ikinci paragrafi.", 1),
    _chunk("b1", "doc-b", "Baska bir dokumanin ilk parcasi.", 0),
    _chunk("t1", "doc-b", "Esit uzaklikta duran birinci parca.", 1),
    _chunk("t2", "doc-b", "Esit uzaklikta duran ikinci parca.", 2),
    # Identical text under two ids, in two documents: a real corpus has these
    # (a boilerplate paragraph repeated in every report) and a store must not
    # collapse them into one row.
    _chunk("same-a", "doc-a", "Ayni metin, iki farkli parcada.", 2),
    _chunk("same-b", "doc-b", "Ayni metin, iki farkli parcada.", 3),
]


def _pgvector(path):
    return PgVectorStore(collection="parity")


def _reference(path):
    from reference_store import ReferenceVectorDB

    return ReferenceVectorDB(path=str(path), collection_name="parity")


STORES = {"pgvector": _pgvector, "reference": _reference}


@pytest.fixture(params=sorted(STORES))
def filled(request, tmp_path):
    """The same corpus, in each implementation."""
    store = STORES[request.param](tmp_path / request.param)
    store.add_chunks(CHUNKS, [VECTORS[c.chunk_id] for c in CHUNKS])
    yield store
    closer = getattr(store, "close", None)
    if callable(closer):
        closer()


def ranking(store, vector, **kwargs):
    return [row["chunk_id"] for row in store.query(vector, **kwargs)]


# ------------------------------------------------------------ top-k order
def test_the_ranking_is_the_same_ranking(filled):
    """The whole point: nearest first, and the same order all the way down."""
    assert ranking(filled, PROBE, top_k=7) == [
        "a1", "a2", "same-a", "same-b", "b1", "t1", "t2",
    ]


def test_top_k_takes_a_prefix_of_that_ranking(filled):
    whole = ranking(filled, PROBE, top_k=7)
    for k in (1, 2, 3, 5):
        assert ranking(filled, PROBE, top_k=k) == whole[:k]


def test_distance_runs_the_same_direction_in_both(filled):
    """Lower is better, 0 for an identical vector, 1 for an orthogonal one.

    The two numbers the console and the retriever derive from this --
    ``1 - distance`` and ``1 - distance / 2`` -- are only meaningful on that
    scale, so the scale is part of the contract and not an implementation
    detail.
    """
    by_id = {row["chunk_id"]: row["distance"]
             for row in filled.query(PROBE, top_k=7)}
    assert by_id["a1"] == pytest.approx(0.0, abs=1e-6)
    assert by_id["b1"] == pytest.approx(1.0, abs=1e-6)
    assert by_id["a2"] < by_id["same-a"] < by_id["b1"]


# ------------------------------------------------------------------- ties
def test_a_tie_breaks_on_chunk_id_in_both(filled):
    """Equal distances must not come back in an arbitrary order: the RRF
    fusion above this turns a rank into a score, so an unstable page would
    make the same question give two different answers."""
    tied = [row for row in filled.query(TIE_PROBE, top_k=2)]
    assert [row["chunk_id"] for row in tied] == ["t1", "t2"]
    assert tied[0]["distance"] == pytest.approx(tied[1]["distance"], abs=1e-6)


def test_the_same_query_twice_gives_the_same_page(filled):
    assert ranking(filled, TIE_PROBE, top_k=3) == ranking(filled, TIE_PROBE, top_k=3)


def test_two_chunks_with_identical_text_stay_two_chunks(filled):
    found = {row["chunk_id"]: row for row in filled.query(VECTORS["same-a"], top_k=7)}
    assert {"same-a", "same-b"} <= set(found)
    assert found["same-a"]["content"] == found["same-b"]["content"]
    assert found["same-a"]["metadata"]["doc_id"] != found["same-b"]["metadata"]["doc_id"]


# --------------------------------------------------------------- filtering
def test_a_filtered_query_sees_one_document_in_both(filled):
    assert ranking(filled, PROBE, top_k=7, filter_dict={"doc_id": "doc-b"}) == [
        "same-b", "b1", "t1", "t2",
    ]


def test_a_filter_on_a_chunker_written_key_works_in_both(filled):
    """Not only ``doc_id``: the store the product replaced compared any
    metadata key, and the routes' pagination filter is written that way."""
    found = filled.get_chunks_paginated(offset=0, limit=10,
                                        filter_dict={"section_title": "Bolum 0"})
    assert sorted(row["chunk_id"] for row in found["chunks"]) == ["a1", "b1"]
    assert found["total"] == 2


def test_a_filter_that_matches_nothing_is_empty_not_an_error(filled):
    assert filled.query(PROBE, top_k=5, filter_dict={"doc_id": "doc-zzz"}) == []
    assert filled.get_chunks_paginated(filter_dict={"doc_id": "doc-zzz"})["total"] == 0


def test_an_empty_store_answers_empty_in_both(tmp_path, request):
    for name, factory in sorted(STORES.items()):
        store = factory(tmp_path / ("empty-" + name))
        assert store.query(PROBE, top_k=5) == []
        assert store.count() == 0


# --------------------------------------------------------------- deletion
def test_deleting_a_document_changes_both_rankings_the_same_way(filled):
    before = ranking(filled, PROBE, top_k=7)
    filled.delete_by_doc_id("doc-a")
    after = ranking(filled, PROBE, top_k=7)

    assert after == [cid for cid in before if cid not in {"a1", "a2", "same-a"}]
    assert filled.count() == 4


def test_deleting_one_chunk_leaves_the_rest_of_its_document(filled):
    filled.delete_chunk("a2")
    assert ranking(filled, PROBE, top_k=7) == [
        "a1", "same-a", "same-b", "b1", "t1", "t2",
    ]


# --------------------------------------------------------------- rebuilding
def test_re_embedding_moves_the_ranking_in_both_the_same_way(filled):
    """What the rebuild endpoint does: same texts, same metadata, new vectors.

    Not every store has to support it -- ``replace_all`` is an optional
    capability -- so a store without it is skipped rather than failed, which
    is the same check the pipeline makes before offering a re-index.
    """
    if not hasattr(filled, "replace_all"):
        pytest.skip(f"{filled.get_name()} does not re-index in place")

    chunks = sorted(filled.get_all_chunks(), key=lambda chunk: chunk.chunk_id)
    # Reversed: what was nearest becomes furthest, so a ranking that did not
    # actually change would be visible.
    flipped = [[-value for value in VECTORS[chunk.chunk_id]] for chunk in chunks]
    filled.replace_all(chunks, flipped)

    assert filled.count() == len(CHUNKS)
    assert ranking(filled, PROBE, top_k=1) == ["b1"]
    stored = {chunk.chunk_id: chunk for chunk in filled.get_all_chunks()}
    assert stored["a1"].content == CHUNKS[0].content, "a re-index rewrites vectors only"
    assert stored["a1"].search_text == "takip alacak 2024"


# ----------------------------------------------------- metadata on the way out
def test_a_hit_carries_the_metadata_the_answer_chain_reads(filled):
    """``search_text`` and ``table_view`` are reconstructed from the metadata a
    hit carries, so a store that dropped them would silently change what the
    answer model is given."""
    hit = filled.query(PROBE, top_k=1)[0]
    assert hit["metadata"]["search_text"] == "takip alacak 2024"
    assert hit["metadata"]["table_view"].startswith("| Donem |")
    assert hit["metadata"]["doc_title"] == "doc-a.pdf"
    assert int(hit["metadata"]["chunk_index"]) == 0


# ------------------------------------------------------------- kb isolation
def test_one_collection_never_sees_another(tmp_path):
    """The isolation a directory per knowledge base used to give.

    Stated on the shipped store only: the reference store *is* a directory, so
    it would pass this by construction and prove nothing about the store that
    replaced it with a key.
    """
    first = PgVectorStore(collection="kb-one")
    second = PgVectorStore(collection="kb-two")
    first.add_chunks(CHUNKS[:2], [VECTORS[c.chunk_id] for c in CHUNKS[:2]])
    second.add_chunks(CHUNKS[2:4], [VECTORS[c.chunk_id] for c in CHUNKS[2:4]])

    assert ranking(first, PROBE, top_k=5) == ["a1", "a2"]
    assert ranking(second, PROBE, top_k=5) == ["b1", "t1"]
    assert first.count() == 2 and second.count() == 2

    first.delete_by_doc_id("doc-a")
    assert first.count() == 0 and second.count() == 2
