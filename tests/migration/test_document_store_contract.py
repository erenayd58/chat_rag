"""What a document store must do, whichever store it is.

The product keeps its chunks in a ``BaseVectorDB``. Today that is Chroma;
the platform migration replaces it with pgvector. Nothing in this repository
said what such a replacement has to *do* -- the store was exercised only
through its own concrete class, so the contract lived in whichever call sites
happened to be covered. A pgvector store written against ``BaseVectorDB``
would satisfy the abstract base class and still break four routes, because
the base class declares six methods and the product calls twelve.

This module is that contract, stated once and run against more than one
implementation -- the store the product ships, and a dependency-free reference
store written to the contract and nothing else (``reference_store.py``). Two
implementations is the point: with one, a contract quietly becomes a
description of that one, which is exactly what a pgvector store would then
fail to be measured against. It is what a new store has to pass to be
finished.

Three things it deliberately does **not** pin:

* how a store computes similarity -- only that a nearer vector ranks first
  and that ``distance`` is lower-is-better, which is what the retrievers and
  the RRF fusion assume;
* how a store persists anything -- files, a sqlite database, a server;
* the exact float value of any score.

What it does pin is what the rest of the product reads back: the result
record, the metadata that survives a round trip (including the two derived
renderings the answer chain depends on), the isolation of one document's
rows from another's, and the pagination shape three routes return verbatim.
"""

from __future__ import annotations

import pytest

from components.vectordb import BaseVectorDB, ChromaVectorDB
from core.models import DocumentChunk

#: The store the product is configured with by default. Whatever else is
#: shipped, this one has to implement everything the routes call.
DEFAULT_STORE = ChromaVectorDB

#: Every method the product calls on a store without first checking that it
#: is there. A store that lacks one of these fails neither at import nor at
#: configuration time -- it fails at the fourth click, in production.
#: ``app.py`` and ``pipeline/rag_pipeline.py`` are where each is called.
REQUIRED_OF_THE_DEFAULT_STORE = (
    "add_chunks",              # ingest
    "query",                   # dense retrieval
    "get_all_chunks",          # keyword index build, stats, re-index
    "get_chunks_paginated",    # /api/chunks, /api/documents/<id>/chunks
    "search_chunks_by_text",   # /api/chunks?search=
    "get_chunk_by_id",         # GET    /api/chunks/<chunk_id>
    "update_chunk",            # PUT    /api/chunks/<chunk_id>
    "delete_chunk",            # DELETE /api/chunks/<chunk_id>
    "add_single_chunk",        # POST   /api/chunks
    "delete_by_doc_id",        # document deletion, ingest rollback
    "count",                   # retriever staleness checks
    "get_name",                # provenance, /api/stats
)

#: Methods the product calls only behind a capability check, so a store may
#: leave them out. Listed to keep the distinction deliberate: moving one of
#: these into the required set is the moment to remove its ``hasattr`` guard.
OPTIONAL = ("replace_all", "close", "bm25_search")


# --------------------------------------------------------------- the stores
def _chroma(path):
    return ChromaVectorDB(path=str(path), collection_name="documents")


def _reference(path):
    from reference_store import ReferenceVectorDB

    return ReferenceVectorDB(path=str(path), collection_name="documents")


#: ``chroma`` is what the product runs. ``reference`` is not shipped and is
#: never configured; it is here so the contract has a second implementation to
#: be a contract against -- see reference_store.py.
STORES = {"chroma": _chroma, "reference": _reference}


def _close(store):
    closer = getattr(store, "close", None)
    if callable(closer):
        closer()


@pytest.fixture(params=sorted(STORES))
def store(request, tmp_path):
    """One empty store of each shipped implementation."""
    made = STORES[request.param](tmp_path / request.param)
    yield made
    _close(made)


# --------------------------------------------------------------- the corpus
#: Four vectors, written down rather than computed, so "the nearest one comes
#: first" is a fact about the store and not about a model.
VECTORS = {
    "a1": [1.0, 0.0, 0.0, 0.0],
    "a2": [0.9, 0.1, 0.0, 0.0],
    "b1": [0.0, 1.0, 0.0, 0.0],
    "b2": [0.0, 0.0, 1.0, 0.0],
}


def _chunk(chunk_id, doc_id, content, index, total, **metadata):
    return DocumentChunk(
        chunk_id=chunk_id,
        content=content,
        doc_id=doc_id,
        doc_title=doc_id + ".pdf",
        chunk_index=index,
        total_chunks=total,
        section_title="Bolum " + str(index),
        metadata={"word_count": len(content.split()), **metadata},
    )


CHUNKS = [
    # The first chunk carries both derived renderings Deep Analysis produces.
    # They are the reason metadata has to survive a round trip: retrieval
    # indexes ``retrieval_text`` and the answer context reads ``table_view``,
    # and both are reconstructed from metadata after the store returns.
    _chunk("a1", "doc-a", "Takipteki alacaklar 2024 yilinda azaldi.", 0, 2,
           search_text="takip alacak 2024 12,4 milyar",
           table_view="| Donem | Tutar |\n| 2024 | 12,4 |"),
    _chunk("a2", "doc-a", "Ayni bolumun ikinci paragrafi.", 1, 2),
    _chunk("b1", "doc-b", "Baska bir dokumanin ilk parcasi.", 0, 2),
    _chunk("b2", "doc-b", "Baska bir dokumanin ikinci parcasi.", 1, 2),
]


@pytest.fixture
def filled(store):
    store.add_chunks(CHUNKS, [VECTORS[c.chunk_id] for c in CHUNKS])
    return store


# ------------------------------------------------------- the result record
def test_a_query_returns_the_record_every_retriever_reads(filled):
    """``chunk_id``, ``content``, ``distance``, ``metadata`` -- the four keys
    the retrievers, the fusion and the lab routes all destructure."""
    results = filled.query(VECTORS["a1"], top_k=4)
    assert results, "a filled store answered nothing"
    for row in results:
        assert {"chunk_id", "content", "distance", "metadata"} <= set(row)
        assert isinstance(row["chunk_id"], str) and row["chunk_id"]
        assert isinstance(row["content"], str)
        assert isinstance(row["distance"], (int, float))
        assert isinstance(row["metadata"], dict)


def test_the_nearest_vector_comes_first_and_distance_is_lower_is_better(filled):
    """RRF and every score threshold in the product assume this ordering."""
    results = filled.query(VECTORS["a1"], top_k=4)
    assert results[0]["chunk_id"] == "a1"
    distances = [row["distance"] for row in results]
    assert distances == sorted(distances), distances


def test_a_query_returns_at_most_the_top_k_asked_for(filled):
    assert len(filled.query(VECTORS["a1"], top_k=2)) <= 2


def test_content_comes_back_verbatim_because_a_citation_quotes_it(filled):
    """The stored ``content`` is the document's own text, and it is what a
    source card and a citation show. A store may index whatever it likes; it
    may not hand back a normalised, truncated or re-rendered version."""
    by_id = {row["chunk_id"]: row for row in filled.query(VECTORS["a1"], top_k=4)}
    for chunk in CHUNKS:
        if chunk.chunk_id in by_id:
            assert by_id[chunk.chunk_id]["content"] == chunk.content


# ----------------------------------------------------------- the round trip
def test_the_metadata_the_product_reads_survives_a_round_trip(filled):
    """Not "all metadata": the fields other layers actually look up."""
    stored = {chunk.chunk_id: chunk for chunk in filled.get_all_chunks()}
    assert set(stored) == {c.chunk_id for c in CHUNKS}
    for original in CHUNKS:
        back = stored[original.chunk_id]
        assert back.content == original.content
        assert back.doc_id == original.doc_id
        assert back.doc_title == original.doc_title
        assert int(back.chunk_index) == original.chunk_index
        assert int(back.total_chunks) == original.total_chunks
        assert back.section_title == original.section_title


def test_the_two_derived_renderings_survive_and_keep_their_meaning(filled):
    """``search_text`` is indexed *beside* the content, and ``table_view``
    rides in the answer context beneath it. Both are carried in metadata
    precisely so a store never has to widen its record for them -- which only
    works if the store gives the metadata back.

    The properties on ``DocumentChunk`` are the contract, not the storage:
    ``retrieval_text`` must still be content + rendering, and a chunk with no
    rendering must still index its content byte for byte.
    """
    stored = {chunk.chunk_id: chunk for chunk in filled.get_all_chunks()}
    enriched, plain = stored["a1"], stored["a2"]

    assert enriched.search_text == "takip alacak 2024 12,4 milyar"
    assert enriched.table_view.startswith("| Donem |")
    assert enriched.retrieval_text == enriched.content + "\n" + enriched.search_text

    assert plain.search_text is None
    assert plain.table_view is None
    assert plain.retrieval_text == plain.content


def test_count_is_the_number_of_chunks_stored(filled):
    assert filled.count() == len(CHUNKS)


def test_an_empty_store_answers_empty_rather_than_raising(store):
    assert store.count() == 0
    assert store.query(VECTORS["a1"], top_k=5) == []
    assert store.get_all_chunks() == []


def test_a_store_names_itself(store):
    assert isinstance(store.get_name(), str) and store.get_name()


# ------------------------------------------------------- document isolation
def test_a_query_can_be_scoped_to_one_document(filled):
    """The scope of a question. Chat filters by ``doc_id`` this way, and an
    answer quoting a document the user did not ask about is the bug this
    prevents."""
    results = filled.query(VECTORS["a1"], top_k=4, filter_dict={"doc_id": "doc-b"})
    assert results, "filtering returned nothing at all"
    assert {row["metadata"]["doc_id"] for row in results} == {"doc-b"}


def test_deleting_a_document_removes_its_rows_and_only_its_rows(filled):
    """Document deletion and ingest rollback both rely on this. Taking a
    neighbour's chunks with it would silently shrink another document."""
    filled.delete_by_doc_id("doc-a")
    left = filled.get_all_chunks()
    assert {chunk.chunk_id for chunk in left} == {"b1", "b2"}
    assert filled.count() == 2
    assert all(row["metadata"]["doc_id"] == "doc-b"
               for row in filled.query(VECTORS["a1"], top_k=4))


def test_deleting_a_document_that_is_not_there_is_not_an_error(filled):
    filled.delete_by_doc_id("doc-that-never-existed")
    assert filled.count() == len(CHUNKS)


# ------------------------------------------------------------- the browsing
def test_pagination_returns_the_shape_three_routes_hand_to_the_browser(filled):
    """``/api/chunks``, ``/api/documents/<id>/chunks`` and the canonical-units
    view return this object nearly verbatim, so its keys are a wire contract."""
    page = filled.get_chunks_paginated(offset=0, limit=2)
    assert set(page) == {"chunks", "total", "offset", "limit"}
    assert page["total"] == len(CHUNKS)
    assert page["offset"] == 0 and page["limit"] == 2
    assert len(page["chunks"]) == 2
    for row in page["chunks"]:
        assert {"chunk_id", "content", "metadata"} <= set(row)


def test_pagination_walks_the_whole_store_without_repeating_a_chunk(filled):
    seen = []
    for offset in (0, 2, 4):
        seen += [row["chunk_id"] for row in
                 filled.get_chunks_paginated(offset=offset, limit=2)["chunks"]]
    assert sorted(seen) == sorted(c.chunk_id for c in CHUNKS)


def test_pagination_can_be_scoped_to_one_document(filled):
    page = filled.get_chunks_paginated(offset=0, limit=10, filter_dict={"doc_id": "doc-a"})
    assert page["total"] == 2
    assert {row["metadata"]["doc_id"] for row in page["chunks"]} == {"doc-a"}


def test_a_text_search_finds_the_chunk_that_contains_the_phrase(filled):
    found = filled.search_chunks_by_text("takipteki", offset=0, limit=10)
    assert set(found) == {"chunks", "total", "offset", "limit"}
    assert [row["chunk_id"] for row in found["chunks"]] == ["a1"]
    assert filled.search_chunks_by_text("yok boyle bir sey")["total"] == 0


# ------------------------------------------------------------- persistence
@pytest.mark.parametrize("name", sorted(STORES))
def test_what_was_written_is_still_there_when_the_store_is_opened_again(name, tmp_path):
    """A restart must not lose an ingest. The store is reopened at the same
    location by a second instance, which is what a process restart is."""
    factory = STORES[name]
    path = tmp_path / name
    first = factory(path)
    first.add_chunks(CHUNKS, [VECTORS[c.chunk_id] for c in CHUNKS])
    _close(first)

    again = factory(path)
    try:
        assert again.count() == len(CHUNKS)
        assert {c.chunk_id for c in again.get_all_chunks()} == {c.chunk_id for c in CHUNKS}
        assert again.query(VECTORS["a1"], top_k=1)[0]["chunk_id"] == "a1"
    finally:
        _close(again)


# ----------------------------------------------- the surface a store needs
def test_the_default_store_implements_everything_the_routes_call():
    """The list above is derived from the unguarded call sites. It is the
    checklist a pgvector store has to finish -- ``BaseVectorDB`` alone is not,
    because it declares six of these twelve."""
    missing = [name for name in REQUIRED_OF_THE_DEFAULT_STORE
               if not callable(getattr(DEFAULT_STORE, name, None))]
    assert missing == [], (
        DEFAULT_STORE.__name__ + " is the configured default and the routes "
        "call these on it: " + repr(missing)
    )


def test_the_abstract_base_class_is_the_narrow_contract_every_store_meets():
    """What any store, default or not, must provide -- and the reason the list
    above exists beside it: this one is much smaller than the real need."""
    declared = set(BaseVectorDB.__abstractmethods__)
    assert declared, "the base class stopped declaring anything"
    assert declared < set(REQUIRED_OF_THE_DEFAULT_STORE), (
        "the base class now declares something the required list does not "
        "name: " + repr(sorted(declared - set(REQUIRED_OF_THE_DEFAULT_STORE)))
    )


def test_an_optional_capability_is_never_also_a_required_one():
    """"Optional" stays a decision rather than an accident: promoting one of
    these is the moment to also remove its ``hasattr`` guard."""
    assert not (set(OPTIONAL) & set(REQUIRED_OF_THE_DEFAULT_STORE))
