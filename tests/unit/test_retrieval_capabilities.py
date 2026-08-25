from __future__ import annotations

from components.retriever import (
    BM25OnlyRetriever,
    NullEmbedding,
    method_is_available,
    retrieval_capabilities,
    unavailable_reason,
)


def available(retriever):
    return {
        entry["name"]
        for entry in retrieval_capabilities(retriever)["methods"]
        if entry["available"]
    }


class FullRetriever:
    """A retriever with a dense leg, like the legacy hybrid one."""

    requires_document_embeddings = True

    def hybrid_search(self, *a, **k): return []
    def vector_search(self, *a, **k): return []
    def keyword_search(self, *a, **k): return []


class HybridOnlyRetriever:
    """Benchmark-aligned: a single frozen entry point, nothing to split out."""

    def hybrid_search(self, *a, **k): return []


def test_a_dense_retriever_offers_every_method():
    assert available(FullRetriever()) == {"hybrid", "vector", "bm25"}


def test_a_lexical_only_retriever_offers_bm25_alone():
    retriever = BM25OnlyRetriever(embedding_model=NullEmbedding(), vector_db=None)
    assert available(retriever) == {"bm25"}


def test_vector_is_refused_because_no_embeddings_are_computed():
    retriever = BM25OnlyRetriever(embedding_model=NullEmbedding(), vector_db=None)
    assert not method_is_available(retriever, "vector")
    assert "embedding" in unavailable_reason(retriever, "vector").lower()


def test_hybrid_is_hidden_when_it_would_duplicate_bm25():
    """Without a dense leg the two run the same code; offering both implies a
    comparison the reviewer cannot actually make."""
    retriever = BM25OnlyRetriever(embedding_model=NullEmbedding(), vector_db=None)
    assert not method_is_available(retriever, "hybrid")
    assert "bm25" in unavailable_reason(retriever, "hybrid").lower()


def test_a_retriever_without_a_keyword_leg_does_not_offer_bm25():
    assert available(HybridOnlyRetriever()) == {"hybrid"}


def test_the_default_is_a_method_that_actually_works():
    retriever = BM25OnlyRetriever(embedding_model=NullEmbedding(), vector_db=None)
    caps = retrieval_capabilities(retriever)
    assert caps["default"] == "bm25"
    assert caps["dense"] is False
    assert retrieval_capabilities(FullRetriever())["default"] == "hybrid"


def test_every_method_is_reported_with_a_reason_when_unavailable():
    caps = retrieval_capabilities(BM25OnlyRetriever(NullEmbedding(), None))
    assert [m["name"] for m in caps["methods"]] == ["hybrid", "vector", "bm25"]
    for entry in caps["methods"]:
        assert entry["available"] or entry.get("reason")


def test_an_unknown_method_is_never_available():
    assert not method_is_available(FullRetriever(), "magic")
    assert "magic" in unavailable_reason(FullRetriever(), "magic")


def test_capabilities_do_not_run_a_search():
    class Exploding:
        requires_document_embeddings = True

        def hybrid_search(self, *a, **k): raise AssertionError("search was run")
        def vector_search(self, *a, **k): raise AssertionError("search was run")
        def keyword_search(self, *a, **k): raise AssertionError("search was run")

    assert available(Exploding()) == {"hybrid", "vector", "bm25"}
