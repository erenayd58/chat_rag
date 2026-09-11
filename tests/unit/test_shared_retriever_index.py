"""A retriever shared by every caller of its knowledge base.

Every caller of a knowledge base shares its one pipeline now, so its
retriever is searched on several threads at once while an ingest on another
rebuilds the index and a delete on a third forgets it. Three properties keep
that honest, and each was false while the index was three separately
assigned attributes:

* a search reads **one** index for the whole of itself: never scores over one
  corpus and chunks out of another, never an ``AttributeError`` on the
  ``None`` an invalidation just wrote;
* a burst of first searches builds the index **once**, reading the store
  once, rather than once per thread;
* ``last_stats`` is the calling thread's own, because the pipeline reads it
  right after its own search and a concurrent search must not overwrite it.

The store here is in memory and the embedding is a stub: what is exercised is
the retriever's bookkeeping, not retrieval quality.
"""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from chat_rag.components.retriever import BM25OnlyRetriever, HybridRRFRetriever
from chat_rag.components.retriever.built_index import BuiltIndex
from chat_rag.core.models import DocumentChunk


def _chunk(index: int, text: str, doc: str = "doc") -> DocumentChunk:
    return DocumentChunk(
        chunk_id=f"{doc}:c{index:03d}", content=text, doc_id=doc, doc_title=doc,
        chunk_index=index, total_chunks=0, metadata={"doc_id": doc},
    )


CORPUS = [_chunk(i, f"kelime{i} takipteki alacaklar azaldi") for i in range(40)]
OTHER = [_chunk(i, f"baska{i} karbon ayak izi", doc="other") for i in range(15)]


class Store:
    """Counts how often the corpus is read, and lets a test hold the read open."""

    def __init__(self, chunks):
        self.chunks = list(chunks)
        self.reads = 0
        self.gate = threading.Event()
        self.gate.set()
        self.lock = threading.Lock()

    def get_all_chunks(self):
        with self.lock:
            self.reads += 1
        self.gate.wait(10)
        return list(self.chunks)

    def count(self):
        return len(self.chunks)

    def _stored_dimension(self):
        return None

    def read_manifest(self):
        return None

    def query(self, *args, **kwargs):  # pragma: no cover - dense leg is off here
        return []


class Embedding:
    def get_name(self):
        return "stub"

    def get_dimension(self):
        return 4

    def encode_queries(self, texts):
        return np.ones((len(texts), 4), dtype=np.float32)

    encode_documents = encode_queries


@pytest.fixture(params=["hybrid", "bm25_only"])
def retriever(request):
    store = Store(CORPUS)
    if request.param == "hybrid":
        built = HybridRRFRetriever(Embedding(), store)
    else:
        built = BM25OnlyRetriever(Embedding(), store)
    built.store = store
    return built


def _consistent(results):
    """Every hit names a chunk of the corpus it was scored against."""
    docs = {result.chunk.doc_id for result in results}
    assert len(docs) <= 1, f"one search answered out of two corpora: {docs}"
    return docs


def test_a_search_never_sees_half_of_a_rebuild(retriever):
    """Searches on eight threads while another thread swaps the index
    between two corpora and forgets it: no exception, and every answer is
    whole -- scored over one corpus, chunks from that corpus."""
    errors: list[BaseException] = []
    seen: set[str] = set()
    stop = threading.Event()

    def search():
        try:
            while not stop.is_set():
                hits = retriever.keyword_search("takipteki karbon", top_k=5)
                seen.update(_consistent(hits))
        except BaseException as error:  # noqa: BLE001 - the assertion is that there is none
            errors.append(error)

    def churn():
        for turn in range(60):
            retriever.build_index(CORPUS if turn % 2 else OTHER)
            retriever.invalidate_index()
            time.sleep(0.001)
        stop.set()

    searchers = [threading.Thread(target=search) for _ in range(8)]
    for thread in searchers:
        thread.start()
    churner = threading.Thread(target=churn)
    churner.start()
    churner.join(30)
    stop.set()
    for thread in searchers:
        thread.join(30)

    assert not errors, errors[:3]
    assert seen, "no search returned anything"


def test_a_burst_of_first_searches_reads_the_store_once(retriever):
    """Twelve threads ask an unbuilt retriever at once. The store is read once
    -- the build runs under the retriever's lock -- and every thread searches
    the one index that came of it."""
    store = retriever.store
    store.gate.clear()  # hold the first read open so every thread arrives before it ends
    results: list[list] = []
    barrier = threading.Barrier(12)

    def ask():
        barrier.wait(10)
        results.append(retriever.keyword_search("takipteki", top_k=3))

    threads = [threading.Thread(target=ask) for _ in range(12)]
    for thread in threads:
        thread.start()
    time.sleep(0.2)
    store.gate.set()
    for thread in threads:
        thread.join(20)

    assert len(results) == 12
    assert store.reads == 1, f"the corpus was read {store.reads} times for one build"
    assert all(hits for hits in results)
    assert retriever.ensure_index() is retriever.ensure_index(), "a second call rebuilt"


def test_an_empty_corpus_is_a_built_index_until_something_changes(retriever):
    """Reading an empty store is still a build: it is not read again on every
    search, only after a rebuild or an invalidation."""
    store = retriever.store
    store.chunks = []
    assert retriever.keyword_search("x", top_k=3) == []
    assert retriever.keyword_search("x", top_k=3) == []
    assert store.reads == 1
    retriever.invalidate_index()
    assert retriever.keyword_search("x", top_k=3) == []
    assert store.reads == 2


def test_the_names_the_product_reads_come_from_the_one_index(retriever):
    """``chunks_list``, ``_chunks_by_id`` and ``_bm25`` are read by the
    pipeline's status probe and by tests; they are views of the one built
    index and cannot disagree with each other."""
    assert retriever.chunks_list == [] and retriever._bm25 is None
    retriever.build_index(CORPUS)
    assert [chunk.chunk_id for chunk in retriever.chunks_list] == sorted(
        chunk.chunk_id for chunk in CORPUS)
    assert set(retriever._chunks_by_id) == {chunk.chunk_id for chunk in CORPUS}
    assert retriever._bm25 is not None
    assert isinstance(retriever.ensure_index(), BuiltIndex)
    retriever.invalidate_index()
    assert retriever.chunks_list == [] and retriever._bm25 is None


def test_last_stats_belong_to_the_thread_that_searched():
    """Two hybrid searches at once, each reading its own ``last_stats``
    afterwards -- the way the pipeline does -- see their own numbers."""
    store = Store(CORPUS)
    retriever = HybridRRFRetriever(Embedding(), store)
    retriever.build_index(CORPUS)
    barrier = threading.Barrier(2)
    outcomes: dict[int, tuple[int, int]] = {}

    def ask(top_k: int):
        barrier.wait(10)
        hits = retriever.hybrid_search("takipteki alacaklar", top_k=top_k)
        barrier.wait(10)  # both searches are over before either reads its stats
        outcomes[top_k] = (len(hits), retriever.last_stats["returned"])

    threads = [threading.Thread(target=ask, args=(k,)) for k in (2, 7)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(20)

    assert outcomes == {2: (2, 2), 7: (7, 7)}, outcomes
    assert retriever.last_stats == {}, "the main thread never searched"
