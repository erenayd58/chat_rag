"""The pipeline cache: bounded, and never evicting something in use.

A ``RAGPipeline`` holds an embedding model, a Chroma handle and the whole
knowledge base's lexical index. Phase 2's report called this cache
"effectively unbounded", and it was: keyed by a browser session id, never
pruned. These tests pin the three things that had to become true -- it
cannot grow past its bound, it cannot evict a pipeline someone is using, and
an evicted pipeline's store handle is closed rather than leaked -- plus the
index drift that having several pipelines per knowledge base caused.

The doubles are deliberately not pipelines: the cache must not know what it
holds beyond "it may have a vector_db to close and a retriever to invalidate".
"""

from __future__ import annotations

import threading

import pytest

from chat_rag.components.ingest.pipelines import PipelineCache


class Store:
    def __init__(self):
        self.closed = 0

    def close(self):
        self.closed += 1


class Retriever:
    def __init__(self):
        self.invalidated = 0
        self._bm25 = object()
        self.chunks_list = [1, 2, 3]

    def invalidate_index(self):
        self.invalidated += 1
        self._bm25 = None
        self.chunks_list = []


class FakePipeline:
    def __init__(self, kb_id):
        self.kb_id = kb_id
        self.vector_db = Store()
        self.hybrid_retriever = Retriever()


def make(max_entries=3, ttl=1800.0, clock=None):
    built = []

    def build(session_id, kb_id):
        pipeline = FakePipeline(kb_id)
        built.append((session_id, kb_id))
        return pipeline

    cache = PipelineCache(build, max_entries=max_entries, ttl_seconds=ttl,
                          clock=clock or (lambda: 0.0))
    return cache, built


# ------------------------------------------------------------------ basics


def test_a_pipeline_is_built_once_per_session_and_knowledge_base():
    cache, built = make()
    first = cache.get("s1", "kb1")
    again = cache.get("s1", "kb1")
    assert first is again
    assert built == [("s1", "kb1")]
    assert cache.stats["hits"] == 1 and cache.stats["misses"] == 1


def test_a_different_session_gets_its_own_pipeline():
    """Conversation history lives on the pipeline, so sessions must not share
    one. This is why the cache is bounded rather than keyed by knowledge base."""
    cache, built = make()
    assert cache.get("s1", "kb1") is not cache.get("s2", "kb1")
    assert len(built) == 2


def test_concurrent_callers_build_one_pipeline_not_two():
    """Two threads asking at the same moment would otherwise each construct an
    embedding model and a store handle, and one would be thrown away."""
    ready = threading.Barrier(6)
    built = []
    lock = threading.Lock()

    def build(session_id, kb_id):
        with lock:
            built.append(kb_id)
        return FakePipeline(kb_id)

    cache = PipelineCache(build, max_entries=4, ttl_seconds=0)
    results = []

    def ask():
        ready.wait(10)
        results.append(cache.get("s1", "kb1"))

    threads = [threading.Thread(target=ask) for _ in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(10)
    assert len(built) == 1
    assert len({id(pipeline) for pipeline in results}) == 1


# ------------------------------------------------------------------ bounds


def test_the_cache_cannot_grow_past_its_bound():
    cache, built = make(max_entries=3)
    pipelines = [cache.get(f"s{index}", "kb1") for index in range(10)]
    assert cache.snapshot()["size"] == 3
    assert cache.stats["evicted_lru"] == 7
    # The evicted ones had their stores closed rather than merely dropped.
    assert sum(p.vector_db.closed for p in pipelines) == 7
    assert [p.vector_db.closed for p in pipelines[-3:]] == [0, 0, 0], "the newest three"


def test_eviction_is_least_recently_used():
    cache, _ = make(max_entries=2)
    first = cache.get("s1", "kb1")
    second = cache.get("s2", "kb1")
    cache.get("s1", "kb1")  # first is used again, so second is now the oldest
    third = cache.get("s3", "kb1")
    assert second.vector_db.closed == 1
    assert first.vector_db.closed == 0 and third.vector_db.closed == 0
    assert cache.snapshot()["size"] == 2


def test_an_idle_pipeline_is_dropped_after_its_ttl():
    now = {"t": 1000.0}
    cache, _ = make(max_entries=10, ttl=60.0, clock=lambda: now["t"])
    stale = cache.get("s1", "kb1")
    now["t"] += 61
    cache.get("s2", "kb2")  # any access sweeps expired entries
    assert stale.vector_db.closed == 1
    assert cache.stats["evicted_idle"] == 1
    assert cache.snapshot()["size"] == 1


def test_ttl_zero_means_no_idle_eviction():
    now = {"t": 0.0}
    cache, _ = make(max_entries=10, ttl=0, clock=lambda: now["t"])
    cache.get("s1", "kb1")
    now["t"] += 10_000
    cache.get("s2", "kb2")
    assert cache.snapshot()["size"] == 2


# ------------------------------------------------------------------ leases


def test_a_leased_pipeline_is_never_evicted():
    """An ingest job outlives the request that started it. Closing its store
    mid-write to make room for a browser would corrupt the job, not the cache.

    The unleased entries are still evicted normally around it, so a long job
    does not stop the cache doing its work -- it only protects itself.
    """
    cache, _ = make(max_entries=1)
    with cache.lease("worker", "kb1") as busy:
        churn = [cache.get(f"browser{index}", "kb1") for index in range(5)]
        assert busy.vector_db.closed == 0, "the working pipeline survived"
        assert cache.snapshot()["leased"] == 1
        # Each browser evicts the one before it. The last is still held back,
        # because it was just handed out and its caller is about to use it.
        assert sum(p.vector_db.closed for p in churn) == 4
        assert churn[-1].vector_db.closed == 0
    cache.get("someone-else", "kb1")
    assert sum(p.vector_db.closed for p in churn) == 5


def test_when_everything_is_in_use_the_bound_gives_before_a_store_does():
    """The one case where the cache knowingly exceeds its bound. Briefly
    holding one pipeline too many beats closing a store mid-write."""
    cache, _ = make(max_entries=1)
    with cache.lease("worker-a", "kb1") as first:
        with cache.lease("worker-b", "kb2") as second:
            assert cache.snapshot()["size"] == 2 > cache.max_entries
            assert cache.stats["evictions_deferred"] >= 1
            assert first.vector_db.closed == 0 and second.vector_db.closed == 0
    # Both leases are done: the next access brings it back under the bound.
    cache.get("browser", "kb3")
    assert cache.snapshot()["size"] <= cache.max_entries


def test_a_pipeline_handed_out_is_never_closed_during_that_call():
    """The bug this ordering exists to prevent: with the bound reached and
    every other entry leased, eviction would pick the entry being returned --
    closing its store between the build and the caller's first use of it."""
    cache, _ = make(max_entries=1)
    with cache.lease("worker", "kb1"):
        handed_out = cache.get("browser", "kb1")
        assert handed_out.vector_db.closed == 0
    assert cache.stats["evictions_deferred"] >= 1


def test_a_lease_returns_the_same_pipeline_the_cache_holds():
    cache, _ = make()
    with cache.lease("s1", "kb1") as leased:
        assert cache.get("s1", "kb1") is leased
    assert cache.snapshot()["leased"] == 0


def test_a_lease_is_released_even_when_the_work_fails():
    cache, _ = make()
    with pytest.raises(RuntimeError):
        with cache.lease("s1", "kb1"):
            raise RuntimeError("the ingest failed")
    assert cache.snapshot()["leased"] == 0


def test_nested_leases_on_one_pipeline_are_counted():
    cache, _ = make(max_entries=1)
    with cache.lease("s1", "kb1"):
        with cache.lease("s1", "kb1"):
            assert cache.snapshot()["leased"] == 1
        assert cache.snapshot()["leased"] == 1, "the outer lease still holds it"
    assert cache.snapshot()["leased"] == 0


# ------------------------------------------------------- knowledge bases


def test_discarding_a_knowledge_base_closes_its_stores():
    """A deleted knowledge base's store directory cannot be removed on Windows
    while a Chroma handle is open on it."""
    cache, _ = make(max_entries=10)
    first = cache.get("s1", "kb1")
    second = cache.get("s2", "kb1")
    other = cache.get("s3", "kb2")

    assert cache.discard_kb("kb1") == 2
    assert first.vector_db.closed == 1 and second.vector_db.closed == 1
    assert other.vector_db.closed == 0
    assert cache.snapshot()["knowledge_bases"] == ["kb2"]


def test_a_knowledge_base_in_use_is_dropped_without_closing_under_its_user():
    cache, _ = make(max_entries=10)
    with cache.lease("worker", "kb1") as busy:
        assert cache.discard_kb("kb1") == 1
        assert busy.vector_db.closed == 0, "closing it would fail the request using it"
    assert cache.snapshot()["size"] == 0


def test_ingest_invalidates_every_other_pipelines_index():
    """The drift Phase 2 recorded: a document ingested in one session was
    missing from another session's keyword search until a restart."""
    cache, _ = make(max_entries=10)
    ingesting = cache.get("uploader", "kb1")
    watching = cache.get("browser", "kb1")
    elsewhere = cache.get("browser", "kb2")

    invalidated = cache.invalidate_indexes("kb1", except_pipeline=ingesting)

    assert invalidated == 1
    assert watching.hybrid_retriever.invalidated == 1
    assert watching.hybrid_retriever._bm25 is None
    assert ingesting.hybrid_retriever.invalidated == 0, "it rebuilt its own index already"
    assert elsewhere.hybrid_retriever.invalidated == 0, "a different knowledge base"
    assert cache.stats["invalidated"] == 1


def test_a_retriever_that_cannot_be_invalidated_does_not_break_an_ingest():
    class Awkward:
        def invalidate_index(self):
            raise RuntimeError("no index here")

    cache, _ = make(max_entries=10)
    pipeline = cache.get("s1", "kb1")
    pipeline.hybrid_retriever = Awkward()
    assert cache.invalidate_indexes("kb1") == 0


def test_a_store_that_will_not_close_does_not_break_eviction():
    class Stubborn:
        def close(self):
            raise OSError("still in use by something")

    cache, _ = make(max_entries=1)
    first = cache.get("s1", "kb1")
    first.vector_db = Stubborn()
    cache.get("s2", "kb1")  # evicts the first
    assert cache.snapshot()["size"] == 1


def test_the_snapshot_reports_what_an_operator_asks_for():
    cache, _ = make(max_entries=4)
    cache.get("s1", "kb1")
    cache.get("s2", "kb2")
    snapshot = cache.snapshot()
    assert snapshot["size"] == 2 and snapshot["max"] == 4
    assert snapshot["knowledge_bases"] == ["kb1", "kb2"]
    assert set(snapshot["stats"]) >= {"hits", "misses", "evicted_lru", "evicted_idle"}


def test_a_cache_of_zero_is_refused_at_construction():
    with pytest.raises(ValueError):
        PipelineCache(lambda s, k: None, max_entries=0)
