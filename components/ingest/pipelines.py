"""A bounded home for the most expensive object in the process.

A ``RAGPipeline`` is not a small thing. Each one holds an embedding model
(a local sentence-transformers model, when that is the provider), a Chroma
client with the store's sqlite file and hnsw index open, a chunker, possibly
a cross-encoder reranker, and a BM25 index built from **every chunk in the
knowledge base** -- held in memory, as objects. Phase 2's own report called
the cache holding these "effectively unbounded", and it was: the key is
``session_id:kb_id``, ``session_id`` is a uuid4 in a browser cookie, and
nothing ever removed an entry. Every new browser that opened a knowledge
base added one, for the life of the process.

Why the key still has a session in it
-------------------------------------

The obvious fix is one pipeline per knowledge base. It is the wrong fix
here, and the reason is three lines above the cache: ``RAGPipeline`` owns a
``ConversationManager``. Chat history -- the questions a person asked and
the answers they were given -- lives on the pipeline. Sharing one pipeline
between browsers would put one person's conversation into another person's
context window, and ``/api/clear`` would clear everybody's. That is not a
refactor, it is a privacy defect, so the session dimension stays and the
cache is *bounded* instead.

What that leaves is the drift Phase 2 also noted: two pipelines for one
knowledge base each hold their own BM25 index, and an ingest rebuilds only
the index of the pipeline that ran it. Left alone, a document ingested in
one browser is invisible to lexical search in another until the process
restarts. :meth:`PipelineCache.invalidate_indexes` is the answer to that --
the ingest tells the cache the knowledge base changed, every *other*
pipeline drops its index, and each rebuilds from the store on its next
query. Nothing is shared, nothing is stale.

Eviction that cannot corrupt anything
-------------------------------------

An entry is evicted only when it is not in use. "In use" is a count, not a
guess: :meth:`lease` increments it for the duration of a request or a job,
and a leased pipeline is never chosen for eviction, so nothing can close a
Chroma handle out from under a query in progress. Eviction closes the store
explicitly (the handle is what makes a store directory undeletable on
Windows) and then drops the reference.

The bound itself is least-recently-used with a ceiling
(``PIPELINE_CACHE_MAX``) and an idle timeout (``PIPELINE_CACHE_TTL``),
because both failure modes are real: many browsers in one hour, and one
browser that left an entry behind for a week.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Optional

logger = logging.getLogger("RAG.pipelines")


@dataclass
class Entry:
    key: str
    kb_id: str
    pipeline: Any
    created_at: float
    last_used: float
    uses: int = 0
    #: How many callers hold this pipeline right now. Never evicted above zero.
    leases: int = 0


class PipelineCache:
    """Bounded, lease-aware cache of built pipelines."""

    def __init__(
        self,
        build: Callable[[str, Optional[str]], Any],
        *,
        max_entries: int = 8,
        ttl_seconds: float = 1800.0,
        clock: Callable[[], float] = time.monotonic,
    ):
        if max_entries < 1:
            raise ValueError("PIPELINE_CACHE_MAX must be at least 1")
        self._build = build
        self.max_entries = int(max_entries)
        self.ttl_seconds = float(ttl_seconds)
        self._clock = clock
        self._lock = threading.RLock()
        self._entries: "OrderedDict[str, Entry]" = OrderedDict()
        self.stats = {"hits": 0, "misses": 0, "evicted_lru": 0, "evicted_idle": 0,
                      "evictions_deferred": 0, "invalidated": 0, "closed": 0}

    # ------------------------------------------------------------- reading
    @staticmethod
    def key_for(session_id: str, kb_id: Optional[str]) -> str:
        return f"{session_id}:{kb_id or 'default'}"

    def _acquire(self, session_id: str, kb_id: Optional[str], *, lease: bool) -> Any:
        """Find or build one pipeline, optionally leasing it, then evict.

        The order matters and is the whole point of doing this in one place.
        Taking the lease *before* eviction runs is what stops the cache handing
        back a pipeline it has just closed: with the bound reached and every
        other entry in use, the entry being returned is the only one eviction
        could pick, and it would pick it in the gap between the build and the
        lease. Building under the lock is deliberate too -- two threads asking
        for the same knowledge base at once would otherwise each construct an
        embedding model and a store handle, and one would be thrown away after
        being paid for.
        """
        key = self.key_for(session_id, kb_id)
        with self._lock:
            self._expire_locked()
            entry = self._entries.get(key)
            if entry is None:
                self.stats["misses"] += 1
                pipeline = self._build(session_id, kb_id)
                now = self._clock()
                entry = Entry(key=key, kb_id=kb_id or "default", pipeline=pipeline,
                              created_at=now, last_used=now)
                self._entries[key] = entry
            else:
                self.stats["hits"] += 1
            entry.uses += 1
            entry.last_used = self._clock()
            self._entries.move_to_end(key)
            if lease:
                entry.leases += 1
            # Never the entry just handed out: whoever asked for it is about
            # to use it, leased or not.
            self._evict_locked(protect=key)
            return entry.pipeline

    def get(self, session_id: str, kb_id: Optional[str] = None) -> Any:
        """The pipeline for this session and knowledge base, building it once.

        Safe for the length of this call. Work that outlives one call -- an
        ingest job -- must use :meth:`lease`, which keeps it safe for the
        length of the work.
        """
        return self._acquire(session_id, kb_id, lease=False)

    @contextmanager
    def lease(self, session_id: str, kb_id: Optional[str] = None) -> Iterator[Any]:
        """Hold a pipeline for the length of a request or a job.

        Nothing evicts a pipeline while a lease is out, so a long ingest
        cannot have its store closed by a burst of unrelated traffic.
        """
        key = self.key_for(session_id, kb_id)
        pipeline = self._acquire(session_id, kb_id, lease=True)
        try:
            yield pipeline
        finally:
            self._release(key)

    @contextmanager
    def lease_via(
        self, resolve: Callable[[str, Optional[str]], Any],
        session_id: str, kb_id: Optional[str] = None,
    ) -> Iterator[Any]:
        """Lease the pipeline that ``resolve`` returns, if this cache owns it.

        The application has one seam for "give me the pipeline for this
        session" -- ``app.get_pipeline`` -- and everything, including the test
        suite, goes through it. A job needs the pipeline from that seam *and*
        a lease on it, and taking them in two steps would leave a window in
        which the entry could be evicted between the two.

        So both happen here, under one lock: an entry that already exists is
        leased and returned; otherwise ``resolve`` is called to produce one
        (in production that re-enters :meth:`get` on this same reentrant lock
        and creates the entry, which is then leased). When ``resolve`` returns
        something this cache does not hold -- a stub in a test -- there is
        nothing to lease and nothing to protect, and it is simply returned.
        """
        key = self.key_for(session_id, kb_id)
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                pipeline = resolve(session_id, kb_id)
                entry = self._entries.get(key)
            else:
                pipeline = entry.pipeline
            leased = entry is not None and entry.pipeline is pipeline
            if leased:
                entry.leases += 1
                entry.uses += 1
                entry.last_used = self._clock()
                self._entries.move_to_end(key)
                self._evict_locked(protect=key)
        try:
            yield pipeline
        finally:
            if leased:
                self._release(key)

    def _release(self, key: str) -> None:
        with self._lock:
            entry = self._entries.get(key)
            if entry is not None:
                entry.leases = max(0, entry.leases - 1)
                entry.last_used = self._clock()
            self._evict_locked()

    # ------------------------------------------------------------ eviction
    def _expire_locked(self) -> None:
        if not self.ttl_seconds:
            return
        now = self._clock()
        for key in [
            key for key, entry in self._entries.items()
            if entry.leases == 0 and now - entry.last_used > self.ttl_seconds
        ]:
            self._drop_locked(key, "evicted_idle")

    def _evict_locked(self, protect: Optional[str] = None) -> None:
        while len(self._entries) > self.max_entries:
            victim = next(
                (key for key, entry in self._entries.items()
                 if entry.leases == 0 and key != protect),
                None,
            )
            if victim is None:
                # Everything is in use, or the only candidate is the entry
                # being handed out. The cache stays over its bound until a
                # lease ends or the next call arrives: exceeding a memory
                # target briefly beats closing a store somebody is using.
                self.stats["evictions_deferred"] += 1
                logger.warning(
                    "pipeline cache is over its bound (%d/%d) and every entry is in use",
                    len(self._entries), self.max_entries,
                )
                return
            self._drop_locked(victim, "evicted_lru")

    def _drop_locked(self, key: str, reason: str) -> None:
        entry = self._entries.pop(key, None)
        if entry is None:
            return
        self.stats[reason] = self.stats.get(reason, 0) + 1
        self._close(entry.pipeline)

    def _close(self, pipeline: Any) -> None:
        """Let go of the store handle before letting go of the pipeline.

        Dropping the reference is not enough: Chroma keeps the sqlite file and
        the hnsw index open until its client is closed, and on Windows an open
        handle is what makes a store directory undeletable.
        """
        store = getattr(pipeline, "vector_db", None)
        closer = getattr(store, "close", None)
        if callable(closer):
            try:
                closer()
                self.stats["closed"] += 1
            except Exception as error:  # noqa: BLE001 - an unclosable store is not fatal
                logger.warning("could not close a vector store on eviction: %s", error)

    # ------------------------------------------------------- explicit drops
    def discard_kb(self, kb_id: str) -> int:
        """Forget every pipeline for one knowledge base, closing its stores.

        Used when a knowledge base is deleted or re-indexed. A leased entry is
        dropped from the cache but not closed -- its user is mid-request, and
        closing the store under them would fail the request rather than tidy
        it; the last reference goes when they finish.
        """
        with self._lock:
            keys = [key for key, entry in self._entries.items() if entry.kb_id == kb_id]
            for key in keys:
                entry = self._entries.pop(key)
                if entry.leases:
                    logger.info("dropping in-use pipeline %s without closing its store", key)
                    continue
                self._close(entry.pipeline)
            return len(keys)

    def invalidate_indexes(self, kb_id: str, *, except_pipeline: Any = None) -> int:
        """Tell every other pipeline for this knowledge base that it changed.

        The lexical index is built once per pipeline and rebuilt only by the
        pipeline that ingested. Without this, a document ingested in one
        browser session is missing from keyword search in another until the
        process restarts -- the drift Phase 2 recorded.
        """
        invalidated = 0
        with self._lock:
            entries = [e for e in self._entries.values() if e.kb_id == kb_id]
        for entry in entries:
            if entry.pipeline is except_pipeline:
                continue
            retriever = getattr(entry.pipeline, "hybrid_retriever", None)
            if retriever is None:
                continue
            dropper = getattr(retriever, "invalidate_index", None)
            try:
                if callable(dropper):
                    dropper()
                else:
                    # Every retriever in this product builds from these two;
                    # clearing them makes ``ensure_index`` rebuild on demand.
                    retriever._bm25 = None
                    retriever.chunks_list = []
                invalidated += 1
            except Exception as error:  # noqa: BLE001 - a stale index is not worth a 500
                logger.warning("could not invalidate the index for %s: %s", kb_id, error)
        with self._lock:
            self.stats["invalidated"] += invalidated
        return invalidated

    def clear(self) -> None:
        with self._lock:
            for key in list(self._entries):
                self._drop_locked(key, "evicted_lru")

    # -------------------------------------------------------------- status
    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            now = self._clock()
            return {
                "size": len(self._entries),
                "max": self.max_entries,
                "ttl_seconds": self.ttl_seconds,
                "leased": sum(1 for e in self._entries.values() if e.leases),
                "knowledge_bases": sorted({e.kb_id for e in self._entries.values()}),
                "idle_seconds": {
                    "max": round(max((now - e.last_used for e in self._entries.values()), default=0.0), 1),
                },
                "stats": dict(self.stats),
            }
