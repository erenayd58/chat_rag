"""One built index, as one value, so a search never reads half of a rebuild.

A retriever's lexical index used to be three or four attributes -- the ordered
chunk list, the by-id map, the by-position map and the BM25 (or the frozen
hybrid index) over them -- assigned one after another by ``build_index`` and
cleared one after another by ``invalidate_index``. That was fine while every
request built a pipeline of its own. It is not fine now that the pipeline for
a knowledge base is shared by every caller who did not bring a session: an
ingest finishing on one thread rebuilds the index while a question on another
is between reading the BM25 and reading the chunk list, and the question is
answered with scores over one corpus and chunks out of another -- or with an
``AttributeError`` on the ``None`` the invalidation just wrote.

So the index is one immutable value. A retriever holds *one* reference to it,
a search reads that reference once and uses what it got, and a rebuild or an
invalidation replaces the reference. Reading a reference is atomic; nothing a
search holds can change under it. Building is serialised by the retriever's
own lock so two first questions do not read the whole store twice.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Optional

from chat_rag.core.models import DocumentChunk


@dataclass(frozen=True)
class BuiltIndex:
    """The chunks in chunk-id order and the engine built over them.

    ``engine`` is whatever the retriever searches with -- a BM25, the frozen
    hybrid index -- and is ``None`` when there were no chunks to build it
    from. An empty index is still a built one: the store was read and found
    empty, and it is not read again until something invalidates this.
    """

    chunks: tuple[DocumentChunk, ...]
    engine: Any
    by_id: Mapping[str, DocumentChunk]
    by_position: Mapping[tuple[str, int], DocumentChunk]

    @classmethod
    def over(cls, chunks: Iterable[DocumentChunk],
             build: Callable[[tuple[DocumentChunk, ...]], Any]) -> "BuiltIndex":
        ordered = tuple(sorted(chunks, key=lambda chunk: chunk.chunk_id))
        return cls(
            chunks=ordered,
            engine=build(ordered) if ordered else None,
            by_id={chunk.chunk_id: chunk for chunk in ordered},
            by_position={(chunk.doc_id, int(chunk.chunk_index)): chunk for chunk in ordered},
        )

    def __len__(self) -> int:
        return len(self.chunks)


class IndexHolder:
    """The one reference a retriever keeps, and the lock a build runs under."""

    def __init__(self) -> None:
        self._built: Optional[BuiltIndex] = None
        self._lock = threading.Lock()

    @property
    def current(self) -> Optional[BuiltIndex]:
        """What is built right now, or ``None``. Read once per search."""
        return self._built

    def replace(self, built: Optional[BuiltIndex]) -> None:
        """Publish a new index, or forget the one there is."""
        with self._lock:
            self._built = built

    def ensure(self, build: Callable[[], BuiltIndex]) -> BuiltIndex:
        """The current index, built by ``build`` first if there is none.

        Built under the lock, so a burst of first questions on a shared
        pipeline reads the store once; every waiter then gets the one index.
        """
        built = self._built
        if built is not None:
            return built
        with self._lock:
            if self._built is None:
                self._built = build()
            return self._built
