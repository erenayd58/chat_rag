"""The indexing chunkers a knowledge base may be created with, in one table.

Deliberately not the same registry as the chunking methods a document is
*analysed* with (``amsc.methods``, offered through
``components/viewer/methods.py``). An analysis method is run over a document
for the Viewer to compare; an indexing chunker decides what a knowledge
base's retrieval index actually holds. ``structure_first`` is the product's
chunker; ``v4`` is the frozen AMSC V4/A4 package, kept so a knowledge base
can be indexed with exactly what the library's benchmark measured.

``tests/unit/test_method_wire_contract.py`` pins that no analysis-method key
or engine name is ever accepted here, so a Viewer selection can never quietly
become a retrieval change. The factory and the knowledge-base manager both
read this table rather than spelling the names again.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class IndexingChunker:
    id: str
    #: Every spelling a record or a setting may use for it, lower-cased.
    aliases: frozenset[str]
    description: str
    #: Whether ``chunker.params`` may carry runtime tuning for it.
    accepts_params: bool
    #: The wording the manager and the factory refuse params with.
    params_refusal: str = ""


INDEXING_CHUNKERS: tuple[IndexingChunker, ...] = (
    IndexingChunker(
        id="v4",
        aliases=frozenset({"v4", "frozenv4chunker", "frozen_v4_chunker"}),
        description="Frozen AMSC V4/A4 at Phase 5",
        accepts_params=False,
        params_refusal="Frozen V4 accepts no runtime",
    ),
    IndexingChunker(
        id="structure_first",
        aliases=frozenset({
            "structure_first", "structurefirst", "structural",
            "structuralchunker", "structural_chunker",
        }),
        description="Structure-first chunking (no embeddings) - default demo profile",
        accepts_params=False,
        params_refusal="Structure-first accepts no runtime",
    ),
)

#: What an unset or empty type means.
DEFAULT_ID = "structure_first"


def ids() -> tuple[str, ...]:
    return tuple(chunker.id for chunker in INDEXING_CHUNKERS)


def resolve(raw: Any) -> Optional[IndexingChunker]:
    """The chunker a type string names, or ``None`` for an unknown one."""
    wanted = str(raw or "").strip().lower()
    for chunker in INDEXING_CHUNKERS:
        if wanted in chunker.aliases:
            return chunker
    return None


def expected() -> str:
    """``'v4' or 'structure_first'`` -- for an error message."""
    names = [f"'{name}'" for name in ids()]
    return ", ".join(names[:-1]) + " or " + names[-1]


def describe() -> list[dict[str, str]]:
    """What the creation form is offered."""
    return [{"name": chunker.id, "description": chunker.description} for chunker in INDEXING_CHUNKERS]
