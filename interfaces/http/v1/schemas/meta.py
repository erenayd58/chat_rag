"""What this deployment can offer, and whether it is well.

The chunking-method list is the important one. It is a projection of the
library's registry and there is no second catalogue anywhere -- not here, not
in the console's JavaScript, not in the Viewer. A method registered in the
library appears here on the next request, which is what makes
"implementation + registration + tests" the whole cost of adding one. Nothing
in this module names a method, and nothing may.
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import Field

from .common import Collection, Schema


class ChunkingMethod(Schema):
    """One chunking method, exactly as the library's registry describes it.

    Nothing is added and nothing is filtered. ``available`` is this machine's
    answer, not the library's -- a method needing an embedder it cannot load
    is offered as unavailable with the reason, never hidden, because a picker
    that silently drops an option cannot explain why it is missing.
    """

    key: str
    label: str
    summary: str
    engine: str
    available: bool
    unavailable_reason: Optional[str] = None
    uses_model: bool
    default: bool
    orchestration: bool = Field(
        default=False,
        description="an orchestration runs over a baseline partition rather "
                    "than being one; a picker that shows the two the same way "
                    "misleads",
    )
    baseline: Optional[str] = None

    @classmethod
    def of(cls, entry: dict) -> "ChunkingMethod":
        return cls(
            key=entry["key"],
            label=entry["label"],
            summary=entry["summary"],
            engine=entry["engine"],
            available=entry["available"],
            unavailable_reason=entry.get("reason") or None,
            uses_model=entry["uses_model"],
            default=entry["default"],
            orchestration=bool(entry.get("orchestration")),
            baseline=entry.get("baseline"),
        )


class ChunkingMethodCollection(Collection[ChunkingMethod]):
    pass


class RetrievalMethod(Schema):
    """One retrieval method, and whether this knowledge base's retriever can
    actually serve it. Reporting only -- no search runs."""

    name: Optional[str] = None
    label: Optional[str] = None
    available: bool = False
    unavailable_reason: Optional[str] = None

    @classmethod
    def of(cls, entry: dict) -> "RetrievalMethod":
        return cls(
            name=entry.get("name"),
            label=entry.get("label"),
            available=bool(entry.get("available")),
            unavailable_reason=entry.get("reason") or None,
        )


class RetrievalMethodCollection(Collection[RetrievalMethod]):
    default: Optional[str] = None


class ModelChain(Schema):
    """The configured model chain: chunking, embedding, answer. Never a key.

    ``chain`` is the pipeline's own report and is passed through: which
    provider fields a chain has is a deployment's business, and the one thing
    this endpoint promises is that no credential is among them.
    """

    chain: dict[str, Any] = Field(default_factory=dict)


class IngestCapacityLine(Schema):
    running: int
    queued: int
    queue_capacity: int
    workers: int


class QueryCapacityLine(Schema):
    active: int
    max_active: int


class Capacity(Schema):
    ingest: IngestCapacityLine
    query: QueryCapacityLine


class Health(Schema):
    """Liveness, readiness and one line of capacity.

    Deliberately small and always answerable, including while degraded --
    which it reports rather than fails on. Detail is an operator's question,
    and this endpoint is polled every few seconds by something that only needs
    to know whether to look further.
    """

    state: str
    ready: bool
    reasons: list[Any] = Field(default_factory=list)
    checked_at: Optional[str] = None
    capacity: Capacity

    @classmethod
    def of(cls, state: dict) -> "Health":
        return cls(
            state=state["state"],
            ready=state["ready"],
            reasons=state["reasons"],
            checked_at=state["timestamp"],
            capacity=Capacity(
                ingest=IngestCapacityLine(
                    running=state["ingest"]["running"],
                    queued=state["ingest"]["queued"],
                    queue_capacity=state["ingest"]["queue_capacity"],
                    workers=state["ingest"]["workers"],
                ),
                query=QueryCapacityLine(
                    active=state["query"]["active"],
                    max_active=state["query"]["max_active"],
                ),
            ),
        )
