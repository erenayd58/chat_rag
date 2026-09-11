"""What the public API answers with: values, not the engine's own dictionaries.

Every use case in :mod:`chat_rag.application` returns a plain ``dict`` -- which
is exactly right for a use case, because the adapter above it decides what to
call each field. ``/api/v1`` decides one way (``interfaces/http/v1/schemas``);
this module decides the other, for a caller writing Python.

The two projections are deliberately independent. They read the same
dictionaries and they will usually agree, but the wire contract is versioned
and this one is not the same promise: freezing them together would mean a
Python rename is an HTTP break, and an HTTP addition is a Python one. Nothing
here imports a schema, and nothing in ``interfaces`` imports this.

Frozen dataclasses, and ``raw`` on the ones with an open-ended payload. The
named fields are what this library promises; ``raw`` is the dictionary it was
built from, so a caller who needs something the promise does not cover can
reach it without waiting for a release -- and knows, by having to say ``raw``,
that it is not part of the promise.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator, Mapping, Optional, Sequence


@dataclass(frozen=True)
class Source:
    """One passage an answer was given, and whether the answer used it.

    ``label`` is what the answer cites (``[S1]``); ``used`` is the difference
    between a passage the model read and one it was merely offered.
    """

    label: Optional[str] = None
    chunk_id: Optional[str] = None
    document_id: Optional[str] = None
    document: Optional[str] = None
    section: Optional[str] = None
    pages: tuple[Any, ...] = ()
    chunking_mode: Optional[str] = None
    used: bool = False
    score: Optional[float] = None
    content: str = ""
    raw: Mapping[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def of(cls, row: Mapping[str, Any]) -> "Source":
        return cls(
            label=row.get("label"),
            chunk_id=row.get("chunk_id"),
            document_id=row.get("doc_id") or row.get("document_id"),
            document=row.get("document"),
            section=row.get("heading") or row.get("section"),
            pages=tuple(row.get("pages") or ()),
            chunking_mode=row.get("chunking_mode"),
            used=bool(row.get("used")),
            score=row.get("score"),
            content=row.get("content") or row.get("content_preview") or "",
            raw=dict(row),
        )


@dataclass(frozen=True)
class Answer:
    """One answered question, with the evidence it was answered from.

    ``grounded`` is false when the model answered without citing any of the
    passages it was given. A caller that shows ``text`` without reading this
    cannot tell an answer from a guess, which is why it is a field here and
    not something to dig for in ``diagnostics``.
    """

    text: str = ""
    sources: tuple[Source, ...] = ()
    grounded: bool = False
    knowledge_base_id: Optional[str] = None
    retrieval_method: Optional[str] = None
    query_id: Optional[str] = None
    seconds: Optional[float] = None
    #: The pipeline's own metadata. Useful, and explicitly not contractual:
    #: it is how a retrieval change is investigated, and pinning it here would
    #: freeze the internals this facade exists to keep free.
    diagnostics: Mapping[str, Any] = field(default_factory=dict, repr=False)

    def __str__(self) -> str:
        """Printing an answer prints the answer."""
        return self.text

    @classmethod
    def of(cls, result: Mapping[str, Any], *,
           knowledge_base_id: Optional[str] = None) -> "Answer":
        metadata: Mapping[str, Any] = result.get("metadata") or {}
        generation: Mapping[str, Any] = metadata.get("answer") or {}
        timing: Mapping[str, Any] = metadata.get("query") or {}
        return cls(
            text=result.get("answer") or "",
            sources=tuple(Source.of(row) for row in (result.get("sources") or ())),
            grounded=bool(generation.get("grounded")),
            knowledge_base_id=knowledge_base_id,
            retrieval_method=metadata.get("retrieval_method"),
            query_id=timing.get("query_id"),
            seconds=timing.get("total_seconds"),
            diagnostics=dict(metadata),
        )


@dataclass(frozen=True)
class Hit:
    """One retrieved chunk and the score that ranked it.

    A search result rather than a stored row: it carries the method that found
    it, which is the thing two searches of the same corpus differ by.
    """

    chunk_id: Optional[str] = None
    content: str = ""
    score: Optional[float] = None
    method: Optional[str] = None
    document_id: Optional[str] = None
    document: Optional[str] = None
    metadata: Mapping[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def of(cls, row: Mapping[str, Any], *, method: Optional[str] = None) -> "Hit":
        metadata: Mapping[str, Any] = row.get("metadata") or {}
        return cls(
            chunk_id=row.get("chunk_id"),
            content=row.get("content") or "",
            score=row.get("score"),
            method=row.get("retrieval_method") or method,
            document_id=metadata.get("doc_id") or row.get("doc_id"),
            document=metadata.get("doc_title") or metadata.get("document"),
            metadata=dict(metadata),
        )


@dataclass(frozen=True)
class Chunk:
    """One stored chunk of an indexed corpus: what a question actually searches.

    Not a :class:`Hit`: nothing ranked it, so it has no score and no method.
    """

    chunk_id: Optional[str] = None
    content: str = ""
    document_id: Optional[str] = None
    index: Optional[int] = None
    metadata: Mapping[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def of(cls, row: Mapping[str, Any]) -> "Chunk":
        metadata: Mapping[str, Any] = row.get("metadata") or {}
        return cls(
            chunk_id=row.get("chunk_id"),
            content=row.get("content") or "",
            document_id=metadata.get("doc_id"),
            index=metadata.get("chunk_index"),
            metadata=dict(metadata),
        )


@dataclass(frozen=True)
class Health:
    """What one engine can do right now.

    Three states, because they call for three different things: ``ok`` is
    serving with room to accept work, ``overloaded`` is serving but refusing
    new uploads or questions (wait, do not restart), and ``degraded`` is
    serving reads with something an operator should look at. ``reasons`` says
    which, in words.
    """

    state: str = "ok"
    ready: bool = True
    reasons: tuple[str, ...] = ()
    llm_provider: Optional[str] = None
    embedding_model: Optional[str] = None
    ingest: Mapping[str, Any] = field(default_factory=dict, repr=False)
    query: Mapping[str, Any] = field(default_factory=dict, repr=False)
    raw: Mapping[str, Any] = field(default_factory=dict, repr=False)

    @property
    def ok(self) -> bool:
        """True only in the state that is serving *and* has room."""
        return self.state == "ok"

    @classmethod
    def of(cls, report: Mapping[str, Any]) -> "Health":
        return cls(
            state=report.get("state") or "ok",
            ready=bool(report.get("ready")),
            reasons=tuple(report.get("reasons") or ()),
            llm_provider=report.get("llm_provider"),
            embedding_model=report.get("embedding_model"),
            ingest=dict(report.get("ingest") or {}),
            query=dict(report.get("query") or {}),
            raw=dict(report),
        )


@dataclass(frozen=True)
class Migration:
    """What :meth:`Engine.migrate` did to the schema.

    ``outcome`` is one of three words. ``created``: the database was empty
    and has the schema now. ``upgraded``: it was at an earlier revision and is
    at head now. ``current``: it was already at head and nothing was applied
    -- said as its own answer rather than as silence, because a program that
    runs this at every start needs "nothing to do" to look different from
    "did not run". ``before`` is ``None`` for an empty database.
    """

    before: Optional[str] = None
    after: Optional[str] = None
    outcome: str = "current"

    @property
    def changed(self) -> bool:
        """Whether anything was applied."""
        return self.outcome != "current"

    @classmethod
    def of(cls, report: Mapping[str, Any]) -> "Migration":
        return cls(
            before=report.get("before"),
            after=report.get("after"),
            outcome=report.get("outcome") or "current",
        )


@dataclass(frozen=True)
class Method:
    """One chunking method, as this deployment can offer it.

    ``available`` is a fact about this machine, not about the registry: a
    method needing a local embedding model is offered only where one can be
    run, and carries ``reason`` when it cannot -- so a caller never asks for a
    comparison the engine would then refuse.
    """

    key: str
    label: str = ""
    summary: str = ""
    engine: Optional[str] = None
    available: bool = True
    reason: str = ""
    uses_model: bool = False
    needs_embedder: bool = False
    #: True for a method that orchestrates a baseline partition rather than
    #: being one. ``baseline`` names the partition it starts from.
    orchestration: bool = False
    baseline: Optional[str] = None

    @classmethod
    def of(cls, entry: Mapping[str, Any]) -> "Method":
        return cls(
            key=entry.get("key") or "",
            label=entry.get("label") or "",
            summary=entry.get("summary") or "",
            engine=entry.get("engine") or entry.get("kind"),
            available=bool(entry.get("available", True)),
            reason=entry.get("reason") or "",
            uses_model=bool(entry.get("uses_model")),
            needs_embedder=bool(entry.get("needs_embedder")),
            orchestration=bool(entry.get("orchestration")),
            baseline=entry.get("baseline"),
        )


@dataclass(frozen=True)
class Arm:
    """One chunking method's answer, inside a comparison.

    ``status`` is what to branch on: ``ok``, ``insufficient`` (nothing
    retrieved, or the model said the passages were not enough),
    ``no_answer_model`` or ``answer_error``. An arm that could not answer
    still carries its sources, and never fails the comparison -- in a
    comparison the other arms are still the result.
    """

    method: str
    status: str = "ok"
    text: str = ""
    sufficient: Optional[bool] = None
    sources: tuple[Source, ...] = ()
    engine: Optional[str] = None
    label: Optional[str] = None
    error: Optional[str] = None
    #: How much of this arm's retrieved context, at the canonical-unit level,
    #: the other arms also retrieved. ``None`` when only one method ran,
    #: because a comparison of one has nothing to overlap with.
    unit_overlap: Optional[float] = None
    #: False when this arm's index is BM25 alone. An answer, not a failure.
    dense: bool = False
    note: Optional[str] = None
    seconds: Optional[float] = None

    @property
    def answered(self) -> bool:
        return self.status == "ok"

    @classmethod
    def of(cls, method: str, result: Mapping[str, Any], *,
           unit_overlap: Optional[float] = None) -> "Arm":
        answer: Mapping[str, Any] = result.get("answer") or {}
        retrieval: Mapping[str, Any] = result.get("retrieval") or {}
        timing: Mapping[str, Any] = result.get("timing_seconds") or {}
        return cls(
            method=method,
            status=str(result.get("status") or "ok"),
            text=answer.get("text") or "",
            sufficient=answer.get("sufficient"),
            sources=tuple(Source.of(row) for row in (result.get("sources") or ())),
            engine=result.get("arm_kind"),
            label=result.get("arm_label"),
            error=result.get("error") or None,
            unit_overlap=unit_overlap,
            dense=bool(retrieval.get("dense")),
            note=retrieval.get("note") or None,
            seconds=timing.get("total"),
        )


@dataclass(frozen=True)
class Comparison:
    """One question, put to one document chunked several ways.

    The comparison a knowledge-base question cannot make: everywhere else a
    question searches a corpus chunked the one way its knowledge base ingests,
    and here only the chunker differs between arms. Everything downstream --
    the index, the retriever, the context budget, the answer model -- is
    shared, which is the whole claim the result makes.

    Indexable and iterable by method, because the first thing a caller does
    with one is look up an arm by name.
    """

    document_id: str
    question: str = ""
    arms: tuple[Arm, ...] = ()
    embedding_model: Optional[str] = None
    answer_model: Optional[str] = None
    seconds: Optional[float] = None

    @property
    def methods(self) -> tuple[str, ...]:
        return tuple(arm.method for arm in self.arms)

    def __getitem__(self, method: str) -> Arm:
        for arm in self.arms:
            if arm.method == method:
                return arm
        raise KeyError(method)

    def __iter__(self) -> Iterator[Arm]:
        return iter(self.arms)

    def __len__(self) -> int:
        return len(self.arms)

    @classmethod
    def of(cls, result: Mapping[str, Any], *, document_id: str) -> "Comparison":
        methods: Sequence[str] = list(result.get("methods") or ())
        answers: Mapping[str, Any] = result.get("arms") or {}
        overlap: Mapping[str, Any] = result.get("unit_overlap_with_other_arms") or {}
        first: Mapping[str, Any] = (answers.get(methods[0]) if methods else {}) or {}
        # A retrieval-only comparison asked no answer model and reports none;
        # its embedding model is still on the retrieval block, so one field
        # means one thing in both modes.
        models: Mapping[str, Any] = first.get("models") or {
            "embedding": (first.get("retrieval") or {}).get("embedding_model")
        }
        return cls(
            document_id=document_id,
            question=str(result.get("question") or ""),
            arms=tuple(
                Arm.of(method, answers[method],
                       unit_overlap=overlap.get(method) if len(methods) > 1 else None)
                for method in methods if method in answers
            ),
            embedding_model=models.get("embedding"),
            answer_model=models.get("answer"),
            seconds=(result.get("timing") or {}).get("total_seconds"),
        )
