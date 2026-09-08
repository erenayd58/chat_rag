"""The two things the Viewer needs that no other screen does.

**The prepared payload.** One document's whole analysis as a reader's view of
it: the parser's canonical units in reading order, and per chunking method the
chunks plus the unit-offset mapping that says where each one starts and ends
*inside* a unit. That mapping is the Viewer -- it is what lets three methods be
printed into one grid, aligned on the text rather than on a chunk number -- and
nothing else on this contract carries it. ``GET .../analysis/methods/{m}/chunks``
answers the rows of one method; it cannot answer where they cut.

**A question over the arms.** Everywhere else a query searches a knowledge
base's corpus, chunked the one way its knowledge base was ingested. Here the
same question runs through several chunking methods of one document, with only
the chunker differing, so the answer is a comparison of chunkers.

``payload`` and each arm's ``sources`` are **pass-through**, for the same
reason a chunk's ``metadata`` is and said the same way in ``docs/api-v1.md``:
they carry what the packager and the retrieval engine produced. Pinning either
here would freeze the internals this contract exists to leave free -- and a
render model that has to grow a field per chunker feature is exactly the kind
of thing a version number should not be spent on.
"""

from __future__ import annotations

from typing import Any, Optional, Union

from pydantic import Field

from .common import Schema


class AnalysisPayload(Schema):
    """One document's prepared analysis, as the Viewer reads it."""

    document_id: str
    content_id: Optional[str] = None
    label: str = ""
    ready_methods: list[str] = Field(
        default_factory=list,
        description="the methods this upload selected that are built, in "
                    "product order; the payload's arms are exactly these",
    )
    payload: dict[str, Any] = Field(
        default_factory=dict,
        description="the render model: units, arms, mapping, measurements. "
                    "Pass-through and not contractual",
    )

    @classmethod
    def of(cls, built: dict, *, document_id: str, state: dict) -> "AnalysisPayload":
        arms = built.get("arms") or {}
        return cls(
            document_id=document_id,
            content_id=state.get("key"),
            label=str(built.get("label") or state.get("label") or document_id),
            ready_methods=[m for m in (state.get("available_methods") or []) if m in arms],
            payload=built,
        )


class AnalysisQueryRequest(Schema):
    """One question, over one document, through one or more chunking methods.

    ``methods`` absent means every method this upload has ready -- which is
    the comparison someone opening the screen is asking for. Naming a method
    the deployment does not know is **400** with the supported list; naming
    one that is known and not built for this upload is **404**, because the
    two call for different things from a client.
    """

    document_id: str = ""
    question: str = ""
    methods: Optional[Union[list[str], str]] = None
    top_k: int = 5
    answer: bool = Field(
        default=True,
        description="false stops after retrieval: the ranked sources, no "
                    "answer-model call",
    )


class AnalysisAnswer(Schema):
    """What the answer model said about one arm's sources, if it was asked."""

    text: str = ""
    sufficient: Optional[bool] = Field(
        default=None,
        description="the model's own judgement of whether the sources "
                    "contained the answer; null when the reply was not JSON",
    )
    sources_used: list[str] = Field(default_factory=list)


class AnalysisArmResult(Schema):
    """One chunking method's answer to the question.

    ``status`` is what a client branches on: ``ok``, ``insufficient`` (nothing
    retrieved, or the model said the sources were not enough),
    ``no_answer_model`` (this deployment answers nothing; the sources are
    still here) or ``answer_error`` (the model was asked and did not answer;
    the sources are still here). An arm never fails the whole request.
    """

    method: str
    engine: Optional[str] = None
    label: Optional[str] = None
    status: str = "ok"
    error: Optional[str] = None
    answer: Optional[AnalysisAnswer] = None
    sources: list[dict[str, Any]] = Field(
        default_factory=list,
        description="the retrieved passages, in context order, each with its "
                    "citation label, pages, token count and whether the "
                    "answer used it. Pass-through and not contractual",
    )
    unit_overlap: Optional[float] = Field(
        default=None,
        description="how much of this arm's retrieved context, at the "
                    "canonical-unit level, the other arms also retrieved; "
                    "null when only one method ran",
    )
    dense: bool = Field(
        default=False,
        description="whether this arm's index has a dense leg; false is BM25 "
                    "alone, which is an answer and not a failure",
    )
    note: Optional[str] = Field(
        default=None,
        description="why retrieval was degraded for this arm, when it was",
    )
    seconds: Optional[float] = None

    @classmethod
    def of(cls, method: str, result: dict, *, overlap: Optional[float]) -> "AnalysisArmResult":
        answer = result.get("answer") or None
        retrieval = result.get("retrieval") or {}
        timing = result.get("timing_seconds") or {}
        return cls(
            method=method,
            engine=result.get("arm_kind"),
            label=result.get("arm_label"),
            status=str(result.get("status") or "ok"),
            error=result.get("error") or None,
            answer=AnalysisAnswer(
                text=answer.get("text") or "",
                sufficient=answer.get("sufficient"),
                sources_used=list(answer.get("sources_used") or []),
            ) if answer else None,
            sources=list(result.get("sources") or []),
            unit_overlap=overlap,
            dense=bool(retrieval.get("dense")),
            note=retrieval.get("note") or None,
            seconds=timing.get("total"),
        )


class AnalysisQueryResult(Schema):
    """The question, and what each method made of it."""

    document_id: str
    question: str
    methods: list[str]
    arms: list[AnalysisArmResult]
    embedding_model: Optional[str] = None
    answer_model: Optional[str] = None
    total_seconds: Optional[float] = None

    @classmethod
    def of(cls, found: dict, *, document_id: str) -> "AnalysisQueryResult":
        methods = list(found.get("methods") or [])
        results = found.get("arms") or {}
        overlap = found.get("unit_overlap_with_other_arms") or {}
        first = results.get(methods[0]) if methods else None
        # A retrieval-only run reports no answer model because it asked none;
        # its embedding model is still on the retrieval block, and reading it
        # from there keeps one field meaning one thing in both modes.
        models = (first or {}).get("models") or {
            "embedding": ((first or {}).get("retrieval") or {}).get("embedding_model")
        }
        return cls(
            document_id=document_id,
            question=str(found.get("question") or ""),
            methods=methods,
            arms=[
                AnalysisArmResult.of(method, results[method],
                                     overlap=overlap.get(method) if len(methods) > 1 else None)
                for method in methods if method in results
            ],
            embedding_model=models.get("embedding"),
            answer_model=models.get("answer"),
            total_seconds=(found.get("timing") or {}).get("total_seconds"),
        )
