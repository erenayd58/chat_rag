"""What a fallback is for, and what it must never swallow.

Both enhancement paths in this system degrade rather than fail on purpose: a
clarification the model could not produce falls back to the original
question, a strategy it could not choose falls back to hybrid, a document
summary it could not write falls back to the title. That is product
behaviour and these tests keep it.

It must not extend to the limits. A query past its deadline, a budget or a
queue that refused, an operator's cancellation -- none of them mean "this
step failed, carry on without it". Swallowed into a heuristic fallback they
become invisible: the query runs on past its deadline, makes further calls
that are refused in turn, and answers from heuristics as though the limit
had never fired. ``RESOURCE_CONTROL_EXCEPTIONS`` names them once and every
fallback handler re-raises them first; what follows proves the distinction
at each handler and then end to end through the legacy retrieval path.

The legacy path's own trace is checked here too, because it is the other
half of the same file: it used to print the question, the clarified rewrite,
every generated variation and the previous turns of the conversation to
stdout -- a deployment's log stream -- on every query.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

from components.contextual_enhancer import ContextualRAGEnhancer
from components.conversation import ConversationManager
from components.ingest import limits as L
from components.llm.base import BaseLLM
from components.query import limits as Q
from components.query_processor import QueryEnhancer
from components.reranker import LLMReranker
from core.exceptions import (
    IngestInterrupted, IngestOverloaded, LLMException, QueryOverloaded, QueryTimeout,
    RESOURCE_CONTROL_EXCEPTIONS,
)
from core.models import DocumentChunk, RetrievalResult
from pipeline.rag_pipeline import RAGPipeline

QUESTION = "Ornitorenk tarifesi kac kurus"
CLARIFIED = "Ornitorenk tarifesi kac kurus (2024)"

#: One of each kind of limit, exactly as the wrappers raise them.
LIMITS = [
    pytest.param(QueryTimeout(), id="query-deadline"),
    pytest.param(QueryOverloaded("no answer slot", reason="answer_capacity"), id="answer-capacity"),
    pytest.param(QueryOverloaded("no query slot"), id="admission"),
    pytest.param(IngestInterrupted("timed_out", "the job's deadline passed"), id="ingest-deadline"),
    pytest.param(IngestInterrupted("cancelled", "an operator cancelled it"), id="cancelled"),
    pytest.param(IngestOverloaded("the queue is full"), id="ingest-overloaded"),
]


class Raising(BaseLLM):
    """A model that raises whatever it was given, and counts the attempts."""

    provider_id = "test"

    def __init__(self, error: BaseException):
        self.error = error
        self.calls = 0

    def generate(self, messages, temperature=0.3, max_tokens=200, **kwargs):
        self.calls += 1
        raise self.error

    def get_name(self):
        return "Raising"

    def get_model_name(self):
        return "test/raising"


class Answering(Raising):
    def __init__(self, reply: str):
        super().__init__(RuntimeError("unused"))
        self.reply = reply

    def generate(self, messages, temperature=0.3, max_tokens=200, **kwargs):
        self.calls += 1
        return self.reply


def enhancer(error: BaseException) -> QueryEnhancer:
    return QueryEnhancer(Raising(error))


#: Every enhancer entry point that has a fallback, with the call and the
#: property of the fallback answer that proves it was taken.
CALLS = {
    "clarify_query_with_context": (
        lambda e: e.clarify_query_with_context(QUESTION, "Kullanici: onceki soru"),
        lambda answer: answer.clarified_query == QUESTION and answer.confidence == "low",
    ),
    "determine_search_strategy": (
        lambda e: e.determine_search_strategy(QUESTION, CLARIFIED),
        lambda answer: answer.recommended_strategy == "hybrid",
    ),
    "generate_search_queries": (
        lambda e: e.generate_search_queries(
            QUESTION, CLARIFIED,
            SimpleNamespace(recommended_strategy="hybrid", query_type="factual"),
        ),
        lambda answer: [q.text for q in answer] == [CLARIFIED, QUESTION],
    ),
    "expand_query": (
        lambda e: e.expand_query(QUESTION),
        lambda answer: answer == [QUESTION],
    ),
    "understand_intent": (
        lambda e: e.understand_intent(QUESTION),
        lambda answer: answer["main_topic"] == QUESTION,
    ),
}


# ------------------------------------------------- the fallback that stays
@pytest.mark.parametrize("name", sorted(CALLS))
def test_an_ordinary_model_failure_still_falls_back(name):
    """A gateway that is down, a model that answered nonsense: the query
    goes on without the enhancement, which is what it is designed to do."""
    call, degraded = CALLS[name]
    enhancement = enhancer(LLMException("openrouter returned HTTP 502"))
    answer = call(enhancement)
    assert degraded(answer), answer
    assert enhancement.llm_model.calls == 1


@pytest.mark.parametrize("name", sorted(CALLS))
def test_a_malformed_answer_still_falls_back(name):
    """Not an exception at all -- the model replied, with something that is
    not the JSON asked for."""
    call, degraded = CALLS[name]
    enhancement = QueryEnhancer(Answering("I am afraid I cannot do that."))
    assert degraded(call(enhancement)), name


@pytest.mark.parametrize("name", sorted(CALLS))
def test_an_empty_answer_still_falls_back(name):
    call, degraded = CALLS[name]
    assert degraded(call(QueryEnhancer(Answering("   ")))), name


# ------------------------------------------------ the limits that do not
@pytest.mark.parametrize("limit", LIMITS)
@pytest.mark.parametrize("name", sorted(CALLS))
def test_a_limit_is_never_turned_into_a_fallback(name, limit):
    """The whole point: these say *stop*, so they leave by the front door."""
    call, _ = CALLS[name]
    enhancement = enhancer(limit)
    with pytest.raises(type(limit)) as stopped:
        call(enhancement)
    assert stopped.value is limit, "raised as it was, not re-wrapped"


def test_the_named_set_is_what_the_handlers_use():
    """One list, so a new limit cannot be added and quietly swallowed."""
    assert set(RESOURCE_CONTROL_EXCEPTIONS) == {
        IngestInterrupted, IngestOverloaded, QueryTimeout, QueryOverloaded,
    }
    assert all(issubclass(kind, Exception) for kind in RESOURCE_CONTROL_EXCEPTIONS)


# -------------------------------------------------------- the other paths
def test_a_document_summary_falls_back_but_an_interrupted_ingest_does_not():
    """The ingest-side twin. A summary is optional; a job that must stop is
    not, and its guard now reaches this code through the transport."""
    degrading = ContextualRAGEnhancer(Raising(LLMException("gateway down")))
    assert degrading.generate_document_summary("govde", "Rapor") == "Document: Rapor"

    stopping = ContextualRAGEnhancer(Raising(IngestInterrupted("timed_out", "out of time")))
    with pytest.raises(IngestInterrupted):
        stopping.generate_document_summary("govde", "Rapor")


def results(count: int = 3):
    return [
        RetrievalResult(
            chunk=DocumentChunk(chunk_id=f"c{index}", content=f"parca {index}", doc_id="d1",
                                doc_title="Rapor", chunk_index=index, total_chunks=count),
            score=1.0 - index / 10, retrieval_method="hybrid", rank=index,
        )
        for index in range(count)
    ]


def test_the_llm_reranker_falls_back_but_not_past_a_limit():
    candidates = results(4)
    degrading = LLMReranker(Raising(LLMException("gateway down")))
    assert len(degrading.rerank(CLARIFIED, candidates, top_k=2)) == 2

    stopping = LLMReranker(Raising(QueryTimeout()))
    with pytest.raises(QueryTimeout):
        stopping.rerank(CLARIFIED, candidates, top_k=2)


# ------------------------------------------------------------ end to end
def legacy_pipeline(model: BaseLLM) -> RAGPipeline:
    """The legacy retrieval profile with doubles under it. Built by hand
    because ``RAGPipeline.__init__`` would load an embedding model and open
    a store, and neither is what this file is about."""
    pipeline = RAGPipeline.__new__(RAGPipeline)
    pipeline.settings = SimpleNamespace(
        vector_weight=0.7, bm25_weight=0.3,
        include_vector_results_n=5, include_bm25_results_n=5,
        default_top_k=5, query_limits=None,
    )
    pipeline.retrieval_profile = "legacy"
    pipeline.llm_model = model
    pipeline.query_enhancer = QueryEnhancer(pipeline.answer_model)
    pipeline.enable_conversation = False
    pipeline.conversation = ConversationManager(max_history=5)
    pipeline.hybrid_retriever = SimpleNamespace(
        hybrid_search=lambda *a, **k: results(6),
        vector_search=lambda *a, **k: results(6),
        keyword_search=lambda *a, **k: results(6),
    )
    pipeline.reranker = SimpleNamespace(
        rerank=lambda query, candidates, top_k: candidates[:top_k],
        get_name=lambda: "StubReranker",
    )
    pipeline.vector_db = SimpleNamespace(get_name=lambda: "StubStore")
    pipeline.embedding_model = SimpleNamespace(get_name=lambda: "StubEmbedding")
    return pipeline


def test_the_legacy_path_degrades_when_the_model_is_down():
    """No model, no clarification, no strategy -- and still an answer set,
    because retrieval itself needs none of them."""
    pipeline = legacy_pipeline(Raising(LLMException("gateway down")))
    found, metadata = pipeline.retrieve(QUESTION, top_k=3)
    assert len(found) == 3
    assert metadata["refined_query"] == QUESTION
    assert metadata["strategy"]["recommended_strategy"] == "hybrid"


@pytest.mark.parametrize("limit", LIMITS[:3])
def test_the_legacy_path_stops_when_a_limit_fires(limit):
    """Where the swallowed exception used to become a silent degradation:
    the query would have gone on to make more calls and answered from
    heuristics. Now it stops at the first refusal."""
    model = Raising(limit)
    pipeline = legacy_pipeline(model)
    with pytest.raises(type(limit)):
        pipeline.retrieve(QUESTION, top_k=3)
    assert model.calls == 1, "and makes no further call after being refused"


def test_a_query_that_runs_out_of_time_mid_retrieval_stops_there():
    """The deadline reaching the enhancer the way it does in production:
    through the budgeted answer model, on the query's own guard."""
    model = Answering('{"recommended_strategy": "bm25"}')
    pipeline = legacy_pipeline(model)
    pipeline.llm_model = Q.LimitedAnswerModel(model, L.ProviderBudget(1))
    pipeline.query_enhancer = QueryEnhancer(pipeline.llm_model)

    clock = lambda: 1000.0  # noqa: E731 - a stopped clock, already past the deadline
    with L.use_guard(Q.QueryGuard(deadline=999.0, clock=clock)):
        with pytest.raises(QueryTimeout):
            pipeline.retrieve(QUESTION, top_k=3)
    assert model.calls == 0, "out of time: the call was never made"


# ----------------------------------------------------------- log hygiene
def test_the_legacy_trace_does_not_reach_stdout_or_the_default_log(capsys, caplog):
    """It used to print the question, the clarified rewrite, every generated
    variation and the conversation so far. stdout is a deployment's log
    stream, so that was the questions and the corpus leaving the process by
    default."""
    pipeline = legacy_pipeline(Answering("not json"))
    pipeline.enable_conversation = True
    pipeline.conversation.add_turn(user_query="Onceki soru: ornitorenk nedir?",
                                   assistant_response="Gizli cevap metni")

    with caplog.at_level(logging.INFO, logger="RAG"):
        pipeline.retrieve(QUESTION, top_k=3)

    printed = capsys.readouterr()
    assert printed.out == "", "the query path prints nothing"
    logged = "\n".join(record.getMessage() for record in caplog.records
                       if record.levelno >= logging.INFO)
    for forbidden in (QUESTION, "ornitorenk", "Gizli cevap metni", "parca 0"):
        assert forbidden not in logged, forbidden


def test_the_trace_is_still_there_for_a_developer_who_asks(caplog):
    """Debugging capability is preserved, at the level Phase 3 put prompts
    and retrieved chunks at: LOG_FILE_LEVEL=DEBUG."""
    pipeline = legacy_pipeline(Answering("not json"))
    with caplog.at_level(logging.DEBUG, logger="RAG.RAGPipeline"):
        pipeline.retrieve(QUESTION, top_k=3)
    debugged = "\n".join(record.getMessage() for record in caplog.records)
    assert QUESTION in debugged, "the trace still exists, one level down"
    assert "PROCESSING QUERY" in debugged
