"""A model failure degrades; a limit stops. The difference, pinned.

The answer path is written to degrade: a gateway that is down turns into an
answer that says so, not into a broken request. That must never extend to the
resource controls. A deadline that has passed, a queue or a budget that
refused, an operator's cancellation -- none of them mean "this step failed,
carry on". They mean the work must stop, and a bare ``except Exception`` that
swallows one turns a limit into a silent degradation: the query keeps running
past its deadline, making further calls that will be refused in turn.

``core.exceptions.RESOURCE_CONTROL_EXCEPTIONS`` is the one list every handler
re-raises before it degrades, so a limit added later cannot be quietly
swallowed by a handler nobody revisited.
"""

from __future__ import annotations

import pytest

from components.llm.base import BaseLLM
from core.exceptions import (
    IngestInterrupted, IngestOverloaded, LLMException, QueryOverloaded, QueryTimeout,
    RESOURCE_CONTROL_EXCEPTIONS,
)
from core.models import DocumentChunk, RetrievalResult
from pipeline.rag_pipeline import RAGPipeline

QUESTION = "Ornitorenk tarifesi kac kurus"

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


def results(count: int = 3):
    return [
        RetrievalResult(
            chunk=DocumentChunk(chunk_id=f"c{index}", content=f"parca {index}", doc_id="d1",
                                doc_title="Rapor", chunk_index=index, total_chunks=count),
            score=1.0 - index / 10, retrieval_method="bm25_only", rank=index,
        )
        for index in range(count)
    ]


def answering(error: BaseException) -> RAGPipeline:
    """A pipeline whose answer model raises, and nothing else built."""
    pipeline = RAGPipeline.__new__(RAGPipeline)
    pipeline.settings = None
    pipeline.llm_model = Raising(error)
    pipeline.vector_db = None
    pipeline.embedding_model = None
    return pipeline


# ------------------------------------------------- the fallback that stays


def test_an_ordinary_model_failure_becomes_an_answer_that_says_so():
    """A gateway that is down must not take the request down with it."""
    pipeline = answering(LLMException("openrouter returned HTTP 502"))

    answer = pipeline.generate_answer(QUESTION, results())

    assert "error" in answer.lower()
    assert "HTTP 502" in answer
    assert pipeline.llm_model.calls == 1


def test_no_sources_is_answered_without_calling_the_model_at_all():
    pipeline = answering(LLMException("never reached"))

    answer = pipeline.generate_answer(QUESTION, [])

    assert "don't have enough information" in answer
    assert pipeline.llm_model.calls == 0


# ------------------------------------------------ the limits that do not


@pytest.mark.parametrize("limit", LIMITS)
def test_a_limit_is_never_turned_into_an_answer(limit):
    """The whole point: these say *stop*, so they leave by the front door."""
    pipeline = answering(limit)

    with pytest.raises(type(limit)) as stopped:
        pipeline.generate_answer(QUESTION, results())

    assert stopped.value is limit, "raised as it was, not re-wrapped"


def test_the_named_set_is_what_the_handlers_use():
    """One list, so a new limit cannot be added and quietly swallowed."""
    assert set(RESOURCE_CONTROL_EXCEPTIONS) == {
        IngestInterrupted, IngestOverloaded, QueryTimeout, QueryOverloaded,
    }
    assert all(issubclass(kind, Exception) for kind in RESOURCE_CONTROL_EXCEPTIONS)


def test_every_handler_that_degrades_re_raises_the_named_set_first():
    """Read off the source, so a new ``except Exception`` is noticed.

    A handler that degrades must name ``RESOURCE_CONTROL_EXCEPTIONS`` above
    its own catch-all. This is the check that survives however many of them
    there are.
    """
    import inspect

    from pipeline import rag_pipeline

    source = inspect.getsource(rag_pipeline)
    broad = source.count("except Exception")
    guarded = source.count("except RESOURCE_CONTROL_EXCEPTIONS")
    assert guarded >= 2, "retrieval and the answer each guard their own"
    assert broad >= guarded, source.count("except")
