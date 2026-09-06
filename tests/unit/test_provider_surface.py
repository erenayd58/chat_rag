"""Every answer-model adapter this console ships is one it can actually select.

The audit this pins, done once in Phase 8 and kept true by these tests:

``OpenAICompatibleLLM``  the product's primary. ``ANSWER_PROVIDER=openrouter``
    / ``openai_compatible``; the demo answers with it.
``OllamaLLM``            the local fallback the demo configures
    (``ANSWER_FALLBACK_PROVIDER=ollama``), and a primary in its own right --
    it is what ``.env.docker`` runs, because the image ships with no key.
``AzureOpenAILLM``       ``ANSWER_PROVIDER=azure``, and the historical
    ``LLM_PROVIDER`` default. Kept because it is a supported deployment shape
    with its own documented settings, not because it is what is running here.
``UnavailableLLM``       what an unreachable provider becomes, so building a
    pipeline never fails on a model that ingestion and lexical retrieval do
    not need. It carries the reason and raises it if something asks for text.
``FallbackLLM``          the pair, not a provider.

The rule these tests exist for: an adapter nothing can select is dead
optionality. If a provider is removed from ``_build_answer_model``, its module
goes with it; if a module stays, something must be able to reach it.
"""

from __future__ import annotations

import ast
import inspect
import textwrap
from pathlib import Path

import pytest

from components import llm as llm_package
from pipeline.rag_pipeline import RAGPipeline

REPO = Path(__file__).resolve().parents[2]

#: What ``BaseLLM`` subclasses are for. ``FallbackLLM`` composes two of the
#: others rather than talking to anything, so it is selected by
#: ``_create_llm`` rather than by a provider name.
COMPOSITE = {"FallbackLLM"}
#: Not selected by name either: it is the failure of one of the others.
FAILURE_CARRIER = {"UnavailableLLM"}


def _selection_source() -> str:
    return inspect.getsource(RAGPipeline._build_answer_model) + inspect.getsource(
        RAGPipeline._create_llm
    )


def _shipped_adapters() -> set[str]:
    return {
        name
        for name in llm_package.__all__
        if name != "BaseLLM" and isinstance(getattr(llm_package, name), type)
    }


def test_every_shipped_adapter_can_be_selected():
    """No adapter is carried that nothing can reach."""
    source = _selection_source()
    unreachable = sorted(name for name in _shipped_adapters() if name not in source)
    assert unreachable == [], (
        "these answer models ship but nothing selects them; wire each one into "
        f"RAGPipeline._build_answer_model or delete its module: {unreachable}"
    )


def test_every_selectable_provider_name_is_documented():
    """A provider name the code accepts and ``env.example`` never mentions is
    a capability only the source says exists."""
    documented = (REPO / "env.example").read_text(encoding="utf-8")
    tree = ast.parse(textwrap.dedent(inspect.getsource(RAGPipeline._build_answer_model)))
    names = {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    providers = {"openrouter", "openai_compatible", "ollama", "azure"}
    assert providers <= names, "the selection table changed shape: " + str(sorted(names))
    for provider in sorted(providers):
        assert provider in documented, f"{provider} is selectable but undocumented"


@pytest.mark.parametrize("name", sorted(_shipped_adapters() - COMPOSITE - FAILURE_CARRIER))
def test_every_transport_honours_the_query_deadline(name):
    """One deadline policy, one place.

    ``components.ingest.limits`` owns it: ``current_guard()`` says whether a
    deadline is running and ``deadline_timeout()`` clamps a socket to what is
    left. Every transport reads the guard before it calls out; the two that
    can pass a per-call timeout also clamp it, and ``OllamaLLM`` documents in
    its own source why it cannot. What must not happen is a transport that
    knows about neither -- that is a call the deadline cannot stop.
    """
    module = Path(inspect.getfile(getattr(llm_package, name)))
    source = module.read_text(encoding="utf-8")

    assert "current_guard" in source, f"{name} makes provider calls outside the deadline"
    assert "from components.ingest.limits import" in source, (
        f"{name} must read the deadline from the one module that owns it"
    )


def test_the_fallback_pair_is_not_itself_a_provider():
    """``FallbackLLM`` composes; it must not grow a transport of its own."""
    source = Path(inspect.getfile(llm_package.FallbackLLM)).read_text(encoding="utf-8")
    for transport in ("urllib", "httpx", "requests", "ollama", "openai"):
        assert f"import {transport}" not in source, transport
