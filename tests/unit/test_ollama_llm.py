from __future__ import annotations

import httpx
import pytest

from chat_rag.components.llm import ollama_llm
from chat_rag.core.exceptions import LLMException
from chat_rag.core.models import DocumentChunk, RetrievalResult
from chat_rag.pipeline.rag_pipeline import RAGPipeline


class FakeOllamaClient:
    instances: list["FakeOllamaClient"] = []

    def __init__(self, host: str, timeout: int):
        self.host = host
        self.timeout = timeout
        self.messages: list[list[dict[str, str]]] = []
        self.__class__.instances.append(self)

    def list(self):
        return {"models": []}

    def chat(self, *, messages, **kwargs):
        self.messages.append(messages)
        return {"message": {"content": "Ollama response"}}


def _ollama(monkeypatch) -> ollama_llm.OllamaLLM:
    FakeOllamaClient.instances.clear()
    monkeypatch.setattr(ollama_llm.ollama, "Client", FakeOllamaClient)
    return ollama_llm.OllamaLLM(
        model="test-model",
        base_url="http://ollama.test:11434",
        timeout=7,
    )


def test_an_answer_over_retrieved_context_uses_the_ollama_client(monkeypatch):
    llm = _ollama(monkeypatch)
    client = FakeOllamaClient.instances[0]
    assert (client.host, client.timeout) == ("http://ollama.test:11434", 7)

    pipeline = RAGPipeline.__new__(RAGPipeline)
    pipeline.settings = None
    pipeline.llm_model = llm
    pipeline.vector_db = object()
    pipeline.embedding_model = object()
    result = RetrievalResult(
        chunk=DocumentChunk(
            chunk_id="chunk-1",
            content="Retrieved context",
            doc_id="doc-1",
            doc_title="Test document",
            chunk_index=0,
            total_chunks=1,
        ),
        score=0.9,
        retrieval_method="bm25_only",
        rank=1,
    )
    answer = pipeline.generate_answer("What is in the document?", [result])

    assert answer == "Ollama response"
    assert len(client.messages) == 1
    assert "Retrieved context" in client.messages[0][1]["content"]


def test_ollama_timeout_is_wrapped_without_requests_attribute_error(monkeypatch):
    llm = _ollama(monkeypatch)

    def raise_timeout(**kwargs):
        request = httpx.Request("POST", "http://ollama.test:11434/api/chat")
        raise httpx.ReadTimeout("timed out", request=request)

    monkeypatch.setattr(llm.client, "chat", raise_timeout)

    with pytest.raises(LLMException, match="Request timeout after 7s"):
        llm.generate([{"role": "user", "content": "Hello"}])


def test_unavailable_ollama_service_raises_connection_error(monkeypatch):
    class UnavailableOllamaClient(FakeOllamaClient):
        def list(self):
            raise ConnectionError("Ollama is unavailable")

    monkeypatch.setattr(ollama_llm.ollama, "Client", UnavailableOllamaClient)

    with pytest.raises(LLMException, match="Failed to connect to Ollama") as exc_info:
        ollama_llm.OllamaLLM(base_url="http://ollama.test:11434")

    assert "requests" not in str(exc_info.value)
