"""The upload route's Deep Analysis contract, end to end over HTTP.

A misconfigured Deep Analysis request is refused before any document work
(503, configuration_missing), an unsupported chunker is refused as a client
error, Standard uploads ignore the judge configuration entirely, and a
served Deep Analysis upload records the judge's report — counts and model
id, never key material — in the document record.
"""

from __future__ import annotations

import io
import json
import os
from types import SimpleNamespace

import pytest

import app as flask_app
from components.knowledgebase.manager import KnowledgeBaseManager


def judge_settings(configured: bool) -> SimpleNamespace:
    if configured:
        os.environ.setdefault("FAKE_JUDGE_KEY", "placeholder")
        return SimpleNamespace(
            boundary_judge_model="test/model",
            boundary_judge_endpoint="http://localhost:9/v1/chat/completions",
            boundary_judge_api_key_env="FAKE_JUDGE_KEY",
            boundary_judge_timeout=5.0,
        )
    return SimpleNamespace(
        boundary_judge_model="",
        boundary_judge_endpoint="",
        boundary_judge_api_key_env="MISSING_JUDGE_KEY",
        boundary_judge_timeout=5.0,
    )


class StandardOnlyChunker:
    def get_name(self):
        return "SemanticChunker"


class DeepCapableChunker(StandardOnlyChunker):
    def get_name(self):
        return "StructuralChunker"

    def chunk_text_deep(self, *args, **kwargs):  # pragma: no cover - stub
        raise AssertionError("the route must not chunk directly")


class StubPipeline:
    """Only what the upload route touches."""

    def __init__(self, *, configured_judge: bool, deep_capable: bool):
        self.settings = judge_settings(configured_judge)
        self.chunker = DeepCapableChunker() if deep_capable else StandardOnlyChunker()
        self.last_deep_analysis_report = None
        self.seen_deep_analysis = None

    def ingest_document_from_file(self, file_path, doc_title, deep_analysis=False):
        self.seen_deep_analysis = deep_analysis
        if deep_analysis:
            self.last_deep_analysis_report = {
                "mode": "deep_analysis",
                "boundary_judge_model": "test/model",
                "judge_call_count": 3,
                "split_votes": 1,
                "keep_votes": 2,
                "fallback_count": 0,
            }
        return [SimpleNamespace(doc_id="doc-under-test")]


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    manager = KnowledgeBaseManager(str(tmp_path / "kbs.json"))
    monkeypatch.setattr(flask_app, "kb_manager", manager)
    flask_app.app.config.update(TESTING=True)
    kb = manager.create("deep-kb", chunker={"type": "structure_first"})
    with flask_app.app.test_client() as test_client:
        yield test_client, kb["kb_id"]


def upload(test_client, kb_id, deep):
    return test_client.post(
        "/api/documents/upload",
        data={
            "file": (io.BytesIO(b"kucuk bir test belgesi"), "belge.txt"),
            "kb_id": kb_id,
            "deep_analysis": "true" if deep else "false",
        },
        content_type="multipart/form-data",
    )


def use_pipeline(monkeypatch, pipeline):
    monkeypatch.setattr(flask_app, "get_pipeline", lambda *a, **k: pipeline)
    return pipeline


def test_missing_judge_config_refuses_before_any_work(client, monkeypatch):
    test_client, kb_id = client
    pipeline = use_pipeline(
        monkeypatch, StubPipeline(configured_judge=False, deep_capable=True)
    )
    response = upload(test_client, kb_id, deep=True)
    body = response.get_json()
    assert response.status_code == 503
    assert body["deep_analysis_unavailable"] is True
    assert body["configuration_missing"] is True
    assert "BOUNDARY_JUDGE_MODEL" in body["error"]
    assert pipeline.seen_deep_analysis is None, "no ingest may have started"


def test_a_chunker_without_deep_support_is_a_client_error(client, monkeypatch):
    test_client, kb_id = client
    pipeline = use_pipeline(
        monkeypatch, StubPipeline(configured_judge=True, deep_capable=False)
    )
    response = upload(test_client, kb_id, deep=True)
    body = response.get_json()
    assert response.status_code == 400
    assert body["deep_analysis_unavailable"] is True
    assert "SemanticChunker" in body["error"]
    assert pipeline.seen_deep_analysis is None


def test_standard_uploads_ignore_the_judge_configuration(client, monkeypatch):
    test_client, kb_id = client
    pipeline = use_pipeline(
        monkeypatch, StubPipeline(configured_judge=False, deep_capable=True)
    )
    response = upload(test_client, kb_id, deep=False)
    body = response.get_json()
    assert response.status_code == 200
    assert body["chunking_mode"] == "standard"
    assert body["boundary_judge"] is None
    assert pipeline.seen_deep_analysis is False


def test_a_served_deep_upload_records_the_judge_report(client, monkeypatch):
    test_client, kb_id = client
    pipeline = use_pipeline(
        monkeypatch, StubPipeline(configured_judge=True, deep_capable=True)
    )
    response = upload(test_client, kb_id, deep=True)
    body = response.get_json()
    assert response.status_code == 200
    assert pipeline.seen_deep_analysis is True
    assert body["chunking_mode"] == "deep_analysis"
    assert body["boundary_judge"]["judge_call_count"] == 3

    # The tracker record carries the mode first-class and the report in
    # metadata; nothing key-shaped is persisted anywhere.
    records = json.load(open(".ingested_documents.json", encoding="utf-8"))
    (record,) = records.values()
    assert record["chunking_mode"] == "deep_analysis"
    assert record["status"] == "indexed"
    assert record["metadata"]["boundary_judge"]["split_votes"] == 1
    serialized = json.dumps(records).lower()
    assert "placeholder" not in serialized
    assert "api_key" not in serialized
