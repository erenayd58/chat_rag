"""The upload route's Deep Analysis contract, end to end over HTTP.

A chunker with no Deep Analysis path is refused as a client error before
any document work; Standard uploads carry no Deep Analysis fields; a served
Deep Analysis upload records the pipeline's status and report -- counts,
model ids and product wording, never key material -- on the document
record and the summary in the provenance snapshot; and a fallback status
still completes the upload, labelled as what it is rather than as Standard.
"""

from __future__ import annotations

import io
import json
from types import SimpleNamespace

import pytest

from fastapi.testclient import TestClient

import asgi as entrypoint
import interfaces.http as http
from chat_rag.components.viewer import methods as M

V1 = http.v1.PREFIX
from chat_rag.application import workspace as app_workspace
from chat_rag.components.knowledgebase.manager import KnowledgeBaseManager
from chat_rag.config import paths


def deep_report(status: str) -> dict:
    """A report in the shape ``StructuralChunker.chunk_text_deep`` returns."""
    llm = status in ("ok", "degraded")
    return {
        "mode": "deep_analysis",
        "pipeline_mode": "live" if llm else "deterministic",
        "status": status,
        "model_id": "test/proposer" if llm else None,
        "verifier_model_id": "test/proposer" if llm else None,
        "uses_llm": llm,
        "chunk_count": {"standard": 47, "deep": 45},
        "smell_total": {"standard": 9, "deep": 4},
        "totals": {
            "standard": {"orphan_label": 3, "lead_in_cut": 6, "below_min": 2, "above_soft_max": 0},
            "deep": {"orphan_label": 1, "lead_in_cut": 3, "below_min": 2, "above_soft_max": 0},
        },
        "structural_regression_count": 0,
        "strict_regression_count": 0,
        "size_trade_count": 1,
        "change_group_count": 5,
        "selection": {"sections_moved": 4, "sections_reverted": 1, "revert_reasons": {}},
        "llm_effect": {
            "sections_changed_by_llm": 2 if llm else 0,
            "verifier_accepted_groups": 2 if llm else 0,
            "verifier_reverted_groups": 1 if llm else 0,
        },
        "proposer": {
            "call_count": 6, "boundary_count": 30, "accepted_boundary_count": 24,
            "forbidden_boundary_count": 2,
            "transport_status": {"ok": 6} if status == "ok" else {"ok": 4, "provider_error": 2},
        } if llm else None,
        "verifier": {"group_count": 3, "accepted": 2, "reverted": 1} if llm else None,
        "checks": {"hard_max_tokens": 1126, "max_token_count": 1120, "hard_cap_ok": True, "coverage_ok": True},
        "configuration": {"use_llm": llm, "verify": True, "api_key_env": "FAKE_DEEP_KEY"},
        "timing_seconds": {"standard": 0.1, "llm_calls": 30.0, "verifier_calls": 12.0, "selection": 0.4},
        **({"fallback_reason": "Deep Analysis ran without a language model: DEEP_ANALYSIS_MODEL is not set."}
           if status == "fallback_no_provider" else {}),
    }


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

    def __init__(self, *, deep_capable: bool, deep_status: str = "ok"):
        self.settings = SimpleNamespace(deep_analysis_api_key_env="FAKE_DEEP_KEY")
        self.chunker = DeepCapableChunker() if deep_capable else StandardOnlyChunker()
        self.last_deep_analysis_report = None
        self.seen_deep_analysis = None
        self._deep_status = deep_status

    def ingest_document_from_file(self, file_path, doc_title, deep_analysis=False):
        self.seen_deep_analysis = deep_analysis
        if deep_analysis:
            self.last_deep_analysis_report = deep_report(self._deep_status)
        return [SimpleNamespace(doc_id="doc-under-test")]


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("FAKE_DEEP_KEY", "sk-placeholder-secret")
    manager = KnowledgeBaseManager(str(tmp_path / "kbs.json"))
    monkeypatch.setattr(entrypoint.services, "kb_manager", manager)
    # These tests are about what the upload route returns and records. Viewer
    # packaging is a separate contract with its own tests, and it runs on a
    # background thread that outlives the request -- so left real, it reaches
    # back into this test's stubbed pipeline for a vector store that a stub has
    # no reason to own, and logs a failure for every upload here. Staging is
    # stubbed rather than fed a fake store: the boundary being exercised ends
    # at the response.
    monkeypatch.setattr(app_workspace, "stage_analysis", lambda *a, **k: {"status": "queued"})
    kb = manager.create("deep-kb", chunker={"type": "structure_first"})
    with TestClient(http.create_app(entrypoint.services),
                    raise_server_exceptions=False) as test_client:
        yield test_client, kb["kb_id"]


def upload(test_client, kb_id, deep):
    """One upload, always asynchronous: this contract answers 202 with the job.

    Deep Analysis is a *method* here rather than a boolean, which is the same
    decision spelled the way the registry spells it.
    """
    return test_client.post(
        f"{V1}/documents",
        files={"file": ("belge.txt", io.BytesIO(b"kucuk bir test belgesi"), "text/plain")},
        data={"knowledge_base_id": kb_id,
              "methods": [M.DEEP] if deep else [M.STANDARD]},
    )


def settle(test_client, submitted):
    """Wait for the submitted job, then read it back the way a client polls."""
    manager = entrypoint.services.ingest_jobs
    job = manager.get(submitted["id"])
    assert job is not None, submitted
    assert manager.wait(job, 30), "the job did not settle"
    read = test_client.get(f"{V1}/ingest-jobs/{submitted['id']}")
    assert read.status_code == 200, read.text
    return read.json()


def use_pipeline(monkeypatch, pipeline):
    monkeypatch.setattr(entrypoint.services, "get_pipeline", lambda *a, **k: pipeline)
    return pipeline


def records():
    """The ledger, read through the store that owns it.

    Resolved through the same API the application resolves it with, rather
    than named as a file in the working directory -- which since Step 8 is not
    where it is at all. What these tests assert is what a document record
    carries, which is the same question either way.
    """
    from chat_rag.utils import DocumentTracker

    return DocumentTracker().ingested_docs


def test_a_chunker_without_deep_support_is_a_client_error(client, monkeypatch):
    test_client, kb_id = client
    pipeline = use_pipeline(monkeypatch, StubPipeline(deep_capable=False))
    response = upload(test_client, kb_id, deep=True)
    error = response.json()["error"]
    assert response.status_code == 400
    assert error["type"] == "invalid_request"
    assert error["details"]["deep_analysis_unavailable"] is True
    assert "SemanticChunker" in error["message"]
    assert pipeline.seen_deep_analysis is None, "no ingest may have started"


def test_standard_uploads_carry_no_deep_analysis_fields(client, monkeypatch):
    test_client, kb_id = client
    pipeline = use_pipeline(monkeypatch, StubPipeline(deep_capable=True))
    response = upload(test_client, kb_id, deep=False)
    assert response.status_code == 202, response.text
    job = settle(test_client, response.json())
    assert job["status"] == "succeeded", job
    assert job["result"]["chunking_mode"] == "standard"
    assert job["result"]["deep_analysis"] is None
    assert pipeline.seen_deep_analysis is False
    (record,) = records().values()
    assert record["chunking_mode"] == "standard"
    assert "deep_analysis" not in record["metadata"]


def test_a_served_deep_upload_records_status_report_and_summary(client, monkeypatch):
    test_client, kb_id = client
    pipeline = use_pipeline(monkeypatch, StubPipeline(deep_capable=True))
    response = upload(test_client, kb_id, deep=True)
    assert response.status_code == 202, response.text
    job = settle(test_client, response.json())
    assert job["status"] == "succeeded", job
    assert pipeline.seen_deep_analysis is True
    report = job["result"]["deep_analysis"]
    assert job["result"]["chunking_mode"] == "deep_analysis"
    assert report["status"] == "ok"
    assert report["label"] == "Quality checks passed"
    assert report["chunk_count"] == {"standard": 47, "deep": 45}
    assert report["smell_total"] == {"standard": 9, "deep": 4}
    assert report["proposer"]["call_count"] == 6
    assert report["verifier"]["accepted"] == 2

    (record,) = records().values()
    assert record["chunking_mode"] == "deep_analysis"
    assert record["status"] == "indexed"
    assert record["metadata"]["deep_analysis_status"] == "ok"
    assert record["metadata"]["deep_analysis"]["structural_regression_count"] == 0
    assert record["metadata"]["deep_analysis"]["model_id"] == "test/proposer"
    snapshot = record.get("pipeline_snapshot")
    if snapshot is not None:  # the stub pipeline may not be snapshot-able
        options = snapshot["ingest_options"]
        assert options["deep_analysis_status"] == "ok"
        assert options["deep_analysis_summary"]["label"] == "Quality checks passed"


@pytest.mark.parametrize("status, label", [
    ("fallback_no_provider", "Completed with deterministic fallback"),
    ("fallback_provider_error", "Completed with deterministic fallback"),
    ("degraded", "Completed with partial fallback"),
])
def test_a_fallback_still_completes_and_is_labelled_as_such(client, monkeypatch, status, label):
    test_client, kb_id = client
    use_pipeline(monkeypatch, StubPipeline(deep_capable=True, deep_status=status))
    response = upload(test_client, kb_id, deep=True)
    assert response.status_code == 202, response.text
    job = settle(test_client, response.json())
    assert job["status"] == "succeeded", job
    report = job["result"]["deep_analysis"]
    assert job["result"]["chunking_mode"] == "deep_analysis", "never passed off as Standard"
    assert report["status"] == status
    assert report["label"] == label
    assert report["tone"] == "warn"
    assert "Traceback" not in json.dumps(job)
    (record,) = records().values()
    assert record["metadata"]["deep_analysis_status"] == status


def test_nothing_key_shaped_is_persisted_or_returned(client, monkeypatch):
    test_client, kb_id = client
    use_pipeline(monkeypatch, StubPipeline(deep_capable=True))
    job = settle(test_client, upload(test_client, kb_id, deep=True).json())
    serialized = (json.dumps(records()) + json.dumps(job)).lower()
    assert "sk-placeholder-secret" not in serialized
    assert "authorization" not in serialized
    assert '"api_key"' not in serialized
    # Only the variable's *name* travels with the record.
    (record,) = records().values()
    assert record["metadata"]["deep_analysis"]["configuration"]["api_key_env"] == "FAKE_DEEP_KEY"
