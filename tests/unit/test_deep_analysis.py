"""Deep Analysis on ``amsc.deep.pipeline``: the product's premium ingest mode.

What these tests pin:

* Standard stays Standard -- ``chunk_text`` is byte-identical to the frozen
  ``amsc.chunking.structural.chunk_units`` walk and knows nothing of Deep;
* Deep Analysis goes through ``amsc.deep.pipeline.chunk_document`` and only
  there: rows in the structural schema, the hard cap, full coverage;
* every pipeline status reaches the product: ``ok`` with a well-formed
  provider, ``fallback_provider_error`` when every call fails,
  ``degraded`` when some do, ``fallback_no_provider`` when the backend is
  not configured, ``deterministic`` when the LLM was not requested -- and
  none of them raises or is passed off as Standard;
* configuration is backend-only and the key is never read here; no prompt
  text, no key and no document text beyond the rows reach the report;
* the query path has no idea Deep Analysis exists, and the retired
  boundary judge / guard modules are gone.
"""

from __future__ import annotations

import importlib
import inspect
import json
import re
from types import SimpleNamespace

import pytest

from amsc.deep import proposer as deep_proposer
from amsc.deep.pipeline import (
    STATUS_DEGRADED,
    STATUS_DETERMINISTIC,
    STATUS_FALLBACK_NO_PROVIDER,
    STATUS_FALLBACK_PROVIDER_ERROR,
    STATUS_OK,
)
from amsc.chunking.structural import chunk_units
from amsc.document.tokenization import TiktokenTokenCounter
from components.chunker import deep_analysis as product
from components.chunker.normalization_adapter import CanonicalUnitAdapter
from components.chunker.structural_chunker import (
    HARD_MAX_TOKENS,
    MIN_TOKENS,
    SOFT_MAX_TOKENS,
    TARGET_TOKENS,
    StructuralChunker,
)


# ---------------------------------------------------------------- fixtures
def unit(unit_id, order, text, kind="paragraph", section=("BOLUM",)):
    return {
        "unit_id": unit_id,
        "order": order,
        "text": text,
        "type": kind,
        "heading_level": 2 if kind == "heading" else None,
        "section_path": list(section),
        "source": {"page": 1},
    }


WORDS = "veri kalite gosterge donem sonuc analiz kapsam yontem bulgu deger "


def section(heading_id, heading, first_order, path):
    """One oversized section with a lead-in sentence and a list run, so the
    deterministic contract and the proposer both have real work."""
    rows = [unit(heading_id, first_order, f"**{heading}**", "heading", path)]
    order = first_order + 1
    for _ in range(4):
        rows.append(unit(f"p-{order:05d}", order, WORDS * 12, section=path))
        order += 1
    rows.append(unit(f"p-{order:05d}", order, "Asagidaki maddeler dikkate alinir:", section=path))
    order += 1
    for _ in range(4):
        rows.append(unit(f"l-{order:05d}", order, "- " + WORDS * 10, "list", path))
        order += 1
    for _ in range(4):
        rows.append(unit(f"p-{order:05d}", order, WORDS * 12, section=path))
        order += 1
    return rows, order


def quality_document():
    first, order = section("h-00001", "1. GENEL DEGERLENDIRME", 1, ("1. GENEL DEGERLENDIRME",))
    second, _ = section("h-00020", "2. FAALIYETLER", order, ("2. FAALIYETLER",))
    return first + second


def deep_config():
    return product.deep_config(
        min_tokens=MIN_TOKENS,
        target_tokens=TARGET_TOKENS,
        soft_max_tokens=SOFT_MAX_TOKENS,
        hard_max_tokens=HARD_MAX_TOKENS,
    )


def backend_settings(**overrides):
    base = dict(
        deep_analysis_model="test/proposer",
        deep_analysis_verifier_model="",
        deep_analysis_endpoint="",
        deep_analysis_api_key_env="DEEP_TEST_KEY",
        deep_analysis_use_llm=True,
        deep_analysis_verify=True,
        deep_analysis_timeout=5.0,
        deep_analysis_concurrency=2,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


@pytest.fixture
def configured(monkeypatch):
    monkeypatch.setenv("DEEP_TEST_KEY", "sk-test-secret-value")
    return product.build_configuration(backend_settings(), deep_config())


class AnsweringProvider:
    """Answers every marked boundary with a complete, neutral vote and
    calls every verifier comparison a tie."""

    model_id = "test/answering"

    def __init__(self):
        self.prompts = []

    def complete(self, prompt):
        self.prompts.append(prompt)
        if "DIVISION ONE" in prompt:
            return '{"better": "EQUAL", "confidence": "high"}'
        labels = sorted(set(re.findall(r"\[(B\d+)\]", prompt)))
        return json.dumps({"boundaries": [
            {"id": label, "strength": 2, "before": "finished", "after": "standalone"}
            for label in labels
        ]})


class FailingProvider:
    model_id = "test/failing"

    def complete(self, prompt):
        raise ConnectionError("endpoint unreachable")


class FlakyProvider(AnsweringProvider):
    """Fails the first proposer call, answers the rest."""

    model_id = "test/flaky"

    def complete(self, prompt):
        if len(self.prompts) == 0 and "DIVISION ONE" not in prompt:
            self.prompts.append(prompt)
            raise ConnectionError("first call dropped")
        return super().complete(prompt)


def standard_chunks(rows):
    return StructuralChunker().chunk_text("", "doc", "rapor.pdf", parsed_units=rows)


def deep_chunks(configuration, rows, provider=None, verifier=None):
    return StructuralChunker().chunk_text_deep(
        "", "doc", "rapor.pdf",
        configuration=configuration,
        provider=provider,
        verifier_provider=verifier,
        parsed_units=rows,
    )


def unit_ids_of(chunks):
    return [uid for chunk in chunks for uid in json.loads(chunk.metadata["unit_ids_json"])]


# -------------------------------------------------------------- standard
def test_standard_is_the_frozen_structural_walk_byte_for_byte():
    rows = quality_document()
    chunks = standard_chunks(rows)
    units = CanonicalUnitAdapter().normalize(text="", document_id="doc", parsed_units=rows)
    reference = chunk_units(
        units, counter=TiktokenTokenCounter("cl100k_base"),
        min_tokens=MIN_TOKENS, target_tokens=TARGET_TOKENS,
        soft_max_tokens=SOFT_MAX_TOKENS, hard_max_tokens=HARD_MAX_TOKENS,
    )
    assert [c.content for c in chunks] == [r["text"] for r in reference]
    assert all("deep_analysis_status" not in c.metadata for c in chunks)
    assert all("chunking_mode" not in c.metadata for c in chunks)


def test_the_standard_path_never_touches_the_deep_pipeline():
    source = inspect.getsource(StructuralChunker.chunk_text) + inspect.getsource(
        StructuralChunker.chunk_canonical
    )
    assert "deep_pipeline" not in source and "chunk_document" not in source


# ------------------------------------------------------------------ deep
def test_deep_analysis_runs_through_amsc_deep_pipeline(configured):
    rows = quality_document()
    chunks, report = deep_chunks(configured, rows, provider=AnsweringProvider())
    # Rows come from the Deep selector (its own chunk-id family), the report
    # is the pipeline's, and the product adds its checks on top.
    assert all(":d-chunk-" in c.chunk_id for c in chunks)
    assert report["prompt_template_version"] == deep_proposer.PROMPT_TEMPLATE_VERSION
    assert report["mode"] == "deep_analysis"
    assert report["pipeline_mode"] == "live"
    assert report["checks"]["hard_cap_ok"] and report["checks"]["coverage_ok"]


def test_a_well_formed_provider_completes_with_status_ok(configured):
    rows = quality_document()
    provider = AnsweringProvider()
    chunks, report = deep_chunks(configured, rows, provider=provider)
    assert report["status"] == STATUS_OK
    assert report["uses_llm"] is True
    assert report["proposer"]["call_count"] >= 2
    assert provider.prompts, "the fixture must reach the model"
    assert report["structural_regression_count"] == 0
    summary = product.product_summary(report)
    assert summary["label"] == "Quality checks passed"
    assert summary["tone"] == "success"
    assert summary["model_id"] == "test/answering"
    assert summary["proposer"]["failed_count"] == 0


def test_deep_chunks_carry_their_mode_status_and_model(configured):
    chunks, _ = deep_chunks(configured, quality_document(), provider=AnsweringProvider())
    for chunk in chunks:
        assert chunk.metadata["chunking_mode"] == "deep_analysis"
        assert chunk.metadata["deep_analysis_status"] == STATUS_OK
        assert chunk.metadata["proposer_model"] == "test/answering"
        assert chunk.metadata["chunker_type"] == "structure_first"


def test_the_hard_cap_and_coverage_hold_under_every_provider(configured):
    rows = quality_document()
    expected = unit_ids_of(standard_chunks(rows))
    for provider in (AnsweringProvider(), FailingProvider(), FlakyProvider()):
        chunks, report = deep_chunks(configured, rows, provider=provider)
        assert all(c.metadata["token_count"] <= HARD_MAX_TOKENS for c in chunks)
        assert unit_ids_of(chunks) == expected
        assert report["checks"]["max_token_count"] <= HARD_MAX_TOKENS


# ----------------------------------------------------------- failure policy
def test_a_failing_provider_falls_back_without_raising(configured):
    rows = quality_document()
    chunks, report = deep_chunks(configured, rows, provider=FailingProvider())
    assert report["status"] == STATUS_FALLBACK_PROVIDER_ERROR
    assert report["uses_llm"] is False
    # The deterministic contract, not Standard: the same partition the
    # pipeline produces with the LLM switched off.
    deterministic = product.build_configuration(
        backend_settings(deep_analysis_use_llm=False), deep_config()
    )
    expected, _ = deep_chunks(deterministic, rows)
    assert [c.content for c in chunks] == [c.content for c in expected]
    summary = product.product_summary(report)
    assert summary["label"] == "Completed with deterministic fallback"
    assert summary["tone"] == "warn"
    assert summary["proposer"]["failed_count"] == summary["proposer"]["call_count"]


def test_a_partially_failing_provider_is_degraded_not_failed(configured):
    rows = quality_document()
    _, report = deep_chunks(configured, rows, provider=FlakyProvider())
    assert report["status"] == STATUS_DEGRADED
    summary = product.product_summary(report)
    assert summary["label"] == "Completed with partial fallback"
    assert 0 < summary["proposer"]["failed_count"] < summary["proposer"]["call_count"]


def test_missing_configuration_completes_deterministically_and_says_so(monkeypatch):
    monkeypatch.delenv("DEEP_TEST_KEY", raising=False)
    rows = quality_document()
    for settings, named in (
        (backend_settings(deep_analysis_model=""), "DEEP_ANALYSIS_MODEL"),
        (backend_settings(), "DEEP_TEST_KEY"),
    ):
        configuration = product.build_configuration(settings, deep_config())
        assert not configuration.llm_available
        assert named in configuration.missing
        chunks, report = deep_chunks(configuration, rows)
        assert chunks, "the ingest completes"
        assert report["status"] == STATUS_FALLBACK_NO_PROVIDER
        assert named in report["fallback_reason"]
        assert report["uses_llm"] is False
        assert report["configuration"]["proposer_model"] is None
        summary = product.product_summary(report)
        assert summary["label"] == "Completed with deterministic fallback"
        assert summary["fallback_reason"] == report["fallback_reason"]


def test_an_unrequested_llm_is_a_deterministic_pass(monkeypatch):
    monkeypatch.setenv("DEEP_TEST_KEY", "present")
    configuration = product.build_configuration(
        backend_settings(deep_analysis_use_llm=False), deep_config()
    )
    assert configuration.missing == ()
    chunks, report = deep_chunks(configuration, quality_document())
    assert report["status"] == STATUS_DETERMINISTIC
    assert report["pipeline_mode"] == "deterministic"
    assert product.product_summary(report)["label"] == "Deterministic quality pass"
    assert all("proposer_model" not in c.metadata for c in chunks)


# ------------------------------------------------------------ configuration
def test_build_configuration_maps_backend_settings(monkeypatch):
    monkeypatch.setenv("DEEP_TEST_KEY", "present")
    configuration = product.build_configuration(
        backend_settings(
            deep_analysis_verifier_model="test/verifier",
            deep_analysis_endpoint="http://gateway.local/v1/chat/completions",
            deep_analysis_verify=False,
            deep_analysis_timeout=42.0,
            deep_analysis_concurrency=3,
        ),
        deep_config(),
    )
    settings = configuration.settings
    assert configuration.llm_available
    assert settings.proposer_model == "test/proposer"
    assert settings.effective_verifier_model == "test/verifier"
    assert settings.endpoint == "http://gateway.local/v1/chat/completions"
    assert settings.api_key_env == "DEEP_TEST_KEY"
    assert settings.verify is False
    assert settings.timeout_seconds == 42.0
    assert settings.concurrency == 3
    assert settings.config.hard_max_tokens == HARD_MAX_TOKENS


def test_the_default_endpoint_comes_from_amsc_not_from_here(monkeypatch):
    monkeypatch.setenv("DEEP_TEST_KEY", "present")
    from amsc.deep.pipeline import DEFAULT_ENDPOINT

    configuration = product.build_configuration(backend_settings(), deep_config())
    assert configuration.settings.endpoint == DEFAULT_ENDPOINT
    source = inspect.getsource(product)
    assert "openrouter.ai" not in source, "no vendor URL is hardcoded in the product"


def test_settings_read_deep_analysis_and_fall_back_to_the_legacy_names(monkeypatch):
    from config.settings import Settings

    for name in (
        "DEEP_ANALYSIS_MODEL", "DEEP_ANALYSIS_ENDPOINT", "DEEP_ANALYSIS_API_KEY_ENV",
        "DEEP_ANALYSIS_VERIFY", "DEEP_ANALYSIS_USE_LLM", "DEEP_ANALYSIS_TIMEOUT",
        "BOUNDARY_JUDGE_MODEL", "BOUNDARY_JUDGE_ENDPOINT", "BOUNDARY_JUDGE_API_KEY_ENV",
        "BOUNDARY_JUDGE_TIMEOUT",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("BOUNDARY_JUDGE_MODEL", "legacy/model")
    monkeypatch.setenv("BOUNDARY_JUDGE_TIMEOUT", "77")
    legacy = Settings()
    assert legacy.deep_analysis_model == "legacy/model"
    assert legacy.deep_analysis_timeout == 77.0
    assert legacy.deep_analysis_api_key_env == "OPENROUTER_API_KEY"
    assert legacy.deep_analysis_verify is True and legacy.deep_analysis_use_llm is True

    monkeypatch.setenv("DEEP_ANALYSIS_MODEL", "new/model")
    monkeypatch.setenv("DEEP_ANALYSIS_API_KEY_ENV", "GATEWAY_KEY")
    monkeypatch.setenv("DEEP_ANALYSIS_VERIFY", "false")
    current = Settings()
    assert current.deep_analysis_model == "new/model"
    assert current.deep_analysis_api_key_env == "GATEWAY_KEY"
    assert current.deep_analysis_verify is False
    assert not hasattr(current, "boundary_judge_model")


# ---------------------------------------------------------------- wording
def test_every_pipeline_status_has_product_wording():
    for status in product.STATUSES:
        text = product.describe_status(status)
        assert text["label"] and text["headline"] and text["detail"]
        assert text["tone"] in {"success", "neutral", "warn"}
        assert "Traceback" not in text["detail"]
    assert product.describe_status("something_new")["tone"] == "warn"
    failed = product.describe_status(STATUS_OK, checks_ok=False)
    assert failed["label"] == "Quality checks failed" and failed["tone"] == "danger"


def test_the_summary_is_compact_and_serialisable(configured):
    _, report = deep_chunks(configured, quality_document(), provider=AnsweringProvider())
    summary = product.product_summary(report)
    json.dumps(summary)
    assert {"status", "label", "tone", "chunk_count", "smell_total", "smells",
            "proposer", "verifier", "checks", "sections"} <= set(summary)
    assert summary["chunk_count"]["standard"] >= 1 and summary["chunk_count"]["deep"] >= 1
    assert all({"standard", "deep"} == set(v) for v in summary["smells"].values())
    assert "totals" not in summary and "config" not in summary


# ------------------------------------------------------------------ hygiene
def test_the_report_and_chunks_carry_no_prompt_text_or_secret(configured):
    provider = AnsweringProvider()
    chunks, report = deep_chunks(configured, quality_document(), provider=provider)
    assert provider.prompts, "the fixture must have produced prompts"
    dumped = json.dumps(report, ensure_ascii=False) + json.dumps(
        [c.metadata for c in chunks], ensure_ascii=False
    )
    assert "sk-test-secret-value" not in dumped
    assert "[U1]" not in dumped and "[B1]" not in dumped and "DIVISION" not in dumped
    assert "Authorization" not in dumped and "api_key\"" not in dumped.replace("api_key_env", "")
    # Only the *name* of the key variable is recorded.
    assert report["configuration"]["api_key_env"] == "DEEP_TEST_KEY"


def test_the_configuration_never_reads_the_key_value():
    source = inspect.getsource(product)
    assert "os.environ.get(key_env" in source  # presence check ...
    assert "settings.api_key" not in source.replace("api_key_env", "")  # ... and nothing else


def test_the_query_path_has_no_deep_analysis():
    from pipeline.rag_pipeline import RAGPipeline

    source = "".join(
        inspect.getsource(getattr(RAGPipeline, name))
        for name in ("query", "retrieve", "generate_answer")
    )
    for token in ("deep_pipeline", "chunk_text_deep", "proposer", "verifier", "boundary_judge"):
        assert token not in source


@pytest.mark.parametrize("module", [
    "components.chunker.boundary_judge",
    "components.chunker.boundary_guard",
])
def test_the_retired_boundary_judge_modules_are_gone(module):
    with pytest.raises(ImportError):
        importlib.import_module(module)


def test_nothing_in_the_product_calls_the_retired_judge_entry_point():
    import components.chunker.structural_chunker as chunker
    import pipeline.rag_pipeline as pipeline_module

    for module in (chunker, pipeline_module, product):
        source = inspect.getsource(module)
        assert "chunk_with_product_mode" not in source
        assert "llm_boundary_judge" not in source
        assert "StructurallyGuardedJudge" not in source
