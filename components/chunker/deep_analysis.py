"""Deep Analysis: the product's premium ingest mode, on ``amsc.deep_pipeline``.

There is exactly one Deep Analysis implementation and it lives upstream:
``amsc.deep_pipeline.chunk_document`` runs the frozen structure-first walk,
the LLM proposer, the deterministic quality selector, the double-order
verifier and the quality measurement, and returns rows plus a report. This
module only does what an application has to do around that call:

* turn backend configuration into :class:`DeepAnalysisSettings` without
  ever touching the API key (only the *name* of its variable is configured;
  the provider reads it at request time);
* say up front what an LLM run would need when it is not configured, so
  the ingest can complete deterministically and *say so* rather than
  silently looking like Standard;
* translate the pipeline's ``status`` into product wording, and reduce the
  report to the summary the console shows.

Nothing here is consulted at query time. Chat reads the chunks that were
indexed at upload; no proposer, verifier or judge runs during a question.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from amsc.deep_analysis import DeepConfig
from amsc.deep_pipeline import (
    DEFAULT_ENDPOINT,
    STATUS_DEGRADED,
    STATUS_DETERMINISTIC,
    STATUS_FALLBACK_NO_PROVIDER,
    STATUS_FALLBACK_PROVIDER_ERROR,
    STATUS_OK,
    DeepAnalysisSettings,
)

#: The mode name the product records on documents and chunks.
CHUNKING_MODE = "deep_analysis"

#: Every status the pipeline can report, in the order the console ranks them.
STATUSES = (
    STATUS_OK,
    STATUS_DETERMINISTIC,
    STATUS_DEGRADED,
    STATUS_FALLBACK_PROVIDER_ERROR,
    STATUS_FALLBACK_NO_PROVIDER,
)


@dataclass(frozen=True)
class DeepAnalysisConfiguration:
    """What one ingest will run with, and what it lacks for a model run.

    ``missing`` names the configuration an LLM run would need and does not
    have (a model id, a key in the named variable). When it is non-empty the
    pipeline runs the deterministic quality contract alone and the ingest is
    recorded as ``fallback_no_provider`` -- never as Standard.
    """

    settings: DeepAnalysisSettings
    missing: tuple[str, ...] = ()

    @property
    def llm_available(self) -> bool:
        return self.settings.use_llm and not self.missing

    @property
    def fallback_reason(self) -> str | None:
        if not self.missing:
            return None
        return (
            "Deep Analysis ran without a language model: "
            + ", ".join(self.missing)
            + " is not set in the backend environment. The deterministic "
            "quality contract produced this result."
        )


def deep_config(
    *, min_tokens: int, target_tokens: int, soft_max_tokens: int, hard_max_tokens: int
) -> DeepConfig:
    """The pipeline budgets, taken from the Standard chunker's constants so
    both modes cut under one set of numbers."""
    return DeepConfig(
        min_tokens=min_tokens,
        target_tokens=target_tokens,
        soft_max_tokens=soft_max_tokens,
        hard_max_tokens=hard_max_tokens,
    )


def build_configuration(app_settings: Any, config: DeepConfig) -> DeepAnalysisConfiguration:
    """Backend settings -> pipeline settings. Reads the key's presence only."""
    model = (getattr(app_settings, "deep_analysis_model", "") or "").strip()
    verifier_model = (getattr(app_settings, "deep_analysis_verifier_model", "") or "").strip()
    endpoint = (getattr(app_settings, "deep_analysis_endpoint", "") or "").strip() or DEFAULT_ENDPOINT
    key_env = (getattr(app_settings, "deep_analysis_api_key_env", "") or "").strip()
    use_llm = bool(getattr(app_settings, "deep_analysis_use_llm", True))
    verify = bool(getattr(app_settings, "deep_analysis_verify", True))
    timeout = float(getattr(app_settings, "deep_analysis_timeout", 120.0) or 120.0)
    concurrency = int(getattr(app_settings, "deep_analysis_concurrency", 8) or 8)

    missing: list[str] = []
    if use_llm:
        if not model:
            missing.append("DEEP_ANALYSIS_MODEL")
        if not key_env:
            missing.append("DEEP_ANALYSIS_API_KEY_ENV")
        elif not os.environ.get(key_env, "").strip():
            missing.append(key_env)

    settings = DeepAnalysisSettings(
        config=config,
        # An unconfigured model run is not attempted: the pipeline runs the
        # deterministic contract and the ingest is labelled accordingly.
        use_llm=use_llm and not missing,
        verify=verify,
        proposer_model=model or DeepAnalysisSettings.proposer_model,
        verifier_model=verifier_model or None,
        endpoint=endpoint,
        api_key_env=key_env or DeepAnalysisSettings.api_key_env,
        concurrency=concurrency,
        timeout_seconds=timeout,
    )
    return DeepAnalysisConfiguration(settings=settings, missing=tuple(missing))


# -------------------------------------------------------------- transports
def build_transports(settings: DeepAnalysisSettings) -> tuple[Any, Any | None]:
    """The real proposer and verifier transports, from amsc.

    Kept as a module attribute so a test can replace it with doubles and
    still go through :func:`limited_providers` -- the budget wrapping is the
    thing under test, not the HTTP client.
    """
    from amsc.deep_pipeline import build_providers

    return build_providers(settings)


def limited_providers(configuration: DeepAnalysisConfiguration) -> tuple[Any | None, Any | None]:
    """The proposer and verifier this ingest will call, under the budget.

    ``(None, None)`` when no model run is possible: the pipeline then runs the
    deterministic contract alone, exactly as before. Otherwise each transport
    is wrapped so every ``complete()`` takes one slot of the process-wide
    provider budget and asks the current job's guard first. This is the
    narrowest boundary there is -- one call, one slot -- so a Deep job with a
    pool of eight still shares the budget call by call with every other Deep
    job, rather than holding eight slots for the length of its run.

    Both the job's guard and its trace are read *here* -- on the ingest
    worker, before the Deep pipeline starts -- and handed to the wrappers,
    because ``collect_votes`` makes every call on a ``ThreadPoolExecutor``
    thread where neither thread-local exists. Binding them at this point is
    what keeps a call's deadline and its measurement attached to the job that
    asked for it.
    """
    if not configuration.llm_available:
        return None, None
    from components.observability import telemetry as T
    from components.ingest.limits import LimitedProvider, current_guard, provider_budget

    proposer, verifier = build_transports(configuration.settings)
    if proposer is None:
        return None, None
    budget = provider_budget()
    guard = current_guard()
    trace = T.current_trace()
    limited_proposer = LimitedProvider(proposer, budget, guard, trace=trace)
    limited_verifier = (
        LimitedProvider(verifier, budget, guard, trace=trace) if verifier is not None else None
    )
    return limited_proposer, limited_verifier


# ---------------------------------------------------------------- wording
_STATUS_TEXT: dict[str, dict[str, str]] = {
    STATUS_OK: {
        "label": "Quality checks passed",
        "headline": "Deep Analysis completed",
        "tone": "success",
        "detail": (
            "The language model proposed boundaries, the verifier confirmed "
            "every accepted change, and the deterministic quality contract held."
        ),
    },
    STATUS_DETERMINISTIC: {
        "label": "Deterministic quality pass",
        "headline": "Completed without a language model",
        "tone": "neutral",
        "detail": (
            "The language model was not requested for this ingest; the "
            "deterministic quality contract ran on its own."
        ),
    },
    STATUS_FALLBACK_NO_PROVIDER: {
        "label": "Completed with deterministic fallback",
        "headline": "No model provider configured",
        "tone": "warn",
        "detail": (
            "No language-model provider was configured, so the deterministic "
            "quality contract produced this result. Standard was not used."
        ),
    },
    STATUS_FALLBACK_PROVIDER_ERROR: {
        "label": "Completed with deterministic fallback",
        "headline": "Model provider unavailable",
        "tone": "warn",
        "detail": (
            "The language-model provider could not be used for any section, "
            "so every section kept its deterministic result."
        ),
    },
    STATUS_DEGRADED: {
        "label": "Completed with partial fallback",
        "headline": "Some model calls failed",
        "tone": "warn",
        "detail": (
            "Some language-model calls failed; those sections kept their "
            "deterministic result, the rest were verified normally."
        ),
    },
}


def describe_status(status: str, *, checks_ok: bool = True) -> dict[str, str]:
    """Product wording for a pipeline status. Never a traceback."""
    text = dict(_STATUS_TEXT.get(status) or {
        "label": "Completed with unknown status",
        "headline": f"Deep Analysis reported '{status}'",
        "tone": "warn",
        "detail": "The pipeline reported a status this console does not know.",
    })
    if not checks_ok:
        text.update(
            label="Quality checks failed",
            headline="A structural check did not hold",
            tone="danger",
            detail=(
                "The result breached a structural check (hard token cap, "
                "coverage or a quality regression). The document is indexed; "
                "review the details before relying on it."
            ),
        )
    text["status"] = status
    return text


# ---------------------------------------------------------------- summary
def _smell_table(totals: Any) -> dict[str, dict[str, int]] | None:
    """``{smell: {standard, deep}}`` for the smell types only."""
    if not isinstance(totals, dict):
        return None
    standard = totals.get("standard") or {}
    deep = totals.get("deep") or {}
    table: dict[str, dict[str, int]] = {}
    for key in standard:
        if key in ("below_min", "above_soft_max"):
            continue
        table[key] = {"standard": int(standard.get(key, 0)), "deep": int(deep.get(key, 0))}
    return table


def _size_table(totals: Any) -> dict[str, dict[str, int]] | None:
    if not isinstance(totals, dict):
        return None
    standard = totals.get("standard") or {}
    deep = totals.get("deep") or {}
    return {
        key: {"standard": int(standard.get(key, 0)), "deep": int(deep.get(key, 0))}
        for key in ("below_min", "above_soft_max")
        if key in standard
    }


def checks_passed(report: dict[str, Any]) -> bool:
    checks = report.get("checks") or {}
    return (
        bool(checks.get("hard_cap_ok", True))
        and bool(checks.get("coverage_ok", True))
        and int(report.get("structural_regression_count") or 0) == 0
    )


def product_summary(report: dict[str, Any]) -> dict[str, Any]:
    """The part of the report the console and the snapshot carry.

    Counts, model ids and wording only. The full report stays on the
    document record; per-section audit never leaves the pipeline.
    """
    status = str(report.get("status") or "")
    wording = describe_status(status, checks_ok=checks_passed(report))
    proposer = report.get("proposer") or None
    verifier = report.get("verifier") or None
    selection = report.get("selection") or {}
    llm_effect = report.get("llm_effect") or {}
    transports = (proposer or {}).get("transport_status") or {}
    answered = int(transports.get("ok", 0)) + int(transports.get("cached", 0))
    call_count = int((proposer or {}).get("call_count", 0)) if proposer else 0
    return {
        **wording,
        "chunking_mode": CHUNKING_MODE,
        "pipeline_mode": report.get("pipeline_mode"),
        "uses_llm": bool(report.get("uses_llm")),
        "model_id": report.get("model_id"),
        "verifier_model_id": report.get("verifier_model_id"),
        "chunk_count": report.get("chunk_count"),
        "smell_total": report.get("smell_total"),
        "smells": _smell_table(report.get("totals")),
        "size_counters": _size_table(report.get("totals")),
        "structural_regression_count": report.get("structural_regression_count"),
        "strict_regression_count": report.get("strict_regression_count"),
        "size_trade_count": report.get("size_trade_count"),
        "change_group_count": report.get("change_group_count"),
        "sections": {
            "moved_by_contract": selection.get("sections_moved"),
            "reverted_by_contract": selection.get("sections_reverted"),
            "changed_by_llm": llm_effect.get("sections_changed_by_llm"),
        },
        "proposer": {
            "call_count": call_count,
            "answered_count": answered,
            "failed_count": max(call_count - answered, 0),
            "boundary_count": proposer.get("boundary_count"),
            "accepted_boundary_count": proposer.get("accepted_boundary_count"),
            "forbidden_boundary_count": proposer.get("forbidden_boundary_count"),
        } if proposer else None,
        "verifier": {
            "group_count": verifier.get("group_count"),
            "accepted": verifier.get("accepted"),
            "reverted": verifier.get("reverted"),
        } if verifier else None,
        "checks": report.get("checks"),
        "fallback_reason": report.get("fallback_reason"),
        "timing_seconds": report.get("timing_seconds"),
    }
