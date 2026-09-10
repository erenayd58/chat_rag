"""Operational visibility: one way to measure a job, one way to log an event.

``telemetry`` measures (stages, traces, counters, bounded summaries);
``events`` writes the operator-facing log lines, with a redaction rule that
keeps document content and credentials out of them.
"""

from . import events, telemetry  # noqa: F401
from .events import emit, error, warn  # noqa: F401
from .telemetry import (  # noqa: F401
    CHUNK, DEEP, EMBED, INDEX, LEDGER, PARSE, QUEUE_WAIT, STAGES, VIEWER,
    JobTrace, MetricsRegistry, annotate, categorise, current_trace, metrics,
    stage, use_trace,
)
