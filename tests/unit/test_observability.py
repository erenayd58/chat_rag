"""Telemetry that tells the truth, and logs that cannot leak a document.

Two claims are worth testing here rather than trusting. The first is that
the numbers mean what they say: a stage that took a known amount of time
reports it, a stage that failed is recorded as failed with a category rather
than dropped, and the summaries the metrics endpoint serves are computed from
real traces and stay bounded however many jobs run. The second is the rule
that makes operational logging safe to leave on: no document text and no
credential can reach a log line through this path, whatever a caller passes.
"""

from __future__ import annotations

import logging
import threading

import pytest

from components.observability import events
from components.observability import telemetry as T


class Clock:
    """A perf counter this test drives, so a duration is exact, not timed."""

    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


@pytest.fixture
def clock(monkeypatch):
    fake = Clock()
    monkeypatch.setattr(T.time, "perf_counter", fake)
    return fake


@pytest.fixture
def registry(monkeypatch):
    fresh = T.MetricsRegistry(window=5)
    monkeypatch.setattr(T, "_registry", fresh)
    return fresh


# ------------------------------------------------------------------ stages


def test_a_stage_records_the_time_it_actually_took(clock):
    trace = T.JobTrace(job_id="j1", kb_id="kb1")
    with T.use_trace(trace):
        with T.stage(T.PARSE):
            clock.advance(2.5)
        with T.stage(T.CHUNK):
            clock.advance(0.25)
    assert trace.seconds_for(T.PARSE) == 2.5
    assert trace.seconds_for(T.CHUNK) == 0.25
    assert trace.as_dict()["stages"] == {"parse": 2.5, "chunk": 0.25}


def test_a_stage_that_runs_twice_is_summed_not_overwritten(clock):
    trace = T.JobTrace(job_id="j1")
    with T.use_trace(trace):
        for _ in range(3):
            with T.stage(T.EMBED):
                clock.advance(1.0)
    assert trace.seconds_for(T.EMBED) == 3.0


def test_a_failed_stage_is_recorded_with_its_category_and_still_raises(clock):
    """"Parse took forty seconds and then failed" is the single most useful
    line in an incident; a plain timer would have thrown it away."""
    from core.exceptions import ChunkerException

    trace = T.JobTrace(job_id="j1")
    with T.use_trace(trace):
        with pytest.raises(ChunkerException):
            with T.stage(T.CHUNK):
                clock.advance(40.0)
                raise ChunkerException("bad canonical")
    (recorded,) = trace.stages
    assert recorded.name == "chunk" and recorded.seconds == 40.0
    assert recorded.ok is False and recorded.error_category == "chunking"
    assert trace.as_dict()["failed_stages"] == ["chunk"]


def test_annotations_attach_counts_to_the_stage_that_just_ran(clock):
    trace = T.JobTrace(job_id="j1")
    with T.use_trace(trace):
        with T.stage(T.PARSE):
            clock.advance(1.0)
            T.annotate(characters=4096, units=37)
    assert trace.stages[0].detail == {"characters": 4096, "units": 37}


def test_a_stage_outside_a_job_measures_nothing_and_costs_nothing():
    assert T.current_trace() is None
    with T.stage(T.PARSE):
        pass  # a CLI ingest or a unit test: no trace, no error
    T.annotate(anything=1)


def test_a_trace_belongs_to_its_thread():
    trace = T.JobTrace(job_id="j1")
    seen = {}

    def other():
        seen["trace"] = T.current_trace()

    with T.use_trace(trace):
        thread = threading.Thread(target=other)
        thread.start()
        thread.join(5)
    assert seen["trace"] is None


def test_provider_and_embedding_work_is_counted_apart_from_stages():
    """Those calls happen on a pool several layers below the stage that
    opened, so they are recorded on the trace, not on a stage."""
    trace = T.JobTrace(job_id="j1")
    trace.record_provider(seconds=2.0, wait_seconds=0.5)
    trace.record_provider(seconds=1.0, wait_seconds=0.25)
    trace.record_embedding(seconds=0.5, wait_seconds=0.0)
    described = trace.as_dict()
    assert described["provider"] == {"calls": 2, "seconds": 3.0, "wait_seconds": 0.75}
    assert described["embedding"] == {"calls": 1, "seconds": 0.5, "wait_seconds": 0.0}


def test_a_trace_carries_no_document_content():
    trace = T.JobTrace(job_id="j1", kb_id="kb1", mode="deep_analysis")
    with T.use_trace(trace):
        with T.stage(T.PARSE):
            T.annotate(characters=100_000)
    rendered = str(trace.as_dict())
    assert "characters" not in rendered, "counts live on the stage, not in the summary"
    assert set(trace.as_dict()) == {
        "job_id", "kb_id", "mode", "kind", "status", "error_category", "queue_seconds",
        "total_seconds", "stages", "failed_stages", "provider", "embedding",
    }


# ---------------------------------------------------------------- categories


@pytest.mark.parametrize("error, expected", [
    (None, "none"),
    (TimeoutError("slow"), "timeout"),
    (OSError("disk"), "storage"),
    (ValueError("?"), "unknown"),
])
def test_errors_are_categorised_by_what_an_operator_would_do_next(error, expected):
    assert T.categorise(error) == expected


def test_ingest_specific_errors_keep_their_own_names():
    from core.exceptions import ConfigurationException, IngestInterrupted, IngestOverloaded

    assert T.categorise(IngestInterrupted("timed_out")) == "timed_out"
    assert T.categorise(IngestInterrupted("cancelled")) == "cancelled"
    assert T.categorise(IngestOverloaded("full")) == "overloaded"
    assert T.categorise(ConfigurationException("no chunker")) == "configuration"


# ----------------------------------------------------------------- registry


def test_the_registry_summarises_real_traces(registry, clock):
    for index in range(4):
        trace = T.JobTrace(job_id=f"j{index}")
        trace.queue_seconds = float(index)
        trace.total_seconds = float(index) * 2
        with T.use_trace(trace):
            with T.stage(T.PARSE):
                clock.advance(1.0 + index)
        registry.finish(trace)

    snapshot = registry.snapshot()
    assert snapshot["jobs"]["measured"] == 4
    assert snapshot["jobs"]["queue_wait_seconds"]["max"] == 3.0
    assert snapshot["jobs"]["total_seconds"]["p50"] in (2.0, 4.0)
    assert snapshot["stages"]["parse"]["count"] == 4
    assert snapshot["stages"]["parse"]["max"] == 4.0
    assert snapshot["stages"]["parse"]["total"] == 1 + 2 + 3 + 4


def test_the_trace_window_cannot_grow_with_uptime(registry):
    for index in range(500):
        trace = T.JobTrace(job_id=f"j{index}")
        trace.total_seconds = 1.0
        registry.finish(trace)
    snapshot = registry.snapshot()
    assert snapshot["jobs"]["measured"] == 5, "the window is a fixed length"
    assert snapshot["jobs"]["window"] == 5
    assert len(snapshot["recent"]) <= 5


def test_the_recent_list_is_capped_however_much_is_asked_for(registry):
    for index in range(5):
        registry.finish(T.JobTrace(job_id=f"j{index}"))
    assert len(registry.snapshot(recent=1000)["recent"]) <= 5
    assert registry.snapshot(recent=0)["recent"] == []


def test_error_messages_per_category_are_bounded(registry):
    for index in range(50):
        registry.record_error("provider", f"gateway said {index}")
    snapshot = registry.snapshot()
    assert snapshot["errors"]["by_category"]["provider"] == 50
    assert len(snapshot["errors"]["recent_messages"]["provider"]) == T.MESSAGES_PER_CATEGORY


def test_a_very_long_error_message_is_truncated(registry):
    registry.record_error("provider", "x" * 5000)
    (message,) = registry.snapshot()["errors"]["recent_messages"]["provider"]
    assert len(message) <= 200


def test_counters_add_up(registry):
    registry.count("ingest.accepted", 3)
    registry.count("ingest.rejected")
    assert registry.snapshot()["counters"] == {"ingest.accepted": 3, "ingest.rejected": 1}


# ------------------------------------------------------------------- events


def test_an_event_renders_as_stable_key_value_pairs():
    line = events.render("ingest.job.started", job_id="abc", kb_id="kb1", queue_seconds=1.5)
    assert line == "event=ingest.job.started job_id=abc kb_id=kb1 queue_seconds=1.5"


def test_a_field_that_would_carry_document_text_is_redacted():
    line = events.render("ingest.job.succeeded", job_id="abc",
                         content="Bu belgenin gizli icerigi burada yaziyor.")
    assert "gizli" not in line
    assert "content=<redacted>" in line


@pytest.mark.parametrize("field", ["text", "chunk", "prompt", "api_key", "password", "secret"])
def test_every_forbidden_field_name_is_refused(field):
    line = events.render("x", **{field: "something sensitive"})
    assert "sensitive" not in line


def test_a_credential_shaped_value_is_redacted_whatever_field_it_arrives_in():
    assert "sk-" not in events.render("x", note="sk-livekey1234567890abcd")
    assert "Bearer" not in events.render("x", header="Bearer abcdef123456")
    # A long run of letters *and* digits is what a key looks like. A long run
    # of one repeated character is a message, and is truncated instead.
    opaque = "aZ3" * 14
    assert opaque not in events.render("x", trace_id=opaque)


def test_a_long_message_is_truncated_rather_than_pasted_whole():
    line = events.render("x", error="the gateway refused this request " * 200)
    assert len(line) < 300
    assert line.endswith('..."')


def test_a_long_opaque_blob_is_treated_as_a_secret_not_a_message():
    """Where truncation and redaction meet, this errs towards redaction: an
    over-redacted message costs information, a leaked key costs a credential."""
    assert "<redacted>" in events.render("x", note="a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6q7")


def test_a_multiline_value_stays_on_one_line():
    line = events.render("x", error="first line\nsecond line")
    assert "\n" not in line
    assert 'error="first line second line"' in line


def test_the_emitted_line_reaches_the_log(caplog):
    with caplog.at_level(logging.INFO, logger="RAG.ops"):
        events.emit("ingest.job.accepted", job_id="j1", kb_id="kb1")
    assert "event=ingest.job.accepted job_id=j1 kb_id=kb1" in caplog.text


def test_an_operational_log_of_a_whole_ingest_carries_no_document_text(caplog):
    """The end-to-end version of the rule: everything a job emits, checked
    against the document it was given."""
    secret_sentence = "Sirket 2024 yilinda gizli bir anlasma imzaladi"
    with caplog.at_level(logging.DEBUG, logger="RAG.ops"):
        events.emit("ingest.job.accepted", job_id="j1", kb_id="kb1", filename="rapor.pdf")
        events.emit("ingest.job.started", job_id="j1", queue_seconds=0.4)
        events.emit("ingest.job.succeeded", job_id="j1", chunks=42, t_parse=1.2,
                    text=secret_sentence)
    assert secret_sentence not in caplog.text
    assert "<redacted>" in caplog.text
    assert "rapor.pdf" in caplog.text, "a filename is operationally necessary"
