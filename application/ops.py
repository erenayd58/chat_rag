"""What this process can do right now, and what it has been doing.

Two read models over the same facts, kept apart because they are asked at
different rates by different callers. :func:`health` is polled by a load
balancer every few seconds and must stay cheap and always answerable --
including while the service is degraded, which it reports rather than fails
on. :func:`metrics` is what an operator reads once health has told them to
look closer, and carries the history health deliberately does not.
"""

from __future__ import annotations

import sys
from datetime import datetime

import storage as database
from components.observability import events, telemetry as T
from components.query import QUERY_DEADLINE_SEMANTICS, answer_budget

from . import workspace

#: How many consecutive failed jobs it takes before the service calls itself
#: degraded. Small enough to notice a broken provider, large enough that one
#: unreadable upload is not an incident.
DEGRADED_AFTER_JOBS = 5


def service_state(services) -> tuple[str, bool, list]:
    """``(status, ready, reasons)`` -- what this process can do right now.

    Three states, because they call for three different actions and nothing
    finer would be acted on differently:

    * ``ok`` -- serving, with room to accept work.
    * ``overloaded`` -- serving, but the ingest queue is full, so an upload
      would be refused. Retrying later is the answer, not a restart.
    * ``degraded`` -- serving reads, but something an operator should look at:
      no knowledge base can be reached, or every recent ingest failed.

    ``ready`` is separate from ``status`` on purpose: it answers "may this
    process be sent traffic", which stays true while overloaded (uploads are
    refused politely, chat and search still work) and while degraded.
    """
    reasons = []
    status = 'ok'
    capacity = services.ingest_jobs.snapshot()
    if capacity['queued'] >= capacity['queue_capacity'] and capacity['queue_capacity'] >= 0:
        if capacity['running'] >= capacity['workers']:
            status = 'overloaded'
            reasons.append('the ingest queue is full; uploads are being refused')
    # The same state for the other path: every query slot in use means a
    # question arriving now is refused. Transient, like a full queue.
    if services.query_admission.saturated:
        status = 'overloaded'
        reasons.append('every query slot is in use; questions are being refused')
    # After the queue check, because the two states are not equal: being full
    # is transient and needs patience, while every recent job failing needs a
    # person. Degraded therefore wins when both are true.
    measured, failed = T.metrics().recent_outcomes()
    if measured >= DEGRADED_AFTER_JOBS and failed == measured:
        status = 'degraded'
        reasons.append(f'the last {measured} ingest jobs all failed')
    try:
        services.kb_manager.list()
    except Exception as error:  # noqa: BLE001 - reported, not raised
        status = 'degraded'
        # Redacted: this reason is served by /api/health and /api/ops/metrics,
        # and a storage error names the deployment's database or data root in
        # full -- a PostgreSQL DSN carries a password.
        reasons.append(
            'the knowledge base records could not be read: '
            + events.redact_message(error)
        )
    # Ready means "this process can serve"; it is not made false by being
    # busy, because refusing traffic would make the overload worse.
    return status, True, reasons


def query_capacity(services) -> dict:
    """One line of query capacity: callers in a query, answer calls in flight,
    and both limits. Snapshot reads under short locks, so this answers while
    every query slot is busy -- which is when it is asked."""
    admission = services.query_admission.snapshot()
    budget = answer_budget().snapshot()
    return {
        'active': admission['active'],
        'max_active': admission['limit'],
        'answer_inflight': budget['inflight'],
        'answer_limit': budget['limit'],
    }


def health(services) -> dict:
    """Liveness, readiness and one line of capacity. Deliberately small."""
    status, ready, reasons = service_state(services)
    capacity = services.ingest_jobs.snapshot()
    return {
        # 'healthy' is kept as the historical value of this field so existing
        # probes and the serve smoke keep working; 'state' is the new one.
        'status': 'healthy',
        'state': status,
        'ready': ready,
        'reasons': reasons,
        'timestamp': datetime.now().isoformat(),
        'llm_provider': services.settings.llm_provider,
        'embedding_model': services.settings.embedding_model_name,
        'ingest': {
            'running': capacity['running'],
            'queued': capacity['queued'],
            'queue_capacity': capacity['queue_capacity'],
            'workers': capacity['workers'],
            'provider_inflight': capacity['budgets']['deep_analysis']['inflight'],
            'provider_limit': capacity['budgets']['deep_analysis']['limit'],
            'embedding_inflight': capacity['budgets']['embedding']['inflight'],
            'embedding_limit': capacity['budgets']['embedding']['limit'],
        },
        'query': query_capacity(services),
    }


def metrics(services, *, recent: int = 10) -> dict:
    """Everything an operator needs when health says to look closer.

    Counters, utilisation, stage and job latency over a bounded window of
    recent jobs, error categories with a few example messages, and the state
    of the caches that could otherwise grow. All of it is bounded by
    construction: the trace window has a fixed length and the recent-job list
    is capped, so this answer cannot grow with uptime.
    """
    state, ready, reasons = service_state(services)
    return {
        'state': state,
        'ready': ready,
        'reasons': reasons,
        'ingest': services.ingest_jobs.snapshot(),
        'query': {
            'admission': services.query_admission.snapshot(),
            'answer_budget': answer_budget().snapshot(),
            'limits': services.settings.query_limits.to_dict(),
            'deadline_semantics': QUERY_DEADLINE_SEMANTICS,
        },
        'metrics': T.metrics().snapshot(recent=recent),
        # Whether this process can reach PostgreSQL, and how much of its
        # connection pool is checked out. The number an operator wants when
        # "the console is slow" turns out to be "every connection is held".
        'database': database.health(),
        'caches': {
            'pipelines': services.pipeline_cache.snapshot(),
            'viewer_boundary_model': workspace.boundary_model_stats(),
            'local_models': local_model_stats(),
        },
        # The effective non-secret configuration, from the same method the
        # start-up banner prints, so "what is this instance running with" has
        # one answer whether you read the log or call the endpoint. Never a
        # credential and never an environment dump.
        'configuration': services.settings.effective_configuration(),
    }


def local_model_stats() -> dict:
    """The shared sentence-transformers models resident at query time, by name,
    with how often each was actually loaded. Only a module already imported is
    asked, so this costs nothing on a deployment that loads none."""
    stats = {}
    embedding = sys.modules.get('components.embedding.sentence_transformer_embedding')
    if embedding is not None:
        stats['sentence_transformers'] = embedding.model_stats()
    return stats
