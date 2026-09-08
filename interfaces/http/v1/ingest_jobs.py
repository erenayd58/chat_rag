"""`/api/v1/ingest-jobs` -- what became of a submitted upload.

A job id outlives the process that minted it. Every transition is journalled
and start-up settles anything that was in flight against the ingest ledger, so
a **404** here means one thing only: the job finished longer ago than jobs are
kept. It never means "we lost it".
"""

from __future__ import annotations

from flask import Blueprint, request

from application import ingest as use_case

from ..context import flag, services
from . import resources
from .envelope import collection, install, page_request, resource

bp = Blueprint('v1_ingest_jobs', __name__)
install(bp)


@bp.route('/ingest-jobs', methods=['GET'])
def list_jobs():
    """``?knowledge_base_id=`` narrows; ``?active=true`` leaves out the finished."""
    offset, limit = page_request()
    found = use_case.list_jobs(services(), request.args.get('knowledge_base_id') or None,
                               active_only=flag(request.args.get('active')))
    records = found["jobs"]
    items = [resources.ingest_job(record) for record in records[offset:offset + limit]]
    capacity = found["capacity"]
    return collection(items, offset=offset, limit=limit, total=len(records),
                      capacity={
                          "running": capacity["running"],
                          "queued": capacity["queued"],
                          "queue_capacity": capacity["queue_capacity"],
                          "workers": capacity["workers"],
                      })


@bp.route('/ingest-jobs/<job_id>', methods=['GET'])
def get_job(job_id):
    return resource(resources.ingest_job(use_case.job(services(), job_id)))


@bp.route('/ingest-jobs/<job_id>', methods=['DELETE'])
def cancel_job(job_id):
    """Cancel a job: a queued one at once, a running one at its next seam.

    A running job that has already written its document finishes as
    ``succeeded`` -- the returned ``status`` says which happened, so this is a
    resource and not a 204.
    """
    return resource(resources.ingest_job(use_case.cancel(services(), job_id)))
