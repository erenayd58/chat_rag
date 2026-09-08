"""HTTP for the operational surfaces."""

from __future__ import annotations

from flask import Blueprint, jsonify, request

from application import ops as use_case

from ..context import services
from .responses import install, ok

bp = Blueprint('ops', __name__)
install(bp)


@bp.route('/api/health', methods=['GET'])
def health_check():
    """Liveness, readiness and one line of capacity. Deliberately small.

    Detail belongs at ``/api/ops/metrics``: an endpoint a load balancer polls
    every few seconds must not carry a history, and an operator reading this
    one wants to know whether to look further, not to read everything.
    """
    return jsonify(use_case.health(services()))


@bp.route('/api/ops/metrics', methods=['GET'])
def ops_metrics():
    """``?recent=N`` sets how many individual job traces come back."""
    try:
        recent = int(request.args.get('recent', 10))
    except ValueError:
        recent = 10
    return ok(**use_case.metrics(services(), recent=recent))
