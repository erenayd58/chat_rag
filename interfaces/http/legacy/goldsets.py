"""HTTP for the gold set."""

from __future__ import annotations

from flask import Blueprint, request

from application import goldsets as use_case

from ..context import services
from .responses import install, ok

bp = Blueprint('goldsets', __name__)
install(bp)


@bp.route('/api/goldset', methods=['GET'])
def list_goldset():
    return ok(entries=use_case.list_all(services(), request.args.get('kb_id')))


@bp.route('/api/goldset', methods=['POST'])
def upsert_goldset():
    return ok(entry=use_case.upsert(services(), request.get_json(silent=True) or {}))


@bp.route('/api/goldset/<entry_id>', methods=['DELETE'])
def delete_goldset(entry_id):
    use_case.delete(services(), entry_id)
    return ok()
