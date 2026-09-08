"""HTTP for what this deployment can offer: methods, models, retrieval."""

from __future__ import annotations

from flask import Blueprint, request

from application import catalogue as use_case

from ..context import services, session_id
from .responses import install, ok

bp = Blueprint('catalogue', __name__)
install(bp)


@bp.route('/api/demo/methods', methods=['GET'])
def demo_methods():
    return ok(methods=use_case.chunking_methods())


@bp.route('/api/models', methods=['GET'])
def model_chain():
    return ok(chain=use_case.model_chain(services(), session_id=session_id(),
                                         kb_id=request.args.get('kb_id') or None))


@bp.route('/api/retrieval/capabilities', methods=['GET'])
def retrieval_capabilities_api():
    return ok(**use_case.retrieval(services(), session_id=session_id(),
                                   kb_id=request.args.get('kb_id') or None))
