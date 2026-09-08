"""HTTP for documents: their rows, their chunks, their parser view, their deletion."""

from __future__ import annotations

from flask import Blueprint, request

from application import documents as use_case

from ..context import services, session_id
from .responses import install, ok

bp = Blueprint('documents', __name__)
install(bp)


@bp.route('/api/documents', methods=['GET'])
def get_documents():
    kb_id = request.args.get('kb_id', None)
    return ok(documents=use_case.list_all(services(), kb_id), kb_id=kb_id)


@bp.route('/api/stats', methods=['GET'])
def get_stats():
    return ok(stats=use_case.statistics(
        services(), kb_id=request.args.get('kb_id', None), session_id=session_id(),
    ))


@bp.route('/api/documents/<doc_id>/chunks', methods=['GET'])
def get_document_chunks(doc_id):
    return ok(**use_case.chunks_of(services(), doc_id,
                                   kb_id=request.args.get('kb_id'),
                                   session_id=session_id()))


@bp.route('/api/documents/<doc_id>/canonical-units', methods=['GET'])
def get_document_canonical_units(doc_id):
    return ok(**use_case.canonical_units(
        services(), doc_id,
        kb_id=request.args.get('kb_id'),
        session_id=session_id(),
        page_from=request.args.get('page_from', type=int),
        page_to=request.args.get('page_to', type=int),
        unit_type=request.args.get('unit_type') or None,
        offset=request.args.get('offset', default=0, type=int),
        limit=request.args.get('limit', default=100, type=int),
    ))


@bp.route('/api/documents/<doc_id>', methods=['DELETE'])
def delete_document(doc_id):
    use_case.delete(services(), doc_id, kb_id=request.args.get('kb_id'),
                    session_id=session_id())
    return ok(message='Document deleted successfully')
