"""HTTP for knowledge bases and their vectors."""

from __future__ import annotations

from flask import Blueprint, request

from application import knowledge_bases as use_case

from .context import services, session_id
from .responses import install, ok

bp = Blueprint('knowledge_bases', __name__)
install(bp)


@bp.route('/api/kb', methods=['GET'])
def list_kb():
    return ok(knowledge_bases=use_case.list_all(services()))


@bp.route('/api/kb', methods=['POST'])
def create_kb():
    return ok(kb=use_case.create(services(), request.get_json(silent=True) or {}))


@bp.route('/api/kb/<kb_id>', methods=['GET'])
def get_kb(kb_id):
    return ok(kb=use_case.get(services(), kb_id))


@bp.route('/api/kb/<kb_id>', methods=['PUT'])
def update_kb(kb_id):
    return ok(kb=use_case.update(services(), kb_id, request.get_json(silent=True) or {}))


@bp.route('/api/kb/<kb_id>', methods=['DELETE'])
def delete_kb(kb_id):
    return ok(**use_case.delete(services(), kb_id))


@bp.route('/api/kb/<kb_id>/embedding-index', methods=['GET'])
def embedding_index_status(kb_id):
    return ok(index=use_case.embedding_index(services(), kb_id, session_id=session_id()))


@bp.route('/api/kb/<kb_id>/reindex-embeddings', methods=['POST'])
def reindex_embeddings(kb_id):
    return ok(**use_case.reindex_embeddings(services(), kb_id, session_id=session_id()))
