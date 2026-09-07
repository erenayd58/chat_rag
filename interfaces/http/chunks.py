"""HTTP for chunk inspection and the Lab's searches."""

from __future__ import annotations

from flask import Blueprint, request

from application import chunks as use_case

from .context import services, session_id
from .responses import install, ok

bp = Blueprint('chunks', __name__)
install(bp)


@bp.route('/api/chunks', methods=['GET'])
def get_chunks():
    return ok(**use_case.browse(
        services(),
        kb_id=request.args.get('kb_id'),
        session_id=session_id(),
        offset=int(request.args.get('offset', 0)),
        limit=int(request.args.get('limit', 20)),
        search_text=request.args.get('search', ''),
    ))


@bp.route('/api/chunks/search-vector', methods=['POST'])
def search_chunks_vector():
    data = request.get_json(silent=True) or {}
    return ok(**use_case.search_vector(
        services(),
        query=data.get('query', ''),
        kb_id=data.get('kb_id'),
        session_id=session_id(),
        offset=data.get('offset', 0),
        limit=data.get('limit', 20),
    ))


@bp.route('/api/chunks/search-bm25', methods=['POST'])
def search_chunks_bm25():
    data = request.get_json(silent=True) or {}
    return ok(**use_case.search_bm25(
        services(),
        query=data.get('query', ''),
        kb_id=data.get('kb_id'),
        session_id=session_id(),
        offset=int(data.get('offset', 0)),
        limit=int(data.get('limit', 20)),
    ))


@bp.route('/api/experiment/search_chunks', methods=['POST'])
def experiment_search_chunks():
    data = request.get_json(silent=True) or {}
    return ok(**use_case.experiment_search(
        services(),
        query=data.get('query', ''),
        method=data.get('method', 'hybrid'),
        kb_id=data.get('kb_id'),
        session_id=session_id(),
        top_k=int(data.get('top_k', 20)),
    ))


@bp.route('/api/chunks/<chunk_id>', methods=['GET'])
def get_chunk(chunk_id):
    return ok(chunk=use_case.read(services(), chunk_id,
                                  kb_id=request.args.get('kb_id'),
                                  session_id=session_id()))


@bp.route('/api/chunks/<chunk_id>', methods=['PUT'])
def update_chunk(chunk_id):
    data = request.get_json(silent=True) or {}
    use_case.update(services(), chunk_id, content=data.get('content'),
                    metadata=data.get('metadata'), kb_id=request.args.get('kb_id'),
                    session_id=session_id())
    return ok(message='Chunk updated successfully')


@bp.route('/api/chunks/<chunk_id>', methods=['DELETE'])
def delete_chunk(chunk_id):
    use_case.delete(services(), chunk_id, kb_id=request.args.get('kb_id'),
                    session_id=session_id())
    return ok(message='Chunk deleted successfully')
