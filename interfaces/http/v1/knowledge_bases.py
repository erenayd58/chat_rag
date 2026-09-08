"""`/api/v1/knowledge-bases` -- the collection a document is ingested into."""

from __future__ import annotations

from flask import Blueprint, request

from application import chunks as chunk_use_case
from application import knowledge_bases as use_case

from ..context import services, session_id
from . import resources
from .envelope import collection, install, page_request, resource

bp = Blueprint('v1_knowledge_bases', __name__)
install(bp)


@bp.route('/knowledge-bases', methods=['GET'])
def list_knowledge_bases():
    offset, limit = page_request()
    records = use_case.list_all(services())
    items = [resources.knowledge_base(record) for record in records[offset:offset + limit]]
    return collection(items, offset=offset, limit=limit, total=len(records))


@bp.route('/knowledge-bases', methods=['POST'])
def create_knowledge_base():
    """Create one. Its chunker, embedding model and storage are fixed here and
    not editable afterwards, because the corpus that gets ingested depends on
    them -- changing one later would describe the vectors wrongly."""
    body = request.get_json(silent=True) or {}
    record = use_case.create(services(), {
        "name": body.get("name"),
        "chunker": body.get("chunker"),
        "embedding_model_name": body.get("embedding_model"),
        "retrieval_method": body.get("retrieval_method"),
        "extra": body.get("extra"),
    })
    return resource(resources.knowledge_base(record), status=201,
                    headers={"Location": f"/api/v1/knowledge-bases/{record['kb_id']}"})


@bp.route('/knowledge-bases/<kb_id>', methods=['GET'])
def get_knowledge_base(kb_id):
    return resource(resources.knowledge_base(use_case.get(services(), kb_id)))


@bp.route('/knowledge-bases/<kb_id>', methods=['PATCH'])
def update_knowledge_base(kb_id):
    """Only ``name`` and ``extra``: everything else was decided at creation."""
    body = request.get_json(silent=True) or {}
    updates = {key: body[key] for key in ("name", "extra") if key in body}
    return resource(resources.knowledge_base(use_case.update(services(), kb_id, updates)))


@bp.route('/knowledge-bases/<kb_id>', methods=['DELETE'])
def delete_knowledge_base(kb_id):
    """Delete the knowledge base, and its corpus with it.

    Its documents' ledger rows and analyses deliberately survive -- a user
    does not lose the record that a file was ever ingested. **409** when the
    corpus is still needed by something else.
    """
    use_case.delete(services(), kb_id)
    return '', 204


@bp.route('/knowledge-bases/<kb_id>/embedding-index', methods=['GET'])
def embedding_index(kb_id):
    """Whether the stored vectors belong to the embedding model configured now."""
    return resource(use_case.embedding_index(services(), kb_id, session_id=session_id()))


@bp.route('/knowledge-bases/<kb_id>/embedding-index/rebuild', methods=['POST'])
def rebuild_embedding_index(kb_id):
    """Re-embed every stored chunk with the current embedding model.

    Synchronous, and the one long write this API does inline: it is an
    operator action on a knowledge base nobody is querying, not a user action.
    """
    return resource(use_case.reindex_embeddings(services(), kb_id, session_id=session_id()))


@bp.route('/knowledge-bases/<kb_id>/chunks', methods=['GET'])
def browse_chunks(kb_id):
    """The corpus, a page at a time. ``?search=`` filters by phrase.

    A substring scan inside the store: no embedding call and no index build,
    which is why -- unlike the searches -- it runs under no query bound.
    """
    offset, limit = page_request()
    found = chunk_use_case.browse(
        services(), kb_id=kb_id, session_id=session_id(),
        offset=offset, limit=limit, search_text=request.args.get('search', ''),
    )
    return collection([resources.chunk(row) for row in found["chunks"]],
                      offset=found["offset"], limit=found["limit"], total=found["total"])
