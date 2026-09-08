"""What this deployment can offer, and whether it is well.

The chunking-method list is the important one. It is a projection of the
library's registry (``amsc.chunking.registry``) and there is no second
catalogue anywhere -- not here, not in the console's JavaScript, not in the
Viewer. A method registered in the library appears here on the next request,
which is what makes "implementation + registration + tests" the whole cost of
adding one.
"""

from __future__ import annotations

from flask import Blueprint, request

from application import catalogue, ops

from ..context import services, session_id
from . import resources
from .envelope import collection, install, resource

bp = Blueprint('v1_meta', __name__)
install(bp)


@bp.route('/meta/chunking-methods', methods=['GET'])
def chunking_methods():
    """Every chunking method this deployment knows about.

    Unavailable methods are listed with their reason rather than hidden: a
    picker that silently drops an option cannot explain why it is missing, and
    "this machine has no embedding model for Hybrid" is the answer the user
    needs.
    """
    items = [resources.chunking_method(entry) for entry in catalogue.chunking_method_facts()]
    return collection(items, offset=0, limit=len(items), total=len(items))


@bp.route('/meta/retrieval-methods', methods=['GET'])
def retrieval_methods():
    """Which retrieval methods a knowledge base's retriever can actually serve.

    Reporting only -- no search runs. A profile that computes no embeddings
    says so here rather than failing a search later.
    """
    found = catalogue.retrieval(services(), session_id=session_id(),
                               kb_id=request.args.get('knowledge_base_id') or None)
    items = [{
        "name": entry.get("name"),
        "label": entry.get("label"),
        "available": bool(entry.get("available")),
        "unavailable_reason": entry.get("reason") or None,
    } for entry in found.get("methods") or []]
    return collection(items, offset=0, limit=len(items), total=len(items),
                      default=found.get("default"))


@bp.route('/meta/models', methods=['GET'])
def models():
    """The configured model chain: chunking, embedding, answer. Never a key."""
    return resource({"chain": catalogue.model_chain(
        services(), session_id=session_id(),
        kb_id=request.args.get('knowledge_base_id') or None)})


@bp.route('/health', methods=['GET'])
def health():
    """Liveness, readiness and one line of capacity.

    Deliberately small and always answerable, including while degraded --
    which it reports rather than fails on. Detail is an operator's question,
    and this endpoint is polled every few seconds by something that only needs
    to know whether to look further.
    """
    state = ops.health(services())
    return resource({
        "state": state["state"],
        "ready": state["ready"],
        "reasons": state["reasons"],
        "checked_at": state["timestamp"],
        "capacity": {
            "ingest": {
                "running": state["ingest"]["running"],
                "queued": state["ingest"]["queued"],
                "queue_capacity": state["ingest"]["queue_capacity"],
                "workers": state["ingest"]["workers"],
            },
            "query": {
                "active": state["query"]["active"],
                "max_active": state["query"]["max_active"],
            },
        },
    })
