"""HTTP for the companion Viewer: the workspace, and each document's analysis."""

from __future__ import annotations

from flask import Blueprint, request

from application import workspace as use_case

from ..context import flag, services
from .responses import install, ok

bp = Blueprint('viewer', __name__)
install(bp)


@bp.route('/api/demo/viewer', methods=['GET'])
def demo_viewer_status():
    """Whether the companion Agentic Chunking Viewer is up (sidebar status dot)."""
    return ok(**use_case.probe_viewer(services().settings.viewer_url))


@bp.route('/api/demo/workspace', methods=['GET'])
def demo_workspace():
    """Live knowledge base / document state for the Viewer workspace panel.

    ``?prepare=1`` also queues an analysis for every document that has none,
    which is what the Viewer's refresh asks for. Queuing is all it does: the
    packaging runs on a worker, so this call never waits on it.
    """
    container = services()
    if flag(request.args.get('prepare')):
        use_case.prepare_missing(container)
    return ok(**use_case.snapshot(container, console_url=request.host_url.rstrip('/')))


@bp.route('/api/demo/viewer-analysis/<doc_id>', methods=['GET', 'POST'])
def demo_viewer_analysis(doc_id):
    """Where this document's Viewer analysis got to; POST queues or retries it."""
    if request.method == 'POST':
        return ok(state=use_case.request_analysis(services(), doc_id))
    return ok(state=use_case.analysis_state(doc_id))


@bp.route('/api/demo/viewer-analysis/<doc_id>/methods', methods=['POST'])
def demo_add_methods(doc_id):
    return ok(state=use_case.add_methods(doc_id, (request.get_json(silent=True) or {}).get('methods')))


@bp.route('/api/demo/viewer-analysis/<doc_id>/payload', methods=['GET'])
def demo_viewer_payload(doc_id):
    return ok(doc_id=doc_id, payload=use_case.payload(doc_id))


@bp.route('/api/demo/viewer-analysis/<doc_id>/chunks', methods=['GET'])
def demo_viewer_chunks(doc_id):
    return ok(**use_case.chunk_rows(doc_id, request.args.get('method') or ''))
