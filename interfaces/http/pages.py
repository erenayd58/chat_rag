"""The rendered screens. Not part of the API contract -- these are what a
Next.js front end replaces -- so they keep Flask's own error behaviour."""

from __future__ import annotations

from flask import Blueprint, redirect, render_template

from .context import ensure_session, services

bp = Blueprint('pages', __name__)


@bp.app_context_processor
def inject_companion_links():
    """The research viewer's address, for the sidebar link. Empty hides it."""
    return {'viewer_url': services().settings.viewer_url}


@bp.route('/')
def kb_list_page():
    """Knowledge Bases: the product landing page."""
    ensure_session()
    return render_template('kb_list.html', active_nav='kb')


@bp.route('/kb/<kb_id>')
def kb_detail_page(kb_id):
    """Knowledge base detail: Overview | Documents | Settings."""
    ensure_session()
    if not services().kb_manager.get(kb_id):
        return redirect('/')
    return render_template('kb_detail.html', active_nav='kb', kb_id=kb_id)


@bp.route('/chat')
def chat_page():
    """Conversational QA over a selected knowledge base."""
    ensure_session()
    return render_template('chat.html', active_nav='chat')


@bp.route('/lab')
def lab_page():
    """Technical tools: retrieval quality review, chunk and parser views."""
    ensure_session()
    return render_template('lab.html', active_nav='lab')
