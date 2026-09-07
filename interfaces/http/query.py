"""HTTP for the one question the product answers."""

from __future__ import annotations

from flask import Blueprint, request

from application import query as use_case

from .context import fresh_session_id, services
from .responses import install, ok

bp = Blueprint('query', __name__)
install(bp)


@bp.route('/api/query', methods=['POST'])
def query():
    """Answer one question.

    The bounds this runs under -- admission, the answer budget and the
    deadline -- belong to the use case; what is here is the request body, and
    the three refusals the table in ``responses.py`` turns into **503**
    ``overloaded``, **503** ``generation_unavailable`` and **504**.
    """
    data = request.get_json(silent=True) or {}
    return ok(**use_case.answer(
        services(),
        question=data.get('question', ''),
        session_id=fresh_session_id(),
        kb_id=data.get('kb_id'),
        top_k=data.get('top_k', 5),
        temperature=data.get('temperature', 0.3),
        max_tokens=data.get('max_tokens', 500),
    ))
