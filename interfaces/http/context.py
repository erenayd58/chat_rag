"""What a Flask request carries that the application does not want to know.

The session cookie holds one thing -- a per-browser id used to pick a cached
pipeline -- and the application takes it as an ordinary string. The container
is looked up from the running application rather than imported, so a test that
replaces a seam on it is honoured everywhere without any module holding its
own reference.
"""

from __future__ import annotations

import uuid

from flask import current_app, session

from application.services import Services

#: Where the container lives on the Flask application.
EXTENSION = 'chat_rag'


def services() -> Services:
    return current_app.extensions[EXTENSION]


def ensure_session() -> None:
    """Create the per-browser session id used to cache pipelines."""
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())


def session_id(default: str = 'global') -> str:
    return session.get('session_id', default)


def fresh_session_id() -> str:
    """The session id, or a new one for a caller that never took a page.

    Used by ``/api/query``, which is reachable by a client that never loaded a
    screen; a fresh id gives that client its own cache entry rather than
    sharing the one every anonymous caller would share.
    """
    return session.get('session_id', str(uuid.uuid4()))


def flag(value) -> bool:
    return (value or '').strip().lower() in {'1', 'true', 'yes', 'on'}
