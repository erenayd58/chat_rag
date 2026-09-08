"""The HTTP adapters. Two surfaces, one application underneath.

``v1`` is the product contract -- the surface a client should build against.
It is a FastAPI application: typed request and response models, one table that
turns a refusal into a status code, and a generated OpenAPI document. ``legacy``
is the Flask-era surface the current templates and the Viewer's relay still
speak; it stays until they do not, and then it is one directory to delete.

Neither holds product behaviour. Both read a request, call a use case in
:mod:`application`, and turn what comes back into their own wire shape -- so
the two can disagree about field names and status codes, as they do, without
either being a second implementation of anything, and without anything being
synchronised between them.

They are two frameworks for one step. :mod:`interfaces.http.coexistence`
mounts the ASGI application inside the Flask one so a single process serves
both over a single container; ``asgi.py`` is the same FastAPI application as
a standalone entrypoint, for the deployment that no longer needs the console's
screens.
"""

from __future__ import annotations

from . import coexistence, legacy, v1
from .context import EXTENSION


def register(app, services) -> None:
    """Bind one application container to one Flask app, and mount both surfaces.

    The container is stored on the app for the legacy blueprints and handed to
    the FastAPI factory for the contract routes -- one object either way, so a
    test that replaces a seam on it is honoured by every route on both
    surfaces at once.
    """
    app.extensions[EXTENSION] = services
    for blueprint in legacy.BLUEPRINTS:
        app.register_blueprint(blueprint)
    coexistence.mount(app, v1.create_app(services))
