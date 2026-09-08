"""The HTTP adapters. Two surfaces, one application underneath.

``v1`` is the product contract -- the surface a client should build against,
and the one a FastAPI port has to reproduce. ``legacy`` is the Flask-era
surface the current templates and the Viewer's relay still speak; it stays
until they do not, and then it is one directory to delete.

Neither holds product behaviour. Both read a request, call a use case in
:mod:`application`, and turn what comes back into their own wire shape -- so
the two can disagree about field names and status codes, as they do, without
either being a second implementation of anything.
"""

from __future__ import annotations

from . import legacy, v1
from .context import EXTENSION


def register(app, services) -> None:
    """Bind one application container to one Flask app, and mount both surfaces.

    The container is stored on the app rather than imported by the blueprints,
    so nothing in this package holds a reference of its own and a replaced
    seam is seen by every route at once.
    """
    app.extensions[EXTENSION] = services
    for blueprint in legacy.BLUEPRINTS:
        app.register_blueprint(blueprint)
    v1.register(app)
