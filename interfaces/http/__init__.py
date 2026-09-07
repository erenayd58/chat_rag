"""The Flask adapter: nine blueprints, one per behaviour group.

Each module here does four things and no more -- read the path, query and
body, call one use case, and let :mod:`interfaces.http.responses` turn the
answer or the refusal into a response. The behaviour groups are the same ones
:mod:`application` is divided into, and the names match, so "where does this
endpoint's decision live" has an answer you can read off the tree.

``pages`` is the exception: it renders the console's own screens and is not
part of the API contract.
"""

from __future__ import annotations

from . import (
    catalogue, chunks, documents, goldsets, ingest, knowledge_bases, ops, pages, query,
    viewer,
)
from .context import EXTENSION

#: Registration order is display order in the routing table and nothing else;
#: no two blueprints claim the same rule.
BLUEPRINTS = (
    pages.bp,
    knowledge_bases.bp,
    documents.bp,
    chunks.bp,
    query.bp,
    ingest.bp,
    viewer.bp,
    goldsets.bp,
    catalogue.bp,
    ops.bp,
)


def register(app, services) -> None:
    """Bind one application container to one Flask app.

    The container is stored on the app rather than imported by the blueprints,
    so nothing in this package holds a reference of its own and a replaced
    seam is seen by every route at once.
    """
    app.extensions[EXTENSION] = services
    for blueprint in BLUEPRINTS:
        app.register_blueprint(blueprint)
