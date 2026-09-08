"""The FastAPI application that serves `/api/v1`.

One factory, called with the application container this process composed. The
container is put on ``app.state`` rather than imported by anything under this
package, so the same routers serve a test's container, the console process's
container and a future standalone deployment's, and a test that replaces a
seam on it is honoured by every route at once.

What is here is what belongs to *this framework*: the application object, the
routers it mounts, the exception table it installs, and the lifespan. The
product's behaviour is in :mod:`application`; the wire shapes are in
:mod:`interfaces.http.v1.schemas`. Neither imports this module, which is what
lets the same contract be served by something other than FastAPI later --
exactly as it was served by Flask until this step.

**The OpenAPI document is generated, not maintained.** ``/api/v1/openapi.json``
is produced from these routers and their schemas, so it cannot drift from what
is served. The interactive documentation pages are deliberately off: they load
their JavaScript from a CDN, and nothing else this product serves needs the
network to render. Point any OpenAPI viewer at the document instead.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any, Callable, Optional

from fastapi import FastAPI

from application.services import Services

from . import errors
from .routers import ROUTERS

logger = logging.getLogger("RAG.api.v1")

#: The version prefix, in one place. Bumping the contract means adding a
#: package beside this one, not editing these routes.
PREFIX = "/api/v1"
#: Where the generated contract is served.
OPENAPI_PATH = f"{PREFIX}/openapi.json"

TITLE = "chat_rag"
SUMMARY = "Knowledge bases, documents and their analyses, ingest jobs, questions and searches."
VERSION = "1"

#: What the tags mean, so the generated document groups itself the way the
#: contract document does.
TAGS = [
    {"name": "meta", "description": "what this deployment can offer"},
    {"name": "health", "description": "liveness, readiness and one line of capacity"},
    {"name": "knowledge bases", "description": "the collection a document is ingested into"},
    {"name": "documents", "description": "an upload and its corpus"},
    {"name": "analysis", "description": "a document's chunking variants"},
    {"name": "ingest jobs", "description": "what became of a submitted upload"},
    {"name": "queries", "description": "asking, and looking"},
]

#: Every refusal this surface can make, published on every operation so a
#: generated client has the taxonomy rather than only the happy path.
REFUSAL_RESPONSES: dict[int | str, dict[str, Any]] = {
    400: {"description": "invalid_request -- the request or its payload is wrong"},
    404: {"description": "not_found -- the resource is not here"},
    409: {"description": "not_ready or conflict -- it exists and its state refuses this"},
    500: {"description": "internal -- the server failed"},
    503: {"description": "unavailable or overloaded -- capacity, or a capability, is gone"},
    504: {"description": "timeout -- the deadline passed"},
}


def _lifespan(services: Services,
              on_start: Optional[Callable[[Services], Any]],
              on_stop: Optional[Callable[[Services], Any]]):
    """Start-up and shut-down, run once per application.

    Both hooks are what a *standalone* deployment passes: settling the
    previous process's ingest jobs, resuming interrupted analyses and sweeping
    staged uploads on the way in, stopping the workers on the way out. They
    are deliberately absent when this application is mounted inside the Flask
    console, because that process has already done all of it and owns the
    container -- running the recovery twice would resume every unfinished
    document a second time, and closing the workers would close them under the
    legacy surface.

    Nothing is logged on the way out. Shutdown runs while the process is
    tearing down, which is exactly when a stream a handler was writing to may
    already be closed.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if on_start is not None:
            on_start(services)
        logger.info("api v1 ready")
        yield
        if on_stop is not None:
            on_stop(services)

    return lifespan


#: FastAPI documents a **422** on every operation that reads a body, a form or
#: a query parameter -- its own answer to a payload it could not validate.
#: This surface does not send one: a payload it cannot read is a **400**
#: ``invalid_request``, which is the taxonomy every other refusal on it uses
#: and the one a client branches on. A generated document that advertised a
#: status the server never returns would be wrong in the direction that costs
#: a client the most, so it is taken back out along with the two schemas that
#: existed only to describe it.
UNSENT = ("422",)
UNSENT_SCHEMAS = ("HTTPValidationError", "ValidationError")


def _publish_only_what_is_answered(app: FastAPI) -> None:
    """Generate the document, then remove the answers this surface cannot give.

    A correction to the generation, not a second document: everything in the
    schema still comes from the routers and their models, and nothing is
    written down twice.
    """
    generated = app.openapi

    def openapi() -> dict[str, Any]:
        schema = generated()
        for operations in schema.get("paths", {}).values():
            for operation in operations.values():
                for status in UNSENT:
                    operation.get("responses", {}).pop(status, None)
        for name in UNSENT_SCHEMAS:
            schema.get("components", {}).get("schemas", {}).pop(name, None)
        return schema

    app.openapi = openapi


def create_app(services: Services, *,
               on_start: Optional[Callable[[Services], Any]] = None,
               on_stop: Optional[Callable[[Services], Any]] = None) -> FastAPI:
    """One application, over one container."""
    app = FastAPI(
        title=TITLE,
        summary=SUMMARY,
        version=VERSION,
        openapi_tags=TAGS,
        openapi_url=OPENAPI_PATH,
        # Off on purpose: see the module docstring.
        docs_url=None,
        redoc_url=None,
        swagger_ui_oauth2_redirect_url=None,
        lifespan=_lifespan(services, on_start, on_stop),
        responses=REFUSAL_RESPONSES,
    )
    app.state.services = services
    errors.install(app)
    _publish_only_what_is_answered(app)
    for router in ROUTERS:
        app.include_router(router, prefix=PREFIX)
    return app
