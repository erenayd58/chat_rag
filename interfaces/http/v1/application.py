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
is served; what a generated document cannot infer by itself -- the shape of a
refusal, the two headers, the status this surface never sends -- is in
:mod:`interfaces.http.v1.openapi`. The interactive documentation pages are
deliberately off: they load their JavaScript from a CDN, and nothing else this
product serves needs the network to render. Point any OpenAPI viewer at the
document instead.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any, Callable, Optional

from fastapi import FastAPI

from application.services import Services

from . import errors, openapi
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
        responses=openapi.REFUSAL_RESPONSES,
    )
    app.state.services = services
    errors.install(app)
    openapi.install(app)
    for router in ROUTERS:
        app.include_router(router, prefix=PREFIX)
    return app
