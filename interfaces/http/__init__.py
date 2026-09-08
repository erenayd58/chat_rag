"""The HTTP adapter. One surface, one application underneath.

``v1`` is the product contract -- the surface a client builds against. It is a
FastAPI application: typed request and response models, one table that turns a
refusal into a status code, and a generated OpenAPI document. Beside it, off
the contract on purpose, is one operator route (``operator``): the detail an
operator reads once ``GET /api/v1/health`` has told them to look closer.

There used to be a second, Flask-era surface here -- the console's own
screens, their JavaScript and the Viewer's relay over ``/api/demo/*`` -- and a
bridge that mounted this application inside that one so a single process could
serve both. The console is a Next.js application over ``/api/v1`` and the
Viewer is one of its screens, so all three are gone (``docs/legacy-removal.md``).
What is left is what that plan said would be left: this package, and
``asgi.py`` as the entrypoint.

Nothing here holds product behaviour. A router reads a request, calls a use
case in :mod:`application`, and turns what comes back into a wire shape.
"""

from __future__ import annotations

from typing import Any, Callable, Iterator, Optional

from starlette.routing import Route

from application.services import Services

from . import operator, v1

#: The documents spell a path parameter ``<kb_id>``; FastAPI spells it
#: ``{kb_id}``. One substitution, in one place, so a test comparing the two
#: does not carry its own.
def documented(path: str) -> str:
    return path.replace("{", "<").replace("}", ">")


def served_routes(routes, prefix: str = "") -> Iterator[tuple[str, list[str], str]]:
    """Every concrete route a router table serves, with its full path.

    FastAPI does not necessarily flatten an included router into the
    application's route list -- recent versions keep the router itself in the
    table and match through it -- so this walks whatever nesting it used and
    yields the leaves. Written against the shapes rather than against one
    version: a router that carries its own ``routes`` (a mount, an older
    ``include_router``) and one that carries the router it included are both
    handled, and anything that is neither is skipped.

    It is here rather than in a test because it is the answer to "what does
    this application serve", and two suites and one document depend on that
    being derived from the table rather than written out again.
    """
    for route in routes:
        included = getattr(route, "original_router", None)
        if included is not None:
            context = getattr(route, "include_context", None)
            yield from served_routes(
                included.routes, prefix + (getattr(context, "prefix", "") or ""))
        elif isinstance(route, Route):
            yield prefix + route.path, sorted(route.methods or {"GET"}), route.name
        elif hasattr(route, "routes"):
            yield from served_routes(route.routes, prefix + getattr(route, "path", ""))


def surface(app) -> set[tuple[str, str]]:
    """``{(verb, path)}`` for everything an application serves, as the
    documents spell it. ``HEAD`` is Starlette's own answer to a ``GET`` and is
    not part of what anybody published."""
    return {(method, documented(path))
            for path, methods, _ in served_routes(app.routes)
            for method in methods if method != "HEAD"}


def create_app(services: Services, *,
               on_start: Optional[Callable[[Services], Any]] = None,
               on_stop: Optional[Callable[[Services], Any]] = None):
    """The application this process serves: the contract, and the operator route.

    One container, handed in rather than imported, so a test that replaces a
    seam on it is honoured by every route at once.
    """
    app = v1.create_app(services, on_start=on_start, on_stop=on_stop)
    app.include_router(operator.router)
    return app


__all__ = ["create_app", "documented", "operator", "served_routes", "surface", "v1"]
