"""Serving an ASGI application from inside the WSGI process, for one step.

`/api/v1` is FastAPI now and the console's screens, its JavaScript and the
Viewer's relay are still Flask. Both have to run, in one process, over **one**
application container -- that is the whole requirement, and it rules out the
obvious answers. Two processes would need the state that lives in module
globals to be shared (the packaging queue, the pipeline cache, the embedded
vector store); two containers would need the two surfaces to be kept in step,
which is a synchronisation layer and is exactly what a migration must not
grow.

So the FastAPI application is *mounted inside* the Flask one:

    waitress -> Flask -> this bridge -> FastAPI -> application -> stores
                   \\-> the legacy blueprints -^

Every `/api/v1` path is registered on the Flask routing table with a view that
forwards the request into the ASGI application and returns what came back.
The routes are read from FastAPI's own table rather than written out again, so
there is one list of what `/api/v1` serves and Flask's table is derived from
it; ``tests/migration/test_http_surface.py`` reads that table and holds it
against ``docs/api-v1.md``.

Two things are carried across the boundary and nothing else. The request
itself -- method, path, query, headers, body -- and the console's session id,
which selects a cached pipeline: a browser that has both surfaces open gets
one pipeline rather than two. Nothing is copied, cached or translated: the
FastAPI side reads the same ``Services`` object the Flask side was given.

**This module is temporary.** When the console's screens are served by
something that speaks to `/api/v1` directly, the FastAPI application in
``asgi.py`` becomes the process and this file is deleted. Nothing else in
either surface knows it exists.
"""

from __future__ import annotations

import asyncio
import atexit
import logging
import threading
from typing import Any, Iterable, Mapping, Optional

from flask import Flask, Response, request, session
from starlette.routing import Route

logger = logging.getLogger("RAG.http.coexistence")

#: Flask's own converters do not appear in FastAPI paths, so the translation
#: is one substitution: ``{name}`` becomes ``<name>``.
_OPEN, _CLOSE = "{", "}"

#: Headers a WSGI server sets from the connection and an ASGI response must
#: not try to set again through it.
_HOP_BY_HOP = frozenset({
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailers", "transfer-encoding", "upgrade", "content-length",
})


class _Loop:
    """One event loop, on one daemon thread, for the life of the process.

    A request thread hands a coroutine to it and blocks until the answer comes
    back. The thread is a daemon and the loop is ours, so the process can exit
    without waiting for either -- which a pooled executor would not allow, and
    which matters because this loop is started lazily by whichever request
    arrives first.

    FastAPI's own concurrency is unaffected: every handler on this surface is
    a synchronous ``def``, so Starlette runs it in its thread pool rather than
    on this loop, and two requests are still served at once.
    """

    def __init__(self) -> None:
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._run, name="api-v1-asgi", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def call(self, coroutine):
        return asyncio.run_coroutine_threadsafe(coroutine, self._loop).result()

    def close(self) -> None:
        self._loop.call_soon_threadsafe(self._loop.stop)


class Bridge:
    """One ASGI application, callable from a WSGI request thread."""

    def __init__(self, app) -> None:
        self._app = app
        self._loop: Optional[_Loop] = None
        self._lifespan = None
        self._lock = threading.Lock()

    # ----------------------------------------------------------- lifecycle
    def _started(self) -> _Loop:
        """The loop, started with the application's lifespan, once.

        Lazily, because a module import must not start a thread, and under a
        lock, because the first two requests can arrive together.
        """
        if self._loop is None:
            with self._lock:
                if self._loop is None:
                    loop = _Loop()
                    self._lifespan = self._app.router.lifespan_context(self._app)
                    loop.call(self._lifespan.__aenter__())
                    atexit.register(self._stop)
                    self._loop = loop
        return self._loop

    def _stop(self) -> None:  # pragma: no cover - process exit
        loop, lifespan = self._loop, self._lifespan
        self._loop, self._lifespan = None, None
        if loop is None:
            return
        try:
            if lifespan is not None:
                loop.call(lifespan.__aexit__(None, None, None))
        except Exception as error:  # noqa: BLE001 - never fail an exit on this
            logger.warning(f"api v1 shutdown raised: {error}")
        finally:
            loop.close()

    # ------------------------------------------------------------- request
    def respond(self, environ: Mapping[str, Any], state: Mapping[str, Any]) -> Response:
        """Serve one WSGI request through the ASGI application."""
        scope = _scope(environ, state)
        body = _body(environ)
        started: dict[str, Any] = {}
        chunks: list[bytes] = []

        async def receive() -> dict:
            return {"type": "http.request", "body": body, "more_body": False}

        async def send(message: dict) -> None:
            if message["type"] == "http.response.start":
                started["status"] = message["status"]
                started["headers"] = message.get("headers") or []
            elif message["type"] == "http.response.body":
                chunks.append(message.get("body") or b"")

        try:
            self._started().call(self._app(scope, receive, send))
        except Exception:
            # Starlette's outermost middleware answers, then re-raises so a
            # server can log the traceback. It has already been logged by the
            # handler in ``interfaces.http.v1.errors``; if it answered, that
            # answer is the response.
            if "status" not in started:
                raise
            logger.debug("api v1 answered and re-raised", exc_info=True)

        return Response(
            b"".join(chunks),
            status=started.get("status", 500),
            headers=[(name.decode("latin-1"), value.decode("latin-1"))
                     for name, value in started.get("headers", [])
                     if name.decode("latin-1").lower() not in _HOP_BY_HOP],
        )


def _scope(environ: Mapping[str, Any], state: Mapping[str, Any]) -> dict:
    path = (environ.get("SCRIPT_NAME", "") or "") + (environ.get("PATH_INFO", "") or "")
    return {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": (environ.get("SERVER_PROTOCOL") or "HTTP/1.1").split("/")[-1],
        "method": (environ.get("REQUEST_METHOD") or "GET").upper(),
        "scheme": environ.get("wsgi.url_scheme", "http"),
        "path": path,
        "raw_path": path.encode("utf-8"),
        "query_string": (environ.get("QUERY_STRING") or "").encode("latin-1"),
        "root_path": "",
        "headers": list(_headers(environ)),
        "client": (environ.get("REMOTE_ADDR"), 0),
        "server": (environ.get("SERVER_NAME"), _port(environ)),
        # Starlette merges this into ``request.state``; it is how the console's
        # session reaches a surface that has no cookies of its own.
        "state": dict(state),
    }


def _headers(environ: Mapping[str, Any]) -> Iterable[tuple[bytes, bytes]]:
    for key, value in environ.items():
        if key.startswith("HTTP_"):
            yield (key[5:].lower().replace("_", "-").encode("latin-1"),
                   str(value).encode("latin-1"))
    for key, name in (("CONTENT_TYPE", b"content-type"),
                      ("CONTENT_LENGTH", b"content-length")):
        if environ.get(key):
            yield name, str(environ[key]).encode("latin-1")


def _port(environ: Mapping[str, Any]) -> int:
    try:
        return int(environ.get("SERVER_PORT") or 0)
    except (TypeError, ValueError):
        return 0


def _body(environ: Mapping[str, Any]) -> bytes:
    stream = environ.get("wsgi.input")
    if stream is None:
        return b""
    try:
        length = int(environ.get("CONTENT_LENGTH") or 0)
    except (TypeError, ValueError):
        length = 0
    return stream.read(length) if length > 0 else b""


def served_routes(routes, prefix: str = ""):
    """Every concrete route an ASGI router table serves, with its full path.

    FastAPI does not necessarily flatten an included router into the
    application's route list -- recent versions keep the router itself in the
    table and match through it -- so this walks whatever nesting it used and
    yields the leaves. Written against the shapes rather than against one
    version: a router that carries its own ``routes`` (a mount, an older
    ``include_router``) and one that carries the router it included are both
    handled, and anything that is neither is skipped.
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


def rule_for(path: str) -> str:
    """A FastAPI path as a Werkzeug rule: ``{kb_id}`` becomes ``<kb_id>``."""
    return path.replace(_OPEN, "<").replace(_CLOSE, ">")


def mount(flask_app: Flask, api, *, session_key: str = "session_id") -> None:
    """Register every route the ASGI application serves on the Flask table.

    The list comes from FastAPI's own routing table, so the two cannot
    disagree about what is served -- which is what lets
    ``tests/migration/test_http_surface.py`` keep reading one table and still
    be checking the whole API.
    """
    bridge = Bridge(api)

    def view(**_values):
        # The only thing taken from the Flask request beyond the raw environ.
        # Read, never written: this surface does not start sessions.
        return bridge.respond(request.environ, {session_key: session.get(session_key, "")})

    for path, methods, name in served_routes(api.routes):
        flask_app.add_url_rule(
            rule_for(path),
            endpoint=f"api_v1:{name}",
            view_func=view,
            methods=methods,
            # Werkzeug would otherwise redirect ``/api/v1/documents/`` to the
            # rule without the slash and change a POST into a GET.
            strict_slashes=True,
        )
