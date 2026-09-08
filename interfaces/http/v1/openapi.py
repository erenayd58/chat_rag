"""What the generated document says that the models cannot say themselves.

`/api/v1/openapi.json` is generated from the routers and their schemas, and
that is the point of it: there is no second document to keep in step. But a
generated document is only as complete as what the framework can infer, and
three things it cannot infer are part of this contract:

* **what a refusal looks like.** FastAPI knows a handler's success model; it
  does not know that every operation here can answer
  ``{"error": {"type", "message", "details"}}``, because that body is produced
  by the exception handlers in :mod:`interfaces.http.v1.errors` and never
  returned by a route. Declared without it, a generated client types the happy
  path and leaves the taxonomy a client actually branches on as an untyped
  blob.
* **the headers two answers carry.** A created knowledge base and an accepted
  upload both say where to look next in ``Location``, and an overload refusal
  says how long to wait in ``Retry-After``. A client generated from a document
  that does not mention them has to know to look anyway.
* **the answer this surface never gives.** FastAPI documents a **422** on
  every operation that reads a body, a form or a query parameter. This one
  does not send it -- a payload it cannot read is **400** ``invalid_request``,
  which is the taxonomy every other refusal uses -- and a document advertising
  a status the server never returns is wrong in the direction that costs a
  client the most.

None of this is a second source of truth. The refusal *body* is the same
:class:`~interfaces.http.v1.schemas.ErrorResponse` the handlers build, the
statuses are the ones :data:`interfaces.http.v1.errors.REFUSALS` maps to, and
the removal below is a correction applied to the generated document rather
than a document written by hand.
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI

from .schemas import ErrorResponse

#: What each refusal status means, published on **every** operation so a
#: generated client has the whole taxonomy rather than only the happy path.
#: The body is the one the error handlers build; ``type`` inside it is what a
#: client branches on, and ``docs/api-v1.md`` maps each name to its status.
REFUSAL_RESPONSES: dict[int | str, dict[str, Any]] = {
    400: {"description": "invalid_request -- the request or its payload is wrong",
          "model": ErrorResponse},
    404: {"description": "not_found -- the resource is not here",
          "model": ErrorResponse},
    409: {"description": "not_ready or conflict -- it exists and its state refuses this",
          "model": ErrorResponse},
    500: {"description": "internal -- the server failed",
          "model": ErrorResponse},
    503: {"description": "unavailable or overloaded -- capacity, or a capability, is gone",
          "model": ErrorResponse,
          "headers": {
              "Retry-After": {
                  "description": "seconds to wait before retrying; sent with an "
                                 "``overloaded`` refusal, which is never queued",
                  "schema": {"type": "integer"},
              },
          }},
    504: {"description": "timeout -- the deadline passed",
          "model": ErrorResponse},
}

#: The header a created or accepted resource answers with. Declared on the
#: operation that sends it rather than globally, because it is the answer to
#: "where do I look next" and only two answers have one.
LOCATION_HEADER = {
    "Location": {
        "description": "the path of the resource this call created or accepted",
        "schema": {"type": "string"},
    },
}

#: FastAPI's own answer to a payload it could not validate. This surface does
#: not send it; see the module docstring.
UNSENT = ("422",)
UNSENT_SCHEMAS = ("HTTPValidationError", "ValidationError")


def install(app: FastAPI) -> None:
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
