"""`/api/v1` -- the product contract, served by FastAPI.

The surface a client is meant to build against, and the one that has to keep
working while everything under it is replaced: files by PostgreSQL, Chroma by
pgvector, the templates by a Next.js front end. Flask has already been
replaced here -- these routes are FastAPI, over the same use cases the Flask
ones called, with the same URLs, statuses and bodies, held to that by
``tests/migration/test_api_v1_contract.py``.

It is not a renamed copy of the Flask-era surface. What is here is the product
-- knowledge bases, documents and their analyses, ingest jobs, questions and
searches, and enough discovery for a client to know what this deployment can
do. What is deliberately not here is listed in ``docs/api-v1.md``, with the
reason for each.

Four kinds of module, and nothing else:

``routers/``     one per product concept. Read inputs, call one use case,
                 return a schema.
``schemas/``     the API's own Pydantic types -- the boundary between what a
                 client sees and what a store holds.
``errors.py``    the one table that turns a refusal into a status code.
``application``  the FastAPI object, and the lifespan a deployment runs.

No product decision is taken in any of them, and none is duplicated from the
legacy adapter: both surfaces call the same use cases, over the same container.
"""

from __future__ import annotations

from .application import OPENAPI_PATH, PREFIX, create_app
from .envelope import DEFAULT_LIMIT, MAX_LIMIT

__all__ = ["DEFAULT_LIMIT", "MAX_LIMIT", "OPENAPI_PATH", "PREFIX", "create_app"]
