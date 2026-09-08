"""`/api/v1` -- the product contract, served by FastAPI.

The surface a client is meant to build against, and the one that has to keep
working while everything under it is replaced: files by PostgreSQL, Chroma by
pgvector, the console's own templates by a Next.js front end. All three have
happened underneath it, and the Flask-era surface it was written beside is
gone; what a client sees is held still by
``tests/migration/test_api_v1_contract.py``.

It was never a renamed copy of that surface. What is here is the product
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

No product decision is taken in any of them: a router calls a use case in
:mod:`application`, over the container the process composed, and nothing here
holds behaviour of its own.
"""

from __future__ import annotations

from .application import OPENAPI_PATH, PREFIX, create_app
from .envelope import DEFAULT_LIMIT, MAX_LIMIT

__all__ = ["DEFAULT_LIMIT", "MAX_LIMIT", "OPENAPI_PATH", "PREFIX", "create_app"]
