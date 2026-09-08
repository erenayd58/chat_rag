"""`/api/v1` -- the product contract.

The surface a client is meant to build against, and the one that has to keep
working while everything under it is replaced: Flask by FastAPI, files by
PostgreSQL, Chroma by pgvector, the templates by a Next.js front end.

It is not a renamed copy of the Flask-era surface. What is here is the product
-- knowledge bases, documents and their analyses, ingest jobs, questions and
searches, and enough discovery for a client to know what this deployment can
do. What is deliberately not here is listed in ``docs/api-v1.md``, with the
reason for each.

Every route reads its inputs, calls one use case in :mod:`application`, and
lets :mod:`interfaces.http.v1.envelope` and :mod:`interfaces.http.v1.resources`
turn the answer into the wire shape. No product decision is taken here, and
none is duplicated from the legacy adapter: both call the same use cases.
"""

from __future__ import annotations

from . import documents, ingest_jobs, knowledge_bases, meta, queries

#: The version prefix, in one place. Bumping the contract means adding a
#: package beside this one, not editing these routes.
PREFIX = "/api/v1"

BLUEPRINTS = (
    meta.bp,
    knowledge_bases.bp,
    documents.bp,
    ingest_jobs.bp,
    queries.bp,
)


def register(app) -> None:
    for blueprint in BLUEPRINTS:
        app.register_blueprint(blueprint, url_prefix=PREFIX)
