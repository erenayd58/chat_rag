"""One router per product concept, and nothing that is not a router.

Split by the resource a client asks about rather than by the module a use case
lives in, so a reader looking for "what happens when a document is deleted"
opens ``documents.py`` and finds one function. There is no monolithic router
and there is no ``misc``: a route that fits none of these is a resource nobody
has named yet.

Every handler here reads its inputs, calls **one** use case in
:mod:`application`, and returns a schema. It does not catch exceptions
(``..errors`` holds the whole table), it does not know a status code beyond
the one its decorator declares, and it takes no product decision -- every one
of those belongs to the use case it calls.
"""

from __future__ import annotations

from . import documents, ingest_jobs, knowledge_bases, meta, queries

#: Mount order is the order they appear in the OpenAPI document and nothing
#: else; no two routers claim the same path.
ROUTERS = (
    meta.router,
    knowledge_bases.router,
    documents.router,
    ingest_jobs.router,
    queries.router,
)

__all__ = ["ROUTERS", "documents", "ingest_jobs", "knowledge_bases", "meta", "queries"]
