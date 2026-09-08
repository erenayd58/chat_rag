"""The Flask-era HTTP surface, kept while the console still speaks it.

This is the compatibility adapter, not the contract. Its shapes are what the
current templates and the Viewer's relay were written against: a `success`
envelope, `kb_id` / `doc_id` field names, a synchronous upload, and a route
layout that grew one endpoint at a time. `/api/v1` (`..v1`) is the surface a
new client should use, and this package is the one to delete when nothing
speaks the old one any more.

Both adapters call the same use cases. Nothing here holds product behaviour.
"""

from __future__ import annotations

from . import (
    catalogue, chunks, documents, goldsets, ingest, knowledge_bases, ops, pages, query,
    viewer,
)

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
