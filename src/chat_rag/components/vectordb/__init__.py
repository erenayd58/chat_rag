"""The document store: chunk text, metadata and embeddings.

One implementation ships, and it is PostgreSQL with pgvector. What a store has
to do is stated outside this package, in
``tests/migration/test_document_store_contract.py``: the abstract base class
below declares six methods and the product calls twelve, so the contract --
run against the shipped store and against a dependency-free reference written
to it -- is what a replacement is measured by, not this module.
"""
from .base import BaseVectorDB
from .pgvector_store import PgVectorStore

__all__ = ['BaseVectorDB', 'PgVectorStore']
