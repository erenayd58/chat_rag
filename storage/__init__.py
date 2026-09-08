"""PostgreSQL: where this application's relational state lives.

Since Step 8, PostgreSQL is authoritative for every durable record this
product keeps *about* documents -- the knowledge bases, the ingest ledger, the
content identities and their analysis state, the ingest journal and the gold
set. The vector store is still authoritative for embeddings; Step 9 replaces
it with pgvector, and until then the two are separate on purpose.

What is deliberately *not* here: the artifacts of document processing. The
canonical units a parse produced, each chunking method's packaged
``chunks.jsonl``, a Deep Analysis run tree, the assembled Viewer payload, the
embedding caches and the logs are all still files. They are large, they are
regenerable from an ingest, nothing queries across them, and a row that
pointed at them would only be a second name for a path. What moved into the
database is the state that says which of them exist and what they mean.

Three modules:

``models``        the schema, and the reasons behind the edges that are not
                  obvious (``storage/models.py``)
``engine``        one pooled engine per process, one session per unit of work
``repositories``  every SQL statement, behind the persistence API the
                  application already had

Migrations live in ``storage/migrations`` and are run by Alembic. Nothing in
this package creates a table at import or at start-up.

One name to know about: ``storage.engine`` is the accessor re-exported below
(``storage.engine()`` returns the process's engine), which shadows the
submodule of the same name. Reach the module itself with
``from storage.engine import ...``.
"""

from .engine import (
    DatabaseNotConfigured, DatabaseUnavailable, describe, dispose, engine,
    health, pool_status, require_reachable, session_scope,
)
from .repositories import (
    ContentRepository, DocumentRepository, GoldSetRepository,
    IngestJobRepository, KnowledgeBaseRepository,
)

__all__ = [
    "ContentRepository",
    "DatabaseNotConfigured",
    "DatabaseUnavailable",
    "DocumentRepository",
    "GoldSetRepository",
    "IngestJobRepository",
    "KnowledgeBaseRepository",
    "describe",
    "dispose",
    "engine",
    "health",
    "pool_status",
    "require_reachable",
    "session_scope",
]
