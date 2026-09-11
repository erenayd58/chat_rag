"""The public Python API: what this engine offers a program rather than a browser.

    from chat_rag import Engine, EngineConfig

    with Engine(EngineConfig(database_url=...)) as engine:
        engine.migrate()                      # the schema, created or brought to head
        kb = engine.knowledge_bases.create("Reports")
        document = kb.ingest("report.pdf")
        document.analysis().request()
        hits = kb.search("liquidity")
        answer = kb.ask("What changed?")

Everything here delegates. The behaviour lives in
:mod:`chat_rag.application`, where it has lived since L1, and this package
adds three things and no fourth: an object that owns a container and a session,
an activation around every call so a second engine is a second engine, and
value types instead of dictionaries. A rule this facade never invents lives in
a use case; a refusal it raises is that use case's own.

Why it is a package of its own rather than functions on ``application``
----------------------------------------------------------------------

Because the use cases are the *inside* and this is the outside, and the two
have different obligations. A use case may take a ``Services``, a session id
and eight keyword arguments, and may return whatever dictionary suits the next
change; a published surface may not. Keeping them in separate modules is what
lets ``application`` keep changing shape while ``chat_rag.api`` keeps its word
-- and what makes the promise something a later step can enforce, because it
is a list of names in one place rather than a habit.

Nothing here imports a web framework. That is checked, not asserted, by
``tests/application/test_without_a_framework.py``, which reads this package
under the same rule it holds the use cases to.
"""

from __future__ import annotations

from chat_rag.application.errors import (
    ApplicationError, Conflict, InvalidRequest, NotFound, NotReady,
    ProcessingFailed, Unavailable,
)
from chat_rag.core.exceptions import (
    IngestInterrupted, IngestOverloaded, QueryOverloaded, QueryTimeout,
)

from chat_rag.config import Settings

from .config import EngineConfig
from .engine import Engine, migrate_database, open_engine
from .resources import Analysis, Document, IngestJob, KnowledgeBase, KnowledgeBases
from .results import Answer, Arm, Chunk, Comparison, Health, Hit, Method, Migration, Source

#: The published names. Two groups, and the second is why it is worth writing
#: this list out: a caller has to be able to *catch* what this library refuses
#: with, and the refusals are the application's own six meanings plus the four
#: resource-control exceptions the limits subsystems raise. Re-exporting them
#: here says they are part of the surface rather than an implementation detail
#: somebody found by reading a traceback.
__all__ = [
    # the engine and how it is configured
    "Engine",
    "EngineConfig",
    "Settings",
    "open_engine",
    "migrate_database",
    # the things it holds
    "KnowledgeBase",
    "KnowledgeBases",
    "Document",
    "IngestJob",
    "Analysis",
    # what its calls answer with
    "Answer",
    "Arm",
    "Chunk",
    "Comparison",
    "Health",
    "Hit",
    "Method",
    "Migration",
    "Source",
    # what its calls refuse with
    "ApplicationError",
    "InvalidRequest",
    "NotFound",
    "Conflict",
    "Unavailable",
    "NotReady",
    "ProcessingFailed",
    # and what the bounds refuse with, which is not the same thing: these mean
    # "not now", never "not like that"
    "IngestOverloaded",
    "IngestInterrupted",
    "QueryOverloaded",
    "QueryTimeout",
]
