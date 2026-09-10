"""The RAG engine: ingest a document, analyse it, search it, answer over it.

This package is the part of the product that is not a web server. It holds the
public API (:mod:`chat_rag.api`), the use cases underneath it
(:mod:`chat_rag.application`), the pipeline they run (:mod:`chat_rag.pipeline`),
the components that pipeline is built from (:mod:`chat_rag.components`), the
configuration those read (:mod:`chat_rag.config`) and the relational
persistence underneath (:mod:`chat_rag.storage`). Nothing here imports a web
framework.

What runs *beside* it lives outside this package, in the repository root, and
is deliberately not shipped in the wheel: ``interfaces/`` is the FastAPI
adapter that serves ``/api/v1``, ``asgi.py`` is its entry point, ``cli/`` and
``tools/`` are operator programs, ``frontend/`` is the console. Each of them
imports this package; this package imports none of them.

The public API::

    from chat_rag import Engine, EngineConfig

    with Engine(EngineConfig(database_url=...)) as engine:
        kb = engine.knowledge_bases.create("Reports")
        document = kb.ingest("report.pdf")
        answer = kb.ask("What changed?")

Importing ``chat_rag`` itself still costs nothing. The names above are
resolved on first *use* rather than pulled in here (:func:`__getattr__`,
PEP 562), because reaching :mod:`chat_rag.api` reaches the pipeline and so
loads sentence-transformers and torch -- which a program that imported this
module to read a version number should not pay for. ``from chat_rag import
Engine`` works exactly as if it were imported at the top; it simply happens
when it is asked for.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # for a type checker and an IDE, never at run time
    from chat_rag.api import (
        Analysis, Answer, ApplicationError, Arm, Chunk, Comparison, Conflict,
        Document, Engine, EngineConfig, Health, Hit, IngestInterrupted,
        IngestJob, IngestOverloaded, InvalidRequest, KnowledgeBase,
        KnowledgeBases, Method, NotFound, NotReady, ProcessingFailed,
        QueryOverloaded, QueryTimeout, Settings, Source, Unavailable,
        open_engine,
    )

#: The public API, re-exported from :mod:`chat_rag.api`. One list rather than
#: two: it is read by :func:`__getattr__` below, so a name published there and
#: forgotten here is a name that does not resolve -- which a test can see.
__all__ = [
    "Engine",
    "EngineConfig",
    "Settings",
    "open_engine",
    "KnowledgeBase",
    "KnowledgeBases",
    "Document",
    "IngestJob",
    "Analysis",
    "Answer",
    "Arm",
    "Chunk",
    "Comparison",
    "Health",
    "Hit",
    "Method",
    "Source",
    "ApplicationError",
    "InvalidRequest",
    "NotFound",
    "Conflict",
    "Unavailable",
    "NotReady",
    "ProcessingFailed",
    "IngestOverloaded",
    "IngestInterrupted",
    "QueryOverloaded",
    "QueryTimeout",
]


def _version() -> str:
    """The installed distribution's version, or the placeholder for a checkout.

    Read from the metadata rather than written here, so there is one place a
    release number lives (``pyproject.toml``) and no second one to forget.
    A source tree that was never installed has no metadata and says so, which
    is more honest than reporting a version nobody published.
    """
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("chat-rag")
    except PackageNotFoundError:  # pragma: no cover - an uninstalled checkout
        return "0.0.0.dev0"


def __getattr__(name: str):
    """Resolve a public name by importing the facade, once, on first use."""
    if name == "__version__":
        value = globals()["__version__"] = _version()
        return value
    if name in __all__:
        import chat_rag.api as api

        value = getattr(api, name)
        # Cached in this module's namespace, so the import above happens once
        # and every later lookup is an ordinary attribute read.
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """``dir(chat_rag)`` should show the public API without importing it."""
    return sorted({*globals(), *__all__, "__version__"})
