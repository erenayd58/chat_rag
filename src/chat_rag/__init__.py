"""The RAG engine: ingest a document, analyse it, search it, answer over it.

This package is the part of the product that is not a web server. It holds the
use cases (:mod:`chat_rag.application`), the pipeline they run
(:mod:`chat_rag.pipeline`), the components that pipeline is built from
(:mod:`chat_rag.components`), the configuration those read
(:mod:`chat_rag.config`) and the relational persistence underneath
(:mod:`chat_rag.storage`). Nothing here imports a web framework.

What runs *beside* it lives outside this package, in the repository root, and
is deliberately not shipped in the wheel: ``interfaces/`` is the FastAPI
adapter that serves ``/api/v1``, ``asgi.py`` is its entry point, ``cli/`` and
``tools/`` are operator programs, ``frontend/`` is the console. Each of them
imports this package; this package imports none of them.

Importing ``chat_rag`` itself costs nothing -- no submodule is pulled in here,
because several of them load an ML library on the way. Ask for the one you
want::

    from chat_rag.application import services

The public facade (``Engine``, ``EngineConfig``) is not here yet; until it
arrives the use cases in :mod:`chat_rag.application` are the Python API, and
they already take plain arguments and return plain data.
"""

__all__: list[str] = []
