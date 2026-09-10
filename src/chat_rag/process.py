"""Process-wide defaults an entry point sets, and a library must not.

One function, and it writes to ``os.environ``. That is exactly why it is here
rather than folded into an import: a package that mutates the environment when
it is imported has changed the behaviour of every other library in the process
before its caller had a chance to say anything.

It used to be four lines at the top of ``chat_rag/application/__init__.py``,
which ran the moment anything reached a use case. That worked because every
way into the application went through that package first -- and it meant that
importing a use case to read its docstring reconfigured OpenMP.

The entry points call it now: ``asgi.py``, ``python -m cli``, the smoke tools
and the test session. Each of them calls it *before* importing the application,
because that is what the setting requires (see below).
"""

from __future__ import annotations

import os

#: Thread-pool variables and the values this application wants.
#:
#: Several embedding models can be instantiated in one process -- one per
#: knowledge base in the pipeline cache -- and without these OpenMP aborts the
#: process rather than sharing its thread pool. Retrieval and ingest get their
#: concurrency from the request threads and the ingest workers, not from an
#: intra-op thread pool, so one thread each is what this application wants
#: from the numeric libraries underneath it.
THREAD_DEFAULTS = {
    "OMP_NUM_THREADS": "1",
    "TOKENIZERS_PARALLELISM": "false",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}


def apply_thread_defaults(env: dict | None = None) -> dict:
    """Set the thread-pool variables this process has not already been given.

    ``setdefault``, never an override: an operator who set ``OMP_NUM_THREADS``
    meant it, and a container that sized its own pools keeps them.

    **Call this before importing the application.** The libraries these
    variables configure read them when *they* are imported, and
    ``chat_rag.pipeline`` pulls in sentence-transformers -- and so torch --
    at module scope. Setting them afterwards changes nothing.

    Returns the values that were applied, so an entry point can report them.
    """
    target = os.environ if env is None else env
    applied = {}
    for name, value in THREAD_DEFAULTS.items():
        if not target.get(name):
            target[name] = value
            applied[name] = value
    return applied
