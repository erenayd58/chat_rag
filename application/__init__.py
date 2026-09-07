"""The product's behaviour, with no web framework under it.

Every use case here takes ordinary Python in and gives ordinary Python back:
a :class:`~application.services.Services` container, plain arguments, dicts
and dataclasses. Nothing imports Flask, reads a request or builds a response.
What refuses, refuses by raising one of :mod:`application.errors` or one of
the resource-control exceptions the limits subsystem owns; turning that into
a status code is the adapter's job.

The modules are the product's behaviour groups, not layers:

    knowledge_bases   a named collection, its chunker, its store
    documents         an ingest: its rows, its chunks, its deletion
    ingest            the upload decision and the job that carries it out
    chunks            chunk inspection and the Lab's bounded retrieval
    query             one question, under admission and a deadline
    workspace         the Viewer read model and the analysis lifecycle
    catalogue         what this deployment can offer: methods, models, profiles
    goldsets          the confirmed answers a knowledge base is evaluated on
    ops               what this process can do right now
"""

import os

# Before anything imports an ML library, which happens the moment
# ``application.services`` pulls in the pipeline. Several embedding models can
# be instantiated in one process -- one per knowledge base in the cache -- and
# without these OpenMP aborts the process rather than sharing its thread pool.
# Here rather than in an entrypoint because every way into the application --
# the Flask app, the CLI, a test, a future adapter -- imports this package
# first, and each of them owning its own copy is how one of them ends up
# without it.
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')
