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
    analysis_query    one question, through several chunkings of one document
    workspace         the Viewer read model and the analysis lifecycle
    catalogue         what this deployment can offer: methods, models, profiles
    goldsets          the confirmed answers a knowledge base is evaluated on
    ops               what this process can do right now

Importing this package does nothing but define those modules. It used to set
four OpenMP thread-pool variables first, on the way past, because everything
reaching a use case came through here -- which made importing a use case a
change to the process's environment. That decision belongs to whoever started
the process, so it moved to :func:`chat_rag.process.apply_thread_defaults`,
which the entry points call before they import this package
(``asgi.py``, ``python -m cli``, the smoke tools, the test session).
"""
