"""Command-line entry points for the retrieval proof of concept.

This package is a *process*, so the two decisions a process makes are made
here, before any submodule imports the application:

* the thread-pool defaults, which the numeric libraries under
  ``chat_rag.pipeline`` read when they themselves are imported;
* the log handlers, because ``chat_rag`` installs none of its own.

Both used to happen by accident -- the first at the top of
``chat_rag/application/__init__.py``, the second when ``chat_rag.utils.logger``
was first imported. ``python -m cli`` and ``import cli.runtime`` both come
through this file, so both still get them, and now by saying so.
"""

from chat_rag.process import apply_thread_defaults

apply_thread_defaults()

from chat_rag.utils.logger import configure_logging  # noqa: E402

configure_logging()
