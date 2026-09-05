"""Viewer packaging for the documents this console ingests.

Two modules, one job each:

* :mod:`~components.viewer.methods` -- this deployment's *view* of the
  library method registry (``amsc.methods``): availability, product order,
  the default selection. It registers nothing of its own.
* :mod:`~components.viewer.analysis` -- the packaging lifecycle and the
  state that goes with it: stage, queue, build, publish, recover.

See ``chunk/docs/viewer-architecture.md`` for how these reach the browser.
"""

from . import analysis

__all__ = ["analysis"]
