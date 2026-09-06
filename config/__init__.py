"""Configuration for the RAG console.

Six owners, one each, and one precedence rule:

    the real process environment  >  accepted values from ``.env``  >  the
    application default written on the setting's own dataclass field

``paths``    where runtime state lives (``CHAT_RAG_DATA_DIR`` and everything
             under it). State paths have one extra rule -- see
             :func:`config.paths.load_env_file`.
``runtime``  the server process: host, port, request threads, channel timeout.
``ingest``   the ingest path's workers, queue, deadlines and provider budgets.
``query``    the query path's admission, answer budget and deadlines.

``settings`` reads the rest (models, endpoints, retrieval, parsing) and holds
the three limit objects, validated at construction so a bad value stops the
process at start-up rather than at the first request.

The sixth owner is outside this package: ``utils/logger.py`` resolves
``LOG_LEVEL``, ``LOG_FILE_LEVEL`` and the rotation limits, and deliberately
falls back rather than refusing to start. ``docs/configuration.md`` says why.

The class, not an instance: importing ``config`` reads the ``.env`` file and
nothing else. A process builds its own :class:`Settings` when it is ready to
(``app.py`` does, and hands a per-knowledge-base copy to each pipeline), so a
tool that only wants ``config.paths`` does not pay for -- or fail on -- a full
configuration it never asked for.

``.env`` is applied **here**, once, before any config module reads anything.
It used to be applied by ``config.settings``, which meant a module that
imported ``config.paths`` or ``utils.logger`` first could read the environment
before the file had been applied to it. Package initialisation runs before any
submodule, so this is the one point where that cannot happen.
"""

import os

from . import paths

#: The developer's local file. A deployment sets real environment variables
#: instead; those always win (see ``paths.load_env_file``).
ENV_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")

#: What the file actually supplied, for diagnostics. Applied through
#: ``config.paths`` rather than ``dotenv`` directly, because a declared data
#: root owns the state paths and a ``.env`` left over from local development
#: must not move them.
APPLIED_ENV_FILE = paths.load_env_file(ENV_FILE)

from .settings import Settings  # noqa: E402  (after the .env is applied)

__all__ = ["Settings", "paths", "ENV_FILE", "APPLIED_ENV_FILE"]
