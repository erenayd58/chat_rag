#!/bin/sh
# What happens between "the container started" and "the server serves".
#
# One step, and then out of the way: bring the schema to head, then `exec` the
# command the image was given. `exec` matters -- it makes the server PID 1, so
# the SIGTERM `docker stop` sends reaches uvicorn itself. A shell that stayed
# in front of it would take the signal and leave the server to be killed at the
# end of the grace period, which is exactly the graceful shutdown this
# deployment has (draining ingest jobs, then returning the database pool).
#
# The migration step is `tools/migrate.py`, not a bare `alembic upgrade head`:
# it waits for a database that is accepting connections but still recovering,
# it holds an advisory lock so two containers starting together cannot both
# run the same migration, and it prints the revision it moved from and to.
# CHAT_RAG_MIGRATE_ON_START=0 turns it off for a deployment that applies its
# schema as a separate reviewed step.
set -e

python -m tools.migrate

exec "$@"
