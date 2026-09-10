"""The operator surface: what this process has been doing, in detail.

One route, and it is deliberately **not** on `/api/v1`. The contract is what a
client builds against and is versioned for that reason; this is what a person
reads when :func:`application.ops.health` has told them to look closer, and its
contents are free to change with the internals it reports on -- counters, stage
latencies, cache occupancy, the effective configuration. Promoting it would
mean promising that shape, which is the opposite of what it is for.

It kept its path and its body through the removal of the Flask surface that
used to serve it (``docs/legacy-removal.md``, wave 3): an operator's script or
dashboard polling ``/api/ops/metrics`` sees no difference, which is the whole
reason the route was not simply deleted with the rest.

``application.ops`` holds the read model. Nothing here decides anything.
"""

from __future__ import annotations

from typing import Annotated, Optional

from fastapi import APIRouter, Query

from chat_rag.application import ops as use_case

from .v1.dependencies import Container

#: Off the generated contract on purpose: ``/api/v1/openapi.json`` is the list
#: of what a client may build against, and this is not on it.
router = APIRouter(include_in_schema=False)


#: How many individual job traces come back when nobody says.
DEFAULT_RECENT = 10


@router.get("/api/ops/metrics")
def metrics(services: Container,
            recent: Annotated[Optional[str], Query(
                description="how many individual job traces to include")] = None) -> dict:
    """``?recent=N`` sets how many individual job traces come back.

    ``recent`` is read forgivingly -- a value that is not a number falls back
    to the default rather than refusing the request. That is deliberate and it
    is the opposite of what the contract does with a bad parameter: this is
    the endpoint somebody curls at three in the morning, and refusing to
    report anything because a shell mangled a query string would be the wrong
    answer to give them.

    The ``success`` field is the shape this endpoint has always answered with
    and is kept for the scripts that read it; it is not the contract's
    envelope, which this route deliberately does not use.
    """
    try:
        wanted = int(recent) if recent is not None and recent.strip() else DEFAULT_RECENT
    except ValueError:
        wanted = DEFAULT_RECENT
    return {"success": True, **use_case.metrics(services, recent=max(0, wanted))}
