"""What a router is given, and where it comes from.

Three things, and no fourth: the application container, the caller's session
id, and a page request. None of them is a product decision -- they are the
wiring a transport has to do before a use case can be called, kept here so
that a router reads as "take these, call one use case, return the schema".

The container is read from the running application rather than imported, so a
test that replaces a seam on it is honoured by every route at once and no
module in this package holds a reference of its own. It is the one object the
process composed, handed in by ``interfaces.http.create_app``.

The session id selects a cached pipeline and nothing else -- it is not
identity, it is not authorisation, and this surface neither sets nor reads
one. Every caller of a knowledge base shares its one pipeline: the store
handle and the lexical index are the knowledge base's, not the caller's, and
a pipeline built per caller was a pipeline built per request once the Flask
cookie that used to name a browser went (Step 13) -- every question and every
search read the whole corpus and rebuilt its index. The shared entry is what
the pipeline cache invalidates on an ingest and a delete, and the library's
``Engine`` still has a session of its own because two engines in one process
must not share a cache entry. Real sessions belong to the authentication work
that is deliberately not in this step.
"""

from __future__ import annotations

from typing import Annotated, Optional

from fastapi import Depends, Query, Request

from chat_rag.application.services import Services

from .envelope import clamp, whole_number

#: The cache entry every caller of this surface shares, per knowledge base.
SHARED_SESSION = "global"


def container(request: Request) -> Services:
    return request.app.state.services


def session_id(_request: Request) -> str:
    """The pipeline-cache key this surface uses: the shared one, always.

    A dependency rather than a constant so a route reads as "the session it
    was given", and so the one place that decides what a session is on this
    surface stays this function.
    """
    return SHARED_SESSION


class Pagination:
    """``offset`` and ``limit``, clamped and never refused.

    Declared as strings on purpose. FastAPI would answer ``?limit=abc`` with a
    validation error; this surface answers it with the default, because a page
    size is a presentation decision and refusing one breaks a link somebody
    pasted without teaching anybody anything. ``interfaces.http.v1.envelope``
    owns the ceiling and the defaults.
    """

    def __init__(
        self,
        offset: Annotated[Optional[str], Query(
            description="index of the first item; a whole number, defaulting "
                        "to 0 when it is not one")] = None,
        limit: Annotated[Optional[str], Query(
            description="how many items to return; a whole number, clamped to "
                        "the maximum page size")] = None,
    ):
        self.offset, self.limit = clamp(offset, limit)

    @property
    def stop(self) -> int:
        return self.offset + self.limit


def optional_number(raw: Optional[str]) -> Optional[int]:
    """A whole number in a query string, or nothing. Never a refusal, for the
    same reason :class:`Pagination` is not."""
    return whole_number(raw, None)


Container = Annotated[Services, Depends(container)]
SessionId = Annotated[str, Depends(session_id)]
Page = Annotated[Pagination, Depends(Pagination)]
