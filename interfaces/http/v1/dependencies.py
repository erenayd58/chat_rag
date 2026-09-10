"""What a router is given, and where it comes from.

Three things, and no fourth: the application container, the caller's session
id, and a page request. None of them is a product decision -- they are the
wiring a transport has to do before a use case can be called, kept here so
that a router reads as "take these, call one use case, return the schema".

The container is read from the running application rather than imported, so a
test that replaces a seam on it is honoured by every route at once and no
module in this package holds a reference of its own. It is the one object the
process composed, handed in by ``interfaces.http.create_app``.

The session id is the one thing this API takes from the transport that it did
not ask for. It selects a cached pipeline and nothing else -- it is not
identity, it is not authorisation, and this surface never sets it. A caller
that carries one on the ASGI scope gets its own cache entry; there is no
cookie to read otherwise, so the shared 'global' entry is used instead. Real
sessions belong to the authentication work that is deliberately not in this
step.
"""

from __future__ import annotations

import uuid
from typing import Annotated, Optional

from fastapi import Depends, Query, Request

from chat_rag.application.services import Services

from .envelope import clamp, whole_number

#: Where a caller's session id is read from on the ASGI scope.
SESSION_STATE = "session_id"
#: The cache entry a caller with no session of its own shares.
SHARED_SESSION = "global"


def container(request: Request) -> Services:
    return request.app.state.services


def session_id(request: Request) -> str:
    """The caller's pipeline-cache key, or the shared one."""
    return getattr(request.state, SESSION_STATE, "") or SHARED_SESSION


def fresh_session_id(request: Request) -> str:
    """The session id, or a new one for a caller that never took a page.

    Used by the two POSTs, which are reachable by a client that never loaded a
    screen; a fresh id gives that client its own cache entry rather than
    sharing the one every anonymous caller would share.
    """
    return getattr(request.state, SESSION_STATE, "") or str(uuid.uuid4())


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
FreshSessionId = Annotated[str, Depends(fresh_session_id)]
Page = Annotated[Pagination, Depends(Pagination)]
