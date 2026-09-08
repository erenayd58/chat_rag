"""The numbers a page is made of, and the lenient reading of a page request.

Kept apart from the routers and from Pydantic on purpose: a page size is a
*policy* of this contract, not a fact about any one resource, and it is the
one thing outside the schemas that a client can observe directly. The shapes
themselves -- a resource, a collection, a refusal -- are declared in
:mod:`interfaces.http.v1.schemas`; the refusal taxonomy is in
:mod:`interfaces.http.v1.errors`.

``offset`` and ``limit`` are read leniently and never raise. That is a
deliberate departure from FastAPI's default, which would answer a request for
``?limit=abc`` with a validation error: refusing a page size teaches nobody
anything and breaks a link somebody pasted, so nonsense gets the default and
an over-large ask gets the ceiling. Without that ceiling, ``?limit=100000``
is a way to make any list endpoint as expensive as the caller likes.
"""

from __future__ import annotations

from typing import Optional, Sequence, TypeVar

#: How many items a collection returns when the caller does not say.
DEFAULT_LIMIT = 50
#: The most a caller may ask for in one page. A ceiling, not a suggestion.
MAX_LIMIT = 200

T = TypeVar("T")


def whole_number(raw: Optional[str], fallback: Optional[int] = None) -> Optional[int]:
    """One query-string number, or ``fallback`` when it is not one.

    The same forgiving reading Flask's ``request.args.get(name, type=int)``
    gave every optional filter on this surface.
    """
    if raw is None:
        return fallback
    try:
        return int(str(raw).strip())
    except (TypeError, ValueError):
        return fallback


def clamp(offset: Optional[str], limit: Optional[str]) -> tuple[int, int]:
    """``offset`` and ``limit`` as this contract promises them: clamped, never
    raising."""
    resolved_offset = max(0, whole_number(offset, 0) or 0)
    resolved_limit = max(1, min(MAX_LIMIT, whole_number(limit, DEFAULT_LIMIT) or DEFAULT_LIMIT))
    return resolved_offset, resolved_limit


def slice_of(rows: Sequence[T], *, offset: int, limit: int) -> list[T]:
    """One page out of a list the use case returned whole."""
    return list(rows[offset:offset + limit])


def flag(value: Optional[str]) -> bool:
    """A boolean in a query string, read the way this surface has always read one."""
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}
