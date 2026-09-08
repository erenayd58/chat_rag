"""The three shapes every `/api/v1` answer has, and no fourth.

* **a resource** -- the object itself, at the top level. No envelope, no
  ``success`` flag: the status line already says whether it worked, and a
  client that has to unwrap every answer to reach the thing it asked for is
  one that will unwrap wrongly somewhere.
* **a collection** -- ``{"items": [...], "page": {...}}``. Always both, even
  when everything fits on one page, so a client never has to branch on which
  kind of list it received. A few collections carry one extra top-level key
  (a capacity line, the retrieval default); each of those is its own subclass
  below or beside its resource, so the extra is declared rather than smuggled.
* **a refusal** -- ``{"error": {"type", "message", "details"}}``. ``type`` is
  the machine-readable name; it is what a client branches on, and it does not
  change when a message is reworded or a status is reconsidered.

These are the API's own types, not the application's. Nothing in
:mod:`application` imports this package, and nothing here is a projection of a
store row that happens to be convenient: what a client sees is decided here,
which is what lets the persistence underneath be replaced without the wire
moving.
"""

from __future__ import annotations

from typing import Any, Generic, Optional, TypeVar

from pydantic import BaseModel, ConfigDict, Field

T = TypeVar("T")


class Schema(BaseModel):
    """The base every model on this surface shares.

    ``extra="forbid"`` is the point of it: a field that reached a response
    without being declared here is a leak, and this turns that into a failure
    at the boundary rather than a discovery in somebody's logs.
    """

    model_config = ConfigDict(extra="forbid")


class Page(Schema):
    """Where in a list this answer came from, and how long the list is."""

    offset: int = Field(description="the index of the first item returned")
    limit: int = Field(description="how many items were asked for, after clamping")
    total: int = Field(description="how many there are in all")


class Collection(Schema, Generic[T]):
    """A page of a list, with the numbers a client needs to ask for the next."""

    items: list[T]
    page: Page

    @classmethod
    def of(cls, items, *, offset: int, limit: int, total: int, **extra):
        return cls(items=items, page=Page(offset=offset, limit=limit, total=total),
                   **extra)


class ApiError(Schema):
    """One refusal. ``type`` is the contract; the message is not."""

    type: str = Field(description="the machine-readable name a client branches on")
    message: str = Field(description="what went wrong, for a person")
    details: Optional[dict[str, Any]] = Field(
        default=None,
        description="what the refusal carries beyond its name; absent when it "
                    "carries nothing",
    )


class ErrorResponse(Schema):
    """The body of every refusal this API makes."""

    error: ApiError

    def body(self) -> dict[str, Any]:
        """The wire form.

        Written out rather than dumped with ``exclude_none``: ``details`` is
        absent rather than null when a refusal carries nothing, but what it
        *does* carry is a pass-through dict (a not-ready analysis state, the
        supported values of a rejected field) and no key inside it is dropped
        for being null.
        """
        error: dict[str, Any] = {"type": self.error.type, "message": self.error.message}
        if self.error.details:
            error["details"] = self.error.details
        return {"error": error}
