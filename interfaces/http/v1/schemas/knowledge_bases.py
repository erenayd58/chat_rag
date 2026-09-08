"""A named collection with its own chunker and its own corpus."""

from __future__ import annotations

from typing import Any, Optional

from pydantic import ConfigDict, Field

from .common import Collection, Schema


class KnowledgeBase(Schema):
    """No storage location and no store provider: where the vectors live is a
    deployment's business, and it is the thing the pgvector migration changes.
    The embedding model *is* here -- it is a product fact, because the vectors
    belong to it and re-indexing is how you change it.
    """

    id: Optional[str] = None
    name: Optional[str] = None
    chunker: dict[str, Any] = Field(default_factory=dict)
    retrieval_method: Optional[str] = None
    embedding_model: Optional[str] = None
    extra: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def of(cls, record: dict) -> "KnowledgeBase":
        return cls(
            id=record.get("kb_id"),
            name=record.get("name"),
            chunker=record.get("chunker") or {},
            retrieval_method=record.get("retrieval_method"),
            embedding_model=record.get("embedding_model_name"),
            extra=record.get("extra") or {},
        )


class KnowledgeBaseCollection(Collection[KnowledgeBase]):
    pass


class KnowledgeBaseCreate(Schema):
    """What creating one takes.

    Every field is optional here and refused by the use case rather than by
    the schema, because "a knowledge base needs a name" is a product rule with
    a product's answer -- **400** ``invalid_request`` -- and not a shape the
    transport should be inventing a different code for.
    """

    name: Optional[str] = None
    chunker: Optional[dict[str, Any]] = None
    embedding_model: Optional[str] = None
    retrieval_method: Optional[str] = None
    extra: Optional[dict[str, Any]] = None

    def as_payload(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "chunker": self.chunker,
            "embedding_model_name": self.embedding_model,
            "retrieval_method": self.retrieval_method,
            "extra": self.extra,
        }


class KnowledgeBaseUpdate(Schema):
    """Only ``name`` and ``extra``: everything else was decided at creation,
    because the corpus that was ingested depends on it.

    A field left out is left alone -- which is why this is read with
    ``exclude_unset`` rather than by its defaults.
    """

    name: Optional[str] = None
    extra: Optional[dict[str, Any]] = None

    def changes(self) -> dict[str, Any]:
        return self.model_dump(exclude_unset=True)


class EmbeddingIndex(Schema):
    """Whether the stored vectors belong to the embedding model configured now.

    The whole body is the store's own manifest report, passed through
    unchanged and explicitly not contractual: what a manifest records is an
    implementation's business, and this endpoint exists to answer one
    question -- re-index, or do not. It is declared as an open object rather
    than as a set of fields precisely so that a profile which reports more (or
    less) is not silently padded with nulls by this adapter.

    The keys every profile answers today are ``state``, ``compatible``,
    ``dense_available`` and ``reason``.
    """

    model_config = ConfigDict(extra="allow")


class EmbeddingReindex(Schema):
    """What a re-index did, and where the index stands afterwards.

    Both halves are pass-through: how many chunks were re-embedded and by
    which model is an operator's report, not a shape a client branches on.
    """

    result: dict[str, Any] = Field(default_factory=dict)
    index: dict[str, Any] = Field(default_factory=dict)
