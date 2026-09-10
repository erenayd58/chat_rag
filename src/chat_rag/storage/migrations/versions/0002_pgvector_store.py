"""The vector store, moved into PostgreSQL.

Step 8 left one path in the relational schema and one store outside it: a
knowledge base was a row, but its vectors were a Chroma directory named by
``knowledge_bases.vector_db_path``. This revision closes that. It

* installs the ``vector`` extension, which is what makes an embedding a column
  rather than a blob a query cannot order by;
* creates ``vector_collections`` -- one searchable corpus, owned by a
  knowledge base, carrying the embedding manifest that used to be an
  ``embedding_index.json`` beside the directory;
* creates ``chunk_vectors`` -- the chunk text, its metadata and its embedding,
  cascading from the collection and, through it, from the knowledge base;
* drops ``knowledge_bases.vector_db_path`` and makes ``pgvector`` the provider
  every record names.

``CREATE EXTENSION`` needs a role that may install one (a superuser, or the
``rds_superuser``-equivalent on a managed service). It is here rather than in
a deployment note because a schema this application cannot create is a schema
nobody can reproduce -- see ``docs/database.md`` for the one-line grant a
locked-down installation needs instead.

The embedding column is declared ``vector`` with no width. The application
supports several embedding spaces at once -- 384 dimensions locally, 4096 from
the demo's gateway, and a 1-wide placeholder where a lexical-only profile
stores no embedding at all -- and there is no width that would not break one
of them. That decision is also why there is no ANN index here: pgvector can
only index a column of fixed width, and correctness across every configured
model is worth more than an approximate scan of a corpus this size.
``docs/database.md`` records the index and the condition under which it
becomes correct to add one.

Revision ID: 0002_pgvector_store
Revises: 0001_initial_schema
Create Date: 2026-09-08
"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql

revision: str = "0002_pgvector_store"
down_revision: Union[str, None] = "0001_initial_schema"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "vector_collections",
        sa.Column("collection", sa.String(length=128), nullable=False),
        sa.Column("kb_id", sa.String(length=128), nullable=True),
        sa.Column("embedding_provider", sa.Text(), nullable=True),
        sa.Column("embedding_model", sa.Text(), nullable=True),
        sa.Column("embedding_endpoint", sa.Text(), nullable=True),
        sa.Column("embedding_dimension", sa.Integer(), nullable=True),
        sa.Column("embedding_fingerprint", sa.Text(), nullable=True),
        sa.Column("manifest_chunk_count", sa.Integer(), nullable=True),
        sa.Column("written_at", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True),
                  server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True),
                  server_default=sa.text("now()"), nullable=False),
        # The one referential edge a vector has. It cascades, unlike the
        # ``kb_id`` on a document or a content: a ledger row is worth keeping
        # after its knowledge base is gone, a vector is not.
        sa.ForeignKeyConstraint(["kb_id"], ["knowledge_bases.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("collection"),
    )
    op.create_index(op.f("ix_vector_collections_kb_id"), "vector_collections",
                    ["kb_id"], unique=False)

    op.create_table(
        "chunk_vectors",
        sa.Column("collection", sa.String(length=128), nullable=False),
        sa.Column("chunk_id", sa.Text(), nullable=False),
        sa.Column("doc_id", sa.Text(), server_default="", nullable=False),
        sa.Column("doc_title", sa.Text(), server_default="", nullable=False),
        sa.Column("chunk_index", sa.Integer(), server_default="0", nullable=False),
        sa.Column("total_chunks", sa.Integer(), server_default="0", nullable=False),
        sa.Column("section_title", sa.Text(), nullable=True),
        sa.Column("method", sa.String(length=64), nullable=True),
        sa.Column("content", sa.Text(), server_default="", nullable=False),
        sa.Column("metadata", postgresql.JSONB(astext_type=sa.Text()),
                  server_default="{}", nullable=False),
        sa.Column("embedding", Vector(), nullable=True),
        sa.Column("embedding_dim", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True),
                  server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True),
                  server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(["collection"], ["vector_collections.collection"],
                                ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("collection", "chunk_id"),
    )
    op.create_index("ix_chunk_vectors_collection_doc", "chunk_vectors",
                    ["collection", "doc_id"], unique=False)

    # The last path in the relational schema. A knowledge base is identified by
    # its id; where its vectors are is no longer a question with a filesystem
    # answer.
    op.drop_column("knowledge_bases", "vector_db_path")
    op.alter_column("knowledge_bases", "vector_db_provider",
                    server_default="pgvector")
    op.execute("UPDATE knowledge_bases SET vector_db_provider = 'pgvector'")


def downgrade() -> None:
    op.execute("UPDATE knowledge_bases SET vector_db_provider = 'chroma'")
    op.alter_column("knowledge_bases", "vector_db_provider", server_default="chroma")
    op.add_column("knowledge_bases", sa.Column("vector_db_path", sa.Text(),
                                               nullable=True))
    op.drop_index("ix_chunk_vectors_collection_doc", table_name="chunk_vectors")
    op.drop_table("chunk_vectors")
    op.drop_index(op.f("ix_vector_collections_kb_id"), table_name="vector_collections")
    op.drop_table("vector_collections")
    # The extension is deliberately left installed: another schema in the same
    # database may be using it, and dropping it would take their columns with
    # it.
