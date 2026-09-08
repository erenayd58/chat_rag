"""The relational schema this application was migrated onto in Step 8.

Everything durable the product knew about documents lived in JSON files until
this revision: ``.knowledge_bases.json``, ``.ingested_documents.json``,
``.gold_set.json``, one ``state.json`` per analysed content and one file per
ingest job. This creates the seven tables that replace them.

Read ``storage/models.py`` for why the edges are drawn as they are -- in
particular why ``documents.kb_id`` and ``contents.kb_id`` are deliberately
*not* foreign keys, and why an upload's selected methods live on the
membership row rather than on the content.

Nothing here stores a vector. Embeddings remain in the vector store.

Revision ID: 0001_initial_schema
Revises:
Create Date: 2026-09-08
"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = "0001_initial_schema"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table('contents',
    sa.Column('id', sa.String(length=128), nullable=False),
    sa.Column('content_key', sa.Text(), nullable=False),
    sa.Column('content_sha256', sa.Text(), nullable=True),
    sa.Column('status', sa.Text(), server_default='pending', nullable=False),
    sa.Column('label', sa.Text(), nullable=True),
    sa.Column('kb_id', sa.String(length=128), nullable=True),
    sa.Column('kb_name', sa.Text(), nullable=True),
    sa.Column('chunking_mode', sa.Text(), nullable=True),
    sa.Column('requested_methods', postgresql.ARRAY(sa.Text()), server_default='{}', nullable=False),
    sa.Column('ready_methods', postgresql.ARRAY(sa.Text()), nullable=True),
    sa.Column('failed_methods', postgresql.ARRAY(sa.Text()), nullable=True),
    sa.Column('unit_count', sa.Integer(), nullable=True),
    sa.Column('parse_seconds', sa.Float(), nullable=True),
    sa.Column('payload_bytes', sa.BigInteger(), nullable=True),
    sa.Column('deep_source', sa.Text(), nullable=True),
    sa.Column('error', sa.Text(), nullable=True),
    sa.Column('traceback', sa.Text(), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('content_key', name='uq_contents_content_key'),
    sa.UniqueConstraint('content_sha256', name='uq_contents_sha256')
    )
    op.create_index(op.f('ix_contents_kb_id'), 'contents', ['kb_id'], unique=False)
    op.create_table('documents',
    sa.Column('id', sa.String(length=128), nullable=False),
    sa.Column('doc_id', sa.String(length=128), nullable=False),
    sa.Column('kb_id', sa.String(length=128), nullable=True),
    sa.Column('file_name', sa.Text(), nullable=False),
    sa.Column('source_path', sa.Text(), nullable=True),
    sa.Column('file_hash', sa.Text(), nullable=True),
    sa.Column('file_size', sa.BigInteger(), server_default='0', nullable=False),
    sa.Column('chunk_count', sa.Integer(), server_default='0', nullable=False),
    sa.Column('status', sa.Text(), server_default='indexed', nullable=False),
    sa.Column('chunking_mode', sa.Text(), nullable=True),
    sa.Column('ingested_at', sa.Text(), nullable=False),
    sa.Column('metadata', postgresql.JSONB(astext_type=sa.Text()), server_default='{}', nullable=False),
    sa.Column('pipeline_snapshot', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    sa.Column('ingest_job_id', sa.String(length=128), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('doc_id', name='uq_documents_doc_id')
    )
    op.create_index(op.f('ix_documents_file_hash'), 'documents', ['file_hash'], unique=False)
    op.create_index('ix_documents_ingest_job_id', 'documents', ['ingest_job_id'], unique=False)
    op.create_index('ix_documents_ingested_at', 'documents', ['ingested_at'], unique=False)
    op.create_index(op.f('ix_documents_kb_id'), 'documents', ['kb_id'], unique=False)
    op.create_table('gold_set_entries',
    sa.Column('entry_id', sa.String(length=128), nullable=False),
    sa.Column('schema_version', sa.Integer(), server_default='1', nullable=False),
    sa.Column('kb_id', sa.String(length=128), nullable=False),
    sa.Column('question', sa.Text(), nullable=False),
    sa.Column('document_id', sa.Text(), nullable=True),
    sa.Column('document_title', sa.Text(), nullable=True),
    sa.Column('document_sha256', sa.Text(), nullable=True),
    sa.Column('correct_chunk_id', sa.Text(), nullable=True),
    sa.Column('section', sa.Text(), nullable=True),
    sa.Column('pages', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    sa.Column('unit_ids', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    sa.Column('evidence', sa.Text(), server_default='', nullable=False),
    sa.Column('retrieval_method', sa.Text(), nullable=True),
    sa.Column('found_at_rank', sa.Integer(), nullable=True),
    sa.Column('created_at', sa.Text(), nullable=False),
    sa.Column('updated_at', sa.Text(), nullable=False),
    sa.PrimaryKeyConstraint('entry_id')
    )
    op.create_index(op.f('ix_gold_set_entries_kb_id'), 'gold_set_entries', ['kb_id'], unique=False)
    op.create_table('ingest_jobs',
    sa.Column('job_id', sa.String(length=128), nullable=False),
    sa.Column('kb_id', sa.String(length=128), nullable=True),
    sa.Column('status', sa.Text(), nullable=False),
    sa.Column('doc_id', sa.String(length=128), nullable=True),
    sa.Column('filename', sa.Text(), nullable=True),
    sa.Column('journalled_at', sa.Float(), server_default='0', nullable=False),
    sa.Column('restart_recovered', sa.Boolean(), server_default='false', nullable=False),
    sa.Column('snapshot', postgresql.JSONB(astext_type=sa.Text()), server_default='{}', nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.PrimaryKeyConstraint('job_id')
    )
    op.create_index('ix_ingest_jobs_journalled_at', 'ingest_jobs', ['journalled_at'], unique=False)
    op.create_index(op.f('ix_ingest_jobs_kb_id'), 'ingest_jobs', ['kb_id'], unique=False)
    op.create_index(op.f('ix_ingest_jobs_status'), 'ingest_jobs', ['status'], unique=False)
    op.create_table('knowledge_bases',
    sa.Column('id', sa.String(length=128), nullable=False),
    sa.Column('name', sa.Text(), nullable=False),
    sa.Column('name_key', sa.Text(), nullable=False),
    sa.Column('chunker_type', sa.Text(), nullable=False),
    sa.Column('chunker_params', postgresql.JSONB(astext_type=sa.Text()), server_default='{}', nullable=False),
    sa.Column('embedding_model_name', sa.Text(), nullable=True),
    sa.Column('embedding_provider', sa.Text(), nullable=True),
    sa.Column('vector_db_provider', sa.Text(), server_default='chroma', nullable=False),
    sa.Column('vector_db_path', sa.Text(), nullable=True),
    sa.Column('retrieval_method', sa.Text(), server_default='hybrid', nullable=False),
    sa.Column('extra', postgresql.JSONB(astext_type=sa.Text()), server_default='{}', nullable=False),
    sa.Column('attributes', postgresql.JSONB(astext_type=sa.Text()), server_default='{}', nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.CheckConstraint('length(name) > 0', name='ck_knowledge_bases_name_not_blank'),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('name_key', name='uq_knowledge_bases_name_key')
    )
    op.create_table('content_documents',
    sa.Column('content_id', sa.String(length=128), nullable=False),
    sa.Column('doc_id', sa.String(length=128), nullable=False),
    sa.Column('selected_methods', postgresql.ARRAY(sa.Text()), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.ForeignKeyConstraint(['content_id'], ['contents.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('content_id', 'doc_id'),
    sa.UniqueConstraint('doc_id', name='uq_content_documents_doc_id')
    )
    op.create_table('content_variants',
    sa.Column('content_id', sa.String(length=128), nullable=False),
    sa.Column('method', sa.String(length=64), nullable=False),
    sa.Column('status', sa.Text(), nullable=False),
    sa.Column('source', sa.Text(), nullable=True),
    sa.Column('chunk_count', sa.Integer(), nullable=True),
    sa.Column('seconds', sa.Float(), nullable=True),
    sa.Column('error', sa.Text(), nullable=True),
    sa.Column('details', postgresql.JSONB(astext_type=sa.Text()), server_default='{}', nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.ForeignKeyConstraint(['content_id'], ['contents.id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('content_id', 'method')
    )


def downgrade() -> None:
    op.drop_table('content_variants')
    op.drop_table('content_documents')
    op.drop_table('knowledge_bases')
    op.drop_index(op.f('ix_ingest_jobs_status'), table_name='ingest_jobs')
    op.drop_index(op.f('ix_ingest_jobs_kb_id'), table_name='ingest_jobs')
    op.drop_index('ix_ingest_jobs_journalled_at', table_name='ingest_jobs')
    op.drop_table('ingest_jobs')
    op.drop_index(op.f('ix_gold_set_entries_kb_id'), table_name='gold_set_entries')
    op.drop_table('gold_set_entries')
    op.drop_index(op.f('ix_documents_kb_id'), table_name='documents')
    op.drop_index('ix_documents_ingested_at', table_name='documents')
    op.drop_index('ix_documents_ingest_job_id', table_name='documents')
    op.drop_index(op.f('ix_documents_file_hash'), table_name='documents')
    op.drop_table('documents')
    op.drop_index(op.f('ix_contents_kb_id'), table_name='contents')
    op.drop_table('contents')
