"""Persist nearby PDF text and vectors for each extracted image.

Revision ID: 0005_image_text_context
Revises: 0004_image_embeddings
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector

revision: str = "0005_image_text_context"
down_revision: Union[str, None] = "0004_image_embeddings"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "document_images",
        sa.Column("context_text", sa.Text(), nullable=True),
    )
    op.add_column(
        "document_images",
        sa.Column("context_embedding", Vector(dim=384), nullable=True),
    )
    op.create_index(
        "ix_document_images_context_embedding_hnsw",
        "document_images",
        ["context_embedding"],
        unique=False,
        postgresql_using="hnsw",
        postgresql_ops={"context_embedding": "vector_cosine_ops"},
    )


def downgrade() -> None:
    op.drop_index(
        "ix_document_images_context_embedding_hnsw",
        table_name="document_images",
    )
    op.drop_column("document_images", "context_embedding")
    op.drop_column("document_images", "context_text")
