"""Add CLIP-compatible embeddings for stored PDF images.

Revision ID: 0004_image_embeddings
Revises: 0003_document_images
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector

revision: str = "0004_image_embeddings"
down_revision: Union[str, None] = "0003_document_images"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "document_images",
        sa.Column("image_embedding", Vector(dim=512), nullable=True),
    )
    op.create_index(
        "ix_document_images_embedding_hnsw",
        "document_images",
        ["image_embedding"],
        unique=False,
        postgresql_using="hnsw",
        postgresql_ops={"image_embedding": "vector_cosine_ops"},
    )


def downgrade() -> None:
    op.drop_index(
        "ix_document_images_embedding_hnsw",
        table_name="document_images",
    )
    op.drop_column("document_images", "image_embedding")
