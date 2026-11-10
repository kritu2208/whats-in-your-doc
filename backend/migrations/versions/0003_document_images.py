"""Store extracted PDF images with their source pages.

Revision ID: 0003_document_images
Revises: 0002_document_metadata
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0003_document_images"
down_revision: Union[str, None] = "0002_document_metadata"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "document_images",
        sa.Column("document_id", sa.Uuid(as_uuid=True), nullable=False),
        sa.Column("page_number", sa.Integer(), nullable=False),
        sa.Column("image_index", sa.Integer(), nullable=False),
        sa.Column("filename", sa.String(length=255), nullable=False),
        sa.Column("media_type", sa.String(length=100), nullable=False),
        sa.Column("data", sa.LargeBinary(), nullable=False),
        sa.Column("id", sa.Uuid(as_uuid=True), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["document_id"],
            ["documents.id"],
            name="fk_document_images_document_id_documents",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_document_images"),
        sa.UniqueConstraint(
            "document_id",
            "page_number",
            "image_index",
            name="uq_document_images_document_page_index",
        ),
    )
    op.create_index(
        "ix_document_images_document_page",
        "document_images",
        ["document_id", "page_number"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_document_images_document_page", table_name="document_images")
    op.drop_table("document_images")
