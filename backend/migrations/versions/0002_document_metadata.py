"""Persist parser metadata on uploaded documents.

Revision ID: 0002_document_metadata
Revises: 0001_initial_schema
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0002_document_metadata"
down_revision: Union[str, None] = "0001_initial_schema"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "documents",
        sa.Column(
            "metadata",
            sa.JSON(),
            server_default=sa.text("'{}'"),
            nullable=False,
        ),
    )


def downgrade() -> None:
    op.drop_column("documents", "metadata")
