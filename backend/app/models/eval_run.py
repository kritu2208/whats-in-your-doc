from typing import Any

from sqlalchemy import Index, JSON, String
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base
from app.models.base import TimestampMixin, UUIDPrimaryKeyMixin


class EvalRun(Base, UUIDPrimaryKeyMixin, TimestampMixin):
    __tablename__ = "eval_runs"
    __table_args__ = (
        Index("ix_eval_runs_retrieval_mode", "retrieval_mode"),
        Index("ix_eval_runs_status", "status"),
    )

    retrieval_mode: Mapped[str] = mapped_column(String(32), nullable=False)
    status: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default="pending",
        server_default="pending",
    )
    metrics: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    configuration: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
