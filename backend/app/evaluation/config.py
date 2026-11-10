from __future__ import annotations

from pydantic import Field, model_validator

from app.evaluation.dataset import StrictModel
from app.retrieval.engine import RetrievalMode


class AblationConfig(StrictModel):
    mode: RetrievalMode
    top_k: int = Field(default=10, ge=1, le=100)
    candidate_k: int | None = Field(default=None, ge=1, le=100)

    @model_validator(mode="after")
    def candidate_limit_matches_mode(self) -> AblationConfig:
        if self.mode is RetrievalMode.HYBRID_RERANK:
            if self.candidate_k is not None and self.candidate_k < self.top_k:
                raise ValueError("candidate_k must be greater than or equal to top_k.")
        elif self.candidate_k is not None:
            raise ValueError("candidate_k is only supported for hybrid_rerank.")
        return self


DEFAULT_ABLATIONS: tuple[AblationConfig, ...] = (
    AblationConfig(mode=RetrievalMode.DENSE),
    AblationConfig(mode=RetrievalMode.BM25),
    AblationConfig(mode=RetrievalMode.HYBRID),
    AblationConfig(mode=RetrievalMode.HYBRID_RERANK, candidate_k=50),
)
