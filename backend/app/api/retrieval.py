from __future__ import annotations

import logging
import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.core.config import get_settings
from app.ingestion.embeddings import EmbeddingError, EmbeddingService, get_embedding_service
from app.retrieval.engine import (
    RetrievalEngine,
    RetrievalMode,
    RetrievalResult,
    SQLAlchemyRetrievalStore,
)
from app.retrieval.reranker import (
    CrossEncoderReranker,
    RerankingError,
    get_reranker_service,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/retrieval", tags=["retrieval"])


class RetrievalRequest(BaseModel):
    query: str = Field(min_length=1, max_length=10_000)
    mode: RetrievalMode = RetrievalMode.HYBRID
    top_k: int = Field(default=5, ge=1, le=100)
    document_ids: list[uuid.UUID] | None = Field(default=None, max_length=100)
    candidate_k: int | None = Field(default=None, ge=1, le=100)


class RetrievalItem(BaseModel):
    document_id: uuid.UUID
    document_name: str
    chunk_id: uuid.UUID
    chunk_text: str
    page_number: int | None
    chunk_index: int
    relevance_score: float


class RetrievalResponse(BaseModel):
    query: str
    mode: RetrievalMode
    results: list[RetrievalItem]


def _serialize_result(result: RetrievalResult) -> RetrievalItem:
    return RetrievalItem(
        document_id=result.document_id,
        document_name=result.document_name,
        chunk_id=result.chunk_id,
        chunk_text=result.chunk_text,
        page_number=result.page_number,
        chunk_index=result.chunk_index,
        relevance_score=result.relevance_score,
    )


@router.post("/query", response_model=RetrievalResponse)
def retrieve(
    payload: RetrievalRequest,
    session: Session = Depends(get_db),
    embedder: EmbeddingService = Depends(get_embedding_service),
    reranker: CrossEncoderReranker = Depends(get_reranker_service),
) -> RetrievalResponse:
    engine = RetrievalEngine(SQLAlchemyRetrievalStore(session), embedder)
    try:
        if payload.mode is RetrievalMode.HYBRID_RERANK:
            candidate_k = payload.candidate_k or get_settings().retrieval_candidate_k
            if candidate_k < payload.top_k:
                raise HTTPException(
                    status_code=422,
                    detail="candidate_k must be greater than or equal to top_k.",
                )
            candidates = engine.retrieve(
                payload.query,
                mode=RetrievalMode.HYBRID,
                top_k=candidate_k,
                document_ids=payload.document_ids,
            )
            results = reranker.rerank(
                payload.query,
                candidates,
                top_k=payload.top_k,
            )
        else:
            results = engine.retrieve(
                payload.query,
                mode=payload.mode,
                top_k=payload.top_k,
                document_ids=payload.document_ids,
            )
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except EmbeddingError as exc:
        logger.exception("Could not embed retrieval query")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Dense retrieval is unavailable.",
        ) from exc
    except RerankingError as exc:
        logger.exception("Cross-encoder reranking failed")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                f"Cross-encoder reranking failed: {exc} "
                "Verify RERANKER_MODEL_NAME, sentence-transformers installation, "
                "and model download/network access before retrying."
            ),
        ) from exc
    except SQLAlchemyError as exc:
        logger.exception("Database retrieval failed")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Document retrieval is unavailable.",
        ) from exc

    return RetrievalResponse(
        query=payload.query.strip(),
        mode=payload.mode,
        results=[_serialize_result(result) for result in results],
    )
