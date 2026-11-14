from __future__ import annotations

import logging
import json
import time
import uuid
from dataclasses import dataclass
from collections.abc import Generator
from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy import exists, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.chat.repository import (
    ChatRepository,
    ConversationNotFoundError,
    DocumentsNotFoundError,
)
from app.core.config import get_settings
from app.db.session import get_db
from app.ingestion.embeddings import EmbeddingError, EmbeddingService, get_embedding_service
from app.ingestion.image_embeddings import (
    ImageEmbeddingError,
    ImageEmbeddingService,
    get_image_embedding_service,
)
from app.llm.groq_service import (
    GroqConfigurationError,
    LLMRequestError,
    VisionImage,
)
from app.llm.provider import get_llm_service
from app.models import Document, DocumentImage
from app.rag.workflow import (
    AnswerGenerator,
    RAGWorkflow,
    build_retrieval_query,
    is_conversational_greeting,
    is_image_request,
)
from app.retrieval.engine import (
    RetrievalEngine,
    RetrievalMode,
    SQLAlchemyRetrievalStore,
)
from app.retrieval.reranker import (
    CrossEncoderReranker,
    RerankingError,
    get_reranker_service,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/chat", tags=["chat"])
MAX_CHAT_IMAGES = 3
MAX_IMAGE_SEARCH_CANDIDATES = 100


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=10_000)
    conversation_id: uuid.UUID | None = None
    document_ids: list[uuid.UUID] | None = Field(default=None, max_length=100)
    retrieval_mode: RetrievalMode = RetrievalMode.HYBRID_RERANK
    top_k: int = Field(default=5, ge=1, le=20)
    candidate_k: int | None = Field(default=None, ge=1, le=100)
    include_images: bool = False


class ChatCitation(BaseModel):
    document_id: uuid.UUID
    document_name: str
    chunk_id: uuid.UUID
    page_number: int | None
    chunk_index: int
    excerpt: str


class ChatImage(BaseModel):
    document_id: uuid.UUID
    document_name: str
    image_id: uuid.UUID
    page_number: int
    filename: str
    media_type: str
    image_url: str
    cited: bool = False
    context_excerpt: str | None = None


@dataclass(frozen=True)
class RetrievedChatImage:
    source: ChatImage
    data: bytes


class ChatResponse(BaseModel):
    answer: str
    citations: list[ChatCitation]
    images: list[ChatImage]
    conversation_id: uuid.UUID
    retrieval_metadata: dict[str, Any]
    latency_ms: float


class ConversationSummaryResponse(BaseModel):
    id: uuid.UUID
    title: str
    message_count: int
    preview: str
    created_at: datetime
    updated_at: datetime


class SavedMessageResponse(BaseModel):
    id: uuid.UUID
    role: str
    content: str
    citations: list[ChatCitation] = Field(default_factory=list)
    created_at: datetime


class ConversationDetailResponse(BaseModel):
    id: uuid.UUID
    title: str
    created_at: datetime
    updated_at: datetime
    messages: list[SavedMessageResponse]


def get_chat_repository(session: Session = Depends(get_db)) -> ChatRepository:
    return ChatRepository(session)


def get_rag_workflow(
    session: Session = Depends(get_db),
    embedder: EmbeddingService = Depends(get_embedding_service),
    reranker: CrossEncoderReranker = Depends(get_reranker_service),
    llm: AnswerGenerator = Depends(get_llm_service),
) -> RAGWorkflow:
    retrieval = RetrievalEngine(SQLAlchemyRetrievalStore(session), embedder)
    return RAGWorkflow(retrieval, reranker, llm)


def get_relevant_images(
    session: Session,
    *,
    question: str,
    document_ids: list[uuid.UUID] | None,
    image_embedder: ImageEmbeddingService,
    text_embedder: EmbeddingService,
) -> list[RetrievedChatImage]:
    missing_images_query = select(DocumentImage).where(
        DocumentImage.image_embedding.is_(None)
    )
    if document_ids is not None:
        missing_images_query = missing_images_query.where(
            DocumentImage.document_id.in_(document_ids)
        )
    missing_images = list(
        session.scalars(
            missing_images_query
            .order_by(DocumentImage.created_at.desc(), DocumentImage.id)
            .limit(MAX_CHAT_IMAGES * 4)
        ).all()
    )
    if missing_images:
        vectors = image_embedder.embed_images([image.data for image in missing_images])
        for image, vector in zip(missing_images, vectors, strict=True):
            image.image_embedding = vector
        session.flush()

    image_query_embedding = image_embedder.embed_text(question)
    image_distance = DocumentImage.image_embedding.cosine_distance(
        image_query_embedding
    )
    image_query = (
        select(
            DocumentImage.document_id,
            Document.filename.label("document_name"),
            DocumentImage.id,
            DocumentImage.page_number,
            DocumentImage.filename,
            DocumentImage.media_type,
            image_distance.label("distance"),
        )
        .join(Document, Document.id == DocumentImage.document_id)
        .where(DocumentImage.image_embedding.is_not(None))
    )
    if document_ids is not None:
        image_query = image_query.where(DocumentImage.document_id.in_(document_ids))
    image_rows = session.execute(
        image_query
        .order_by(image_distance, DocumentImage.document_id, DocumentImage.image_index)
        .limit(MAX_IMAGE_SEARCH_CANDIDATES)
    ).all()
    context_rows: list[Any] = []
    context_exists = select(
        exists(
            select(DocumentImage.id).where(
                DocumentImage.context_embedding.is_not(None),
                *(
                    [DocumentImage.document_id.in_(document_ids)]
                    if document_ids is not None
                    else []
                ),
            )
        )
    )
    if session.scalar(context_exists):
        text_query_embedding = text_embedder.embed_texts([question])[0]
        context_distance = DocumentImage.context_embedding.cosine_distance(
            text_query_embedding
        )
        context_query = (
            select(
                DocumentImage.document_id,
                Document.filename.label("document_name"),
                DocumentImage.id,
                DocumentImage.page_number,
                DocumentImage.filename,
                DocumentImage.media_type,
                context_distance.label("distance"),
            )
            .join(Document, Document.id == DocumentImage.document_id)
            .where(DocumentImage.context_embedding.is_not(None))
        )
        if document_ids is not None:
            context_query = context_query.where(
                DocumentImage.document_id.in_(document_ids)
            )
        context_rows = session.execute(
            context_query
            .order_by(
                context_distance,
                DocumentImage.document_id,
                DocumentImage.image_index,
            )
            .limit(MAX_IMAGE_SEARCH_CANDIDATES)
        ).all()

    image_by_id: dict[uuid.UUID, tuple[Any, ...]] = {}
    fused_scores: dict[uuid.UUID, float] = {}
    for ranking in (image_rows, context_rows):
        for rank, row in enumerate(ranking, start=1):
            image_id = row[2]
            image_by_id.setdefault(image_id, row)
            fused_scores[image_id] = fused_scores.get(image_id, 0.0) + (
                1.0 / (60 + rank)
            )
    ranked_image_ids = sorted(
        fused_scores,
        key=lambda image_id: (-fused_scores[image_id], str(image_id)),
    )[:MAX_CHAT_IMAGES]
    if not ranked_image_ids:
        return []

    selected_rows = [image_by_id[image_id] for image_id in ranked_image_ids]
    image_data = {
        image_id: (data, context_text)
        for image_id, data, context_text in session.execute(
            select(DocumentImage.id, DocumentImage.data, DocumentImage.context_text)
            .where(DocumentImage.id.in_(ranked_image_ids))
        ).all()
    }

    return [
        RetrievedChatImage(
            source=ChatImage(
                document_id=row[0],
                document_name=row[1],
                image_id=row[2],
                page_number=row[3],
                filename=row[4],
                media_type=row[5],
                image_url=f"/api/documents/{row[0]}/images/{row[2]}",
                context_excerpt=image_data[row[2]][1],
            ),
            data=image_data[row[2]][0],
        )
        for row in selected_rows
    ]


def _vision_images(retrieved_images: list[RetrievedChatImage]) -> list[VisionImage]:
    return [
        VisionImage(
            image_id=str(image.source.image_id),
            label=f"{image.source.document_name}, page {image.source.page_number}",
            media_type=image.source.media_type,
            data=image.data,
            context_text=image.source.context_excerpt,
        )
        for image in retrieved_images
    ]


def _order_response_images(
    retrieved_images: list[RetrievedChatImage],
    cited_image_ids: set[str],
) -> list[ChatImage]:
    updated = [
        image.source.model_copy(
            update={"cited": str(image.source.image_id) in cited_image_ids}
        )
        for image in retrieved_images
    ]
    # Surface LLM-cited figures first while preserving retrieval order within each group.
    updated.sort(key=lambda image: not image.cited)
    return updated


@router.post("/query", response_model=ChatResponse)
def query_chat(
    payload: ChatRequest,
    repository: ChatRepository = Depends(get_chat_repository),
    workflow: RAGWorkflow = Depends(get_rag_workflow),
    session: Session = Depends(get_db),
    image_embedder: ImageEmbeddingService = Depends(get_image_embedding_service),
    text_embedder: EmbeddingService = Depends(get_embedding_service),
) -> ChatResponse:
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="Question must not be empty.")
    if (
        payload.retrieval_mode is RetrievalMode.HYBRID_RERANK
        and payload.candidate_k is not None
        and payload.candidate_k < payload.top_k
    ):
        raise HTTPException(
            status_code=422,
            detail="candidate_k must be greater than or equal to top_k.",
        )

    started_at = time.perf_counter()
    try:
        repository.validate_documents(payload.document_ids)
        history: list[dict[str, str]] = []
        if payload.conversation_id is not None:
            repository.get_conversation(payload.conversation_id)
            history = repository.get_recent_history(payload.conversation_id)

        should_include_images = (
            payload.include_images or is_image_request(question)
        ) and not is_conversational_greeting(question)
        image_query = build_retrieval_query(question, history)
        retrieved_images = (
            get_relevant_images(
                session,
                question=image_query,
                document_ids=payload.document_ids,
                image_embedder=image_embedder,
                text_embedder=text_embedder,
            )
            if should_include_images
            else []
        )
        try:
            result = workflow.invoke(
                question=question,
                conversation_id=payload.conversation_id,
                document_ids=payload.document_ids,
                retrieval_mode=payload.retrieval_mode,
                top_k=payload.top_k,
                candidate_k=payload.candidate_k or get_settings().retrieval_candidate_k,
                history=history,
                images=_vision_images(retrieved_images),
            )
        except LLMRequestError as exc:
            if not retrieved_images:
                raise
            logger.warning(
                "Answer generation failed; returning %s retrieved image sources",
                len(retrieved_images),
            )
            result = {
                "answer": (
                    "I found relevant figures, but could not generate an AI answer: "
                    f"{exc} Open View sources to see the figures."
                ),
                "citations": [],
                "cited_image_ids": [],
                "retrieval_metadata": {
                    "mode": payload.retrieval_mode.value,
                    "result_count": 0,
                    "image_count": len(retrieved_images),
                    "answer_status": "unavailable",
                },
            }
        answer = result["answer"]
        citations = result.get("citations", [])
        cited_image_ids = set(result.get("cited_image_ids", []))
        images = _order_response_images(retrieved_images, cited_image_ids)
        conversation_id = repository.persist_exchange(
            conversation_id=payload.conversation_id,
            question=question,
            answer=answer,
            citations=citations,
        )
    except ConversationNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except DocumentsNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except GroqConfigurationError as exc:
        logger.warning("Groq configuration is unavailable")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    except (
        LLMRequestError,
        RerankingError,
        EmbeddingError,
        ImageEmbeddingError,
    ) as exc:
        logger.warning("RAG model service failed: %s", type(exc).__name__)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except SQLAlchemyError as exc:
        logger.warning("Chat database operation failed")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Chat persistence or document retrieval is unavailable.",
        ) from exc

    return ChatResponse(
        answer=answer,
        citations=citations,
        images=images,
        conversation_id=conversation_id,
        retrieval_metadata=result.get("retrieval_metadata", {}),
        latency_ms=round((time.perf_counter() - started_at) * 1000, 2),
    )


def _sse_event(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@router.post("/query/stream")
def stream_chat(
    payload: ChatRequest,
    repository: ChatRepository = Depends(get_chat_repository),
    workflow: RAGWorkflow = Depends(get_rag_workflow),
    session: Session = Depends(get_db),
    image_embedder: ImageEmbeddingService = Depends(get_image_embedding_service),
    text_embedder: EmbeddingService = Depends(get_embedding_service),
) -> StreamingResponse:
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="Question must not be empty.")
    if (
        payload.retrieval_mode is RetrievalMode.HYBRID_RERANK
        and payload.candidate_k is not None
        and payload.candidate_k < payload.top_k
    ):
        raise HTTPException(
            status_code=422,
            detail="candidate_k must be greater than or equal to top_k.",
        )

    def generate_events() -> Generator[str, None, None]:
        started_at = time.perf_counter()
        yield _sse_event("status", {"stage": "retrieving"})
        try:
            repository.validate_documents(payload.document_ids)
            history: list[dict[str, str]] = []
            if payload.conversation_id is not None:
                repository.get_conversation(payload.conversation_id)
                history = repository.get_recent_history(payload.conversation_id)

            should_include_images = (
                payload.include_images or is_image_request(question)
            ) and not is_conversational_greeting(question)
            image_query = build_retrieval_query(question, history)
            retrieved_images = (
                get_relevant_images(
                    session,
                    question=image_query,
                    document_ids=payload.document_ids,
                    image_embedder=image_embedder,
                    text_embedder=text_embedder,
                )
                if should_include_images
                else []
            )
            yield _sse_event("status", {"stage": "generating"})
            stream = workflow.stream(
                question=question,
                conversation_id=payload.conversation_id,
                document_ids=payload.document_ids,
                retrieval_mode=payload.retrieval_mode,
                top_k=payload.top_k,
                candidate_k=payload.candidate_k
                or get_settings().retrieval_candidate_k,
                history=history,
                images=_vision_images(retrieved_images),
            )
            while True:
                try:
                    delta = next(stream)
                except StopIteration as stopped:
                    result = stopped.value
                    break
                if delta:
                    yield _sse_event("delta", {"text": delta})

            answer = result["answer"]
            citations = result.get("citations", [])
            cited_image_ids = set(result.get("cited_image_ids", []))
            images = _order_response_images(retrieved_images, cited_image_ids)
            conversation_id = repository.persist_exchange(
                conversation_id=payload.conversation_id,
                question=question,
                answer=answer,
                citations=citations,
            )
            response = ChatResponse(
                answer=answer,
                citations=citations,
                images=images,
                conversation_id=conversation_id,
                retrieval_metadata=result.get("retrieval_metadata", {}),
                latency_ms=round((time.perf_counter() - started_at) * 1000, 2),
            )
            yield _sse_event("done", response.model_dump(mode="json"))
        except ConversationNotFoundError as exc:
            yield _sse_event("error", {"detail": str(exc)})
        except DocumentsNotFoundError as exc:
            yield _sse_event("error", {"detail": str(exc)})
        except GroqConfigurationError as exc:
            logger.warning("Chat streaming provider configuration is unavailable")
            yield _sse_event("error", {"detail": str(exc)})
        except (
            LLMRequestError,
            RerankingError,
            EmbeddingError,
            ImageEmbeddingError,
        ) as exc:
            logger.warning("Chat streaming model service failed: %s", type(exc).__name__)
            yield _sse_event("error", {"detail": str(exc)})
        except ValueError as exc:
            yield _sse_event("error", {"detail": str(exc)})
        except SQLAlchemyError as exc:
            logger.warning("Chat streaming database operation failed")
            yield _sse_event(
                "error",
                {"detail": "Chat persistence or document retrieval is unavailable."},
            )
        except Exception:
            logger.exception("Unexpected chat streaming failure")
            yield _sse_event(
                "error",
                {"detail": "An unexpected error interrupted the answer stream."},
            )

    return StreamingResponse(
        generate_events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/conversations", response_model=list[ConversationSummaryResponse])
def list_conversations(
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=50, ge=1, le=100),
    repository: ChatRepository = Depends(get_chat_repository),
) -> list[ConversationSummaryResponse]:
    try:
        summaries = repository.list_conversations(offset=offset, limit=limit)
    except SQLAlchemyError as exc:
        logger.warning("Could not list conversations")
        raise HTTPException(
            status_code=503,
            detail="Chat history is unavailable right now.",
        ) from exc
    return [ConversationSummaryResponse(**item) for item in summaries]


@router.get(
    "/conversations/{conversation_id}",
    response_model=ConversationDetailResponse,
)
def get_conversation(
    conversation_id: uuid.UUID,
    repository: ChatRepository = Depends(get_chat_repository),
) -> ConversationDetailResponse:
    try:
        conversation = repository.get_conversation_detail(conversation_id)
    except ConversationNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except SQLAlchemyError as exc:
        logger.warning("Could not load conversation %s", conversation_id)
        raise HTTPException(
            status_code=503,
            detail="Chat history is unavailable right now.",
        ) from exc

    ordered_messages = sorted(
        conversation.messages,
        key=ChatRepository.message_sort_key,
    )
    return ConversationDetailResponse(
        id=conversation.id,
        title=conversation.title or "Untitled chat",
        created_at=conversation.created_at,
        updated_at=conversation.updated_at,
        messages=[
            SavedMessageResponse(
                id=message.id,
                role=message.role,
                content=message.content,
                citations=message.citations or [],
                created_at=message.created_at,
            )
            for message in ordered_messages
        ],
    )


@router.delete(
    "/conversations/{conversation_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
def delete_conversation(
    conversation_id: uuid.UUID,
    repository: ChatRepository = Depends(get_chat_repository),
) -> None:
    try:
        repository.delete_conversation(conversation_id)
    except ConversationNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except SQLAlchemyError as exc:
        logger.warning("Could not delete conversation %s", conversation_id)
        raise HTTPException(
            status_code=503,
            detail="Could not delete the conversation.",
        ) from exc
