from __future__ import annotations

import hashlib
import logging
import uuid
from dataclasses import replace
from datetime import datetime

from fastapi import (
    APIRouter,
    Depends,
    File,
    HTTPException,
    Query,
    Response,
    UploadFile,
    status,
)
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.core.config import get_settings
from app.ingestion.embeddings import EmbeddingError, EmbeddingService, get_embedding_service
from app.ingestion.image_embeddings import (
    ImageEmbeddingError,
    ImageEmbeddingService,
    get_image_embedding_service,
)
from app.ingestion.parsers import DocumentParseError, get_extension
from app.ingestion.service import (
    DocumentTooLargeError,
    prepare_document,
    save_document,
)
from app.models import Chunk, Document, DocumentImage

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/documents", tags=["documents"])


class DocumentResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True, populate_by_name=True)

    id: uuid.UUID
    filename: str
    media_type: str
    status: str
    document_metadata: dict[str, object] = Field(
        validation_alias="document_metadata",
        serialization_alias="metadata",
    )
    created_at: datetime


class ChunkPreviewResponse(BaseModel):
    document_id: uuid.UUID
    document_name: str
    chunk_id: uuid.UUID
    chunk_index: int
    page_number: int | None
    content: str


def _find_duplicate_document(
    session: Session,
    *,
    filename: str,
    file_sha256: str,
) -> tuple[Document, str] | None:
    normalized_name = filename.casefold()
    for existing in session.scalars(select(Document)).all():
        if existing.status == "failed":
            continue
        if existing.filename.casefold() == normalized_name:
            return existing, "name"
        metadata = existing.document_metadata or {}
        if metadata.get("content_sha256") == file_sha256:
            return existing, "content"
    return None


@router.post("", response_model=DocumentResponse, status_code=status.HTTP_201_CREATED)
def upload_document(
    file: UploadFile = File(...),
    session: Session = Depends(get_db),
    embedder: EmbeddingService = Depends(get_embedding_service),
    image_embedder: ImageEmbeddingService = Depends(get_image_embedding_service),
) -> Document:
    filename = (file.filename or "").replace("\\", "/").rsplit("/", 1)[-1]
    if not filename:
        raise HTTPException(status_code=422, detail="A filename is required.")
    if get_extension(filename) not in {".pdf", ".md", ".txt"}:
        raise HTTPException(
            status_code=415,
            detail="Supported file types are PDF, Markdown, and TXT.",
        )

    max_bytes = get_settings().upload_max_bytes
    data = file.file.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise HTTPException(status_code=413, detail="The uploaded file exceeds the size limit.")

    file_sha256 = hashlib.sha256(data).hexdigest()
    duplicate = _find_duplicate_document(
        session,
        filename=filename,
        file_sha256=file_sha256,
    )
    if duplicate is not None:
        existing_doc, reason = duplicate
        detail = (
            f"Can't upload '{filename}': it already exists."
            if reason == "name"
            else f"Can't upload '{filename}': this file already exists as '{existing_doc.filename}'."
        )
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=detail)

    try:
        (
            parsed,
            chunks,
            embeddings,
            image_embeddings,
            image_context_embeddings,
        ) = prepare_document(
            data,
            filename,
            embedder,
            image_embedder,
        )
    except DocumentParseError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except DocumentTooLargeError as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except EmbeddingError as exc:
        logger.exception("Could not generate embeddings for uploaded file %s", filename)
        raise HTTPException(status_code=503, detail="Document embedding is unavailable.") from exc
    except ImageEmbeddingError as exc:
        logger.exception("Could not generate image embeddings for uploaded file %s", filename)
        raise HTTPException(
            status_code=503,
            detail="PDF image search is unavailable. Please try again later.",
        ) from exc

    parsed_with_hash = replace(
        parsed,
        metadata={**parsed.metadata, "content_sha256": file_sha256},
    )

    try:
        return save_document(
            session,
            filename=filename,
            parsed=parsed_with_hash,
            chunks=chunks,
            embeddings=embeddings,
            image_embeddings=image_embeddings,
            image_context_embeddings=image_context_embeddings,
        )
    except SQLAlchemyError as exc:
        session.rollback()
        logger.exception("Could not save uploaded document %s", filename)
        raise HTTPException(status_code=500, detail="Could not save the uploaded document.") from exc


@router.get("", response_model=list[DocumentResponse])
def list_documents(
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=50, ge=1, le=100),
    session: Session = Depends(get_db),
) -> list[Document]:
    statement = (
        select(Document)
        .order_by(Document.created_at.desc(), Document.id)
        .offset(offset)
        .limit(limit)
    )
    return list(session.scalars(statement).all())


@router.get(
    "/{document_id}/chunks/{chunk_id}",
    response_model=ChunkPreviewResponse,
)
def get_document_chunk(
    document_id: uuid.UUID,
    chunk_id: uuid.UUID,
    session: Session = Depends(get_db),
) -> ChunkPreviewResponse:
    row = session.execute(
        select(
            Chunk.id,
            Chunk.document_id,
            Chunk.chunk_index,
            Chunk.page_number,
            Chunk.content,
            Document.filename,
        )
        .join(Document, Document.id == Chunk.document_id)
        .where(Chunk.id == chunk_id, Chunk.document_id == document_id)
    ).one_or_none()
    if row is None:
        raise HTTPException(status_code=404, detail="Source chunk not found.")
    chunk_id_val, doc_id_val, chunk_index, page_number, content, filename = row
    return ChunkPreviewResponse(
        document_id=doc_id_val,
        document_name=filename,
        chunk_id=chunk_id_val,
        chunk_index=chunk_index,
        page_number=page_number,
        content=content,
    )


@router.get("/{document_id}/images/{image_id}", response_class=Response)
def get_document_image(
    document_id: uuid.UUID,
    image_id: uuid.UUID,
    session: Session = Depends(get_db),
) -> Response:
    image = session.scalar(
        select(DocumentImage).where(
            DocumentImage.id == image_id,
            DocumentImage.document_id == document_id,
        )
    )
    if image is None:
        raise HTTPException(status_code=404, detail="Document image not found.")

    return Response(
        content=image.data,
        media_type=image.media_type,
        headers={
            "Cache-Control": "private, max-age=3600",
            "X-Content-Type-Options": "nosniff",
        },
    )


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_document(document_id: uuid.UUID, session: Session = Depends(get_db)) -> None:
    document = session.get(Document, document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found.")

    try:
        session.delete(document)
        session.commit()
    except SQLAlchemyError as exc:
        session.rollback()
        logger.exception("Could not delete document %s", document_id)
        raise HTTPException(status_code=500, detail="Could not delete the document.") from exc
