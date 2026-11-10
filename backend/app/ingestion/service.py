from __future__ import annotations

from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.ingestion.chunking import TextChunk, chunk_pages
from app.ingestion.embeddings import EmbeddingService
from app.ingestion.image_embeddings import (
    ImageEmbeddingService,
    get_image_embedding_service,
)
from app.ingestion.parsers import ParsedDocument, parse_document
from app.models import Chunk, Document, DocumentImage


class DocumentTooLargeError(ValueError):
    """Raised when a document would exceed the configured chunk limit."""


def prepare_document(
    data: bytes,
    filename: str,
    embedder: EmbeddingService,
    image_embedder: ImageEmbeddingService | None = None,
) -> tuple[
    ParsedDocument,
    list[TextChunk],
    list[list[float]],
    list[list[float]],
    list[list[float] | None],
]:
    settings = get_settings()
    parsed = parse_document(data, filename)
    chunks = chunk_pages(
        parsed.pages,
        chunk_size=settings.chunk_size_tokens,
        overlap=settings.chunk_overlap_tokens,
    )
    if not chunks:
        raise ValueError("The document contains no extractable text.")
    if len(chunks) > settings.max_document_chunks:
        raise DocumentTooLargeError(
            f"The document exceeds the limit of {settings.max_document_chunks} chunks."
        )
    image_context_texts = [
        (index, image.associated_text)
        for index, image in enumerate(parsed.images)
        if image.associated_text.strip()
    ]
    embedded_texts = embedder.embed_texts(
        [chunk.content for chunk in chunks]
        + [context_text for _, context_text in image_context_texts]
    )
    vectors = embedded_texts[: len(chunks)]
    image_context_vectors: list[list[float] | None] = [None] * len(parsed.images)
    for (image_index, _), vector in zip(
        image_context_texts,
        embedded_texts[len(chunks) :],
        strict=True,
    ):
        image_context_vectors[image_index] = vector
    image_vectors = (
        (image_embedder or get_image_embedding_service()).embed_images(
            [image.data for image in parsed.images]
        )
        if parsed.images
        else []
    )
    return parsed, chunks, vectors, image_vectors, image_context_vectors


def save_document(
    session: Session,
    *,
    filename: str,
    parsed: ParsedDocument,
    chunks: list[TextChunk],
    embeddings: list[list[float]],
    image_embeddings: list[list[float]],
    image_context_embeddings: list[list[float] | None],
) -> Document:
    document = Document(
        filename=filename,
        media_type=parsed.media_type,
        status="ready",
        document_metadata=parsed.metadata,
    )
    document.chunks = [
        Chunk(
            chunk_index=chunk.chunk_index,
            content=chunk.content,
            page_number=chunk.page_number,
            embedding=embedding,
        )
        for chunk, embedding in zip(chunks, embeddings, strict=True)
    ]
    document.images = [
        DocumentImage(
            page_number=image.page_number,
            image_index=image.image_index,
            filename=image.filename,
            media_type=image.media_type,
            data=image.data,
            image_embedding=image_embedding,
            context_text=image.associated_text or None,
            context_embedding=context_embedding,
        )
        for image, image_embedding, context_embedding in zip(
            parsed.images,
            image_embeddings,
            image_context_embeddings,
            strict=True,
        )
    ]
    session.add(document)
    session.flush()
    session.commit()
    return document
