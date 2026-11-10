from __future__ import annotations

from io import BytesIO
from types import SimpleNamespace
from typing import Any
import uuid

import pytest
from sqlalchemy.dialects import postgresql

from app.api.chat import ChatImage, RetrievedChatImage, get_relevant_images
from app.ingestion.image_embeddings import (
    IMAGE_EMBEDDING_DIMENSION,
    IMAGE_EMBEDDING_BATCH_SIZE,
    ImageEmbeddingError,
    ImageEmbeddingService,
)
from app.models import DocumentImage


class FakeEncoded:
    def __init__(self, rows: list[list[float]]) -> None:
        self.rows = rows

    def tolist(self) -> list[list[float]]:
        return self.rows


class FakeImageTextEncoder:
    def __init__(self) -> None:
        self.calls: list[list[Any]] = []

    def encode(
        self,
        inputs: list[Any],
        *,
        batch_size: int,
        convert_to_numpy: bool,
        normalize_embeddings: bool,
        show_progress_bar: bool,
    ) -> FakeEncoded:
        assert batch_size == IMAGE_EMBEDDING_BATCH_SIZE
        assert convert_to_numpy
        assert normalize_embeddings
        assert not show_progress_bar
        self.calls.append(inputs)
        return FakeEncoded([[0.25] * IMAGE_EMBEDDING_DIMENSION for _ in inputs])


def test_image_text_embeddings_load_lazily_and_share_one_encoder() -> None:
    encoders: list[FakeImageTextEncoder] = []

    def factory(model_name: str) -> FakeImageTextEncoder:
        assert model_name == "test-clip-model"
        encoder = FakeImageTextEncoder()
        encoders.append(encoder)
        return encoder

    service = ImageEmbeddingService("test-clip-model", model_factory=factory)
    assert encoders == []

    assert service.embed_text("a calico cat") == [0.25] * IMAGE_EMBEDDING_DIMENSION

    from PIL import Image

    image_buffer = BytesIO()
    Image.new("RGB", (4, 4), color=(200, 80, 120)).save(image_buffer, format="PNG")
    vectors = service.embed_images([image_buffer.getvalue()])

    assert vectors == [[0.25] * IMAGE_EMBEDDING_DIMENSION]
    assert len(encoders) == 1
    assert len(encoders[0].calls) == 2
    assert isinstance(encoders[0].calls[1][0], Image.Image)
    assert encoders[0].calls[1][0].mode == "RGB"


def test_image_text_embeddings_reject_wrong_dimensions() -> None:
    class WrongDimensionEncoder(FakeImageTextEncoder):
        def encode(self, inputs: list[Any], **kwargs: Any) -> FakeEncoded:
            del kwargs
            return FakeEncoded([[0.0, 1.0] for _ in inputs])

    service = ImageEmbeddingService(
        "wrong-dimension",
        model_factory=lambda _: WrongDimensionEncoder(),
    )

    with pytest.raises(ImageEmbeddingError, match="512-dimensional"):
        service.embed_text("a calico cat")


class FakeSearchSession:
    def __init__(
        self,
        *,
        missing_images: list[DocumentImage] | None = None,
        image_rows: list[tuple[Any, ...]] | None = None,
        context_rows: list[tuple[Any, ...]] | None = None,
        image_data_rows: list[tuple[Any, ...]] | None = None,
        has_context: bool = False,
    ) -> None:
        self.missing_images = missing_images or []
        self.image_rows = image_rows or []
        self.context_rows = context_rows or []
        self.image_data_rows = image_data_rows or []
        self.has_context = has_context
        self.executed_statement: Any = None
        self.executed_statements: list[Any] = []
        self.flushed = False
        self.execute_count = 0

    def scalars(self, statement: Any) -> SimpleNamespace:
        del statement
        return SimpleNamespace(all=lambda: self.missing_images)

    def scalar(self, statement: Any) -> bool:
        del statement
        return self.has_context

    def execute(self, statement: Any) -> SimpleNamespace:
        self.executed_statement = statement
        self.executed_statements.append(statement)
        result_sets = (
            [self.image_rows, self.context_rows, self.image_data_rows]
            if self.has_context
            else [self.image_rows, self.image_data_rows]
        )
        rows = result_sets[self.execute_count]
        self.execute_count += 1
        return SimpleNamespace(all=lambda: rows)

    def flush(self) -> None:
        self.flushed = True


class FakeQueryEmbedder:
    def __init__(self, vectors: list[list[float]] | None = None) -> None:
        self.vectors = vectors or []
        self.image_batches: list[list[bytes]] = []
        self.queries: list[str] = []

    def embed_images(self, images: list[bytes]) -> list[list[float]]:
        self.image_batches.append(images)
        return self.vectors

    def embed_text(self, text: str) -> list[float]:
        self.queries.append(text)
        return [0.5] * IMAGE_EMBEDDING_DIMENSION


class FakeTextEmbedder:
    def __init__(self) -> None:
        self.queries: list[str] = []

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.queries.extend(texts)
        return [[0.5] * 384 for _ in texts]


def test_relevant_image_search_backfills_legacy_images_and_uses_semantic_distance() -> None:
    document_id = uuid.UUID(int=42)
    image_id = uuid.UUID(int=84)
    legacy_image = DocumentImage(
        id=image_id,
        document_id=document_id,
        page_number=3,
        image_index=0,
        filename="page-3-image-1.jpg",
        media_type="image/jpeg",
        data=b"legacy-image-bytes",
    )
    session = FakeSearchSession(
        missing_images=[legacy_image],
        image_rows=[
            (
                document_id,
                "cats.pdf",
                image_id,
                3,
                "page-3-image-1.jpg",
                "image/jpeg",
                0.12,
            )
        ],
        image_data_rows=[(image_id, b"legacy-image-bytes", None)],
    )
    embedder = FakeQueryEmbedder(
        vectors=[[0.75] * IMAGE_EMBEDDING_DIMENSION]
    )
    text_embedder = FakeTextEmbedder()

    images = get_relevant_images(
        session,  # type: ignore[arg-type]
        question="Show me a calico cat",
        document_ids=[document_id],
        image_embedder=embedder,  # type: ignore[arg-type]
        text_embedder=text_embedder,  # type: ignore[arg-type]
    )

    assert embedder.image_batches == [[b"legacy-image-bytes"]]
    assert embedder.queries == ["Show me a calico cat"]
    assert legacy_image.image_embedding == [0.75] * IMAGE_EMBEDDING_DIMENSION
    assert session.flushed
    assert text_embedder.queries == []
    assert images == [
        RetrievedChatImage(
            source=ChatImage(
                document_id=document_id,
                document_name="cats.pdf",
                image_id=image_id,
                page_number=3,
                filename="page-3-image-1.jpg",
                media_type="image/jpeg",
                image_url=f"/api/documents/{document_id}/images/{image_id}",
            ),
            data=b"legacy-image-bytes",
        ),
    ]

    compiled = str(
        session.executed_statements[0].compile(dialect=postgresql.dialect())
    )
    assert "<=>" in compiled
    assert "document_images.image_embedding" in compiled
    assert "document_images.document_id IN" in compiled


def test_image_search_fuses_nearby_text_and_visual_rankings() -> None:
    document_id = uuid.UUID(int=242)
    visual_match_id = uuid.UUID(int=243)
    text_match_id = uuid.UUID(int=244)
    other_match_id = uuid.UUID(int=245)
    session = FakeSearchSession(
        image_rows=[
            (
                document_id,
                "cats.pdf",
                visual_match_id,
                1,
                "image-1.jpg",
                "image/jpeg",
                0.1,
            ),
            (
                document_id,
                "cats.pdf",
                text_match_id,
                1,
                "image-2.jpg",
                "image/jpeg",
                0.2,
            ),
        ],
        context_rows=[
            (
                document_id,
                "cats.pdf",
                text_match_id,
                1,
                "image-2.jpg",
                "image/jpeg",
                0.1,
            ),
            (
                document_id,
                "cats.pdf",
                other_match_id,
                1,
                "image-3.jpg",
                "image/jpeg",
                0.15,
            ),
            (
                document_id,
                "cats.pdf",
                visual_match_id,
                1,
                "image-1.jpg",
                "image/jpeg",
                0.2,
            ),
        ],
        image_data_rows=[
            (text_match_id, b"image-2", "Ocicat, spotted coat"),
            (visual_match_id, b"image-1", "Other breed"),
            (other_match_id, b"image-3", "Unrelated"),
        ],
        has_context=True,
    )
    text_embedder = FakeTextEmbedder()

    images = get_relevant_images(
        session,  # type: ignore[arg-type]
        question="Show the Ocicat",
        document_ids=[document_id],
        image_embedder=FakeQueryEmbedder(),  # type: ignore[arg-type]
        text_embedder=text_embedder,  # type: ignore[arg-type]
    )

    assert images[0].source.image_id == text_match_id
    assert images[0].source.context_excerpt == "Ocicat, spotted coat"
    assert images[0].data == b"image-2"
    assert text_embedder.queries == ["Show the Ocicat"]
    assert len(session.executed_statements) == 3


def test_image_search_can_return_a_page_without_retrieved_text_chunks() -> None:
    document_id = uuid.UUID(int=142)
    image_id = uuid.UUID(int=184)
    session = FakeSearchSession(
        image_rows=[
            (
                document_id,
                "cats.pdf",
                image_id,
                7,
                "page-7-image-1.jpg",
                "image/jpeg",
                0.1,
            )
        ],
        image_data_rows=[(image_id, b"page-image-bytes", None)],
    )

    images = get_relevant_images(
        session,  # type: ignore[arg-type]
        question="calico cat",
        document_ids=[document_id],
        image_embedder=FakeQueryEmbedder(),  # type: ignore[arg-type]
        text_embedder=FakeTextEmbedder(),  # type: ignore[arg-type]
    )

    assert images[0].source.page_number == 7
    assert images[0].data == b"page-image-bytes"
