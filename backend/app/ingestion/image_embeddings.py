from __future__ import annotations

from functools import lru_cache
from importlib import import_module
from io import BytesIO
from threading import Lock
from typing import Any, Callable, Protocol

from app.core.config import get_settings

IMAGE_EMBEDDING_DIMENSION = 512
IMAGE_EMBEDDING_BATCH_SIZE = 8


class ImageTextEncoder(Protocol):
    def encode(
        self,
        inputs: list[Any],
        *,
        batch_size: int,
        convert_to_numpy: bool,
        normalize_embeddings: bool,
        show_progress_bar: bool,
    ) -> Any: ...


class ImageEmbeddingError(RuntimeError):
    """Raised when image or text embedding generation fails."""


class ImageEmbeddingService:
    def __init__(
        self,
        model_name: str,
        model_factory: Callable[[str], ImageTextEncoder] | None = None,
    ) -> None:
        self.model_name = model_name
        self._model_factory = model_factory
        self._model: ImageTextEncoder | None = None
        self._model_lock = Lock()

    def _load_model(self) -> ImageTextEncoder:
        if self._model is None:
            with self._model_lock:
                if self._model is None:
                    if self._model_factory is not None:
                        self._model = self._model_factory(self.model_name)
                    else:
                        try:
                            sentence_transformers = import_module("sentence_transformers")
                        except ImportError as exc:
                            raise ImageEmbeddingError(
                                "sentence-transformers is not installed."
                            ) from exc
                        self._model = sentence_transformers.SentenceTransformer(
                            self.model_name,
                            device="cpu",
                        )
        return self._model

    def _encode(self, inputs: list[Any]) -> list[list[float]]:
        if not inputs:
            return []

        try:
            encoded = self._load_model().encode(
                inputs,
                batch_size=IMAGE_EMBEDDING_BATCH_SIZE,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            vectors = encoded.tolist()
        except ImageEmbeddingError:
            raise
        except Exception as exc:
            raise ImageEmbeddingError("Image-text embedding generation failed.") from exc

        if len(vectors) != len(inputs) or any(
            len(vector) != IMAGE_EMBEDDING_DIMENSION for vector in vectors
        ):
            raise ImageEmbeddingError(
                "The image-text model must return "
                f"{IMAGE_EMBEDDING_DIMENSION}-dimensional vectors."
            )
        return [[float(value) for value in vector] for vector in vectors]

    def embed_text(self, text: str) -> list[float]:
        vectors = self._encode([text])
        return vectors[0]

    def embed_images(self, image_data: list[bytes]) -> list[list[float]]:
        if not image_data:
            return []

        try:
            pillow_image = import_module("PIL.Image")
        except ImportError as exc:
            raise ImageEmbeddingError("Pillow is not installed.") from exc

        images = []
        try:
            for data in image_data:
                with pillow_image.open(BytesIO(data)) as image:
                    images.append(image.convert("RGB"))
            return self._encode(images)
        except ImageEmbeddingError:
            raise
        except Exception as exc:
            raise ImageEmbeddingError("A stored PDF image could not be decoded.") from exc
        finally:
            for image in images:
                image.close()


@lru_cache
def get_image_embedding_service() -> ImageEmbeddingService:
    settings = get_settings()
    return ImageEmbeddingService(settings.image_embedding_model_name)
