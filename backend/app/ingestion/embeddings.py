from __future__ import annotations

from functools import lru_cache
from importlib import import_module
from threading import Lock
from typing import Any, Callable, Protocol

from app.core.config import get_settings

EMBEDDING_DIMENSION = 384


class SentenceEncoder(Protocol):
    def encode(
        self,
        sentences: list[str],
        *,
        batch_size: int,
        convert_to_numpy: bool,
        normalize_embeddings: bool,
        show_progress_bar: bool,
    ) -> Any: ...


class EmbeddingError(RuntimeError):
    """Raised when embedding generation fails or produces invalid dimensions."""


class EmbeddingService:
    def __init__(
        self,
        model_name: str,
        model_factory: Callable[[str], SentenceEncoder] | None = None,
    ) -> None:
        self.model_name = model_name
        self._model_factory = model_factory
        self._model: SentenceEncoder | None = None
        self._model_lock = Lock()

    def _load_model(self) -> SentenceEncoder:
        if self._model is None:
            with self._model_lock:
                if self._model is None:
                    if self._model_factory is not None:
                        self._model = self._model_factory(self.model_name)
                    else:
                        try:
                            sentence_transformers = import_module("sentence_transformers")
                        except ImportError as exc:
                            raise EmbeddingError(
                                "sentence-transformers is not installed."
                            ) from exc
                        self._model = sentence_transformers.SentenceTransformer(
                            self.model_name,
                            device="cpu",
                        )
        return self._model

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        try:
            encoded = self._load_model().encode(
                texts,
                batch_size=16,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            vectors = encoded.tolist()
        except EmbeddingError:
            raise
        except Exception as exc:
            raise EmbeddingError("Embedding generation failed.") from exc

        if len(vectors) != len(texts) or any(
            len(vector) != EMBEDDING_DIMENSION for vector in vectors
        ):
            raise EmbeddingError(
                f"The embedding model must return {EMBEDDING_DIMENSION}-dimensional vectors."
            )
        return [[float(value) for value in vector] for vector in vectors]


@lru_cache
@lru_cache
def get_embedding_service() -> EmbeddingService:
    return EmbeddingService(get_settings().embedding_model_name)
