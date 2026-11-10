from __future__ import annotations

from functools import lru_cache
from importlib import import_module
from math import isfinite
from threading import Lock
from typing import Any, Callable, Protocol

from app.core.config import get_settings
from app.retrieval.engine import RetrievalResult


class CrossEncoderModel(Protocol):
    def predict(
        self,
        sentences: list[tuple[str, str]],
        *,
        batch_size: int,
        show_progress_bar: bool,
        convert_to_numpy: bool,
    ) -> Any: ...


class RerankingError(RuntimeError):
    """Raised when the cross-encoder cannot be loaded or score candidates."""


class CrossEncoderReranker:
    def __init__(
        self,
        model_name: str,
        *,
        batch_size: int,
        model_factory: Callable[[str], CrossEncoderModel] | None = None,
    ) -> None:
        if not model_name.strip():
            raise ValueError("Reranker model name must not be blank.")
        if batch_size <= 0 or batch_size > 64:
            raise ValueError("Reranker batch size must be between 1 and 64.")
        self.model_name = model_name
        self.batch_size = batch_size
        self._model_factory = model_factory
        self._model: CrossEncoderModel | None = None
        self._model_lock = Lock()

    def _load_model(self) -> CrossEncoderModel:
        if self._model is None:
            with self._model_lock:
                if self._model is None:
                    if self._model_factory is not None:
                        try:
                            self._model = self._model_factory(self.model_name)
                        except Exception as exc:
                            raise RerankingError(
                                "Could not load the cross-encoder model "
                                f"'{self.model_name}'. Check the model name and network access."
                            ) from exc
                    else:
                        try:
                            sentence_transformers = import_module("sentence_transformers")
                            cross_encoder_class = sentence_transformers.CrossEncoder
                        except (ImportError, AttributeError) as exc:
                            raise RerankingError(
                                "Cross-encoder support is unavailable. Install the "
                                "sentence-transformers dependency."
                            ) from exc
                        try:
                            self._model = cross_encoder_class(
                                self.model_name,
                                device="cpu",
                            )
                        except Exception as exc:
                            raise RerankingError(
                                "Could not load the cross-encoder model "
                                f"'{self.model_name}'. Check the model name and network access."
                            ) from exc
        return self._model

    def rerank(
        self,
        query: str,
        candidates: list[RetrievalResult],
        *,
        top_k: int,
    ) -> list[RetrievalResult]:
        if not query.strip():
            raise ValueError("Query must not be empty.")
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")
        if not candidates:
            return []

        try:
            raw_scores = self._load_model().predict(
                [(query, candidate.chunk_text) for candidate in candidates],
                batch_size=self.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            score_values = raw_scores.tolist() if hasattr(raw_scores, "tolist") else raw_scores
            scores = [float(score) for score in score_values]
        except RerankingError:
            raise
        except Exception as exc:
            raise RerankingError("The cross-encoder failed to score retrieval candidates.") from exc

        if len(scores) != len(candidates) or any(not isfinite(score) for score in scores):
            raise RerankingError(
                "The cross-encoder returned invalid scores for the candidate set."
            )

        ranked = sorted(
            zip(candidates, scores, strict=True),
            key=lambda item: (-item[1], str(item[0].chunk_id)),
        )
        return [
            RetrievalResult(
                document_id=candidate.document_id,
                document_name=candidate.document_name,
                chunk_id=candidate.chunk_id,
                chunk_text=candidate.chunk_text,
                page_number=candidate.page_number,
                chunk_index=candidate.chunk_index,
                relevance_score=score,
            )
            for candidate, score in ranked[:top_k]
        ]


@lru_cache
def get_reranker_service() -> CrossEncoderReranker:
    settings = get_settings()
    return CrossEncoderReranker(
        settings.reranker_model_name,
        batch_size=settings.reranker_batch_size,
    )
