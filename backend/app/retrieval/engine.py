from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from enum import Enum
from typing import Protocol, Sequence

from rank_bm25 import BM25Okapi
from sqlalchemy import Select, exists, select
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.ingestion.embeddings import EmbeddingError, EmbeddingService
from app.models import Chunk, Document

RRF_K = 60
TOKEN_PATTERN = re.compile(r"\w+", re.UNICODE)


class RetrievalMode(str, Enum):
    DENSE = "dense"
    BM25 = "bm25"
    HYBRID = "hybrid"
    HYBRID_RERANK = "hybrid_rerank"


@dataclass(frozen=True)
class RetrievalResult:
    document_id: uuid.UUID
    document_name: str
    chunk_id: uuid.UUID
    chunk_text: str
    page_number: int | None
    chunk_index: int
    relevance_score: float


class RetrievalStore(Protocol):
    def has_dense_candidates(self, document_ids: Sequence[uuid.UUID] | None) -> bool: ...

    def dense_search(
        self,
        query_embedding: list[float],
        document_ids: Sequence[uuid.UUID] | None,
        limit: int,
    ) -> list[RetrievalResult]: ...

    def bm25_candidates(
        self,
        document_ids: Sequence[uuid.UUID] | None,
    ) -> list[RetrievalResult]: ...


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.casefold())


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[RetrievalResult]],
    *,
    limit: int,
) -> list[RetrievalResult]:
    fused_scores: dict[uuid.UUID, float] = {}
    result_by_id: dict[uuid.UUID, RetrievalResult] = {}

    for ranking in rankings:
        for rank, result in enumerate(ranking, start=1):
            fused_scores[result.chunk_id] = fused_scores.get(result.chunk_id, 0.0) + (
                1.0 / (RRF_K + rank)
            )
            result_by_id[result.chunk_id] = result

    ordered_ids = sorted(
        fused_scores,
        key=lambda chunk_id: (-fused_scores[chunk_id], str(chunk_id)),
    )
    return [
        RetrievalResult(
            document_id=result_by_id[chunk_id].document_id,
            document_name=result_by_id[chunk_id].document_name,
            chunk_id=result_by_id[chunk_id].chunk_id,
            chunk_text=result_by_id[chunk_id].chunk_text,
            page_number=result_by_id[chunk_id].page_number,
            chunk_index=result_by_id[chunk_id].chunk_index,
            relevance_score=fused_scores[chunk_id],
        )
        for chunk_id in ordered_ids[:limit]
    ]


def score_bm25(
    query: str,
    candidates: Sequence[RetrievalResult],
    *,
    limit: int,
) -> list[RetrievalResult]:
    query_tokens = tokenize(query)
    tokenized_documents = [tokenize(candidate.chunk_text) for candidate in candidates]
    if not query_tokens or not any(tokenized_documents):
        return []

    scorer = BM25Okapi(tokenized_documents)
    scores = scorer.get_scores(query_tokens)
    query_term_set = set(query_tokens)
    matching = [
        (float(score), candidate)
        for score, tokens, candidate in zip(
            scores,
            tokenized_documents,
            candidates,
            strict=True,
        )
        if query_term_set.intersection(tokens)
    ]
    if not matching:
        return []

    minimum_score = min(score for score, _ in matching)
    maximum_score = max(score for score, _ in matching)
    ranked = sorted(matching, key=lambda item: (-item[0], str(item[1].chunk_id)))
    return [
        RetrievalResult(
            document_id=candidate.document_id,
            document_name=candidate.document_name,
            chunk_id=candidate.chunk_id,
            chunk_text=candidate.chunk_text,
            page_number=candidate.page_number,
            chunk_index=candidate.chunk_index,
            relevance_score=(
                (score - minimum_score) / (maximum_score - minimum_score)
                if maximum_score > minimum_score
                else 1.0
            ),
        )
        for score, candidate in ranked[:limit]
    ]


class SQLAlchemyRetrievalStore:
    def __init__(self, session: Session) -> None:
        self.session = session

    @staticmethod
    def _filtered_statement(
        document_ids: Sequence[uuid.UUID] | None,
    ) -> Select[tuple[Chunk, str]]:
        statement = (
            select(Chunk, Document.filename)
            .join(Document, Chunk.document_id == Document.id)
            .where(Document.status == "ready")
        )
        if document_ids is not None:
            if not document_ids:
                statement = statement.where(False)
            else:
                statement = statement.where(Document.id.in_(document_ids))
        return statement

    def has_dense_candidates(self, document_ids: Sequence[uuid.UUID] | None) -> bool:
        statement = self._filtered_statement(document_ids).where(Chunk.embedding.is_not(None))
        return bool(self.session.scalar(select(exists(statement))))

    def dense_search(
        self,
        query_embedding: list[float],
        document_ids: Sequence[uuid.UUID] | None,
        limit: int,
    ) -> list[RetrievalResult]:
        distance = Chunk.embedding.cosine_distance(query_embedding)
        statement = (
            self._filtered_statement(document_ids)
            .where(Chunk.embedding.is_not(None))
            .add_columns(distance.label("cosine_distance"))
            .order_by(distance)
            .limit(limit)
        )
        rows = self.session.execute(statement).all()
        return [
            RetrievalResult(
                document_id=chunk.document_id,
                document_name=filename,
                chunk_id=chunk.id,
                chunk_text=chunk.content,
                page_number=chunk.page_number,
                chunk_index=chunk.chunk_index,
                relevance_score=max(0.0, 1.0 - float(distance_score)),
            )
            for chunk, filename, distance_score in rows
        ]

    def bm25_candidates(
        self,
        document_ids: Sequence[uuid.UUID] | None,
    ) -> list[RetrievalResult]:
        settings = get_settings()
        statement = (
            self._filtered_statement(document_ids)
            .order_by(Document.id, Chunk.chunk_index, Chunk.id)
            .limit(settings.retrieval_candidate_limit)
        )
        rows = self.session.execute(statement).all()
        return [
            RetrievalResult(
                document_id=chunk.document_id,
                document_name=filename,
                chunk_id=chunk.id,
                chunk_text=chunk.content,
                page_number=chunk.page_number,
                chunk_index=chunk.chunk_index,
                relevance_score=0.0,
            )
            for chunk, filename in rows
        ]


class RetrievalEngine:
    def __init__(
        self,
        store: RetrievalStore,
        embedder: EmbeddingService,
    ) -> None:
        self.store = store
        self.embedder = embedder

    def retrieve(
        self,
        query: str,
        *,
        mode: RetrievalMode | str = RetrievalMode.HYBRID,
        top_k: int = 5,
        document_ids: Sequence[uuid.UUID] | None = None,
    ) -> list[RetrievalResult]:
        normalized_query = query.strip()
        if not normalized_query:
            raise ValueError("Query must not be empty.")
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")
        try:
            retrieval_mode = RetrievalMode(mode)
        except ValueError as exc:
            raise ValueError(f"Unsupported retrieval mode: {mode}") from exc

        if document_ids is not None and not document_ids:
            return []

        settings = get_settings()
        candidate_limit = min(
            max(top_k, top_k * settings.retrieval_candidate_multiplier),
            settings.retrieval_candidate_limit,
        )
        if retrieval_mode is RetrievalMode.BM25:
            return score_bm25(
                normalized_query,
                self.store.bm25_candidates(document_ids),
                limit=top_k,
            )

        dense_results: list[RetrievalResult] = []
        if self.store.has_dense_candidates(document_ids):
            try:
                query_embedding = self.embedder.embed_texts([normalized_query])[0]
            except (EmbeddingError, IndexError) as exc:
                raise EmbeddingError("Could not embed retrieval query.") from exc
            dense_results = self.store.dense_search(
                query_embedding,
                document_ids,
                candidate_limit if retrieval_mode is RetrievalMode.HYBRID else top_k,
            )
        if retrieval_mode is RetrievalMode.DENSE:
            return dense_results[:top_k]

        bm25_results = score_bm25(
            normalized_query,
            self.store.bm25_candidates(document_ids),
            limit=candidate_limit,
        )
        if not dense_results:
            return bm25_results[:top_k]
        if not bm25_results:
            return dense_results[:top_k]
        return reciprocal_rank_fusion([dense_results, bm25_results], limit=top_k)


def get_retrieval_store(session: Session) -> SQLAlchemyRetrievalStore:
    return SQLAlchemyRetrievalStore(session)
