from __future__ import annotations

import uuid
from collections.abc import Sequence
from typing import Any, cast

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app import main
from app.core.config import get_settings
from app.db.session import get_db
from app.ingestion.embeddings import get_embedding_service
from app.retrieval.engine import (
    RetrievalEngine,
    RetrievalMode,
    RetrievalResult,
    SQLAlchemyRetrievalStore,
    reciprocal_rank_fusion,
    score_bm25,
)


def result(
    index: int,
    text: str,
    *,
    document_id: uuid.UUID | None = None,
    score: float = 0.0,
) -> RetrievalResult:
    return RetrievalResult(
        document_id=document_id or uuid.UUID(int=1),
        document_name="guide.md",
        chunk_id=uuid.UUID(int=index + 1),
        chunk_text=text,
        page_number=index + 1,
        chunk_index=index,
        relevance_score=score,
    )


class FakeEmbedder:
    def __init__(self) -> None:
        self.queries: list[str] = []

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.queries.extend(texts)
        return [[0.25] * 384 for _ in texts]


class FakeStore:
    def __init__(
        self,
        dense: list[RetrievalResult] | None = None,
        bm25: list[RetrievalResult] | None = None,
    ) -> None:
        self.dense = dense or []
        self.bm25 = bm25 or []
        self.requested_document_ids: Sequence[uuid.UUID] | None = None
        self.dense_vector: list[float] | None = None
        self.dense_limit: int | None = None

    def has_dense_candidates(self, document_ids: Sequence[uuid.UUID] | None) -> bool:
        self.requested_document_ids = document_ids
        return bool(self.dense)

    def dense_search(
        self,
        query_embedding: list[float],
        document_ids: Sequence[uuid.UUID] | None,
        limit: int,
    ) -> list[RetrievalResult]:
        self.dense_vector = query_embedding
        self.requested_document_ids = document_ids
        self.dense_limit = limit
        return self.dense[:limit]

    def bm25_candidates(
        self,
        document_ids: Sequence[uuid.UUID] | None,
    ) -> list[RetrievalResult]:
        self.requested_document_ids = document_ids
        if document_ids is None:
            return self.bm25
        return [item for item in self.bm25 if item.document_id in document_ids]


def test_dense_retrieval_uses_query_embedding_and_top_k(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATABASE_URL", "sqlite+pysqlite:///:memory:")
    get_settings.cache_clear()
    dense = [result(index, f"chunk {index}", score=0.9 - index * 0.1) for index in range(4)]
    store = FakeStore(dense=dense)
    embedder = FakeEmbedder()

    try:
        results = RetrievalEngine(store, embedder).retrieve(
            "  relevant question  ",
            mode=RetrievalMode.DENSE,
            top_k=2,
        )
    finally:
        get_settings.cache_clear()

    assert [item.chunk_index for item in results] == [0, 1]
    assert embedder.queries == ["relevant question"]
    assert store.dense_vector == [0.25] * 384
    assert store.dense_limit == 2


def test_bm25_scores_relevant_terms_and_returns_empty_for_no_match() -> None:
    candidates = [
        result(0, "quantum entanglement experimental physics"),
        result(1, "car engine oil and tire maintenance"),
        result(2, "quantum states and entanglement"),
    ]

    ranked = score_bm25("quantum entanglement", candidates, limit=2)
    no_match = score_bm25("volcano geology", candidates, limit=5)

    assert {item.chunk_index for item in ranked} == {0, 2}
    assert max(item.relevance_score for item in ranked) == 1.0
    assert no_match == []


def test_rrf_combines_rankings_with_k_60_and_stable_tie_breaking() -> None:
    first = result(0, "first", score=0.9)
    shared = result(1, "shared", score=0.8)
    third = result(2, "third", score=0.7)

    fused = reciprocal_rank_fusion(
        [[first, shared], [shared, third]],
        limit=3,
    )

    assert [item.chunk_text for item in fused] == ["shared", "first", "third"]
    assert fused[0].relevance_score == pytest.approx((1 / 62) + (1 / 61))
    assert fused[1].relevance_score == pytest.approx(1 / 61)


def test_hybrid_retrieval_fuses_dense_and_bm25_candidates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DATABASE_URL", "sqlite+pysqlite:///:memory:")
    get_settings.cache_clear()
    dense = [result(0, "quantum experiments"), result(1, "classical mechanics")]
    bm25 = [
        result(1, "classical mechanics quantum"),
        result(2, "quantum field experiment"),
    ]
    store = FakeStore(dense=dense, bm25=bm25)

    try:
        retrieved = RetrievalEngine(store, FakeEmbedder()).retrieve(
            "quantum experiment",
            mode="hybrid",
            top_k=3,
        )
    finally:
        get_settings.cache_clear()

    assert {item.chunk_id for item in retrieved} == {
        dense[0].chunk_id,
        dense[1].chunk_id,
        bm25[1].chunk_id,
    }
    assert retrieved[0].relevance_score > 0


def test_document_filter_is_forwarded_and_applied_to_bm25(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DATABASE_URL", "sqlite+pysqlite:///:memory:")
    get_settings.cache_clear()
    selected_id = uuid.uuid4()
    included = result(0, "selected document keyword", document_id=selected_id)
    excluded = result(1, "selected document keyword", document_id=uuid.uuid4())
    store = FakeStore(bm25=[included, excluded])

    try:
        results = RetrievalEngine(store, FakeEmbedder()).retrieve(
            "keyword",
            mode="bm25",
            document_ids=[selected_id],
        )
    finally:
        get_settings.cache_clear()

    assert [item.chunk_id for item in results] == [included.chunk_id]
    assert store.requested_document_ids == [selected_id]


@pytest.mark.parametrize(
    ("query", "mode", "top_k"),
    [
        ("  ", "dense", 5),
        ("question", "unknown", 5),
        ("question", "dense", 0),
    ],
)
def test_empty_query_invalid_mode_and_invalid_top_k_are_rejected(
    query: str,
    mode: str,
    top_k: int,
) -> None:
    with pytest.raises(ValueError):
        RetrievalEngine(FakeStore(), FakeEmbedder()).retrieve(
            query,
            mode=mode,
            top_k=top_k,
        )


def test_dense_retrieval_with_empty_or_missing_embedding_corpus_does_not_load_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DATABASE_URL", "sqlite+pysqlite:///:memory:")
    get_settings.cache_clear()
    embedder = FakeEmbedder()

    try:
        empty_results = RetrievalEngine(FakeStore(), embedder).retrieve(
            "query",
            mode="dense",
        )
        hybrid_results = RetrievalEngine(
            FakeStore(bm25=[result(0, "matching query text")]),
            embedder,
        ).retrieve("query", mode="hybrid")
    finally:
        get_settings.cache_clear()

    assert empty_results == []
    assert [item.chunk_index for item in hybrid_results] == [0]
    assert embedder.queries == []


class FakeDatabaseSession:
    def __init__(self, rows: list[Any] | None = None) -> None:
        self.statements: list[Any] = []
        self.rows = rows or []

    def scalar(self, statement: Any) -> bool:
        self.statements.append(statement)
        return False

    def execute(self, statement: Any) -> FakeResult:
        self.statements.append(statement)
        return FakeResult(self.rows)


class FakeResult:
    def __init__(self, rows: list[Any] | None = None) -> None:
        self.rows = rows or []

    def all(self) -> list[Any]:
        return self.rows


def test_empty_database_retrieval_api_returns_structured_empty_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DATABASE_URL", "sqlite+pysqlite:///:memory:")
    get_settings.cache_clear()
    session = FakeDatabaseSession()
    embedder = FakeEmbedder()
    main.app.dependency_overrides[get_db] = lambda: session
    main.app.dependency_overrides[get_embedding_service] = lambda: embedder

    try:
        with TestClient(main.app) as client:
            response = client.post(
                "/api/retrieval/query",
                json={"query": "not found", "mode": "dense", "top_k": 5},
            )
    finally:
        main.app.dependency_overrides.clear()
        get_settings.cache_clear()

    assert response.status_code == 200
    assert response.json() == {"query": "not found", "mode": "dense", "results": []}
    assert embedder.queries == []


def test_retrieval_api_rejects_invalid_mode() -> None:
    main.app.dependency_overrides[get_db] = lambda: FakeDatabaseSession()
    main.app.dependency_overrides[get_embedding_service] = FakeEmbedder
    try:
        with TestClient(main.app) as client:
            response = client.post(
                "/api/retrieval/query",
                json={"query": "question", "mode": "rerank"},
            )
    finally:
        main.app.dependency_overrides.clear()
    assert response.status_code == 422


def test_sqlalchemy_dense_query_uses_ready_documents_and_cosine_operator() -> None:
    from sqlalchemy.dialects import postgresql

    session = FakeDatabaseSession()
    store = SQLAlchemyRetrievalStore(cast(Session, session))
    store.dense_search([0.0] * 384, [uuid.UUID(int=1)], limit=7)

    compiled = str(session.statements[0].compile(dialect=postgresql.dialect()))
    assert "chunks.embedding <=> " in compiled
    assert "documents.status = " in compiled
    assert "documents.id IN " in compiled
    assert "LIMIT" in compiled


def test_dense_availability_checks_ready_chunks_with_embeddings() -> None:
    from sqlalchemy.dialects import postgresql

    session = FakeDatabaseSession()
    store = SQLAlchemyRetrievalStore(cast(Session, session))

    assert not store.has_dense_candidates(None)

    compiled = str(session.statements[0].compile(dialect=postgresql.dialect()))
    assert "chunks.embedding IS NOT NULL" in compiled
    assert "documents.status = " in compiled


def test_sqlalchemy_dense_store_maps_similarity_and_result_fields() -> None:
    from app.models import Chunk

    doc_id = uuid.uuid4()
    chunk_id = uuid.uuid4()
    chunk = Chunk(
        id=chunk_id,
        document_id=doc_id,
        chunk_index=3,
        content="relevant text",
        page_number=8,
    )
    session = FakeDatabaseSession(rows=[(chunk, "manual.pdf", 0.25)])
    store = SQLAlchemyRetrievalStore(cast(Session, session))

    results = store.dense_search([0.0] * 384, None, limit=1)

    assert results == [
        RetrievalResult(
            document_id=doc_id,
            document_name="manual.pdf",
            chunk_id=chunk_id,
            chunk_text="relevant text",
            page_number=8,
            chunk_index=3,
            relevance_score=0.75,
        )
    ]


def test_sqlalchemy_bm25_corpus_query_applies_document_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DATABASE_URL", "sqlite+pysqlite:///:memory:")
    get_settings.cache_clear()
    session = FakeDatabaseSession()
    store = SQLAlchemyRetrievalStore(cast(Session, session))
    document_id = uuid.uuid4()

    try:
        assert store.bm25_candidates([document_id]) == []
    finally:
        get_settings.cache_clear()

    from sqlalchemy.dialects import postgresql

    compiled = str(session.statements[0].compile(dialect=postgresql.dialect()))
    assert "documents.status = " in compiled
    assert "documents.id IN " in compiled
    assert "LIMIT" in compiled
