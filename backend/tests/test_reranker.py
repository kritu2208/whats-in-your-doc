from __future__ import annotations

import uuid
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app import main
from app.api import retrieval as retrieval_api
from app.core.config import Settings, get_settings
from app.db.session import get_db
from app.ingestion.embeddings import get_embedding_service
from app.retrieval.engine import RetrievalMode, RetrievalResult
from app.retrieval.reranker import (
    CrossEncoderReranker,
    RerankingError,
    get_reranker_service,
)


def candidate(index: int, text: str) -> RetrievalResult:
    return RetrievalResult(
        document_id=uuid.UUID(int=100),
        document_name="reference.txt",
        chunk_id=uuid.UUID(int=index + 1),
        chunk_text=text,
        page_number=index + 1,
        chunk_index=index,
        relevance_score=1 / (60 + index + 1),
    )


class FakeScores:
    def __init__(self, values: list[float]) -> None:
        self.values = values

    def tolist(self) -> list[float]:
        return self.values


class FakeCrossEncoder:
    def __init__(self, scores: list[float] | Exception) -> None:
        self.scores = scores
        self.calls: list[tuple[list[tuple[str, str]], int, bool, bool]] = []

    def predict(
        self,
        sentences: list[tuple[str, str]],
        *,
        batch_size: int,
        show_progress_bar: bool,
        convert_to_numpy: bool,
    ) -> FakeScores:
        self.calls.append((sentences, batch_size, show_progress_bar, convert_to_numpy))
        if isinstance(self.scores, Exception):
            raise self.scores
        return FakeScores(self.scores)


def test_reranker_is_lazy_scores_query_chunk_pairs_and_sorts() -> None:
    model = FakeCrossEncoder([0.1, 4.2, 1.7])
    model_creations: list[str] = []

    def factory(model_name: str) -> FakeCrossEncoder:
        model_creations.append(model_name)
        return model

    reranker = CrossEncoderReranker(
        "fake/model",
        batch_size=7,
        model_factory=factory,
    )
    candidates = [candidate(0, "low score"), candidate(1, "best score"), candidate(2, "middle score")]
    assert model_creations == []

    results = reranker.rerank("find the best", candidates, top_k=2)

    assert model_creations == ["fake/model"]
    assert [item.chunk_text for item in results] == ["best score", "middle score"]
    assert [item.relevance_score for item in results] == [4.2, 1.7]
    assert model.calls == [
        (
            [
                ("find the best", "low score"),
                ("find the best", "best score"),
                ("find the best", "middle score"),
            ],
            7,
            False,
            True,
        )
    ]


def test_reranker_caches_model_and_preserves_structured_candidate_metadata() -> None:
    model = FakeCrossEncoder([0.5, 0.5])
    loads = 0

    def factory(_: str) -> FakeCrossEncoder:
        nonlocal loads
        loads += 1
        return model

    reranker = CrossEncoderReranker("fake/model", batch_size=1, model_factory=factory)
    candidates = [candidate(0, "first"), candidate(1, "second")]
    reranker.rerank("query", candidates, top_k=2)
    results = reranker.rerank("query", candidates, top_k=2)

    assert loads == 1
    assert [item.chunk_index for item in results] == [0, 1]
    assert results[0].document_id == candidates[0].document_id
    assert results[0].page_number == candidates[0].page_number


def test_empty_candidates_do_not_load_model() -> None:
    def unexpected_factory(_: str) -> FakeCrossEncoder:
        raise AssertionError("The model should not be loaded for an empty candidate list.")

    reranker = CrossEncoderReranker("fake/model", batch_size=4, model_factory=unexpected_factory)

    assert reranker.rerank("query", [], top_k=5) == []


@pytest.mark.parametrize(
    ("model_name", "batch_size"),
    [("", 4), ("model", 0), ("model", 65)],
)
def test_invalid_reranker_configuration_is_rejected(
    model_name: str,
    batch_size: int,
) -> None:
    with pytest.raises(ValueError):
        CrossEncoderReranker(model_name, batch_size=batch_size)


def test_invalid_scores_and_model_failures_raise_actionable_errors() -> None:
    candidate_list = [candidate(0, "some text")]
    failing = CrossEncoderReranker(
        "broken/model",
        batch_size=4,
        model_factory=lambda _: _raise_model_load_error(),
    )
    with pytest.raises(RerankingError, match="Could not load the cross-encoder model"):
        failing.rerank("query", candidate_list, top_k=1)

    invalid_scores = CrossEncoderReranker(
        "invalid/model",
        batch_size=4,
        model_factory=lambda _: FakeCrossEncoder([float("nan")]),
    )
    with pytest.raises(RerankingError, match="invalid scores"):
        invalid_scores.rerank("query", candidate_list, top_k=1)


def _raise_model_load_error() -> FakeCrossEncoder:
    raise RuntimeError("model loading failed")


def test_reranking_model_failures_are_not_silently_swallowed() -> None:
    reranker = CrossEncoderReranker(
        "broken/model",
        batch_size=4,
        model_factory=lambda _: FakeCrossEncoder(RuntimeError("inference failed")),
    )

    with pytest.raises(RerankingError):
        reranker.rerank("query", [candidate(0, "candidate")], top_k=1)


class EmptyStore:
    def has_dense_candidates(self, document_ids: Any) -> bool:
        del document_ids
        return False

    def dense_search(self, query_embedding: Any, document_ids: Any, limit: int) -> list[Any]:
        del query_embedding, document_ids, limit
        return []

    def bm25_candidates(self, document_ids: Any) -> list[Any]:
        del document_ids
        return []


class EmptySession:
    pass


class NeverEmbedder:
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        del texts
        raise AssertionError("No dense candidates should require query embeddings.")


def make_empty_store(session: Any) -> EmptyStore:
    del session
    return EmptyStore()


def test_hybrid_rerank_mode_requests_candidate_k_then_returns_final_top_k(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DATABASE_URL", "sqlite+pysqlite:///:memory:")
    get_settings.cache_clear()
    hybrid_candidates = [
        candidate(0, "candidate 0"),
        candidate(1, "candidate 1"),
        candidate(2, "candidate 2"),
        candidate(3, "candidate 3"),
    ]
    monkeypatch.setattr(
        retrieval_api,
        "SQLAlchemyRetrievalStore",
        make_empty_store,
    )

    class FakeRetrievalEngine:
        def __init__(self, store: Any, embedder: Any) -> None:
            del store, embedder

        def retrieve(self, query: str, *, mode: RetrievalMode, top_k: int, document_ids: Any) -> list[RetrievalResult]:
            assert query == "query"
            assert mode is RetrievalMode.HYBRID
            assert top_k == 4
            assert document_ids is None
            return hybrid_candidates[:top_k]

    class FakeReranker:
        def rerank(
            self,
            query: str,
            candidates: list[RetrievalResult],
            *,
            top_k: int,
        ) -> list[RetrievalResult]:
            assert query == "query"
            assert len(candidates) == 4
            assert top_k == 2
            return [candidate(3, "reranked 3"), candidate(1, "reranked 1")][:top_k]

    monkeypatch.setattr(retrieval_api, "RetrievalEngine", FakeRetrievalEngine)
    main.app.dependency_overrides[get_db] = lambda: EmptySession()
    main.app.dependency_overrides[get_embedding_service] = NeverEmbedder
    main.app.dependency_overrides[get_reranker_service] = FakeReranker

    try:
        with TestClient(main.app) as client:
            response = client.post(
                "/api/retrieval/query",
                json={
                    "query": "query",
                    "mode": "hybrid_rerank",
                    "top_k": 2,
                    "candidate_k": 4,
                },
            )
    finally:
        main.app.dependency_overrides.clear()
        get_settings.cache_clear()

    assert response.status_code == 200
    payload = response.json()
    assert payload["mode"] == "hybrid_rerank"
    assert [item["chunk_text"] for item in payload["results"]] == ["reranked 3", "reranked 1"]


def test_hybrid_rerank_api_returns_503_when_reranking_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DATABASE_URL", "sqlite+pysqlite:///:memory:")
    get_settings.cache_clear()
    monkeypatch.setattr(
        retrieval_api,
        "SQLAlchemyRetrievalStore",
        make_empty_store,
    )

    class CandidateEngine:
        def __init__(self, store: Any, embedder: Any) -> None:
            del store, embedder

        def retrieve(self, *args: Any, **kwargs: Any) -> list[RetrievalResult]:
            del args, kwargs
            return [candidate(0, "already retrieved")]

    class BrokenReranker:
        def rerank(self, *args: Any, **kwargs: Any) -> list[RetrievalResult]:
            del args, kwargs
            raise RerankingError("model unavailable")

    monkeypatch.setattr(retrieval_api, "RetrievalEngine", CandidateEngine)
    main.app.dependency_overrides[get_db] = lambda: EmptySession()
    main.app.dependency_overrides[get_embedding_service] = NeverEmbedder
    main.app.dependency_overrides[get_reranker_service] = BrokenReranker

    try:
        with TestClient(main.app, raise_server_exceptions=False) as client:
            response = client.post(
                "/api/retrieval/query",
                json={"query": "query", "mode": "hybrid_rerank", "top_k": 1},
            )
    finally:
        main.app.dependency_overrides.clear()
        get_settings.cache_clear()

    assert response.status_code == 503
    assert "Verify RERANKER_MODEL_NAME" in response.json()["detail"]


def test_candidate_k_must_cover_final_top_k() -> None:
    main.app.dependency_overrides[get_db] = lambda: EmptySession()
    main.app.dependency_overrides[get_embedding_service] = NeverEmbedder
    main.app.dependency_overrides[get_reranker_service] = lambda: None
    try:
        with TestClient(main.app) as client:
            response = client.post(
                "/api/retrieval/query",
                json={
                    "query": "query",
                    "mode": "hybrid_rerank",
                    "top_k": 5,
                    "candidate_k": 3,
                },
            )
    finally:
        main.app.dependency_overrides.clear()

    assert response.status_code == 422
    assert "candidate_k" in response.json()["detail"]


def test_reranker_defaults_and_invalid_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = Settings(_env_file=None, database_url="sqlite://")
    assert settings.reranker_model_name == "cross-encoder/ms-marco-MiniLM-L-6-v2"
    assert settings.reranker_batch_size == 16
    assert settings.retrieval_candidate_k == 20

    monkeypatch.setenv("DATABASE_URL", "sqlite://")
    monkeypatch.setenv("RETRIEVAL_CANDIDATE_K", "0")
    get_settings.cache_clear()
    try:
        with pytest.raises(ValueError):
            get_settings()
    finally:
        get_settings.cache_clear()
