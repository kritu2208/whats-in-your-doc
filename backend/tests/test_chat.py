from __future__ import annotations

import json
import uuid
from collections.abc import Generator
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from app import main
from app.api import chat as chat_api
from app.api.chat import get_chat_repository, get_rag_workflow
from app.chat.repository import (
    ChatRepository,
    ConversationNotFoundError,
    DocumentsNotFoundError,
)
from app.core.config import get_settings
from app.db.base import Base
from app.db.session import get_db
from app.llm.groq_service import (
    GeneratedAnswer,
    GroqConfigurationError,
    GroqLLMService,
    GroqRequestError,
    VisionImage,
)
from app.models import Conversation, Document, DocumentImage, Message
from app.rag.workflow import RAGWorkflow, UNANSWERABLE_ANSWER
from app.retrieval.engine import RetrievalMode, RetrievalResult


def retrieval_result(
    index: int = 0,
    *,
    text: str = "The policy requires annual review.",
) -> RetrievalResult:
    return RetrievalResult(
        document_id=uuid.UUID(int=100),
        document_name="policy.pdf",
        chunk_id=uuid.UUID(int=index + 1),
        chunk_text=text,
        page_number=index + 1,
        chunk_index=index,
        relevance_score=0.5,
    )


@pytest.fixture(autouse=True)
def settings_environment(monkeypatch: pytest.MonkeyPatch) -> Generator[None, None, None]:
    monkeypatch.setenv("DATABASE_URL", "sqlite+pysqlite:///:memory:")
    monkeypatch.setenv("GROQ_API_KEY", "")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


class FakeRetrievalEngine:
    def __init__(self, chunks: list[RetrievalResult]) -> None:
        self.chunks = chunks
        self.calls: list[dict[str, Any]] = []

    def retrieve(self, query: str, **kwargs: Any) -> list[RetrievalResult]:
        self.calls.append({"query": query, **kwargs})
        return list(self.chunks)


class FakeReranker:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def rerank(
        self,
        query: str,
        candidates: list[RetrievalResult],
        *,
        top_k: int,
    ) -> list[RetrievalResult]:
        self.calls.append({"query": query, "candidates": candidates, "top_k": top_k})
        return candidates[:top_k]


class FakeLLM:
    def __init__(self, generated: GeneratedAnswer | None = None) -> None:
        self.generated = generated or GeneratedAnswer(
            "The policy requires annual review.",
            [str(uuid.UUID(int=1))],
            False,
            [],
        )
        self.calls: list[dict[str, Any]] = []

    def generate_answer(self, **kwargs: Any) -> GeneratedAnswer:
        self.calls.append(kwargs)
        return self.generated

    def stream_answer(self, **kwargs: Any) -> Generator[str, None, GeneratedAnswer]:
        generated = self.generate_answer(**kwargs)
        yield generated.answer
        return generated


def workflow_for(
    chunks: list[RetrievalResult],
    *,
    generated: GeneratedAnswer | None = None,
) -> tuple[RAGWorkflow, FakeRetrievalEngine, FakeReranker, FakeLLM]:
    retrieval = FakeRetrievalEngine(chunks)
    reranker = FakeReranker()
    llm = FakeLLM(generated)
    return RAGWorkflow(retrieval, reranker, llm), retrieval, reranker, llm


def test_langgraph_hybrid_rerank_pipeline_and_citation_shape() -> None:
    workflow, retrieval, reranker, llm = workflow_for(
        [retrieval_result(0), retrieval_result(1, text="Another fact.")]
    )

    result = workflow.invoke(
        question="What review is required?",
        conversation_id=None,
        document_ids=[uuid.UUID(int=100)],
        retrieval_mode=RetrievalMode.HYBRID_RERANK,
        top_k=1,
        candidate_k=3,
        history=[],
    )

    assert retrieval.calls[0]["mode"] is RetrievalMode.HYBRID
    assert retrieval.calls[0]["top_k"] == 3
    assert retrieval.calls[0]["document_ids"] == [uuid.UUID(int=100)]
    assert reranker.calls[0]["top_k"] == 1
    assert llm.calls[0]["context"].startswith("[chunk_id=")
    assert result["answer"] == "The policy requires annual review."
    assert result["cited_image_ids"] == []
    assert result["citations"] == [
        {
            "document_id": str(uuid.UUID(int=100)),
            "document_name": "policy.pdf",
            "chunk_id": str(uuid.UUID(int=1)),
            "page_number": 1,
            "chunk_index": 0,
            "excerpt": "The policy requires annual review.",
        }
    ]


def test_workflow_stream_reuses_retrieval_and_validates_final_citations() -> None:
    workflow, retrieval, reranker, llm = workflow_for([retrieval_result()])
    stream = workflow.stream(
        question="What review is required?",
        conversation_id=None,
        document_ids=None,
        retrieval_mode=RetrievalMode.BM25,
        top_k=1,
        candidate_k=5,
        history=[],
    )
    deltas: list[str] = []
    while True:
        try:
            deltas.append(next(stream))
        except StopIteration as stopped:
            result = stopped.value
            break

    assert deltas == ["The policy requires annual review."]
    assert result["answer"] == "The policy requires annual review."
    assert result["citations"][0]["chunk_id"] == str(uuid.UUID(int=1))
    assert retrieval.calls[0]["mode"] is RetrievalMode.BM25
    assert reranker.calls == []
    assert llm.calls[0]["question"] == "What review is required?"


def test_workflow_does_not_call_groq_without_retrieved_chunks() -> None:
    workflow, _, reranker, llm = workflow_for([])

    result = workflow.invoke(
        question="What is in the file?",
        conversation_id=None,
        document_ids=None,
        retrieval_mode=RetrievalMode.HYBRID_RERANK,
        top_k=5,
        candidate_k=20,
        history=[],
    )

    assert result["answer"] == UNANSWERABLE_ANSWER
    assert result["citations"] == []
    assert result["unanswerable"] is True
    assert reranker.calls == []
    assert llm.calls == []


def test_workflow_answers_from_cited_images_without_text_chunks() -> None:
    image_id = str(uuid.UUID(int=901))
    workflow, _, _, llm = workflow_for(
        [],
        generated=GeneratedAnswer(
            "This looks like an Ocicat.",
            [],
            False,
            [image_id],
        ),
    )

    result = workflow.invoke(
        question="What cat is shown?",
        conversation_id=None,
        document_ids=None,
        retrieval_mode=RetrievalMode.HYBRID_RERANK,
        top_k=5,
        candidate_k=20,
        history=[],
        images=[
            VisionImage(
                image_id=image_id,
                label="cats.pdf, page 1",
                media_type="image/jpeg",
                data=b"image-data",
            )
        ],
    )

    assert llm.calls[0]["context"] == ""
    assert result["answer"] == "This looks like an Ocicat."
    assert result["citations"] == []
    assert result["cited_image_ids"] == [image_id]
    assert result["unanswerable"] is False


def test_workflow_rejects_unknown_and_duplicate_citations() -> None:
    workflow, _, _, _ = workflow_for(
        [retrieval_result()],
        generated=GeneratedAnswer(
            "A generated answer.",
            [str(uuid.UUID(int=900)), str(uuid.UUID(int=1)), str(uuid.UUID(int=1))],
            False,
            [],
        ),
    )

    result = workflow.invoke(
        question="Question",
        conversation_id=None,
        document_ids=None,
        retrieval_mode=RetrievalMode.BM25,
        top_k=5,
        candidate_k=20,
        history=[],
    )

    assert result["answer"] == "A generated answer."
    assert len(result["citations"]) == 1
    assert result["citations"][0]["chunk_id"] == str(uuid.UUID(int=1))


def test_context_respects_configured_size_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_MAX_CONTEXT_CHARS", "1000")
    get_settings.cache_clear()
    long_chunk = retrieval_result(text="source " * 1000)

    state = RAGWorkflow.build_context({"chunks": [long_chunk]})

    assert len(state["context"]) <= 1000
    assert str(long_chunk.chunk_id) in state["context"]
    get_settings.cache_clear()


def test_workflow_retrieval_modes_only_rerank_hybrid_rerank() -> None:
    for mode in (RetrievalMode.DENSE, RetrievalMode.BM25, RetrievalMode.HYBRID):
        workflow, retrieval, reranker, _ = workflow_for([retrieval_result()])
        workflow.invoke(
            question="Question",
            conversation_id=None,
            document_ids=None,
            retrieval_mode=mode,
            top_k=1,
            candidate_k=20,
            history=[],
        )
        assert retrieval.calls[0]["mode"] is mode
        assert reranker.calls == []


def fake_groq_response(content: str) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content),
            )
        ]
    )


class FakeGroqClient:
    def __init__(self, response: Any = None, error: Exception | None = None) -> None:
        self.response = response
        self.error = error
        self.calls: list[dict[str, Any]] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return self.response


def test_groq_service_uses_mock_client_and_parses_grounded_response() -> None:
    client = FakeGroqClient(
        fake_groq_response(
            '{"answer":"Supported.","cited_chunk_ids":["chunk-1"],'
            '"cited_image_ids":[],"unanswerable":false}'
        )
    )
    client_options: dict[str, Any] = {}

    def factory(**kwargs: Any) -> FakeGroqClient:
        client_options.update(kwargs)
        return client

    service = GroqLLMService(
        api_key="test-key",
        model="test-model",
        vision_model="test-vision-model",
        timeout_seconds=12,
        client_factory=factory,
    )
    answer = service.generate_answer(
        question="Question",
        context="chunk-1: source text",
        history=[{"role": "user", "content": "Earlier question"}],
    )

    assert answer == GeneratedAnswer("Supported.", ["chunk-1"], False, [])
    assert client_options == {
        "api_key": "test-key",
        "timeout": 12,
        "max_retries": 1,
    }
    assert client.calls[0]["model"] == "test-model"
    assert any(item["role"] == "user" for item in client.calls[0]["messages"])


def test_groq_service_requires_key_and_skips_empty_context() -> None:
    service = GroqLLMService(
        api_key=None,
        model="test-model",
        vision_model="test-vision-model",
        timeout_seconds=12,
        client_factory=lambda **_: pytest.fail("client should not initialize"),
    )

    with pytest.raises(GroqConfigurationError, match="GROQ_API_KEY"):
        service.generate_answer(question="Question", context="context", history=[])
    assert service.generate_answer(question="Question", context="", history=[]).unanswerable


def test_groq_service_reports_provider_and_payload_failures_without_raw_error() -> None:
    service = GroqLLMService(
        api_key="test-key",
        model="test-model",
        vision_model="test-vision-model",
        timeout_seconds=12,
        client_factory=lambda **_: FakeGroqClient(
            error=RuntimeError("secret token leaked in remote exception")
        ),
    )
    with pytest.raises(GroqRequestError, match="request failed") as error:
        service.generate_answer(question="Question", context="source", history=[])
    assert "secret token" not in str(error.value)

    malformed = GroqLLMService(
        api_key="test-key",
        model="test-model",
        vision_model="test-vision-model",
        timeout_seconds=12,
        client_factory=lambda **_: FakeGroqClient(fake_groq_response("not json")),
    )
    with pytest.raises(GroqRequestError, match="malformed"):
        malformed.generate_answer(question="Question", context="source", history=[])


@pytest.mark.parametrize(
    ("status_code", "message"),
    [
        (401, "Groq rejected the API credentials"),
        (403, "Groq denied this request with HTTP 403"),
        (404, "Groq could not find the configured model"),
        (413, "Groq rejected the vision request because its payload was too large"),
        (429, "Groq rate or usage limit reached"),
    ],
)
def test_groq_service_explains_common_provider_statuses(
    status_code: int,
    message: str,
) -> None:
    class ProviderError(RuntimeError):
        def __init__(self) -> None:
            super().__init__("sensitive remote response")
            self.status_code = status_code

    service = GroqLLMService(
        api_key="test-key",
        model="test-model",
        vision_model="test-vision-model",
        timeout_seconds=12,
        client_factory=lambda **_: FakeGroqClient(error=ProviderError()),
    )

    with pytest.raises(GroqRequestError, match=message) as error:
        service.generate_answer(question="Question", context="source", history=[])
    assert "sensitive remote response" not in str(error.value)


def test_groq_service_sends_retrieved_images_to_vision_model() -> None:
    from io import BytesIO

    from PIL import Image

    image_buffer = BytesIO()
    Image.new("RGB", (4, 4), color=(150, 100, 50)).save(image_buffer, format="PNG")
    client = FakeGroqClient(
        fake_groq_response(
            '{"answer":"This appears to be an Ocicat.","cited_chunk_ids":[],'
            '"cited_image_ids":["image-1"],"unanswerable":false}'
        )
    )
    service = GroqLLMService(
        api_key="test-key",
        model="text-model",
        vision_model="vision-model",
        timeout_seconds=12,
        client_factory=lambda **_: client,
    )

    answer = service.generate_answer(
        question="What cat is shown?",
        context="",
        history=[],
        images=[
            VisionImage(
                image_id="image-1",
                label="cats.pdf, page 2",
                media_type="image/png",
                data=image_buffer.getvalue(),
            )
        ],
    )

    user_message = client.calls[0]["messages"][-1]
    image_content = user_message["content"]
    assert client.calls[0]["model"] == "vision-model"
    assert image_content[1]["text"] == (
        "Image source ID: image-1; source: cats.pdf, page 2; nearby document text: "
        "(no nearby text found)."
    )
    assert image_content[2]["type"] == "image_url"
    assert image_content[2]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert answer.cited_image_ids == ["image-1"]


def test_groq_service_bounds_vision_image_payload() -> None:
    import base64
    from io import BytesIO

    from PIL import Image

    noisy_image = Image.effect_noise((2048, 2048), 100).convert("RGB")
    image_buffer = BytesIO()
    noisy_image.save(image_buffer, format="JPEG", quality=95)

    data_url = GroqLLMService._image_data_url(image_buffer.getvalue())
    encoded_image = base64.b64decode(data_url.removeprefix("data:image/jpeg;base64,"))
    with Image.open(BytesIO(encoded_image)) as prepared_image:
        assert max(prepared_image.size) <= 768
    assert len(encoded_image) <= 512 * 1024


class FakeRepository:
    def __init__(self) -> None:
        self.persisted: dict[str, Any] | None = None
        self.history = [{"role": "user", "content": "Earlier question"}]
        self.validated_documents: list[uuid.UUID] | None = None
        self.existing_conversation: uuid.UUID | None = None

    def validate_documents(self, document_ids: list[uuid.UUID] | None) -> None:
        self.validated_documents = document_ids

    def get_conversation(self, conversation_id: uuid.UUID) -> object:
        self.existing_conversation = conversation_id
        return conversation_id

    def get_recent_history(self, conversation_id: uuid.UUID) -> list[dict[str, str]]:
        self.existing_conversation = conversation_id
        return self.history

    def persist_exchange(self, **kwargs: Any) -> uuid.UUID:
        self.persisted = kwargs
        return kwargs.get("conversation_id") or uuid.UUID(int=500)


class FakeWorkflow:
    def __init__(self) -> None:
        self.call: dict[str, Any] | None = None
        self.chunks: list[RetrievalResult] = []
        self.cited_image_ids: list[str] = []
        self.failure: Exception | None = None

    def invoke(self, **kwargs: Any) -> dict[str, Any]:
        self.call = kwargs
        if self.failure is not None:
            raise self.failure
        return {
            "chunks": self.chunks,
            "answer": "Grounded response.",
            "citations": [
                {
                    "document_id": str(uuid.UUID(int=100)),
                    "document_name": "policy.pdf",
                    "chunk_id": str(uuid.UUID(int=1)),
                    "page_number": 1,
                    "chunk_index": 0,
                    "excerpt": "Policy excerpt.",
                }
            ],
            "retrieval_metadata": {"mode": "hybrid_rerank", "result_count": 1},
            "cited_image_ids": self.cited_image_ids,
        }

    def stream(self, **kwargs: Any) -> Generator[str, None, dict[str, Any]]:
        result = self.invoke(**kwargs)
        yield result["answer"]
        return result


def test_chat_api_creates_conversation_persists_and_returns_response() -> None:
    repository = FakeRepository()
    workflow = FakeWorkflow()
    main.app.dependency_overrides[get_chat_repository] = lambda: repository
    main.app.dependency_overrides[get_rag_workflow] = lambda: workflow
    try:
        with TestClient(main.app) as client:
            response = client.post(
                "/api/chat/query",
                json={
                    "question": "What is required?",
                    "document_ids": [str(uuid.UUID(int=100))],
                },
            )
    finally:
        main.app.dependency_overrides.clear()

    assert response.status_code == 200
    payload = response.json()
    assert payload["answer"] == "Grounded response."
    assert payload["conversation_id"] == str(uuid.UUID(int=500))
    assert payload["citations"][0]["chunk_id"] == str(uuid.UUID(int=1))
    assert payload["images"] == []
    assert payload["retrieval_metadata"]["result_count"] == 1
    assert payload["latency_ms"] >= 0
    assert workflow.call is not None
    assert workflow.call["history"] == []
    assert repository.persisted is not None
    assert repository.persisted["question"] == "What is required?"


def test_chat_stream_api_emits_answer_deltas_and_final_sources() -> None:
    repository = FakeRepository()
    workflow = FakeWorkflow()
    main.app.dependency_overrides[get_chat_repository] = lambda: repository
    main.app.dependency_overrides[get_rag_workflow] = lambda: workflow
    try:
        with TestClient(main.app) as client:
            response = client.post(
                "/api/chat/query/stream",
                json={"question": "What is required?"},
            )
    finally:
        main.app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events = [
        block
        for block in response.text.strip().split("\n\n")
        if block.strip()
    ]
    assert [event.splitlines()[0] for event in events] == [
        "event: status",
        "event: status",
        "event: delta",
        "event: done",
    ]
    final = json.loads(events[-1].split("data: ", 1)[1])
    assert final["answer"] == "Grounded response."
    assert final["conversation_id"] == str(uuid.UUID(int=500))
    assert final["citations"][0]["chunk_id"] == str(uuid.UUID(int=1))
    assert repository.persisted is not None


@pytest.mark.parametrize(
    "provider_failure",
    [None, GroqRequestError("Groq rate limit reached.")],
)
def test_chat_api_returns_semantically_retrieved_images(
    monkeypatch: pytest.MonkeyPatch,
    provider_failure: GroqRequestError | None,
) -> None:
    document_id = uuid.UUID(int=100)
    expected_image_id = uuid.UUID(int=801)
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(
        engine,
        tables=[Document.__table__, DocumentImage.__table__],
    )
    session = Session(engine)
    session.add(
        Document(
            id=document_id,
            filename="policy.pdf",
            media_type="application/pdf",
            status="ready",
            document_metadata={"source_format": "pdf"},
        )
    )
    session.add_all(
        [
            DocumentImage(
                id=expected_image_id,
                document_id=document_id,
                page_number=1,
                image_index=0,
                filename="page-1-image-1.jpg",
                media_type="image/jpeg",
                data=b"image-on-retrieved-page",
            ),
            DocumentImage(
                id=uuid.UUID(int=802),
                document_id=document_id,
                page_number=2,
                image_index=0,
                filename="page-2-image-1.jpg",
                media_type="image/jpeg",
                data=b"image-on-unretrieved-page",
            ),
        ]
    )
    session.commit()

    repository = FakeRepository()
    workflow = FakeWorkflow()
    workflow.failure = provider_failure
    workflow.cited_image_ids = [str(expected_image_id)]

    def retrieve_test_image(
        *args: Any,
        **kwargs: Any,
    ) -> list[chat_api.RetrievedChatImage]:
        assert args == (session,)
        assert kwargs["question"] == "What is required?"
        return [
            chat_api.RetrievedChatImage(
                source=chat_api.ChatImage(
                    document_id=document_id,
                    document_name="policy.pdf",
                    image_id=expected_image_id,
                    page_number=1,
                    filename="page-1-image-1.jpg",
                    media_type="image/jpeg",
                    image_url=(
                        f"/api/documents/{document_id}/images/{expected_image_id}"
                    ),
                    context_excerpt="Policy image caption",
                ),
                data=b"image-on-retrieved-page",
            ),
        ]

    monkeypatch.setattr(
        chat_api,
        "get_relevant_images",
        retrieve_test_image,
    )
    main.app.dependency_overrides[get_db] = lambda: session
    main.app.dependency_overrides[get_chat_repository] = lambda: repository
    main.app.dependency_overrides[get_rag_workflow] = lambda: workflow
    try:
        with TestClient(main.app) as client:
            response = client.post(
                "/api/chat/query",
                json={"question": "What is required?", "include_images": True},
            )
            payload = response.json()
            images = payload.get("images", [])
            image_response = client.get(images[0]["image_url"]) if images else None
    finally:
        main.app.dependency_overrides.clear()
        session.close()
        engine.dispose()

    assert response.status_code == 200
    assert len(payload["images"]) == 1
    assert payload["images"][0]["image_id"] == str(expected_image_id)
    assert payload["images"][0]["page_number"] == 1
    assert payload["images"][0]["cited"] is (provider_failure is None)
    if provider_failure is None:
        assert payload["answer"] == "Grounded response."
        assert len(payload["citations"]) == 1
    else:
        assert "could not generate an AI answer" in payload["answer"]
        assert "Groq rate limit reached" in payload["answer"]
        assert payload["citations"] == []
    assert workflow.call is not None
    assert workflow.call["images"][0].image_id == str(expected_image_id)
    assert image_response is not None
    assert image_response.status_code == 200
    assert image_response.headers["content-type"] == "image/jpeg"
    assert image_response.content == b"image-on-retrieved-page"


def test_chat_api_continues_conversation_with_history() -> None:
    conversation_id = uuid.UUID(int=501)
    repository = FakeRepository()
    workflow = FakeWorkflow()
    main.app.dependency_overrides[get_chat_repository] = lambda: repository
    main.app.dependency_overrides[get_rag_workflow] = lambda: workflow
    try:
        with TestClient(main.app) as client:
            response = client.post(
                "/api/chat/query",
                json={
                    "question": "Follow-up?",
                    "conversation_id": str(conversation_id),
                },
            )
    finally:
        main.app.dependency_overrides.clear()

    assert response.status_code == 200
    assert repository.existing_conversation == conversation_id
    assert workflow.call is not None
    assert workflow.call["history"] == repository.history


@pytest.mark.parametrize(
    ("failure", "expected_status"),
    [
        (ConversationNotFoundError("Conversation not found."), 404),
        (DocumentsNotFoundError("Selected documents unavailable."), 404),
        (GroqConfigurationError("Groq API key is not configured."), 503),
        (GroqRequestError("Groq request failed."), 503),
    ],
)
def test_chat_api_maps_dependency_failures_to_http_responses(
    failure: Exception,
    expected_status: int,
) -> None:
    class FailureRepository(FakeRepository):
        def validate_documents(self, _document_ids: list[uuid.UUID] | None) -> None:
            del _document_ids
            if isinstance(failure, DocumentsNotFoundError):
                raise failure

        def get_conversation(self, _conversation_id: uuid.UUID) -> object:
            del _conversation_id
            if isinstance(failure, ConversationNotFoundError):
                raise failure
            return object()

    class FailureWorkflow(FakeWorkflow):
        def invoke(self, **kwargs: Any) -> dict[str, Any]:
            if isinstance(failure, (GroqConfigurationError, GroqRequestError)):
                raise failure
            return super().invoke(**kwargs)

    main.app.dependency_overrides[get_chat_repository] = FailureRepository
    main.app.dependency_overrides[get_rag_workflow] = FailureWorkflow
    try:
        with TestClient(main.app) as client:
            response = client.post(
                "/api/chat/query",
                json={
                    "question": "Question",
                    "conversation_id": str(uuid.UUID(int=123))
                    if isinstance(failure, ConversationNotFoundError)
                    else None,
                    "document_ids": [str(uuid.UUID(int=123))]
                    if isinstance(failure, DocumentsNotFoundError)
                    else None,
                },
            )
    finally:
        main.app.dependency_overrides.clear()

    assert response.status_code == expected_status


class FakeDatabaseSession:
    def __init__(self, *, available_documents: list[uuid.UUID] | None = None) -> None:
        self.available_documents = available_documents or []
        self.added: list[object] = []
        self.committed = False
        self.flushed = False
        self.history: list[Message] = []

    def scalars(self, statement: object) -> SimpleNamespace:
        del statement
        return SimpleNamespace(all=lambda: self.available_documents)

    def get(self, model: type[object], identifier: uuid.UUID) -> object | None:
        del model
        del identifier
        return None

    def add(self, instance: object) -> None:
        if isinstance(instance, Conversation) and instance.id is None:
            instance.id = uuid.UUID(int=700)
        self.added.append(instance)

    def flush(self) -> None:
        self.flushed = True

    def add_all(self, instances: list[object]) -> None:
        self.added.extend(instances)
        self.history.extend(item for item in instances if isinstance(item, Message))

    def commit(self) -> None:
        self.committed = True


def test_chat_repository_validates_selected_documents_and_persists_exchange() -> None:
    document_id = uuid.UUID(int=100)
    session = FakeDatabaseSession(available_documents=[document_id])
    repository = ChatRepository(session)  # type: ignore[arg-type]
    repository.validate_documents([document_id])

    conversation_id = repository.persist_exchange(
        conversation_id=None,
        question="Question?",
        answer="Answer.",
        citations=[{"chunk_id": str(uuid.UUID(int=1))}],
    )

    assert conversation_id == uuid.UUID(int=700)
    assert session.flushed and session.committed
    assert [message.role for message in session.history] == ["user", "assistant"]
    assert session.history[1].citations == [{"chunk_id": str(uuid.UUID(int=1))}]


def test_chat_repository_limits_history_to_recent_configured_messages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CHAT_HISTORY_MESSAGES", "3")
    get_settings.cache_clear()
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine, tables=[Conversation.__table__, Message.__table__])
    conversation_id = uuid.UUID(int=800)
    with Session(engine) as session:
        session.add(Conversation(id=conversation_id, title="History"))
        session.add_all(
            [
                Message(
                    id=uuid.UUID(int=index + 1),
                    conversation_id=conversation_id,
                    role="user" if index % 2 == 0 else "assistant",
                    content=f"message {index}",
                    created_at=datetime(2026, 1, 1, tzinfo=timezone.utc)
                    + timedelta(minutes=index),
                )
                for index in range(8)
            ]
        )
        session.commit()

        history = ChatRepository(session).get_recent_history(conversation_id)

    assert history == [
        {"role": "assistant", "content": "message 5"},
        {"role": "user", "content": "message 6"},
        {"role": "assistant", "content": "message 7"},
    ]
    get_settings.cache_clear()


def test_chat_repository_rejects_missing_documents_and_conversations() -> None:
    session = FakeDatabaseSession()
    repository = ChatRepository(session)  # type: ignore[arg-type]

    with pytest.raises(DocumentsNotFoundError):
        repository.validate_documents([uuid.UUID(int=404)])
    with pytest.raises(ConversationNotFoundError):
        repository.get_conversation(uuid.UUID(int=404))
