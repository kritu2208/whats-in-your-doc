import os

import pytest
from sqlalchemy import Engine
from sqlalchemy.dialects import postgresql
from sqlalchemy.schema import CreateIndex

from app.core.config import Settings, get_settings
from app.db.base import Base
from app.db.session import get_engine
from app.models import Chunk, Conversation, Document, DocumentImage, EvalRun, Message
from pgvector.sqlalchemy import Vector


def test_settings_load_database_url_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    database_url = "postgresql+psycopg://test_user:test_password@localhost/test_db"
    monkeypatch.setenv("DATABASE_URL", database_url)
    get_settings.cache_clear()

    try:
        assert get_settings().database_url == database_url
    finally:
        get_settings.cache_clear()


def test_settings_accept_database_url_directly() -> None:
    settings = Settings(
        _env_file=None,
        database_url="postgresql+psycopg://user:password@localhost/rag",
    )

    assert settings.database_url.endswith("/rag")


def test_settings_support_local_ollama_provider() -> None:
    settings = Settings(
        _env_file=None,
        database_url="postgresql+psycopg://test:test@localhost/rag",
        llm_provider="ollama",
        ollama_base_url="http://host.docker.internal:11434",
        ollama_model="qwen3-vl:2b",
    )

    assert settings.llm_provider == "ollama"
    assert settings.ollama_model == "qwen3-vl:2b"


def test_engine_uses_configured_postgresql_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(
        "DATABASE_URL",
        "postgresql+psycopg://test_user:test_password@localhost/test_db",
    )
    get_settings.cache_clear()
    get_engine.cache_clear()

    try:
        engine = get_engine()
        assert isinstance(engine, Engine)
        assert engine.dialect.name == "postgresql"
        assert engine.url.database == "test_db"
    finally:
        if get_engine.cache_info().currsize:
            get_engine().dispose()
        get_engine.cache_clear()
        get_settings.cache_clear()


def test_models_are_registered_with_expected_tables_and_relationships() -> None:
    assert set(Base.metadata.tables) == {
        "chunks",
        "conversations",
        "documents",
        "document_images",
        "eval_runs",
        "messages",
    }
    assert Document.chunks.property.mapper.class_ is Chunk
    assert Message.conversation.property.mapper.class_ is Conversation
    assert Chunk.document.property.mapper.class_ is Document
    assert DocumentImage.document.property.mapper.class_ is Document
    assert EvalRun.__tablename__ == "eval_runs"


def test_chunk_embedding_is_384_dimensional_and_uses_cosine_hnsw() -> None:
    embedding_type = Chunk.__table__.c.embedding.type
    assert isinstance(embedding_type, Vector)
    assert embedding_type.dim == 384

    hnsw_index = next(
        index for index in Chunk.__table__.indexes if index.name == "ix_chunks_embedding_hnsw"
    )
    compiled = str(CreateIndex(hnsw_index).compile(dialect=postgresql.dialect()))
    assert "USING hnsw" in compiled
    assert "vector_cosine_ops" in compiled

    context_type = DocumentImage.__table__.c.context_embedding.type
    assert isinstance(context_type, Vector)
    assert context_type.dim == 384
    assert "context_text" in DocumentImage.__table__.c


def test_image_embedding_is_512_dimensional_and_uses_cosine_hnsw() -> None:
    embedding_type = DocumentImage.__table__.c.image_embedding.type
    assert isinstance(embedding_type, Vector)
    assert embedding_type.dim == 512

    hnsw_index = next(
        index
        for index in DocumentImage.__table__.indexes
        if index.name == "ix_document_images_embedding_hnsw"
    )
    compiled = str(CreateIndex(hnsw_index).compile(dialect=postgresql.dialect()))
    assert "USING hnsw" in compiled
    assert "vector_cosine_ops" in compiled


def test_foreign_keys_cascade_on_delete() -> None:
    chunk_foreign_key = next(iter(Chunk.__table__.c.document_id.foreign_keys))
    message_foreign_key = next(iter(Message.__table__.c.conversation_id.foreign_keys))
    image_foreign_key = next(iter(DocumentImage.__table__.c.document_id.foreign_keys))
    assert chunk_foreign_key.ondelete == "CASCADE"
    assert message_foreign_key.ondelete == "CASCADE"
    assert image_foreign_key.ondelete == "CASCADE"


@pytest.mark.skipif(
    not os.environ.get("TEST_DATABASE_URL"),
    reason="Set TEST_DATABASE_URL to run the PostgreSQL integration check.",
)
def test_postgresql_connection() -> None:
    from sqlalchemy import create_engine, text

    engine = create_engine(os.environ["TEST_DATABASE_URL"], pool_pre_ping=True)
    try:
        with engine.connect() as connection:
            assert connection.scalar(text("SELECT 1")) == 1
    finally:
        engine.dispose()


def test_default_vision_model_is_current_groq_vision_model() -> None:
    settings = Settings(
        _env_file=None,
        database_url="postgresql+psycopg://test:test@localhost/test",
    )

    assert settings.groq_vision_model == "qwen/qwen3.8-27b"
