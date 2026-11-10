from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    database_url: str
    embedding_model_name: str = "BAAI/bge-small-en-v1.5"
    image_embedding_model_name: str = "sentence-transformers/clip-ViT-B-32"
    upload_max_bytes: int = Field(default=10 * 1024 * 1024, gt=0)
    chunk_size_tokens: int = Field(default=500, gt=0)
    chunk_overlap_tokens: int = Field(default=50, ge=0)
    max_document_chunks: int = Field(default=2000, gt=0)
    retrieval_candidate_limit: int = Field(default=10000, gt=0)
    retrieval_candidate_multiplier: int = Field(default=4, ge=1)
    retrieval_candidate_k: int = Field(default=20, ge=1, le=100)
    reranker_model_name: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        min_length=1,
    )
    reranker_batch_size: int = Field(default=16, ge=1, le=64)
    llm_provider: str = Field(default="groq", min_length=1)
    groq_api_key: str | None = None
    groq_model: str = Field(default="openai/gpt-oss-20b", min_length=1)
    groq_vision_model: str = Field(
        default="qwen/qwen3.8-27b",
        min_length=1,
    )
    groq_timeout_seconds: float = Field(default=45.0, gt=0, le=120)
    ollama_base_url: str = Field(default="http://host.docker.internal:11434", min_length=1)
    ollama_model: str = Field(default="qwen3-vl:2b", min_length=1)
    ollama_timeout_seconds: float = Field(default=300.0, gt=0, le=900)
    chat_history_messages: int = Field(default=6, ge=0, le=20)
    rag_max_context_chars: int = Field(default=30_000, ge=1000, le=100_000)

    @model_validator(mode="after")
    def validate_chunk_overlap(self) -> Settings:
        if self.chunk_overlap_tokens >= self.chunk_size_tokens:
            raise ValueError("chunk_overlap_tokens must be smaller than chunk_size_tokens.")
        if not self.reranker_model_name.strip():
            raise ValueError("reranker_model_name must not be blank.")
        if not self.groq_model.strip():
            raise ValueError("groq_model must not be blank.")
        if not self.groq_vision_model.strip():
            raise ValueError("groq_vision_model must not be blank.")
        if self.llm_provider not in {"groq", "ollama"}:
            raise ValueError("llm_provider must be 'groq' or 'ollama'.")
        if not self.ollama_base_url.strip():
            raise ValueError("ollama_base_url must not be blank.")
        if not self.ollama_model.strip():
            raise ValueError("ollama_model must not be blank.")
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
