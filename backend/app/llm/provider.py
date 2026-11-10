from __future__ import annotations

from app.core.config import get_settings
from app.llm.groq_service import GroqLLMService, get_groq_service
from app.llm.ollama_service import OllamaLLMService


def get_llm_service() -> GroqLLMService | OllamaLLMService:
    settings = get_settings()
    if settings.llm_provider == "ollama":
        return OllamaLLMService(
            base_url=settings.ollama_base_url,
            model=settings.ollama_model,
            timeout_seconds=settings.ollama_timeout_seconds,
        )
    return get_groq_service()
