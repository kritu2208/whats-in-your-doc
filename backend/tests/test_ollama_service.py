from __future__ import annotations

import base64
import json
from io import BytesIO
from typing import Any

import httpx
import pytest
from PIL import Image

from app.llm.groq_service import LLMRequestError, VisionImage
from app.llm.ollama_service import OllamaLLMService


def test_ollama_service_sends_text_and_images_and_parses_grounded_answer() -> None:
    image_buffer = BytesIO()
    Image.new("RGB", (8, 8), color=(20, 80, 210)).save(image_buffer, format="PNG")
    image_id = "image-123"
    captured: dict[str, Any] = {}

    def handle_request(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["payload"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "message": {
                    "content": json.dumps(
                        {
                            "answer": "The figure is blue.",
                            "cited_chunk_ids": ["chunk-1"],
                            "cited_image_ids": [image_id],
                            "unanswerable": False,
                        }
                    )
                }
            },
        )

    transport = httpx.MockTransport(handle_request)
    service = OllamaLLMService(
        base_url="http://ollama.test:11434/",
        model="qwen3-vl:2b",
        timeout_seconds=30,
        client_factory=lambda **kwargs: httpx.Client(
            transport=transport,
            **kwargs,
        ),
    )

    answer = service.generate_answer(
        question="What color is the figure?",
        context="A figure appears in the source.",
        history=[{"role": "user", "content": "Earlier question"}],
        images=[
            VisionImage(
                image_id=image_id,
                label="guide.pdf, page 2",
                media_type="image/png",
                data=image_buffer.getvalue(),
                context_text="Figure 2",
            )
        ],
    )

    payload = captured["payload"]
    user_message = payload["messages"][-1]
    assert captured["url"] == "http://ollama.test:11434/api/chat"
    assert payload["model"] == "qwen3-vl:2b"
    assert payload["stream"] is False
    assert payload["format"] == "json"
    assert payload["messages"][1]["content"] == "Earlier question"
    assert "Figure 2" in user_message["content"]
    with Image.open(BytesIO(base64.b64decode(user_message["images"][0]))) as sent_image:
        assert sent_image.size == (8, 8)
    assert answer.answer == "The figure is blue."
    assert answer.cited_chunk_ids == ["chunk-1"]
    assert answer.cited_image_ids == [image_id]


def test_ollama_service_reports_missing_model() -> None:
    image_buffer = BytesIO()
    Image.new("RGB", (4, 4), color="blue").save(image_buffer, format="PNG")
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            404,
            request=request,
            json={"error": "model not found"},
        )
    )
    service = OllamaLLMService(
        base_url="http://ollama.test:11434",
        model="qwen3-vl:2b",
        timeout_seconds=30,
        client_factory=lambda **kwargs: httpx.Client(
            transport=transport,
            **kwargs,
        ),
    )

    with pytest.raises(LLMRequestError, match="ollama pull qwen3-vl:2b"):
        service.generate_answer(
            question="What is shown?",
            context="",
            history=[],
            images=[
                VisionImage(
                    image_id="image-1",
                    label="guide.pdf, page 1",
                    media_type="image/png",
                    data=image_buffer.getvalue(),
                )
            ],
        )
