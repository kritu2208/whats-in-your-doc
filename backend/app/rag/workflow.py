from __future__ import annotations

import re
import uuid
from collections.abc import Generator
from typing import Any, Protocol, TypedDict

from langgraph.graph import END, START, StateGraph

from app.core.config import get_settings
from app.llm.groq_service import GeneratedAnswer, VisionImage
from app.retrieval.engine import RetrievalEngine, RetrievalMode, RetrievalResult
from app.retrieval.reranker import CrossEncoderReranker

UNANSWERABLE_ANSWER = (
    "The uploaded documents do not provide enough information to answer this question."
)
DEFAULT_GREETING_ANSWER = (
    "Hey! How can I help you with your uploaded documents today?"
)

_GREETING_PATTERN = re.compile(
    r"^(?:"
    r"h+e+y+|h+i+|h+e+l+o+|h+i+y+a+|h+o+l+a+|y+o+|s+u+p+|namaste|"
    r"good\s+(?:morning|afternoon|evening|night)|gm|"
    r"what(?:'s|s)?\s+up|"
    r"how\s+(?:are\s+(?:you|u)|r\s+u|(?:is|'s)\s+it\s+going)|"
    r"thank\s*(?:you|u)|thanks|thx|ty|"
    r"bye|goodbye|see\s+(?:you|ya)"
    r")(?:\s+(?:there|bro|buddy|mate|friend|bot|assistant|ai))?$"
)


_IMAGE_REQUEST_PATTERN = re.compile(
    r"\b(?:images?|pictures?|photos?|pics?|diagrams?|figures?|illustrations?|"
    r"charts?|graphs?|screenshots?|visuals?|show\s+me|looks?\s+like)\b"
)
_FOLLOWUP_REFERENCE_PATTERN = re.compile(
    r"\b(?:it|its|this|that|these|those|they|them|their|above|earlier|"
    r"more|else|simply|simplify|example|summarize|elaborate)\b|"
    r"(?:like\s+(?:a|an)\s+\d+|explain\s+again|tell\s+me\s+more)"
)


def is_conversational_greeting(text: str) -> bool:
    """Return True when input is a pure conversational greeting or pleasantry."""
    normalized = re.sub(r"[^a-z0-9'\s]", " ", text.lower())
    normalized = re.sub(r"\s+", " ", normalized).strip()
    if not normalized:
        return False
    return bool(_GREETING_PATTERN.fullmatch(normalized))


def is_image_request(text: str) -> bool:
    """Return True when the user explicitly asks to see or find an image/figure."""
    if is_conversational_greeting(text):
        return False
    return bool(_IMAGE_REQUEST_PATTERN.search(text.lower()))


def build_retrieval_query(question: str, history: list[dict[str, str]]) -> str:
    """Expand dependent follow-up questions with the prior user turn for retrieval."""
    cleaned = question.strip()
    if not cleaned or not history:
        return cleaned
    lowered = cleaned.lower()
    is_dependent = bool(_FOLLOWUP_REFERENCE_PATTERN.search(lowered)) or len(lowered.split()) <= 3
    if not is_dependent:
        return cleaned
    for item in reversed(history):
        if item.get("role") != "user":
            continue
        prior = (item.get("content") or "").strip()
        if not prior or is_conversational_greeting(prior):
            continue
        if prior.lower() == lowered:
            continue
        return f"{prior[:200]} {cleaned}"
    return cleaned


class AnswerGenerator(Protocol):
    def generate_answer(
        self,
        *,
        question: str,
        context: str,
        history: list[dict[str, str]],
        images: list[VisionImage] | None = None,
    ) -> GeneratedAnswer: ...

    def stream_answer(
        self,
        *,
        question: str,
        context: str,
        history: list[dict[str, str]],
        images: list[VisionImage] | None = None,
    ) -> Generator[str, None, GeneratedAnswer]: ...


class RAGState(TypedDict, total=False):
    question: str
    retrieval_query: str
    conversation_id: uuid.UUID | None
    document_ids: list[uuid.UUID] | None
    retrieval_mode: RetrievalMode
    top_k: int
    candidate_k: int | None
    history: list[dict[str, str]]
    images: list[VisionImage]
    candidates: list[RetrievalResult]
    chunks: list[RetrievalResult]
    context: str
    answer: str
    cited_chunk_ids: list[str]
    cited_image_ids: list[str]
    citations: list[dict[str, Any]]
    conversational: bool
    unanswerable: bool
    retrieval_metadata: dict[str, Any]


class RAGWorkflow:
    def __init__(
        self,
        retrieval_engine: RetrievalEngine,
        reranker: CrossEncoderReranker,
        llm: AnswerGenerator,
    ) -> None:
        self.retrieval_engine = retrieval_engine
        self.reranker = reranker
        self.llm = llm
        self.graph = self._build_graph()

    def _build_graph(self) -> Any:
        graph = StateGraph(RAGState)
        graph.add_node("receive_question", self.receive_question)
        graph.add_node("retrieve_chunks", self.retrieve_chunks)
        graph.add_node("rerank_chunks", self.rerank_chunks)
        graph.add_node("build_context", self.build_context)
        graph.add_node("generate_answer", self.generate_answer)
        graph.add_node("validate_citations", self.validate_citations)
        graph.add_node("finalize_response", self.finalize_response)

        graph.add_edge(START, "receive_question")
        graph.add_edge("receive_question", "retrieve_chunks")
        graph.add_edge("retrieve_chunks", "rerank_chunks")
        graph.add_edge("rerank_chunks", "build_context")
        graph.add_edge("build_context", "generate_answer")
        graph.add_edge("generate_answer", "validate_citations")
        graph.add_edge("validate_citations", "finalize_response")
        graph.add_edge("finalize_response", END)
        return graph.compile()

    def invoke(
        self,
        *,
        question: str,
        conversation_id: uuid.UUID | None,
        document_ids: list[uuid.UUID] | None,
        retrieval_mode: RetrievalMode,
        top_k: int,
        candidate_k: int | None,
        history: list[dict[str, str]],
        images: list[VisionImage] | None = None,
    ) -> RAGState:
        settings = get_settings()
        return self.graph.invoke(
            {
                "question": question,
                "conversation_id": conversation_id,
                "document_ids": document_ids,
                "retrieval_mode": retrieval_mode,
                "top_k": top_k,
                "candidate_k": candidate_k,
                "history": history[-settings.chat_history_messages :]
                if settings.chat_history_messages
                else [],
                "images": images or [],
            }
        )

    def stream(
        self,
        *,
        question: str,
        conversation_id: uuid.UUID | None,
        document_ids: list[uuid.UUID] | None,
        retrieval_mode: RetrievalMode,
        top_k: int,
        candidate_k: int | None,
        history: list[dict[str, str]],
        images: list[VisionImage] | None = None,
    ) -> Generator[str, None, RAGState]:
        settings = get_settings()
        state: RAGState = {
            "question": question,
            "conversation_id": conversation_id,
            "document_ids": document_ids,
            "retrieval_mode": retrieval_mode,
            "top_k": top_k,
            "candidate_k": candidate_k,
            "history": history[-settings.chat_history_messages :]
            if settings.chat_history_messages
            else [],
            "images": images or [],
        }
        state.update(self.receive_question(state))
        for node in (self.retrieve_chunks, self.rerank_chunks, self.build_context):
            state.update(node(state))

        if state.get("conversational"):
            stream_reply = getattr(self.llm, "stream_conversational_reply", None)
            if callable(stream_reply):
                reply = yield from stream_reply(
                    question=state["question"],
                    history=state.get("history", []),
                )
            else:
                reply = DEFAULT_GREETING_ANSWER
                yield reply
            state.update(
                {
                    "answer": reply,
                    "cited_chunk_ids": [],
                    "cited_image_ids": [],
                    "unanswerable": False,
                }
            )
        elif not state.get("context", "").strip() and not state.get("images"):
            state.update(self.generate_answer(state))
            yield state["answer"]
        else:
            generated = yield from self.llm.stream_answer(
                question=state["question"],
                context=state.get("context", ""),
                history=state.get("history", []),
                images=state.get("images", []),
            )
            state.update(
                {
                    "answer": generated.answer,
                    "cited_chunk_ids": generated.cited_chunk_ids,
                    "cited_image_ids": generated.cited_image_ids,
                    "unanswerable": generated.unanswerable,
                }
            )

        state.update(self.validate_citations(state))
        state.update(self.finalize_response(state))
        return state

    @staticmethod
    def receive_question(state: RAGState) -> RAGState:
        question = state.get("question", "").strip()
        if not question:
            raise ValueError("Question must not be empty.")
        retrieval_query = build_retrieval_query(
            question, state.get("history", [])
        )
        return {
            "question": question,
            "retrieval_query": retrieval_query,
            "candidates": [],
            "chunks": [],
            "images": state.get("images", []),
            "citations": [],
            "conversational": is_conversational_greeting(question),
            "retrieval_metadata": {},
        }

    def retrieve_chunks(self, state: RAGState) -> RAGState:
        mode = state["retrieval_mode"]
        if state.get("conversational"):
            return {
                "candidates": [],
                "chunks": [],
                "retrieval_metadata": {
                    "mode": mode.value,
                    "candidate_count": 0,
                    "result_count": 0,
                    "conversational": True,
                },
            }
        candidate_k = state.get("candidate_k") or get_settings().retrieval_candidate_k
        retrieval_limit = (
            max(candidate_k, state["top_k"])
            if mode is RetrievalMode.HYBRID_RERANK
            else state["top_k"]
        )
        retrieval_query = state.get("retrieval_query") or state["question"]
        candidates = self.retrieval_engine.retrieve(
            retrieval_query,
            mode=(
                RetrievalMode.HYBRID
                if mode is RetrievalMode.HYBRID_RERANK
                else mode
            ),
            top_k=retrieval_limit,
            document_ids=state.get("document_ids"),
        )
        return {
            "candidates": candidates,
            "chunks": candidates,
            "retrieval_metadata": {
                "mode": mode.value,
                "candidate_count": len(candidates),
                "result_count": 0,
                "candidate_k": retrieval_limit,
            },
        }

    def rerank_chunks(self, state: RAGState) -> RAGState:
        if state.get("conversational"):
            metadata = dict(state.get("retrieval_metadata", {}))
            metadata["result_count"] = 0
            return {"chunks": [], "retrieval_metadata": metadata}
        candidates = state.get("candidates", [])
        if (
            state["retrieval_mode"] is RetrievalMode.HYBRID_RERANK
            and candidates
        ):
            retrieval_query = state.get("retrieval_query") or state["question"]
            chunks = self.reranker.rerank(
                retrieval_query,
                candidates,
                top_k=state["top_k"],
            )
        else:
            chunks = candidates[: state["top_k"]]
        metadata = dict(state.get("retrieval_metadata", {}))
        metadata["result_count"] = len(chunks)
        return {"chunks": chunks, "retrieval_metadata": metadata}

    @staticmethod
    def build_context(state: RAGState) -> RAGState:
        settings = get_settings()
        blocks: list[str] = []
        chars_remaining = settings.rag_max_context_chars
        for chunk in state.get("chunks", []):
            page = f" page={chunk.page_number}" if chunk.page_number is not None else ""
            header = (
                f"[chunk_id={chunk.chunk_id} document_id={chunk.document_id} "
                f"document_name={chunk.document_name!r} chunk_index={chunk.chunk_index}"
                f"{page}]\n"
            )
            block = header + chunk.chunk_text
            if len(block) > chars_remaining:
                if chars_remaining <= len(header):
                    break
                block = header + chunk.chunk_text[: chars_remaining - len(header)]
            blocks.append(block)
            chars_remaining -= len(block) + 2
            if chars_remaining <= 0:
                break
        return {"context": "\n\n".join(blocks)}

    def generate_answer(self, state: RAGState) -> RAGState:
        if state.get("conversational"):
            conversational_reply = getattr(self.llm, "generate_conversational_reply", None)
            answer = (
                conversational_reply(
                    question=state["question"],
                    history=state.get("history", []),
                )
                if callable(conversational_reply)
                else DEFAULT_GREETING_ANSWER
            )
            return {
                "answer": answer,
                "cited_chunk_ids": [],
                "cited_image_ids": [],
                "unanswerable": False,
            }

        images = state.get("images", [])
        if not state.get("context", "").strip() and not images:
            return {
                "answer": UNANSWERABLE_ANSWER,
                "cited_chunk_ids": [],
                "cited_image_ids": [],
                "unanswerable": True,
            }

        generated = self.llm.generate_answer(
            question=state["question"],
            context=state["context"],
            history=state.get("history", []),
            images=images,
        )
        return {
            "answer": generated.answer,
            "cited_chunk_ids": generated.cited_chunk_ids,
            "cited_image_ids": generated.cited_image_ids,
            "unanswerable": generated.unanswerable,
        }

    @staticmethod
    def validate_citations(state: RAGState) -> RAGState:
        chunks_by_id = {str(chunk.chunk_id): chunk for chunk in state.get("chunks", [])}
        citation_ids = state.get("cited_chunk_ids", [])
        valid_chunks: list[RetrievalResult] = []
        seen: set[str] = set()
        for citation_id in citation_ids:
            if citation_id in chunks_by_id and citation_id not in seen:
                valid_chunks.append(chunks_by_id[citation_id])
                seen.add(citation_id)

        images_by_id = {
            image.image_id: image for image in state.get("images", [])
        }
        valid_image_ids = list(
            dict.fromkeys(
                image_id
                for image_id in state.get("cited_image_ids", [])
                if image_id in images_by_id
            )
        )
        if state.get("conversational"):
            return {
                "citations": [],
                "cited_chunk_ids": [],
                "cited_image_ids": [],
                "unanswerable": False,
            }
        if state.get("unanswerable") or not valid_chunks and not valid_image_ids:
            return {
                "answer": UNANSWERABLE_ANSWER,
                "cited_chunk_ids": [],
                "cited_image_ids": [],
                "citations": [],
                "unanswerable": True,
            }

        return {
            "citations": [
                {
                    "document_id": str(chunk.document_id),
                    "document_name": chunk.document_name,
                    "chunk_id": str(chunk.chunk_id),
                    "page_number": chunk.page_number,
                    "chunk_index": chunk.chunk_index,
                    "excerpt": chunk.chunk_text[:300],
                }
                for chunk in valid_chunks
            ],
            "cited_chunk_ids": [str(chunk.chunk_id) for chunk in valid_chunks],
            "cited_image_ids": valid_image_ids,
            "unanswerable": False,
        }

    @staticmethod
    def finalize_response(state: RAGState) -> RAGState:
        return {
            "answer": state.get("answer", UNANSWERABLE_ANSWER),
            "citations": state.get("citations", []),
            "cited_image_ids": state.get("cited_image_ids", []),
            "retrieval_metadata": state.get("retrieval_metadata", {}),
        }
