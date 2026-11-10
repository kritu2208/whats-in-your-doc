from __future__ import annotations

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

        if not state.get("context", "").strip() and not state.get("images"):
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
        return {
            "question": question,
            "candidates": [],
            "chunks": [],
            "images": state.get("images", []),
            "citations": [],
            "retrieval_metadata": {},
        }

    def retrieve_chunks(self, state: RAGState) -> RAGState:
        mode = state["retrieval_mode"]
        candidate_k = state.get("candidate_k") or get_settings().retrieval_candidate_k
        retrieval_limit = (
            max(candidate_k, state["top_k"])
            if mode is RetrievalMode.HYBRID_RERANK
            else state["top_k"]
        )
        candidates = self.retrieval_engine.retrieve(
            state["question"],
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
        candidates = state.get("candidates", [])
        if (
            state["retrieval_mode"] is RetrievalMode.HYBRID_RERANK
            and candidates
        ):
            chunks = self.reranker.rerank(
                state["question"],
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
