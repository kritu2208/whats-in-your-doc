# Hybrid RAG Document Q&A with Evaluation

## Project objective

Build a full-stack AI-powered web application where users can upload PDF, Markdown and TXT documents and ask questions about their contents through a chat interface.

The system must generate accurate answers grounded strictly in uploaded documents, with clickable citations pointing to the exact source chunks and page numbers.

## Technology stack

### Backend

- Python 3.11+
- FastAPI
- PostgreSQL 16
- pgvector
- SQLAlchemy
- Alembic

### AI and retrieval

- sentence-transformers
- BAAI/bge-small-en-v1.5 embeddings
- BM25 using rank_bm25
- Reciprocal Rank Fusion (RRF)
- Cross-encoder reranking
- LangGraph
- Groq API with Llama model
- Ollama as an optional local LLM provider

### Frontend

- React
- Vite
- TypeScript
- Modern responsive chat interface

### Infrastructure

- Docker
- Docker Compose
- Pytest

## Core features

1. Document upload and processing.
2. PDF parsing with page number preservation.
3. Intelligent text chunking (approximately 500 tokens with 50-token overlap).
4. Generate and store document embeddings.
5. Dense vector retrieval.
6. BM25 keyword retrieval.
7. Hybrid retrieval using RRF.
8. Cross-encoder reranking.
9. LangGraph-based question-answering pipeline.
10. Answers with verifiable clickable citations.
11. Streaming chat responses using SSE.
12. Source document and chunk preview.
13. Evaluation system measuring Recall@5, Recall@10, MRR, nDCG@10 and faithfulness.
14. Compare dense, BM25, hybrid and hybrid+r​​eranking retrieval modes.
15. Optionally extract PDF images and display images from pages that contain retrieved text chunks.

## Development instructions and milestones

Create all required directories and files. Build the application in milestones:

- **M0:** Project structure, backend, frontend and Docker setup.
- **M1:** Database configuration and models.
- **M2:** Document parsing, chunking, embeddings and upload API.
- **M3:** Dense retrieval, BM25 and RRF.
- **M4:** Cross-encoder reranking.
- **M5:** LangGraph pipeline.
- **M6:** Streaming chat API and citations.
- **M7:** React frontend.
- **M8:** Evaluation framework and metrics.
- **M9:** Testing, documentation and final polish.

## Important rules

- Implement only M0 initially. Stop and wait for approval before M1.
- Do not generate the entire application in one go.
- Use clean, modular and maintainable code.
- Include proper error handling and type hints.
- Never hardcode API keys; provide `.env.example`.
- Write tests alongside implementations.
- Check compatibility with a low-RAM laptop.
- Report actual test results and do not claim unverified behavior.
