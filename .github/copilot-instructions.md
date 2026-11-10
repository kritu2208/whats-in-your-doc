# Project development instructions

- Follow `HYBRID_RAG_PROJECT_SPEC.md` as the source of truth for architecture, stack, features and milestones.
- Work only on the currently approved milestone. M0 is the project skeleton; do not add document ingestion, embeddings, retrieval, LLM orchestration, chat, or evaluation until those milestones are approved.
- Keep backend code in `backend/app`, tests in `backend/tests`, and frontend code in `frontend/src`.
- Keep API route definitions separate from application startup. Add type hints, explicit error handling, and focused tests for new behavior.
- Keep secrets out of source control. Add configuration placeholders to `.env.example`; never commit a populated `.env`.
- Use PostgreSQL 16 with the pgvector image in Docker Compose. Do not load embedding or reranking models during application startup; account for low-RAM development machines.
- Prefer small, modular changes that follow the existing project structure. Update the README when setup or development workflows change.
- Run the most relevant tests and build checks after changes, and report only results that were actually observed.
