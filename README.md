# Hybrid RAG Document Q&A with Evaluation

A milestone-driven full-stack project for document-grounded question answering with verifiable citations and retrieval evaluation. The original requirements and planned milestones are captured in [HYBRID_RAG_PROJECT_SPEC.md](./HYBRID_RAG_PROJECT_SPEC.md).

## Current milestone: M8 (human-reviewed faithfulness evaluation)

M0 provides the project structure, FastAPI service, React/Vite/TypeScript starter UI, PostgreSQL 16 with the pgvector extension image, Docker Compose, environment template, and an API health test. M1 adds environment-based SQLAlchemy configuration, database models, and Alembic migrations. M2 adds PDF/Markdown/TXT parsing, token-aware chunking, lazy CPU embeddings, and document upload/list/delete APIs. M3 adds dense, BM25, and hybrid retrieval. M4 adds optional hybrid retrieval with lazy cross-encoder reranking. M5 adds a sequential LangGraph question-answering workflow, Groq or optional local Ollama grounded responses, validated citations, and persistent conversations. M6 adds the responsive React document-library and chat interface. M7 provides the synthetic starter corpus, retrieval/latency metrics, validated ablation configuration, and the offline runner for dense, BM25, hybrid, and hybrid_rerank comparisons. M8 adds a manual claim-level faithfulness scorer and report command; answer relevancy remains unimplemented. No full database retrieval evaluation or production quality measurements have been generated.

The evaluation runner requires a manually verified stable-reference-to-database mapping; no full evaluation or project performance measurements have been run.

## Project layout

```text
.
├── .github/
│   └── copilot-instructions.md
├── backend/
│   ├── app/
│   │   ├── api/
│   │   │   └── health.py
│   │   │   ├── chat.py
│   │   │   └── retrieval.py
│   │   ├── core/
│   │   │   └── config.py
│   │   ├── chat/
│   │   │   └── repository.py
│   │   ├── evaluation/
│   │   │   ├── config.py
│   │   │   ├── dataset.py
│   │   │   ├── metrics.py
│   │   │   ├── quality.py
│   │   │   └── runner.py
│   │   ├── db/
│   │   │   ├── base.py
│   │   │   └── session.py
│   │   ├── ingestion/
│   │   │   ├── parsers.py
│   │   │   ├── chunking.py
│   │   │   ├── embeddings.py
│   │   │   └── service.py
│   │   ├── retrieval/
│   │   │   ├── engine.py
│   │   │   └── reranker.py
│   │   ├── rag/
│   │   │   └── workflow.py
│   │   ├── llm/
│   │   │   └── groq_service.py
│   │   ├── models/
│   │   ├── main.py
│   ├── tests/
│   │   ├── test_health.py
│   │   ├── test_database.py
│   │   ├── test_ingestion.py
│   │   └── test_retrieval.py
│   │   ├── test_chat.py
│   │   ├── test_evaluation_dataset.py
│   │   ├── test_evaluation_metrics.py
│   │   ├── test_evaluation_quality.py
│   │   └── test_evaluation_runner.py
│   ├── evaluation_datasets/
│   │   ├── README.md
│   │   └── starter/
│   │       ├── manifest.json
│   │       ├── questions.jsonl
│   │       ├── review_template.csv
│   │       └── corpus/
│   ├── scripts/
│   │   ├── validate_evaluation_dataset.py
│   │   ├── run_evaluation.py
│   │   └── run_faithfulness_evaluation.py
│   ├── migrations/
│   │   └── versions/
│   │       ├── 0001_initial_schema.py
│   │       └── 0002_document_metadata.py
│   ├── alembic.ini
│   ├── Dockerfile
│   ├── pytest.ini
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── api.ts
│   │   └── App.tsx
│   ├── Dockerfile
│   ├── nginx.conf
│   ├── package.json
│   └── vite.config.ts
├── .env.example
├── docker-compose.yml
└── HYBRID_RAG_PROJECT_SPEC.md
```

## Run with Docker Compose

Requirements: Docker Desktop (or Docker Engine with Compose) and enough disk space for container images. The first run downloads images and frontend/backend dependencies.

1. Copy `.env.example` to `.env` and replace `POSTGRES_PASSWORD` and the matching password in `DATABASE_URL` with the same local development password.
2. Start PostgreSQL and wait for its health check:

   ```sh
   docker compose up -d db
   ```

3. Build the backend image if needed and apply migrations before exposing the API:

   ```sh
   docker compose run --rm backend alembic upgrade head
   ```

4. Start the backend and frontend:

   ```sh
   docker compose up --build -d backend frontend
   ```

5. Open `http://localhost:8080`. The API is also available at `http://localhost:8000/api/health`; API docs are at `http://localhost:8000/docs`.
6. Stop the services with `docker compose down`. To remove the local database volume as well, run `docker compose down -v` (this deletes the persisted development database).

The database uses `pgvector/pgvector:pg16`. The health endpoint remains independent of database connectivity. Run Alembic explicitly to create or upgrade schema; migrations do not run automatically during application startup.

The document API accepts multipart uploads at `POST /api/documents`, lists stored documents at `GET /api/documents`, and removes a document, its chunks, and its extracted images at `DELETE /api/documents/{id}`. Supported filename extensions are `.pdf`, `.md`, and `.txt`; uploads are limited to 10 MiB by default, and one document is limited to 2,000 chunks to bound low-RAM usage. PDF chunk page numbers are one-based; Markdown and text chunks use a null page number. Parser format and Markdown title metadata are persisted with each document.

When PDF images are enabled in the chat composer, image retrieval combines visual similarity with semantic search over text located near each figure on its page. The local, lazy-loaded `sentence-transformers/clip-ViT-B-32` model supplies 512-dimensional visual vectors; nearby text uses the same 384-dimensional BGE embedding model as document chunks. The two ranked lists are fused with reciprocal rank fusion, and up to three matching figures are sent to the selected vision-capable LLM with the question and retrieved document text. For Groq, each image is resized to at most 768 pixels per side and 512 KiB to help keep requests within provider limits. Retrieved figures and nearby text are shown under **View sources**, with an indicator for images cited in the answer. Text proximity is a layout heuristic, not a guarantee of a true caption relationship, and visual identification is not guaranteed exact; verify important details against the linked PDF page. Extraction supports JPEG, PNG, GIF, WebP, and BMP and is bounded to 30 images per PDF, 5 MiB per image, and 20 MiB total per PDF. Image bytes and vectors are stored in PostgreSQL and images are served from `GET /api/documents/{document_id}/images/{image_id}`. Embedding models run locally on CPU and load only when needed. Existing images can receive missing visual embeddings in batches when image search is first requested, but the original PDFs are not retained, so existing images cannot be associated with page text automatically: re-upload those PDFs to populate nearby-text vectors. Configure `IMAGE_EMBEDDING_MODEL_NAME` only if replacing it with a compatible 512-dimensional image-and-text model.

### Optional local Qwen vision model

The backend can use Groq or a local Ollama model. To avoid Groq quotas, install Ollama for Windows, then run:

```powershell
ollama pull qwen3-vl:2b
ollama run qwen3-vl:2b
```

In the root `.env`, set `LLM_PROVIDER=ollama`, `OLLAMA_BASE_URL=http://host.docker.internal:11434`, and `OLLAMA_MODEL=qwen3-vl:2b`. Keep the Ollama app/server running on Windows, then restart the backend with `docker compose up --build -d backend`. The backend connects to the host Ollama service; it does not download the model into the Docker image. The default local timeout is five minutes because first model loads and CPU inference can be slow. The local model is a vision-language model, so it can analyze retrieved PDF figures without a Groq API key. For this low-memory development setup, start with the 2B model; larger models may exceed available RAM/VRAM.

The web app uploads PDF, Markdown, and TXT documents, lists their backend processing status, and lets you select source documents for chat. Chat always uses the combined `hybrid_rerank` flow: dense semantic and BM25 keyword retrieval are fused with RRF, then cross-encoder reranking is applied. The interface does not offer separate retrieval-mode choices; the retrieval API and evaluation runner continue to support individual modes for testing. Questions use `POST /api/chat/query`; the response conversation ID and transcript are retained in browser local storage so a user can continue after reloading. Starting a new chat clears the local transcript and omits the ID on the next query, letting the existing API create a fresh conversation. The interface renders only citations included in the backend response. `VITE_API_BASE_URL` optionally sets an absolute API prefix at frontend build time; leave it empty (the default) to use the existing Vite `/api` development proxy and Docker/Nginx `/api` proxy. For host-run Vite development, set it in the frontend process environment or `frontend/.env.local`; Docker Compose forwards the root `.env` value as a build argument. Do not put secrets in Vite variables because build-time frontend values are public.

The first ingestion request lazily loads `BAAI/bge-small-en-v1.5` on CPU and keeps the model in process memory for reuse. The model is not loaded on application startup. Change `EMBEDDING_MODEL_NAME` only to a compatible 384-dimensional model. Configure `UPLOAD_MAX_BYTES`, `MAX_DOCUMENT_CHUNKS`, `CHUNK_SIZE_TOKENS`, and `CHUNK_OVERLAP_TOKENS` through the environment if needed.

## Retrieval API

Use `POST /api/retrieval/query` after applying the database migrations and uploading documents. Supported modes are `dense`, `bm25`, `hybrid`, and `hybrid_rerank`; the default is `hybrid`.

```json
{
  "query": "What does the document say about evaluation?",
  "mode": "hybrid",
  "top_k": 5,
  "document_ids": null
}
```

`document_ids` is optional and scopes retrieval to those uploaded document UUIDs. The response includes document ID/name, chunk ID/text, page number, chunk index, and relevance score for each result. Dense search orders by pgvector cosine distance. BM25 is keyword-based, and hybrid merges the dense and BM25 rankings using Reciprocal Rank Fusion with `k=60`. `top_k` is limited to 1–100. The BM25 corpus is capped by `RETRIEVAL_CANDIDATE_LIMIT` (default 10,000); hybrid candidate retrieval uses `RETRIEVAL_CANDIDATE_MULTIPLIER` (default 4) within that cap.

For `hybrid_rerank`, the service first retrieves hybrid RRF candidates, then scores them with a CPU cross-encoder and returns the final `top_k`. `candidate_k` can be provided in the request and must be at least `top_k`; it defaults to `RETRIEVAL_CANDIDATE_K` (20). The model defaults to `cross-encoder/ms-marco-MiniLM-L-6-v2` and batch size defaults to 16. Configure these with `RERANKER_MODEL_NAME`, `RETRIEVAL_CANDIDATE_K`, and `RERANKER_BATCH_SIZE`. The cross-encoder is loaded only for a non-empty `hybrid_rerank` candidate list and is cached in-process. Model loading/scoring failures return HTTP 503 with an actionable message; they do not silently fall back to hybrid results.

Dense and hybrid requests load the existing embedding model only if ready chunks with embeddings exist. BM25 works without embeddings. Retrieval/reranking tests use fake stores, encoders, and cross-encoders and do not download model weights or require PostgreSQL.

## Run locally

Requirements: Python 3.11+ and Node.js 20+.

Backend, from the project root (first copy `.env.example` to `.env`, then set the database password and URL; use `localhost` as the URL host for a host-run backend):

```sh
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux:
# source .venv/bin/activate
python -m pip install -r backend/requirements.txt
docker compose up -d db
cd backend
alembic upgrade head
uvicorn app.main:app --reload
```

Frontend, in a second terminal from the project root:

```sh
cd frontend
npm ci
npm run dev
```

The Vite development server proxies `/api` requests to `http://localhost:8000`. Open the URL printed by Vite (usually `http://localhost:5173`).

## Test and build

Backend tests, from the project root:

```sh
cd backend
python -m pytest
```

Parser, chunker, embedding-shape, retrieval/reranking ranking, retrieval API behavior, database model, and configuration tests do not require a running PostgreSQL instance. Ingestion and retrieval API tests use in-memory session doubles and fake encoders/cross-encoders. The optional connectivity test runs only when `TEST_DATABASE_URL` is set. Applying Alembic migrations, dense SQL against pgvector, and actual persistence require a running PostgreSQL 16/pgvector database.

Frontend production build:

```sh
cd frontend
npm ci
npm run build
```

Validate Docker Compose configuration:

```sh
docker compose config
```

## Chat API and frontend

Set `GROQ_API_KEY` in your local, untracked `.env` file to enable answer generation. `.env.example` intentionally contains a blank key. The default configurable model is `openai/gpt-oss-20b`; `GROQ_TIMEOUT_SECONDS`, `CHAT_HISTORY_MESSAGES`, and `RAG_MAX_CONTEXT_CHARS` bound API time, history, and context size. No provider client is created until an answer needs to be generated.

After applying migrations and uploading documents, call `POST /api/chat/query`:

```json
{
  "question": "What does the document say about evaluation?",
  "document_ids": null,
  "retrieval_mode": "hybrid_rerank",
  "top_k": 5,
  "candidate_k": 20
}
```

`conversation_id` is optional; omit it to create a conversation, or pass the returned UUID to continue. Only recent user/assistant messages are included in the prompt. `document_ids` optionally limits retrieval to selected ready documents. Supported retrieval modes are `dense`, `bm25`, `hybrid`, and `hybrid_rerank`; reranking is the default. The response includes the answer, validated citations, conversation ID, retrieval metadata, and elapsed time. Citations are built only from retrieved chunks; if the model returns no valid citations, the response safely reports that the documents do not provide enough information. Missing Groq configuration, provider failures, and reranking failures return actionable HTTP errors. The query API is synchronous; streaming remains a later milestone.

Chat workflow and persistence tests use fake retrieval, reranking, Groq, and database services. They do not make paid API calls or require model weights. A live Groq call and PostgreSQL persistence require user-provided credentials and a running PostgreSQL 16/pgvector database.

## Evaluation dataset design

The starter evaluation set contains 60 questions across eight explicitly synthetic logical Markdown source documents (50 answerable, 10 unanswerable). Each stable source anchor has a separate small ingestion Markdown file so current production chunking yields exactly one database chunk per anchor; original questions and evidence references remain unchanged. Stable document/chunk references are documented in [the evaluation methodology](backend/evaluation_datasets/README.md). The corpus is fictional development material, not external evidence or a production benchmark; every question is pending manual review. Validate the dataset from `backend/` with `python -m scripts.validate_evaluation_dataset`. The offline runner supports all four existing retrieval modes and uses pure metric functions; it does not call Groq or generate answers. Because the database does not store stable evaluation references, first generate and manually verify a mapping file using `python -m scripts.run_evaluation --write-mapping-template evaluation_outputs/reference-mapping.json`. Upload each manifest ingestion unit separately, complete its one-to-one document/chunk mapping, then run `python -m scripts.run_evaluation --mapping evaluation_outputs/reference-mapping.json`. Timestamped reports are written under the Git-ignored `backend/evaluation_outputs/`. Human reviewers can assess claim-level faithfulness using JSONL records and `python -m scripts.run_faithfulness_evaluation`; see the methodology for the schema, rubric, and command. This is separate from the retrieval run, does not call an LLM judge, and does not implement answer relevancy. Unit tests use synthetic/fake inputs and do not measure project performance.

## Low-RAM development

The backend does not load embedding or reranking models at startup. The first upload or dense/hybrid retrieval request when vectors exist loads the embedding model and runs CPU inference; the first non-empty `hybrid_rerank` request additionally loads the cross-encoder. This can take time and memory. Cross-encoder batch size is configurable and capped at 64; keep candidate_k and batch size conservative on a 6 GB laptop. The BM25 candidate corpus is capped to avoid loading an unbounded index into memory. Groq answer generation uses the hosted API and bounded context. The PostgreSQL service and web application can be started independently during development when resources are limited.

## Milestones

M0 — Project structure and service setup; M1 — Database configuration and models; M2 — Document parsing, chunking, embeddings, and upload API; M3 — Dense retrieval, BM25, and RRF; M4 — Cross-encoder reranking; M5 — LangGraph, Groq, and synchronous chat API; M6 — React document Q&A interface; M7 — Evaluation dataset, retrieval metrics, and offline mode-comparison runner (full database run pending); M8 — Human-reviewed faithfulness implemented, answer relevancy pending; M9 — Testing, documentation, and polish.
