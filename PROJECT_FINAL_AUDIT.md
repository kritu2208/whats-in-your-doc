# Hybrid RAG Document Q&A — Final Milestone Audit

**Audit date:** 2026-11-05  
**Scope:** Repository and artifact review of M0–M7. No source, dependency,
environment, dataset, mapping, or evaluation-result files were changed. This
report is the only intentional project artifact created for the audit; the
test run may refresh ignored test/cache metadata.

## Status summary

Status meanings: **VERIFIED** means supported by inspected implementation and
current tests/artifacts within the stated scope; **PARTIALLY VERIFIED** means
the code or configuration exists, but an important runtime or acceptance check
was not performed; **NOT VERIFIED** means there was insufficient current
evidence; **FAILED** means an inspected requirement is demonstrably absent or
a test/check failed. A status does not imply production readiness.

| Milestone | Status | Audit finding |
|---|---|---|
| M0 — Foundation | **VERIFIED** | Project structure, FastAPI startup/health route, React/Vite scaffold, Dockerfiles and Compose configuration are present. Health test passed; Compose configuration parsed successfully. |
| M1 — Database | **PARTIALLY VERIFIED** | SQLAlchemy models, PostgreSQL/pgvector configuration and Alembic migrations are present; model/configuration tests passed. Live PostgreSQL connectivity and applied schema were not checked in this audit. |
| M2 — Ingestion | **PARTIALLY VERIFIED** | PDF/Markdown/TXT parsing, page-aware token chunking, lazy embeddings, and upload/list/delete APIs are implemented and covered by unit/API tests. Live host-model inference and database-backed uploads were not rerun. |
| M3 — Retrieval | **VERIFIED** | Dense, BM25, and RRF hybrid retrieval are implemented and tested. The saved evaluation run has successful results for all three modes. |
| M4 — Reranking | **VERIFIED** | Lazy CPU cross-encoder reranking and error handling are implemented and unit-tested; the saved run completed `hybrid_rerank` for all questions. |
| M5 — RAG pipeline | **PARTIALLY VERIFIED** | A LangGraph workflow, Groq service, citation validation and database-backed conversation repository are implemented and tested with fakes/in-memory storage. No live Groq request or current PostgreSQL persistence check was made. |
| M6 — Frontend | **PARTIALLY VERIFIED** | React UI source includes document upload/selection, chat, retrieval-mode selection, citation cards and transcript localStorage. A frontend build/browser end-to-end check was not run. Citations are displayed as cards, not clickable source links. |
| M7 — Retrieval evaluation | **VERIFIED** | Synthetic dataset validation passed; mapping, raw evaluation, comparative analysis, manual review and final evaluation report are present and internally consistent for the saved run. Human ground-truth review and answer-quality evaluation remain incomplete. |

No backend test failed in the current run. The **FAILED** label is therefore not
assigned to a whole milestone; the absent capabilities and unverified checks
are itemized below rather than conflated with test failures.

## Evidence by milestone

### M0 — Foundation

- The repository contains separate `backend/` and `frontend/` trees, a root
  specification, README, `.env.example`, Docker Compose configuration and
  Copilot instructions.
- FastAPI startup registers separate health, document, retrieval and chat
  routers. `GET /api/health` returns the expected status response; its current
  test passed.
- The frontend has a React/Vite/TypeScript package manifest and a
  multi-stage frontend Dockerfile.
- Docker Compose uses the `pgvector/pgvector:pg16` image, database health
  checks, backend health checks and service health dependencies. The backend
  image is based on Python 3.11.
- `docker compose config --quiet` exited successfully. This verifies Compose
  configuration parsing, not that any container is currently running.

### M1 — Database

- SQLAlchemy models exist for documents, chunks, conversations, messages and
  evaluation runs. Chunk embeddings use a 384-dimensional pgvector column with
  a cosine HNSW index.
- Alembic has an initial schema migration that enables the `vector` extension
  and creates the tables/indexes, plus a migration for document metadata.
- Database configuration is read from settings with environment support;
  `.env.example` provides placeholders. The real `.env` was not printed,
  edited, or otherwise intentionally changed.
- Model and configuration tests passed. The PostgreSQL integration check is
  gated on `TEST_DATABASE_URL` and was skipped in this audit; current database
  connectivity and migration state are consequently **not verified** here.

### M2 — Document ingestion

- The parser handles PDF, Markdown and TXT; PDF pages retain one-based page
  numbers, while Markdown/TXT have no page number.
- Token-aware chunking enforces configured size/overlap and preserves page
  metadata. The configured defaults are 500 tokens and 50-token overlap.
- Sentence-transformer embedding is configured for the 384-dimensional
  `BAAI/bge-small-en-v1.5` model and loaded lazily on CPU, not during startup.
- Document API routes support upload, listing and deletion. Upload validation
  includes supported extensions, file-size/chunk bounds and explicit parse,
  embedding and database error responses.
- Parser, chunker, embedding-shape and API tests passed. Tests use fakes for
  external/model/database behavior; a live upload was not repeated during this
  read-only audit.

### M3 — Retrieval

- Retrieval implementation includes vector dense search, BM25 candidate
  scoring and Reciprocal Rank Fusion. API accepts `dense`, `bm25`, `hybrid`
  and `hybrid_rerank`.
- Unit tests cover query embedding, BM25 ranking, RRF ordering and API behavior
  using fake stores/encoders.
- The latest saved evaluation file contains 60 successful records per mode,
  confirming that all four retrieval configurations completed in that run.

### M4 — Reranking

- Cross-encoder loading is lazy and CPU-based; scores are validated and
  candidates are deterministically sorted. Failures are surfaced rather than
  silently falling back to hybrid results.
- Reranker tests passed. The latest evaluation contains 60 successful
  `hybrid_rerank` records; the observed run also records the reranking latency
  cost.

### M5 — RAG pipeline

- A compiled LangGraph workflow sequences question validation, retrieval,
  optional reranking, context construction, answer generation, citation
  validation and response finalization.
- The Groq service is configured to answer from supplied context and validates
  structured output. It is not initialized or called at application startup.
- Citation IDs are checked against retrieved chunks; invalid, missing, or
  duplicate citations do not become source citations in the response.
- Conversation/message models and a repository persist exchanges and retrieve
  bounded recent history.
- Chat tests passed with fake retrieval, reranker and LLM implementations and
  in-memory database storage. No paid/external LLM call was made; live Groq
  behavior and PostgreSQL conversation persistence are not verified here.

### M6 — Frontend

- React/Vite/TypeScript source implements health status, document library,
  PDF/Markdown/TXT upload, document selection, retrieval-mode selection,
  synchronous chat submission, new-chat action and localStorage transcript
  persistence.
- Assistant citations are rendered with filename, optional page, excerpt,
  chunk index and abbreviated chunk UUID.
- The citation UI renders informational cards rather than clickable links to a
  source/chunk preview. No separate source-preview route is present in the
  inspected API surface.
- The frontend build and browser end-to-end behavior were not exercised in
  this audit. Package metadata, lockfile, source and Dockerfile are present;
  this is not equivalent to a passing frontend build.

### M7 — Retrieval evaluation

- Dataset validator passed for `hybrid-rag-synthetic-starter` v0.1.0:
  8 logical documents, 60 questions (50 answerable, 10 unanswerable), with
  category counts matching the dataset.
- Manifest and completed mapping each contain 25 stable references; the
  mapping is marked reviewed and has 25 unique `(document_ref, chunk_ref)`
  entries. This audit checked file structure and uniqueness but did not
  reconnect to PostgreSQL to revalidate UUID ownership.
- Latest raw run:
  [`evaluation-20261104T195742783359Z.json`](backend/evaluation_outputs/evaluation-20261104T195742783359Z.json)
  contains 240 records (60 questions × 4 modes), with 60 successes and zero
  failures for each mode.
- The comparative analysis, six-question manual evidence review and final M7
  evaluation report are present. The final report's reported summary/category
  results, the mapping cardinality, and the raw record counts were checked
  against the source artifacts in the preceding report validation and this
  audit.
- Retrieval metrics do not measure generated answer correctness. Faithfulness,
  answer relevancy and abstention quality remain unevaluated.

## Latest verified backend test results

Ran from `backend/` using the existing project interpreter:

```text
C:\Users\91812\hybrid-rag\.venv\Scripts\python.exe -m pytest -q
131 passed, 1 skipped in 4.68s
SKIPPED: tests/test_database.py — PostgreSQL integration check requires TEST_DATABASE_URL
```

The skipped test would connect to PostgreSQL; it was correctly skipped because
the optional test URL was not configured. The other tests use local fixtures,
fakes, or in-memory storage and did not require a live database, model download,
or Groq request. The VS Code test tool did not discover tests on its first
attempt; the direct pytest run above is the actual verified result.

## Evaluation summary

| Mode | Recall@5 | Recall@10 | MRR | nDCG@10 | p50 latency | p95 latency |
|---|---:|---:|---:|---:|---:|---:|
| Dense | 0.9933 | 1.0000 | 0.9800 | 0.9804 | 94.67 ms | 181.05 ms |
| BM25 | **1.0000** | 1.0000 | 0.9183 | 0.9287 | **8.86 ms** | **11.73 ms** |
| Hybrid (RRF) | 0.9933 | 1.0000 | 0.9500 | 0.9569 | 101.12 ms | 151.50 ms |
| Hybrid + reranking | 0.9933 | 1.0000 | **1.0000** | **0.9954** | 428.39 ms | 683.21 ms |

Quality scores average over 50 answerable questions; unanswerable retrieval
metrics are null and excluded. BM25 led Recall@5 and speed; hybrid reranking
led MRR and nDCG@10. All modes tied on Recall@10. Reranking's improved ranking
scores came with substantially higher latency. This is a single run against a
small synthetic dataset, not a production benchmark.

Artifacts:

- Raw outputs:
  [`evaluation_outputs/`](backend/evaluation_outputs/)
- Comparative analysis:
  [`comparative_analysis.md`](backend/evaluation_outputs/comparative_analysis.md)
- Manual evidence-quality review:
  [`manual_quality_review.md`](backend/evaluation_outputs/manual_quality_review.md)
- Final M7 report:
  [`M7_FINAL_REPORT.md`](backend/evaluation_outputs/M7_FINAL_REPORT.md)
- Dataset manifest:
  [`manifest.json`](backend/evaluation_datasets/starter/manifest.json)
- Reference mapping:
  [`reference-mapping.json`](backend/evaluation_outputs/reference-mapping.json)

## Known limitations and incomplete tasks

1. **Human review is incomplete.** The evaluation corpus is fictional and
   limited to 8 logical documents, 25 evidence anchors and 60 questions.
   The manual quality review sampled six questions; it marks the unanswerable
   absence judgment `NEEDS_MANUAL_REVIEW`. Do not generalize the scores to
   real documents or claim production readiness.
2. **Answer evaluation remains incomplete.** The run is retrieval-only.
   Faithfulness, answer relevancy, citation correctness in generated answers
   and abstention behavior were not measured.
3. **Live integrations were not revalidated.** Current PostgreSQL connectivity,
   migration application, live Groq calls, real model downloads/inference and
   current container runtime state were not checked in this audit.
4. **Frontend validation is incomplete.** No production build or browser
   end-to-end check was run. Citation cards are not clickable source links, and
   source chunk preview is absent from the inspected routes/UI.
5. **Streaming is not implemented.** The chat endpoint is synchronous; SSE
   streaming listed in the original specification is still outstanding.
6. **Optional Ollama integration is outstanding.** An example URL placeholder
   exists, but the inspected application settings/service wiring does not
   provide a working Ollama provider.
7. **Documentation is stale and milestone names drift.** The README still
   says no full M7 evaluation results exist and that the full dataset run is
   pending, contradicting the generated raw run and reports. The original
   specification assigns streaming to M6, the frontend to M7, and evaluation
   to M8; the current README/user audit taxonomy calls the frontend M6 and
   evaluation M7. `.github/copilot-instructions.md` also still says to work
   only on M0. These statements should be reconciled by a documentation owner.
8. **Database test coverage is bounded.** The current test run skipped the one
   live PostgreSQL connectivity test. Unit/model metadata coverage is not a
   substitute for testing migrations and persistence against PostgreSQL 16
   with pgvector.

## Remaining work

- Reconcile the milestone plan across the project specification, README and
  Copilot instructions; update the README's stale evaluation status.
- Run the frontend production build and browser tests in an approved
  verification pass.
- Verify migrations and persistence against the intended PostgreSQL 16/pgvector
  environment; separately smoke-test upload and retrieval with configured
  models when resources and credentials are available.
- Complete review of all evaluation questions and references; expand beyond
  synthetic data before using results to select a production strategy.
- Implement and evaluate answer faithfulness, relevancy, citation correctness
  and unanswerable abstention as a separate answer-generation evaluation.
- Add clickable citation navigation/source preview and implement SSE streaming
  if these remain required by the product specification.
- Implement or explicitly remove the optional Ollama-provider requirement.

## Overall project completion assessment

The repository has a substantial working prototype: backend foundation,
database schema, ingestion pipeline, retrieval modes, reranking, LangGraph/Groq
integration, React UI source and an executed offline retrieval evaluation.
The current backend unit suite and Compose configuration validation pass.
However, several important acceptance checks are not current/live, answer
quality is unevaluated, and streaming/clickable source navigation are absent.
The synthetic evaluation is too small and pending human review to establish
real-world quality or production readiness.

**Final verdict:** M0–M7 functionality is largely implemented, but the project
is **not fully verified end-to-end and is not production-ready**. M1, M2, M5
and M6 remain partially verified; M7 is verified only as a retrieval
evaluation artifact and not as an answer-quality assessment.

### Items needing human verification

- Confirm PostgreSQL 16/pgvector is running, migrations are applied, and the
  current database still contains the intended corpus.
- Review every synthetic question/reference and resolve the manual-review
  cases, especially unanswerable questions.
- Decide whether SSE, clickable citations/source preview and Ollama remain
  required product features.
- Approve the milestone taxonomy and update stale documentation/instructions.
- Review real-document privacy, provider credentials, latency and resource
  constraints before any production deployment.
