# Evaluation dataset and methodology

The `starter/` corpus is an explicitly fictional dataset for developing and validating evaluation tooling. Its eight Markdown files, questions, expected answers, and fictional operational facts are synthetic; none are real-world evidence. The 60-question starter set is not a benchmark and must be manually reviewed before it is used to compare retrieval or answer quality.

## Files and stable references

- `starter/manifest.json` identifies the corpus and its eight documents. `document_ref` values such as `SYN-OPS-001` are stable dataset identifiers, not database document UUIDs.
- Each source Markdown file contains `<!-- eval-chunk:cNN -->` anchors. A manifest `chunk_ref` points to the source section after that anchor. These are annotation anchors, not promises that production chunking creates identical chunks.
- `starter/questions.jsonl` has one JSON object per question. `question_id` is unique. Answerable entries contain a reference answer and one or more `relevant_evidence` references to document/chunk anchors. Page numbers are null/omitted for this Markdown corpus; for paginated source formats, record the expected one-based page in both manifest and evidence reference.
- Unanswerable entries use `answerability: "unanswerable"`, `category: "unanswerable"`, a null `reference_answer`, an empty evidence list, and a required `unanswerable_reason`. This explicitly records that the corpus cannot support the requested answer; it is not a guessed answer.
- `starter/review_template.csv` has one review row for each of the 60 questions (50 answerable and 10 unanswerable). Keep reviewer corrections here until an approved dataset revision is made.

The schema and reference checker live in `backend/app/evaluation/dataset.py`. They validate structure, answerability rules, unique IDs, stable document/chunk references, page consistency, corpus files, and chunk anchors. Validation only reads lightweight JSON/JSONL/Markdown and does not construct an embedding or reranker service.

From `backend/`, validate the files with:

```powershell
python -m scripts.validate_evaluation_dataset
```

## Ground-truth authoring and review

1. Select the final corpus and assign each source a stable `document_ref`. Preserve original documents and page boundaries; record page numbers as one-based integers where the source has pages.
2. Read the source before writing a question. Write a concise reference answer from source text only, then identify the smallest set of chunks that fully supports it. A question needing multiple facts must reference every supporting chunk. For a cross-document question, list all documents/chunks necessary to answer it.
3. Keep test questions and reference answers out of indexed source documents and out of generated chunks. Ingest only corpus source files. At evaluation time send only the question and permitted conversation context to retrieval/generation; use answer keys and evidence labels only in scoring code. Do not select or tune prompts, thresholds, or retrieval settings using the held-out test answers.
4. Mark a question unanswerable only after checking every corpus document and plausible paraphrase of the requested fact. Leave `reference_answer` null and evidence empty; explain the missing information in `unanswerable_reason`. Do not fill the gap from outside knowledge.
5. A reviewer compares each question, reference answer, and evidence anchor with the source files. Check factual completeness, ambiguous wording, page/chunk mappings, and answerability. Record reviewer, date, decisions, corrected answer/evidence, and notes in `review_template.csv`.
6. Require a second reviewer for disputed/unanswerable items where practical. Resolve comments explicitly. Promote corrections into a new dataset version only after review; never silently edit a reviewed version used for a reported experiment.
7. Before running against ingested rows, manually map each stable `document_ref` and source anchor to actual database document/chunk UUIDs and verify the referenced content. Chunk boundaries can change with tokenizer, settings, or parser version; never assume anchor IDs equal generated chunk UUIDs.

## Review status and quality gates

All 60 rows begin as `pending`. Before a full evaluation is considered ready, every question should have `review_status=approved`, a reviewer and date, and correct answerability/evidence flags. The starter materials intentionally do not claim approval or measured quality.

## Evaluation architecture

The package is kept independent from API endpoints and database migrations. The offline retrieval runner composes current components rather than copying their algorithms:

1. **Dataset loader/validator** — Pydantic schemas and stable-reference checks. No model initialization.
2. **Corpus mapping** — a human-reviewed mapping is checked against manifest/dataset identity, DB document readiness, filename/source-format metadata, and exact chunk UUID ownership/coverage/page numbers. Ingestion does not store stable evaluation IDs, so the runner does not infer a map from chunk order or filename.
3. **Retrieval** — calls `RetrievalEngine.retrieve(query, mode=..., top_k=..., document_ids=...)` directly. For `hybrid_rerank`, it fetches hybrid candidates and passes them to the existing `CrossEncoderReranker.rerank`; it does not recreate BM25, RRF, dense search, or ranking logic.
4. **Metrics** — uses the pure functions in `backend/app/evaluation/metrics.py`. No answer generation, API requests, or provider calls.
5. **Run orchestration/reporting** — sequential per-question/per-mode results and aggregate JSON report. It does not persist to the `EvalRun` table.

## Retrieval metrics (implemented)

- Recall@k is `|unique relevant references ∩ top-k results| / |unique relevant references|`. The module exposes Recall@5, Recall@10, and a generic `recall_at_k`.
- Per-question reciprocal rank is `1 / rank(first relevant result)`, or zero when no relevant result is retrieved. `mean_reciprocal_rank` averages these values across answerable questions and returns `None` when there are no applicable questions.
- nDCG@10 uses binary relevance. DCG is `sum(1 / log2(rank + 1))` for relevant results in the first ten positions; nDCG divides this by ideal DCG for the same number of relevant references.
- Matching uses the pair `(document_ref, chunk_ref)`; page numbers are metadata and do not alter identity. Duplicate ground-truth references and duplicate retrieved references are each counted once; a repeated retrieved reference does not gain a better rank after its first occurrence.
- Answerable questions must have at least one evidence reference. Empty retrieval for an answerable question yields zero for Recall, MRR, and nDCG. Unanswerable questions yield `None` for each per-question retrieval metric and must be excluded from answerable retrieval aggregates, not counted as retrieval failures. Evaluate abstention behavior for unanswerable questions separately in a future answer-quality/abstention report.
- `k` must be a non-negative integer. Malformed stable references and invalid question shapes are rejected rather than silently ignored.

## Answer-quality measures

- `AnswerQualityInput`, `AnswerQualityOutput`, and the `AnswerQualityEvaluator` protocol keep quality results separate from retrieval metrics.
- Each metric has a status (`not_evaluated`, `unavailable`, or `evaluated`) and an optional score. An evaluated score is bounded to `[0, 1]`; an absent score is never represented as zero.
- Faithfulness is implemented as a manual, claim-level review; it does not call an LLM judge. A reviewer lists every externally verifiable answer claim, marks whether the retrieved text supports it, identifies its supporting stable evidence references, and records which of those references are cited for that claim. Supporting references must be retrieved, claim citations must be among those supporting references, and citations must exist in the answer's citation list. Citation references are checked for consistency, but citation coverage is not part of the faithfulness score.
- The per-answer faithfulness score is `supported factual claims / all assessed factual claims`. Unsupported claims count as zero; supported claims count even if the answer omitted a citation. Reviews that are incomplete or contain no factual claims are `unavailable`, not zero. The report's mean averages evaluated answers only and separately counts unavailable answers.
- Answer relevancy and abstention quality are not implemented. Do not treat this manual faithfulness score as a production benchmark until the answer set and claim reviews have been independently checked.

### Running the manual faithfulness scorer

Create a UTF-8 JSONL file with one object per answer. Each object has a `request` and `review`. The request records the generated answer, retrieved context/evidence, and citations returned for that answer; stable evidence references use `(document_ref, chunk_ref)` and may include one-based `page_number`. The review records reviewer/date, whether all factual claims were assessed, and claim assessments. Example:

```json
{
  "request": {
    "question_id": "q001",
    "question": "What is the stated service window?",
    "answer": "The stated service window is 9 AM to 5 PM.",
    "answerability": "answerable",
    "retrieved_contexts": ["Support is available from 9 AM to 5 PM."],
    "retrieved_evidence": [{"document_ref": "SYN-OPS-001", "chunk_ref": "c01"}],
    "cited_evidence": [{"document_ref": "SYN-OPS-001", "chunk_ref": "c01"}]
  },
  "review": {
    "reviewer": "reviewer-1",
    "reviewed_at": "2025-01-02",
    "all_factual_claims_assessed": true,
    "claims": [{
      "claim": "The service window is 9 AM to 5 PM.",
      "supported": true,
      "supporting_evidence": [{"document_ref": "SYN-OPS-001", "chunk_ref": "c01"}],
      "cited_evidence": [{"document_ref": "SYN-OPS-001", "chunk_ref": "c01"}]
    }]
  }
}
```

From `backend/`, score reviewed records and write a new report (existing files are never overwritten):

```powershell
python -m scripts.run_faithfulness_evaluation `
  --input evaluation_outputs/faithfulness-reviews.jsonl `
  --output evaluation_outputs/faithfulness-report.json `
  --dataset-id starter `
  --dataset-version 1
```

The report contains aggregate and per-question scores, reviewer/date, claim counts, and explanations; it does not copy the answers or retrieved text. This scorer validates evidence-reference consistency but cannot independently verify a reviewer's semantic judgment. Tests use synthetic inputs and do not establish measured project performance.

## Latency metrics (implemented)

- Latency utilities accept measured durations in milliseconds and report p50, p95, and arithmetic mean. Empty samples return `None` values and a zero sample count; negative, non-finite, boolean, or non-numeric durations are rejected.
- Percentiles use the nearest-rank method: sort the `n` observations and select rank `ceil(p * n)` using one-based ranks, with rank one as the minimum for p50/p95. This method does not interpolate between observations.
- The runner records retrieval latency including the optional reranker, excluding stable-reference mapping and metric calculation. No generation timing is collected. Exclude warm-up from scored timings or report cold-start and warm timings separately in future reports.

## Ablation configuration

- `AblationConfig` validates mode and top-k settings. `DEFAULT_ABLATIONS` contains exactly the existing API modes:
  - `dense`: vector cosine retrieval.
  - `bm25`: keyword retrieval.
  - `hybrid`: dense plus BM25 combined by existing reciprocal-rank fusion.
  - `hybrid_rerank`: hybrid candidate retrieval followed by the existing cross-encoder reranker. An optional candidate_k must be at least top_k.
- This configuration only names the mode; it does not duplicate or execute retrieval code.

## Preparing the DB reference mapping

`Document` stores filename/source metadata; generated `Chunk` rows store database UUID, page and index. These tables do not contain the evaluation `document_ref` or source `chunk_ref`. Generated chunk boundaries can differ from source anchor boundaries, so no correct mapping can be inferred from `chunk_index`.

The starter retains eight logical source documents and their original anchored files. Each manifest anchor also has a dedicated `ingestion_file` containing exactly that source section. These 25 small Markdown ingestion units are intentionally uploaded separately: the standard parser/chunker must produce exactly one stored chunk per unit, preserving a one-to-one stable evidence reference without changing production chunking. Dataset loading parses each anchored source section and its ingestion file, checks their parsed text is identical, and verifies the ingestion file produces exactly one chunk with the current production chunker. All question evidence continues to reference the original eight logical document IDs and their original stable `chunk_ref`s.

From `backend/`, generate a starting template:

```powershell
python -m scripts.run_evaluation --write-mapping-template evaluation_outputs/reference-mapping.json
```

Upload each of the 25 manifest `ingestion_file` paths separately using its filename as supplied. For each stable anchor, fill the mapping template's `document_id` with that uploaded document UUID and `chunk_id` with its sole stored chunk UUID. Every stable anchor must map to a distinct database document and one distinct database chunk. The runner validates that IDs exist, documents are ready, upload filename and `source_format` match, each mapped document has exactly the mapped chunk, and page numbers are consistent. Compare each retrieved chunk's content with its source section before setting `"reviewed": true`. The DB schema cannot prove semantic content correspondence; the reviewed mapping is the explicit human-verified association.

## Running retrieval evaluation

From `backend/`, run all four configured modes:

```powershell
python -m scripts.run_evaluation --mapping evaluation_outputs/reference-mapping.json
```

Use `python -m scripts.run_evaluation --help` for options. Reports are timestamped JSON files at `backend/evaluation_outputs/evaluation-<UTC timestamp>.json`; `backend/evaluation_outputs/` is Git-ignored. Reports include per-question/mode status, ranked stable and database references, actual retrieval (including reranking when configured) latency, per-question metrics where applicable, sanitized errors, configuration, and mode summaries. Summary retrieval metrics average only successful answerable results. Failed retrievals are not treated as zero; answerable/unanswerable counts are reported independently. Latency summaries include every attempt, including failed attempts. Mapping/dataset validation failures stop before evaluating; individual retrieval failures are recorded, and the CLI returns a nonzero exit after saving the report.

`hybrid_rerank` uses the existing hybrid retrieval engine and lazy cross-encoder reranker. Dense/hybrid modes may load the existing embedding model on first use; `hybrid_rerank` may additionally load reranker weights. No Groq key is needed or read, and the retrieval runner does not generate answers. Faithfulness can be scored separately from manually reviewed JSONL records as described above; answer relevancy remains unimplemented. The retrieval runner is not executed by unit tests, which use fakes and do not establish measured project performance.

Before using any report as a benchmark, manually review the 60 synthetic questions, ingest the exact reviewed corpus, and verify every stable-reference mapping. The starter corpus is fictional and not a production benchmark. Faithfulness requires separately collected human claim reviews; answer relevancy remains future work.
