# M7 Final Evaluation Report

## 1. Project and evaluation overview

This report summarizes the offline retrieval evaluation for the Hybrid RAG
Document Q&A project. It uses the latest evaluation run,
[`evaluation-20261104T195742783359Z.json`](evaluation-20261104T195742783359Z.json),
and cross-references the original evaluation dataset, manifest, completed
stable-reference mapping, comparative analysis, and manual evidence-quality
review.

The evaluation tests retrieval against a small, explicitly synthetic corpus.
It is a development evaluation of evidence retrieval, not a production
benchmark and not a complete evaluation of generated answers.

## 2. Evaluation objectives

The evaluation was intended to compare four configured retrieval strategies
using the dataset's stable `(document_ref, chunk_ref)` ground truth:

- Measure evidence coverage and ranking quality.
- Compare dense, lexical BM25, RRF hybrid, and hybrid with cross-encoder
  reranking.
- Record per-question retrieval latency and summarize it by mode.
- Keep answerable and unanswerable retrieval metrics distinct.

The run did not generate answers. It therefore did not measure answer
faithfulness, answer relevancy, citation correctness in generated answers, or
abstention quality.

## 3. Dataset description

The manifest identifies dataset `hybrid-rag-synthetic-starter`, version
`0.1.0`, and labels it `synthetic_starter`. It defines 8 fictional logical
Markdown documents with 25 stable evidence anchors, uploaded as individual
ingestion units for a one-anchor-to-one-database-chunk mapping.

The question file contains 60 questions: 50 answerable and 10 unanswerable.
The categories and counts are:

| Category | Questions | Answerable |
|---|---:|---:|
| direct_fact (direct) | 13 | 13 |
| multi_chunk | 7 | 7 |
| cross_document | 10 | 10 |
| paraphrased | 8 | 8 |
| specific_detail | 12 | 12 |
| unanswerable | 10 | 0 |
| **Total** | **60** | **50** |

Each answerable question carries a reference answer and one or more stable
relevant evidence references. Unanswerable questions have no reference answer
or evidence references and include a reason. The dataset documentation states
that these are fictional development materials, not real-world evidence; the
questions remain pending manual review.

The completed mapping has `reviewed: true` and maps all 25 stable anchors to
database document and chunk UUIDs. The mapping was checked against the actual
database during mapping validation before the evaluation run.

## 4. Evaluation methodology

The saved report records 240 question-mode results: 60 questions in each of
the four modes. All 240 retrieval records succeeded, with zero mode failures.
The run uses `top_k=10` for every mode and `candidate_k=50` for
`hybrid_rerank`.

The runner calls the existing retrieval engine and existing reranker. It
matches retrieved results with ground truth using unique stable
`(document_ref, chunk_ref)` pairs. Retrieval metrics are averaged over
successful **answerable** questions only. Unanswerable questions have null
retrieval metrics and are not counted as ordinary retrieval failures. Latency
summaries include all 60 attempts per mode. The p50 and p95 use nearest-rank
percentiles as documented in the evaluation methodology.

The report was produced by the offline retrieval runner; it did not call an
LLM or generate answers. The report's raw per-question records and aggregate
summaries were independently checked against the original question evidence
and metric definitions.

## 5. Retrieval strategies tested

1. **Dense:** vector similarity retrieval using stored embeddings.
2. **BM25:** lexical retrieval using BM25 ranking.
3. **Hybrid (RRF):** combines dense and BM25 rankings using reciprocal rank
   fusion.
4. **Hybrid + cross-encoder reranking:** retrieves hybrid candidates and
   reorders them with the existing cross-encoder; the run used 50 candidates
   before returning top 10.

These are retrieval configurations only. The runner does not use a generated
answer stage.

## 6. Metrics and their definitions

- **Recall@5 / Recall@10:** number of unique relevant stable evidence
  references retrieved in the first 5 / 10 results, divided by the number of
  unique relevant references for that question.
- **Reciprocal rank:** `1 / rank` of the first relevant retrieved reference,
  or zero if none is retrieved. **MRR** is its mean over answerable questions.
- **nDCG@10:** normalized discounted cumulative gain over the first 10
  results, using binary relevance; earlier relevant references contribute
  more.
- **Latency p50 / p95:** nearest-rank 50th and 95th percentiles of observed
  retrieval duration in milliseconds. **Mean latency** is the arithmetic
  mean. These include all 60 attempts per mode, including unanswerable
  questions.
- **Answer faithfulness and answer relevancy:** not measured in this run.
  Their evaluator is an interface only; no answer-quality scores are present
  in the raw report. This absence means not evaluated, not a zero score.

For unanswerable questions, retrieval metrics are null because there is no
ground-truth relevant evidence set. The run does not determine if an answer
generator abstains correctly.

## 7. Overall numerical results

The following values are from the latest raw report's `summaries[]`. Retrieval
quality is aggregated over the 50 answerable questions; each mode processed
all 60 questions successfully.

| Mode | Recall@5 | Recall@10 | MRR | nDCG@10 | Successes | Failures |
|---|---:|---:|---:|---:|---:|---:|
| Dense | 0.9933 | 1.0000 | 0.9800 | 0.9804 | 60 | 0 |
| BM25 | **1.0000** | 1.0000 | 0.9183 | 0.9287 | 60 | 0 |
| Hybrid (RRF) | 0.9933 | 1.0000 | 0.9500 | 0.9569 | 60 | 0 |
| Hybrid + reranking | 0.9933 | 1.0000 | **1.0000** | **0.9954** | 60 | 0 |

Metric-specific leaders: BM25 had the highest Recall@5; all modes tied on
Recall@10; hybrid reranking had the highest MRR and nDCG@10. Thus there is no
single strategy that leads every retrieval-quality metric.

## 8. Category-wise results

The table gives category means recomputed from per-question records in the raw
report, matched to categories in the original dataset. Each cell is
`Recall@5 / Recall@10 / MRR / nDCG@10`. `n` is the number of answerable
questions scored in that category. Unanswerable questions are not assigned
retrieval-quality scores.

| Category (n) | Dense | BM25 | Hybrid (RRF) | Hybrid + reranking |
|---|---|---|---|---|
| direct_fact (13) | 1.0000 / 1.0000 / 0.9615 / 0.9716 | 1.0000 / 1.0000 / 0.9231 / 0.9432 | 1.0000 / 1.0000 / 0.9231 / 0.9432 | 1.0000 / 1.0000 / 1.0000 / 1.0000 |
| multi_chunk (7) | 1.0000 / 1.0000 / 1.0000 / 1.0000 | 1.0000 / 1.0000 / 1.0000 / 0.9595 | 1.0000 / 1.0000 / 1.0000 / 0.9771 | 1.0000 / 1.0000 / 1.0000 / 0.9885 |
| cross_document (10) | 0.9667 / 1.0000 / 1.0000 / 0.9757 | 1.0000 / 1.0000 / 1.0000 / 0.9764 | 0.9667 / 1.0000 / 1.0000 / 0.9852 | 0.9667 / 1.0000 / 1.0000 / 0.9852 |
| paraphrased (8) | 1.0000 / 1.0000 / 0.9375 / 0.9539 | 1.0000 / 1.0000 / 0.7604 / 0.8202 | 1.0000 / 1.0000 / 0.8750 / 0.9077 | 1.0000 / 1.0000 / 1.0000 / 1.0000 |
| specific_detail (12) | 1.0000 / 1.0000 / 1.0000 / 1.0000 | 1.0000 / 1.0000 / 0.9028 / 0.9276 | 1.0000 / 1.0000 / 0.9583 / 0.9692 | 1.0000 / 1.0000 / 1.0000 / 1.0000 |
| unanswerable (10) | N/A | N/A | N/A | N/A |

Category leaders are metric-specific: BM25 led cross-document Recall@5; all
modes tied on category Recall@10; reranking led MRR/nDCG for direct_fact and
paraphrased; dense led multi_chunk nDCG; and dense and reranking tied for
specific_detail MRR/nDCG. Unanswerable retrieval quality is not ranked by
these evidence-recall metrics.

## 9. Latency and performance trade-offs

Measured aggregate latency from `summaries[]`:

| Mode | p50 (ms) | p95 (ms) | Mean (ms) |
|---|---:|---:|---:|
| Dense | 94.67 | 181.05 | 569.13 |
| BM25 | **8.86** | **11.73** | **9.28** |
| Hybrid (RRF) | 101.12 | 151.50 | 111.63 |
| Hybrid + reranking | 428.39 | 683.21 | 550.33 |

BM25 was fastest in this run. Compared with unreranked hybrid, reranking
increased p50 latency by 327.27 ms (4.24x) and p95 by 531.70 ms (4.51x).
Against this latency cost, reranking raised overall MRR from 0.9500 to 1.0000
and nDCG@10 from 0.9569 to 0.9954, while Recall@5 and Recall@10 remained
unchanged at 0.9933 and 1.0000.

The mean latency includes observed outliers: dense `q001` took 28,098.63 ms,
and hybrid_rerank `q001` took 5,908.34 ms. These are reflected in the means
but not the p50/p95 values shown. The numbers are from one local run and
should not be interpreted as stable production latency.

## 10. Manual evidence-quality review

The separate manual review inspected six representative questions, one per
category, across all four modes. It inspected the expected source chunks,
retrieved stable references, their ranks, and actual chunk text; 240 ranked
result references for those selected records were checked against the
reference mapping with no UUID mismatches.

Observed examples:

- **q003 (direct_fact):** `SYN-SUP-003/c01`, which states the four-business-hour
  acknowledgement target, ranked first in all modes.
- **q013 (multi_chunk):** `SYN-SUP-003/c01` and `/c03` jointly contain the
  requested acknowledgement and closure details; both appeared in the first
  two results in every mode.
- **q045 (specific_detail):** `SYN-QUAL-007/c02` explicitly lists all three
  requested checks and ranked first in every mode.
- **q025 (cross_document):** BM25 retrieved all three expected evidence
  chunks by rank 4; dense, hybrid and hybrid_rerank placed one expected
  service-hours chunk at ranks 9, 6 and 6 respectively.
- **q031 (paraphrased):** the service-hours evidence ranked 2 for dense and
  hybrid, 3 for BM25, and 1 for reranked hybrid.
- **q051 (unanswerable):** inspected retrieved top-three chunks did not
  support or state a chief executive name. A lexical scan of the corpus found
  no matches for executive-related terms, but semantic absence and
  answer-generator abstention remain `NEEDS_MANUAL_REVIEW`.

This is a small illustrative review. It checks evidence relevance and ranking,
not whether a generated answer is correct. The full dataset remains pending
manual review, and lexical absence checks do not prove semantic absence.
See [`manual_quality_review.md`](manual_quality_review.md) for the detailed
observations.

## 11. Failure analysis

At top 10, all four modes retrieved every annotated relevant evidence set for
the 50 answerable questions: there were **zero top-10 evidence-miss
questions** in each mode.

At top 5:

| Mode | Answerable questions with a relevant evidence miss |
|---|---|
| Dense | 1 (`q025`) |
| BM25 | 0 |
| Hybrid (RRF) | 1 (`q025`) |
| Hybrid + reranking | 1 (`q025`) |

For `q025`, the three expected references are
`SYN-SUP-003/c01`, `SYN-SUP-003/c02`, and `SYN-OPS-001/c01`. Dense and both
hybrid variants retrieved the first two in top 5 but placed the service-hours
and on-call evidence after top 5; BM25 placed it at rank 4. All four retrieved
it by top 10. This is a top-5 evidence coverage shortfall, not proof of an
incorrect generated answer.

All 10 unanswerable questions had non-empty retrieval results in each mode,
but their retrieval metrics were null by policy. The run does not score
whether an answer generator would abstain appropriately.

## 12. Key findings

- **Best Recall@5:** BM25 at 1.0000. Dense, hybrid, and reranked hybrid each
  scored 0.9933. Recall@10 was 1.0000 for every mode.
- **Best overall ranking metrics:** hybrid reranking had MRR 1.0000 and
  nDCG@10 0.9954. Dense was next on these overall ranking metrics (0.9800
  MRR; 0.9804 nDCG@10); BM25 was lower on ranking position despite perfect
  Recall@5.
- **Hybrid versus single methods:** hybrid improved overall MRR and nDCG@10
  over BM25 but not over dense; it matched dense on Recall@5/10 and did not
  exceed BM25's Recall@5. Hybrid therefore did not uniformly improve on both
  single-method baselines.
- **Reranking trade-off:** reranking improved overall MRR/nDCG over hybrid
  without improving aggregate recall, at substantially higher p50 and p95
  latency.
- **Evidence failure pattern:** the only answerable top-5 miss across the run
  was the cross-document `q025`, shared by dense and both hybrid variants.
  No answerable question was missing evidence at top 10.
- **Quality scope:** the evaluation measured retrieval only. It measured no
  generated-answer quality or abstention behavior.

## 13. Limitations

- The corpus is synthetic, fictional, and limited to 8 logical documents and
  25 evidence anchors; the question set has only 60 items.
- Question-level category samples are small (7–13 answerable questions per
  scored category), so a small number of cases can materially affect category
  means.
- The dataset documentation says questions and ground truth are pending
  manual review. Results should not be treated as final benchmark findings.
- This was one run on one local environment. Latency can vary with hardware,
  caches, model warm-up, and database state; the observed outliers strongly
  affect mean latency.
- Recall and ranking scores measure agreement with annotated evidence
  references, not completeness of the annotation or correctness of answers.
- Faithfulness, answer relevancy, citation correctness in generated answers,
  and unanswerable-answer abstention were not evaluated.
- No production readiness or real-world generalization can be concluded from
  this synthetic evaluation.

## 14. Final conclusion

On this specific synthetic run, BM25 provided the best Recall@5 and the
lowest measured latency. Hybrid reranking produced the best overall ranking
scores (MRR and nDCG@10), but did not increase Recall@5/10 over hybrid and
introduced a substantial latency cost. Dense retrieval had strong overall
ranking scores and lower p50/p95 latency than reranked hybrid in this run.
Hybrid RRF was between the baselines on overall ranking metrics and did not
uniformly outperform dense and BM25.

There is no universally best strategy across all measured dimensions. The
choice depends on whether the priority is speed, top-5 evidence coverage, or
ranking order. These conclusions apply only to this saved run and small
synthetic dataset. They do not establish answer quality or production
readiness.

## 15. Future improvements

- Manually review every question, answerability decision, reference answer,
  and evidence annotation; resolve review-template items and publish a
  reviewed dataset version.
- Expand the corpus and questions with representative real, authorized
  documents and substantially larger category samples.
- Repeat runs under controlled warm/cold-cache conditions and record the
  environment, model versions, and resource usage to characterize latency
  variability.
- Add a separate, clearly specified answer-generation evaluation for
  faithfulness, answer relevancy, citation correctness, and abstention,
  without conflating those measures with retrieval metrics.
- Investigate top-5 cross-document coverage for `q025` and similar questions
  without tuning against the held-out test set; use a separate development
  set for configuration changes.

## Sources

- Raw measured run:
  [`evaluation-20261104T195742783359Z.json`](evaluation-20261104T195742783359Z.json)
- Comparative analysis:
  [`comparative_analysis.md`](comparative_analysis.md)
- Manual evidence-quality review:
  [`manual_quality_review.md`](manual_quality_review.md)
- Dataset manifest:
  [`manifest.json`](../evaluation_datasets/starter/manifest.json)
- Dataset questions:
  [`questions.jsonl`](../evaluation_datasets/starter/questions.jsonl)
- Stable-reference mapping:
  [`reference-mapping.json`](reference-mapping.json)
- Evaluation methodology:
  [`evaluation methodology`](../evaluation_datasets/README.md)
