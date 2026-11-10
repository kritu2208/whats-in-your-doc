# Comparative Retrieval Analysis

## Executive summary

This report compares the four retrieval strategies using the latest saved run,
`evaluation-20261104T195742783359Z.json`, and the matching starter dataset
`evaluation_datasets/starter/questions.jsonl`. The run contains 60 questions per
mode (240 question-mode records), with 50 answerable and 10 unanswerable
questions. Every retrieval record succeeded.

Measured highlights:

- All four modes returned all relevant evidence within top 10 for all 50
  answerable questions (Recall@10 = 1.0000).
- BM25 had the highest overall Recall@5 (1.0000); dense, hybrid and hybrid
  reranking each scored 0.9933.
- Hybrid reranking had the highest overall MRR (1.0000) and nDCG@10 (0.9954).
  Its Recall@5/10 matched dense and hybrid, so the observed gain was ranking
  quality rather than greater top-5/top-10 evidence coverage.
- BM25 was fastest (p50 8.86 ms, p95 11.73 ms). Reranking was slower (p50
  428.39 ms, p95 683.21 ms) than hybrid (101.12 ms, 151.50 ms).
- At top 5, `q025` is the only answerable question with a missing evidence
  reference for dense, hybrid and hybrid_rerank. BM25 retrieved all three
  expected references in top 5. All four modes found all `q025` evidence by
  top 10.

These are retrieval results on a small, fictional starter corpus. They do not
measure generated-answer faithfulness, answer relevancy, or abstention quality.

## Evaluation setup

Evidence sources:

- Raw results and aggregate summaries:
  [`evaluation-20261104T195742783359Z.json`](evaluation-20261104T195742783359Z.json)
- Questions, categories, answerability and relevant-evidence labels:
  [`questions.jsonl`](../evaluation_datasets/starter/questions.jsonl)
- Methodology and metric definitions:
  [`evaluation methodology`](../evaluation_datasets/README.md)

The report configuration uses `top_k=10` for all modes and `candidate_k=50` for
`hybrid_rerank`. The four modes are dense vector retrieval, BM25, hybrid
retrieval using RRF, and hybrid retrieval followed by the cross-encoder
reranker. Per-question retrieval metrics match stable `(document_ref,
chunk_ref)` pairs in the dataset. Overall retrieval metrics average answerable
questions only. Latency summaries use all 60 attempts per mode, including
unanswerable questions, and use the methodology's nearest-rank p50/p95 method.

Dataset categories and sizes:

| Dataset category | Questions | Answerable questions scored for retrieval |
|---|---:|---:|
| direct_fact (`direct`) | 13 | 13 |
| multi_chunk | 7 | 7 |
| cross_document | 10 | 10 |
| paraphrased | 8 | 8 |
| specific_detail | 12 | 12 |
| unanswerable | 10 | 0 |

The dataset uses `direct_fact` as its category label; “direct” below refers to
that label.

## Overall metrics comparison

Values below are the saved report's `summaries[]` values. Retrieval metrics are
means over the 50 answerable questions; the unanswerable questions are not
treated as retrieval failures.

| Retrieval mode | Recall@5 | Recall@10 | MRR | nDCG@10 |
|---|---:|---:|---:|---:|
| Dense | 0.9933 | 1.0000 | 0.9800 | 0.9804 |
| BM25 | **1.0000** | 1.0000 | 0.9183 | 0.9287 |
| Hybrid (RRF) | 0.9933 | 1.0000 | 0.9500 | 0.9569 |
| Hybrid + reranking | 0.9933 | 1.0000 | **1.0000** | **0.9954** |

Metric leaders:

- Recall@5: BM25 (1.0000); Recall@10 is tied at 1.0000.
- MRR and nDCG@10: hybrid + reranking (1.0000 and 0.9954).
- No one mode leads every metric: BM25 leads Recall@5 and latency, while
  reranking leads MRR/nDCG.

## Category-wise comparison

Each cell is `Recall@5 / Recall@10 / MRR / nDCG@10`, averaged over the
answerable questions in that category. Category `n` is shown in the first
column. Unanswerable rows have no retrieval-quality score by design.

| Category (n) | Dense | BM25 | Hybrid (RRF) | Hybrid + reranking |
|---|---|---|---|---|
| direct_fact (13) | 1.0000 / 1.0000 / 0.9615 / 0.9716 | 1.0000 / 1.0000 / 0.9231 / 0.9432 | 1.0000 / 1.0000 / 0.9231 / 0.9432 | 1.0000 / 1.0000 / 1.0000 / 1.0000 |
| multi_chunk (7) | 1.0000 / 1.0000 / 1.0000 / **1.0000** | 1.0000 / 1.0000 / 1.0000 / 0.9595 | 1.0000 / 1.0000 / 1.0000 / 0.9771 | 1.0000 / 1.0000 / 1.0000 / 0.9885 |
| cross_document (10) | 0.9667 / 1.0000 / 1.0000 / 0.9757 | **1.0000** / 1.0000 / 1.0000 / 0.9764 | 0.9667 / 1.0000 / 1.0000 / **0.9852** | 0.9667 / 1.0000 / 1.0000 / **0.9852** |
| paraphrased (8) | 1.0000 / 1.0000 / 0.9375 / 0.9539 | 1.0000 / 1.0000 / 0.7604 / 0.8202 | 1.0000 / 1.0000 / 0.8750 / 0.9077 | 1.0000 / 1.0000 / **1.0000** / **1.0000** |
| specific_detail (12) | 1.0000 / 1.0000 / **1.0000** / **1.0000** | 1.0000 / 1.0000 / 0.9028 / 0.9276 | 1.0000 / 1.0000 / 0.9583 / 0.9692 | **1.0000** / 1.0000 / **1.0000** / **1.0000** |
| unanswerable (10) | N/A | N/A | N/A | N/A |

Category leaders (ties are retained):

- **direct_fact:** all modes tie for recall; reranking leads MRR/nDCG.
- **multi_chunk:** all modes tie for Recall@5/10 and MRR; dense leads nDCG.
- **cross_document:** BM25 leads Recall@5; all modes tie for Recall@10/MRR;
  hybrid and reranking tie for nDCG.
- **paraphrased:** all modes tie for recall; reranking leads MRR/nDCG.
- **specific_detail:** all modes tie for recall; dense and reranking tie for
  MRR/nDCG.
- **unanswerable:** retrieval ranking metrics are not applicable. The report
  records 10 unanswerable retrieval attempts per mode with non-empty retrieved
  lists, but this does not establish whether an answer generator would abstain
  correctly.

## Latency comparison

Measured latency values come from `summaries[]` in the raw run report. Latency
includes all 60 attempts in each mode, including unanswerable questions.

| Retrieval mode | p50 (ms) | p95 (ms) | Mean (ms) |
|---|---:|---:|---:|
| Dense | 94.67 | 181.05 | 569.13 |
| BM25 | **8.86** | **11.73** | **9.28** |
| Hybrid (RRF) | 101.12 | 151.50 | 111.63 |
| Hybrid + reranking | 428.39 | 683.21 | 550.33 |

BM25 is the fastest measured option. Relative to hybrid, reranking increases
p50 by 327.27 ms (about 4.24x) and p95 by 531.70 ms (about 4.51x), while
increasing MRR by 0.0500 and nDCG@10 by 0.0385. It does not improve the
aggregate Recall@5/Recall@10 values over hybrid.

Means are especially sensitive to outliers in this run. The raw per-question
records include dense `q001` latency of 28,098.63 ms and hybrid_rerank `q001`
latency of 5,908.34 ms. These are included in the saved summaries, explaining
why those means are much higher than their p50/p95 values. The observed
latencies should not be interpreted as stable production benchmarks.

## Failure analysis

“Top-5 miss” means one or more dataset `relevant_evidence` references are
absent from the first five retrieved results. “Top-10 miss” is defined
similarly. This is evidence retrieval coverage, not answer-generation failure.

| Mode | Answerable questions with a top-5 evidence miss | Answerable questions with a top-10 evidence miss |
|---|---:|---:|
| Dense | 1 (`q025`) | 0 |
| BM25 | 0 | 0 |
| Hybrid (RRF) | 1 (`q025`) | 0 |
| Hybrid + reranking | 1 (`q025`) | 0 |

`q025` is labeled `cross_document` and asks how routine ticket
acknowledgement and P1 response targets differ, including whether P1 response
is limited to staffed service-desk hours. Its three expected references are
`SYN-SUP-003/c01`, `SYN-SUP-003/c02`, and `SYN-OPS-001/c01`.

- Dense, hybrid and hybrid_rerank retrieve the two support-playbook chunks
  within top 5, but miss `SYN-OPS-001/c01` in top 5. Each records Recall@5
  0.6667, MRR 1.0000 and nDCG@10 of 0.9066 (dense) or 0.9325 (hybrid and
  reranked hybrid). Each finds all three references by top 10.
- BM25 retrieves all three references within top 5 (Recall@5 1.0000, MRR
  1.0000, nDCG@10 0.9675).

There are no top-10 relevant-evidence misses among the 50 answerable questions
for any mode. No other answerable failure pattern was observed. In particular,
the evaluated multi-chunk and specific-detail categories had Recall@5/10 of
1.0000 for every strategy. The cross-document category's mean Recall@5 is
0.9667 for dense, hybrid and reranking versus 1.0000 for BM25; all have
Recall@10 1.0000.

The 10 unanswerable questions have no supporting evidence references, so the
runner correctly leaves their retrieval metrics null and excludes them from
the answerable retrieval aggregates. All 10 have non-empty retrieval results
in each mode. The evaluation does not measure abstention correctness because
no answers were generated.

## Key observations

### Does hybrid improve on dense-only and BM25-only?

- **Against dense:** hybrid has the same overall Recall@5/10, but lower MRR
  (0.9500 vs 0.9800) and nDCG@10 (0.9569 vs 0.9804). Its p95 is lower
  (151.50 vs 181.05 ms), while p50 is slightly higher (101.12 vs 94.67 ms).
  The dense mean is dominated by the recorded `q001` outlier.
- **Against BM25:** hybrid has the same Recall@10 but lower Recall@5
  (0.9933 vs 1.0000); its MRR (0.9500 vs 0.9183) and nDCG@10
  (0.9569 vs 0.9287) are higher. Hybrid is slower: p50 101.12 vs 8.86 ms and
  p95 151.50 vs 11.73 ms.

Therefore, hybrid does not uniformly improve on both baselines. It improves
the overall ranking metrics over BM25 but does not beat dense on those overall
ranking metrics, and BM25 has higher Recall@5.

### Does reranking improve retrieval quality?

Relative to hybrid, reranking leaves overall Recall@5/10 unchanged, raises MRR
from 0.9500 to 1.0000, and raises nDCG@10 from 0.9569 to 0.9954. By category,
the strongest observed gains are in paraphrased and direct_fact ranking.
Cross-document MRR/nDCG and multi_chunk MRR do not improve over the hybrid
values. Reranking still shares the `q025` top-5 miss and has the same
cross-document Recall@5 as dense/hybrid.

The trade-off is latency: reranked hybrid has p50 428.39 ms and p95 683.21 ms,
versus hybrid's 101.12 ms and 151.50 ms. Thus the measured reranking benefit is
better rank ordering, not increased overall evidence recall, and it costs
substantially more time on this run.

## Limitations

- The corpus consists of only eight fictional logical source documents and 25
  separately ingested anchor files. The 60-question set is explicitly a
  synthetic starter dataset, not representative production evidence.
- Category samples are small (7–13 answerable questions per scored category).
  Differences in category means can be driven by one question.
- This is one run and one local machine/runtime. Latency, particularly with
  cold model loading and the observed outliers, is not a robust throughput or
  production latency benchmark.
- Retrieval ranking metrics assess overlap with manually specified relevant
  evidence references. They do not prove answer correctness or the quality of
  citations in generated answers.
- Faithfulness and answer relevancy were not evaluated. Unanswerable-answer
  abstention was also not evaluated.
- The starter dataset documentation states that question ground truth remains
  pending manual review. Results should not be treated as final until that
  review is complete.

## Evidence-based conclusion

For this specific run, BM25 is the clear latency and Recall@5 leader, while
hybrid reranking leads overall MRR and nDCG@10. Dense retrieval performs
strongly on ranking metrics and is faster than hybrid reranking; hybrid RRF
does not beat dense across overall ranking measures. Reranking improves
ordering over unreranked hybrid without changing overall Recall@5/10, at a
substantial latency cost. The single observed top-5 cross-document miss for
dense and both hybrid variants is recovered by top 10; BM25 also retrieves it
within top 5.

These comparisons describe only the saved synthetic run. They are not claims
about answer-generation faithfulness, relevancy, real-world corpora, or
production performance.
