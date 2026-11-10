# Manual Evidence-Quality Review

## Scope and evidence

This is a representative review of **6 questions**, one from each dataset
category, across the four modes in the saved evaluation run:

- Raw retrieval output: [`evaluation-20261104T195742783359Z.json`](evaluation-20261104T195742783359Z.json)
- Original questions and reference answers:
  [`questions.jsonl`](../evaluation_datasets/starter/questions.jsonl)
- Stable reference to database UUID mapping:
  [`reference-mapping.json`](reference-mapping.json)
- Source text for each selected evidence anchor:
  [`starter corpus`](../evaluation_datasets/starter/corpus/)

The sample is `q003` (direct_fact), `q013` (multi_chunk), `q025`
(cross_document), `q031` (paraphrased), `q045` (specific_detail), and `q051`
(unanswerable). Selection is illustrative, not statistically representative.

I checked the actual source text for expected references and the saved
retrieved stable references and ranks. The mapping was checked against the
retrieved database document/chunk UUIDs for all ranked results in these six
questions: **240 ranked-result references checked, 0 mapping mismatches**.
Retrieved IDs were not considered relevant solely because they matched; the
text excerpts and the asked-for facts were compared.

This review evaluates **evidence relevance, coverage, and ranking only**. It
does not score generated answers or establish answer correctness.

## Reviewed questions

### q003 — direct_fact

**Question:** What is the routine ticket acknowledgement target?

**Reference answer:** Routine tickets are acknowledged within four business
hours.

**Expected evidence:** `SYN-SUP-003/c01`, whose stored text says routine
tickets are acknowledged within four business hours. It also says an
acknowledgement includes a ticket reference and next expected update time,
without promising resolution.

| Mode | Expected reference rank | Evidence judgment |
|---|---:|---|
| dense | 1 | Direct and sufficient for the asked target. |
| bm25 | 1 | Direct and sufficient. |
| hybrid | 1 | Direct and sufficient. |
| hybrid_rerank | 1 | Direct and sufficient. |

Other top-three items include priority-target, routing, or closure chunks;
their presence does not affect the fact that the required chunk is ranked
first. **Review result: supported evidence, rank 1 in all modes.**

### q013 — multi_chunk

**Question:** What does a routine-ticket acknowledgement include, and when may
that ticket later be closed without requester confirmation?

**Reference answer:** The acknowledgement includes a ticket reference and next
expected update time. Without requester confirmation, closure is permitted
after two unanswered follow-ups sent on separate business days.

**Expected evidence:**

- `SYN-SUP-003/c01`: contains the ticket-reference and next-update details.
- `SYN-SUP-003/c03`: contains the two unanswered follow-ups on separate
  business days condition.

| Mode | c01 rank | c03 rank | Evidence judgment |
|---|---:|---:|---|
| dense | 1 | 2 | Both chunks appear at the top and jointly cover the requested parts. |
| bm25 | 1 | 2 | Both chunks appear at the top and jointly cover the requested parts. |
| hybrid | 1 | 2 | Both chunks appear at the top and jointly cover the requested parts. |
| hybrid_rerank | 2 | 1 | Both chunks appear at the top; reranking swaps their order. |

The source text of both expected chunks directly supports the requested
evidence. The result is **evidence-sufficient in top 2 for all modes**. This
does not establish that a generated answer would combine the facts correctly.

### q025 — cross_document

**Question:** How do routine acknowledgement and P1 response targets differ,
including whether P1 response is limited to staffed service-desk hours?

**Reference answer:** Routine tickets are acknowledged within four business
hours. P1 initial response is within 15 minutes at any hour; the on-call
engineer covers urgent incidents outside staffed hours.

**Expected evidence:**

- `SYN-SUP-003/c01`: routine acknowledgement within four business hours.
- `SYN-SUP-003/c02`: P1 response within 15 minutes at any hour.
- `SYN-OPS-001/c01`: staffed hours are weekdays 08:00–18:00 UTC, with the
  on-call engineer covering urgent incidents outside those hours.

| Mode | Expected evidence ranks | Top-5 evidence judgment |
|---|---|---|
| dense | c02: 1, c01: 2, OPS/c01: 9 | Partially complete in top 5: both support targets found, service-hours/on-call chunk absent until rank 9. |
| bm25 | c02: 1, c01: 2, OPS/c01: 4 | Complete in top 5; all three expected chunks are present. |
| hybrid | c02: 1, c01: 2, OPS/c01: 6 | Partially complete in top 5; all three appear by rank 6. |
| hybrid_rerank | c02: 1, c01: 2, OPS/c01: 6 | Partially complete in top 5; all three appear by rank 6. |

The inspected expected chunk texts directly cover all three elements, so the
distinction is rank/coverage depth, not a mismatch in the evidence labels.
BM25 is the only mode in this sample with complete expected evidence by rank
5. All modes include the full expected set in their top 10. **Top-5 partial
retrieval is not evidence that an answer produced from top 10 would be
incorrect.**

### q031 — paraphrased

**Question:** When does the support desk operate, and who is responsible for
urgent cases after that window?

**Reference answer:** The desk is staffed from 08:00 to 18:00 UTC on weekdays,
except public holidays; the on-call engineer covers urgent incidents outside
those hours.

**Expected evidence:** `SYN-OPS-001/c01` states all of those facts.

| Mode | Expected reference rank | Evidence judgment |
|---|---:|---|
| dense | 2 | Complete expected evidence at rank 2; rank 1 is the adjacent support-routing chunk, not a substitute for service-hours evidence. |
| bm25 | 3 | Complete expected evidence at rank 3; ranks 1–2 are other chunks and do not supply the full hours/on-call rule. |
| hybrid | 2 | Complete expected evidence at rank 2. |
| hybrid_rerank | 1 | Complete expected evidence at rank 1. |

The expected source chunk is directly sufficient. The saved per-question
metrics reflect the target's rank (dense/hybrid reciprocal rank 0.5, BM25
1/3, reranked 1.0); all four nevertheless retrieve it within top 5. This
supports a ranking observation for this sample, not a general paraphrase
performance claim.

### q045 — specific_detail

**Question:** Which three screen and interaction aspects are checked during
manual review?

**Reference answer:** Keyboard navigation, narrow-screen layout, and citation
rendering against the API response.

**Expected evidence:** `SYN-QUAL-007/c02` explicitly lists keyboard navigation,
narrow-screen layout, and citation rendering against the API response.

| Mode | Expected reference rank | Evidence judgment |
|---|---:|---|
| dense | 1 | Exact, sufficient evidence at rank 1. |
| bm25 | 1 | Exact, sufficient evidence at rank 1. |
| hybrid | 1 | Exact, sufficient evidence at rank 1. |
| hybrid_rerank | 1 | Exact, sufficient evidence at rank 1. |

The top-ranked source chunk lists all three requested aspects. **Review
result: supported evidence, rank 1 in all modes.**

### q051 — unanswerable

**Question:** What is the fictional Northstar chief executive's name?

**Dataset annotation:** `answerability="unanswerable"`, no reference answer,
no relevant evidence, with the reason that no chief executive or executive
names are given in the starter-corpus documents.

The top-three retrieved chunks were:

| Mode | Retrieved top three |
|---|---|
| dense | `SYN-SEC-002/c01` identity; `SYN-OPS-001/c01` service hours; `SYN-INC-006/c01` incident severity |
| bm25 | `SYN-OPS-001/c01` service hours; `SYN-COST-008/c01` monthly cap; `SYN-REL-004/c01` release cadence |
| hybrid | `SYN-OPS-001/c01` service hours; `SYN-SEC-002/c01` identity; `SYN-COST-008/c01` monthly cap |
| hybrid_rerank | `SYN-SEC-002/c01` identity; `SYN-INC-006/c01` incident severity; `SYN-REL-004/c01` release cadence |

The inspected retrieved text contains no executive name and does not answer
the question. A text scan of the 8 source Markdown files and 25 ingestion
Markdown files found no occurrences of “chief executive”, “CEO”, “executive”,
or “chief”. This supports the dataset's no-support annotation but cannot by
itself prove semantic absence or test how an answer generator would abstain.
**Review result: NEEDS_MANUAL_REVIEW** to confirm corpus-wide absence and any
abstention criteria with a human reviewer. The runner correctly records null
retrieval metrics for this unanswerable question; those nulls are not zeros.

## Category-wise observations

These are observations from **one sampled question per category**, not
category-level estimates:

- **direct_fact:** the exact target chunk ranked first for all four modes.
- **multi_chunk:** both distinct expected chunks ranked in the first two
  positions for all modes; reranking reversed their order.
- **cross_document:** the `q025` sample exposed a rank-depth difference.
  BM25 placed all three expected chunks by rank 4; dense, hybrid and reranked
  hybrid placed the service-hours chunk at ranks 9, 6 and 6 respectively.
- **paraphrased:** all modes retrieved the needed service-hours chunk in top
  3; reranking raised it to rank 1 in this example.
- **specific_detail:** the exact supporting chunk ranked first for every mode.
- **unanswerable:** inspected results were topically unrelated to an executive
  identity and did not reveal a name. Absence and abstention remain
  `NEEDS_MANUAL_REVIEW`.

## Successful retrieval examples

- **q003:** `SYN-SUP-003/c01` ranks first in every mode and states the exact
  four-business-hour acknowledgement target.
- **q013:** `SYN-SUP-003/c01` and `/c03` jointly provide acknowledgement
  contents and closure conditions; both are in top 2 in every mode.
- **q045:** `SYN-QUAL-007/c02` ranks first in all modes and explicitly lists
  all three requested review checks.

These are evidence-support observations only, not evaluations of generated
answers.

## Retrieval failures and cases needing judgment

- **q025 top-5 partial retrieval:** dense, hybrid, and hybrid_rerank omit
  `SYN-OPS-001/c01` from top 5 (ranks 9/6/6). Its text provides staffed hours
  and the on-call coverage rule. BM25 retrieves it at rank 4. All four include
  it by top 10. This is a genuine top-5 evidence coverage shortfall for three
  modes, but not a top-10 miss.
- **q051 unanswerable:** no retrieved top-three chunk supports a chief
  executive name. Whether the entire corpus is semantically silent and whether
  a future generated answer should abstain are **NEEDS_MANUAL_REVIEW**.
- No reviewed sample establishes generated-answer correctness, citation
  faithfulness, or relevancy.

## Hybrid versus reranked hybrid

- **q003:** both place the target at rank 1; the inspected top-three distractor
  ordering differs.
- **q013:** both retrieve both required chunks in top 2; hybrid ranks
  acknowledgement first, while reranking places closure first.
- **q025:** both rank the two support chunks first and second and the required
  service-hours chunk sixth. Reranking does not repair this sample's top-5
  coverage gap.
- **q031:** hybrid ranks the required chunk second; reranking moves it to first.
- **q045:** both rank the exact supporting chunk first.
- **q051:** both return non-empty results without a supporting executive
  name; unanswerable retrieval metrics are null, and abstention is not tested.

Reranking's differences here are order changes, not additional evidence
coverage. These six examples should not be generalized beyond the inspected
questions.

## Limitations

- Six questions are a small, selected sample; each category has only one
  reviewed example here.
- The starter corpus is fictional and small, and its annotations are pending
  human ground-truth review.
- Stable-ID agreement and text inspection support evidence association but do
  not establish that every relevant fact has been annotated.
- Retrieval evidence review is distinct from answer correctness. No answers
  were generated in this review, and faithfulness, answer relevancy, and
  unanswerable abstention were not evaluated.
- The unanswerable judgment is inherently a claim of absence; lexical scans
  are useful checks but do not replace human review of the full corpus.
