from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import ceil, isfinite, log2
from statistics import fmean
from typing import TypeAlias

from pydantic import Field

from app.evaluation.dataset import EvaluationQuestion, ExpectedEvidence, StrictModel

EvidenceInput: TypeAlias = (
    ExpectedEvidence | tuple[str, str] | Mapping[str, object]
)
EvidenceKey: TypeAlias = tuple[str, str]


class RetrievalMetricScores(StrictModel):
    question_id: str
    answerability: str
    recall_at_5: float | None
    recall_at_10: float | None
    reciprocal_rank: float | None
    ndcg_at_10: float | None


class LatencySummary(StrictModel):
    sample_count: int = Field(ge=0)
    p50_ms: float | None
    p95_ms: float | None
    mean_ms: float | None


def _evidence_key(reference: EvidenceInput) -> EvidenceKey:
    if isinstance(reference, ExpectedEvidence):
        document_ref, chunk_ref = reference.document_ref, reference.chunk_ref
    elif isinstance(reference, tuple):
        if len(reference) != 2:
            raise ValueError("Evidence references must contain document_ref and chunk_ref.")
        document_ref, chunk_ref = reference
    elif isinstance(reference, Mapping):
        try:
            validated = ExpectedEvidence.model_validate(reference)
        except ValueError as exc:
            raise ValueError(f"Malformed evidence reference: {exc}") from exc
        document_ref, chunk_ref = validated.document_ref, validated.chunk_ref
    else:
        raise ValueError("Evidence references must be ExpectedEvidence, a pair, or a mapping.")

    if (
        not isinstance(document_ref, str)
        or not document_ref.strip()
        or not isinstance(chunk_ref, str)
        or not chunk_ref.strip()
    ):
        raise ValueError("Evidence references require non-empty document_ref and chunk_ref.")
    return document_ref.strip(), chunk_ref.strip()


def _unique_references(references: Sequence[EvidenceInput]) -> list[EvidenceKey]:
    unique: list[EvidenceKey] = []
    seen: set[EvidenceKey] = set()
    for reference in references:
        key = _evidence_key(reference)
        if key not in seen:
            unique.append(key)
            seen.add(key)
    return unique


def _validated_question(question: EvaluationQuestion) -> None:
    if not isinstance(question, EvaluationQuestion):
        raise TypeError("question must be an EvaluationQuestion.")
    for evidence in question.relevant_evidence:
        _evidence_key(evidence)


def _retrieval_inputs(
    retrieved: Sequence[EvidenceInput],
    question: EvaluationQuestion,
) -> tuple[list[EvidenceKey], set[EvidenceKey]] | None:
    _validated_question(question)
    retrieved_keys = _unique_references(retrieved)
    if question.answerability == "unanswerable":
        return None
    relevant = set(_unique_references(question.relevant_evidence))
    if not relevant:
        raise ValueError("Answerable questions must have at least one relevant evidence reference.")
    return retrieved_keys, relevant


def recall_at_k(
    retrieved: Sequence[EvidenceInput],
    question: EvaluationQuestion,
    *,
    k: int,
) -> float | None:
    """Return per-question Recall@k; return None for unanswerable questions."""
    if isinstance(k, bool) or not isinstance(k, int) or k < 0:
        raise ValueError("k must be a non-negative integer.")
    inputs = _retrieval_inputs(retrieved, question)
    if inputs is None:
        return None
    retrieved_keys, relevant = inputs
    found = relevant.intersection(retrieved_keys[:k])
    return len(found) / len(relevant)


def recall_at_5(
    retrieved: Sequence[EvidenceInput],
    question: EvaluationQuestion,
) -> float | None:
    return recall_at_k(retrieved, question, k=5)


def recall_at_10(
    retrieved: Sequence[EvidenceInput],
    question: EvaluationQuestion,
) -> float | None:
    return recall_at_k(retrieved, question, k=10)


def reciprocal_rank(
    retrieved: Sequence[EvidenceInput],
    question: EvaluationQuestion,
) -> float | None:
    """Return one question's reciprocal rank, or None if it is unanswerable."""
    inputs = _retrieval_inputs(retrieved, question)
    if inputs is None:
        return None
    retrieved_keys, relevant = inputs
    for rank, reference in enumerate(retrieved_keys, start=1):
        if reference in relevant:
            return 1.0 / rank
    return 0.0


def mean_reciprocal_rank(
    evaluations: Sequence[tuple[Sequence[EvidenceInput], EvaluationQuestion]],
) -> float | None:
    """Average reciprocal rank over answerable questions; None if none are applicable."""
    scores: list[float] = []
    for evaluation in evaluations:
        if not isinstance(evaluation, tuple) or len(evaluation) != 2:
            raise ValueError("Each MRR input must be a (retrieved, question) pair.")
        retrieved, question = evaluation
        score = reciprocal_rank(retrieved, question)
        if score is not None:
            scores.append(score)
    return fmean(scores) if scores else None


def ndcg_at_10(
    retrieved: Sequence[EvidenceInput],
    question: EvaluationQuestion,
) -> float | None:
    """Return binary-relevance nDCG@10, or None for unanswerable questions."""
    inputs = _retrieval_inputs(retrieved, question)
    if inputs is None:
        return None
    retrieved_keys, relevant = inputs
    actual_dcg = sum(
        1.0 / log2(rank + 1)
        for rank, reference in enumerate(retrieved_keys[:10], start=1)
        if reference in relevant
    )
    ideal_dcg = sum(1.0 / log2(rank + 1) for rank in range(1, min(len(relevant), 10) + 1))
    return actual_dcg / ideal_dcg


def retrieval_metric_scores(
    retrieved: Sequence[EvidenceInput],
    question: EvaluationQuestion,
) -> RetrievalMetricScores:
    """Calculate all supported per-question retrieval metrics."""
    return RetrievalMetricScores(
        question_id=question.question_id,
        answerability=question.answerability,
        recall_at_5=recall_at_5(retrieved, question),
        recall_at_10=recall_at_10(retrieved, question),
        reciprocal_rank=reciprocal_rank(retrieved, question),
        ndcg_at_10=ndcg_at_10(retrieved, question),
    )


def _validated_durations(durations_ms: Sequence[float]) -> list[float]:
    values: list[float] = []
    for duration in durations_ms:
        if isinstance(duration, bool) or not isinstance(duration, (int, float)):
            raise ValueError("Latency durations must be finite non-negative numbers.")
        value = float(duration)
        if not isfinite(value) or value < 0:
            raise ValueError("Latency durations must be finite non-negative numbers.")
        values.append(value)
    return values


def percentile_latency(durations_ms: Sequence[float], percentile: float) -> float | None:
    """Return the nearest-rank percentile; percentile must be between 0 and 100."""
    if (
        isinstance(percentile, bool)
        or not isinstance(percentile, (int, float))
        or not isfinite(float(percentile))
        or not 0 <= percentile <= 100
    ):
        raise ValueError("percentile must be a finite number between 0 and 100.")
    values = sorted(_validated_durations(durations_ms))
    if not values:
        return None
    rank = max(1, ceil((float(percentile) / 100) * len(values)))
    return values[rank - 1]


def latency_summary(durations_ms: Sequence[float]) -> LatencySummary:
    """Summarize measured durations in ms; p50/p95 use nearest-rank percentiles."""
    values = _validated_durations(durations_ms)
    if not values:
        return LatencySummary(sample_count=0, p50_ms=None, p95_ms=None, mean_ms=None)
    return LatencySummary(
        sample_count=len(values),
        p50_ms=percentile_latency(values, 50),
        p95_ms=percentile_latency(values, 95),
        mean_ms=fmean(values),
    )


def p50_latency(durations_ms: Sequence[float]) -> float | None:
    return percentile_latency(durations_ms, 50)


def p95_latency(durations_ms: Sequence[float]) -> float | None:
    return percentile_latency(durations_ms, 95)


def mean_latency(durations_ms: Sequence[float]) -> float | None:
    values = _validated_durations(durations_ms)
    return fmean(values) if values else None
