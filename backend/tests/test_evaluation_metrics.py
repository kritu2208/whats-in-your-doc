from __future__ import annotations

import math

import pytest
from pydantic import ValidationError

from app.evaluation.config import AblationConfig, DEFAULT_ABLATIONS
from app.evaluation.dataset import EvaluationQuestion, ExpectedEvidence
from app.evaluation.metrics import (
    latency_summary,
    mean_latency,
    mean_reciprocal_rank,
    ndcg_at_10,
    p50_latency,
    p95_latency,
    percentile_latency,
    recall_at_5,
    recall_at_10,
    recall_at_k,
    reciprocal_rank,
    retrieval_metric_scores,
)
from app.evaluation.quality import (
    AnswerQualityInput,
    AnswerQualityOutput,
    QualityMetricResult,
)
from app.retrieval.engine import RetrievalMode


def make_question(
    refs: list[tuple[str, str]],
    *,
    answerability: str = "answerable",
) -> EvaluationQuestion:
    return EvaluationQuestion(
        question_id="metric-question",
        question="Which service window applies?",
        answerability=answerability,
        reference_answer="The standard window applies." if answerability == "answerable" else None,
        relevant_evidence=[
            {"document_ref": document, "chunk_ref": chunk} for document, chunk in refs
        ],
        unanswerable_reason="The corpus contains no support." if answerability == "unanswerable" else None,
        category="unanswerable" if answerability == "unanswerable" else "direct_fact",
        difficulty="easy",
    )


def refs(*values: tuple[str, str]) -> list[tuple[str, str]]:
    return list(values)


def test_perfect_retrieval_ranking_scores_all_metrics_as_one() -> None:
    question = make_question([("DOC-A", "c01"), ("DOC-B", "c02")])
    retrieved = refs(("DOC-A", "c01"), ("DOC-B", "c02"))

    assert recall_at_5(retrieved, question) == 1.0
    assert recall_at_10(retrieved, question) == 1.0
    assert reciprocal_rank(retrieved, question) == 1.0
    assert mean_reciprocal_rank([(retrieved, question)]) == 1.0
    assert ndcg_at_10(retrieved, question) == 1.0


def test_relevant_chunk_at_lower_rank_reduces_reciprocal_rank_and_ndcg() -> None:
    question = make_question([("DOC-A", "c01")])
    retrieved = refs(("DOC-X", "c01"), ("DOC-Y", "c02"), ("DOC-A", "c01"))

    assert recall_at_5(retrieved, question) == 1.0
    assert reciprocal_rank(retrieved, question) == pytest.approx(1 / 3)
    assert ndcg_at_10(retrieved, question) == pytest.approx(1 / math.log2(4))


def test_multiple_relevant_chunks_count_toward_recall_and_ndcg() -> None:
    question = make_question(
        [("DOC-A", "c01"), ("DOC-A", "c02"), ("DOC-B", "c01")]
    )
    retrieved = refs(("DOC-A", "c01"), ("DOC-X", "c99"), ("DOC-B", "c01"))

    assert recall_at_5(retrieved, question) == pytest.approx(2 / 3)
    assert ndcg_at_10(retrieved, question) == pytest.approx(
        (1 + 1 / math.log2(4))
        / (1 + 1 / math.log2(3) + 1 / math.log2(4))
    )


def test_no_relevant_chunks_retrieved_returns_zero_scores() -> None:
    question = make_question([("DOC-A", "c01")])
    retrieved = refs(("DOC-B", "c01"), ("DOC-C", "c02"))

    assert recall_at_5(retrieved, question) == 0.0
    assert reciprocal_rank(retrieved, question) == 0.0
    assert ndcg_at_10(retrieved, question) == 0.0


def test_empty_retrieval_is_an_answerable_retrieval_failure() -> None:
    question = make_question([("DOC-A", "c01")])

    assert recall_at_5([], question) == 0.0
    assert recall_at_10([], question) == 0.0
    assert reciprocal_rank([], question) == 0.0
    assert ndcg_at_10([], question) == 0.0


def test_unanswerable_questions_are_excluded_not_scored_as_failures() -> None:
    question = make_question([], answerability="unanswerable")
    scores = retrieval_metric_scores(refs(("DOC-A", "c01")), question)

    assert scores.recall_at_5 is None
    assert scores.recall_at_10 is None
    assert scores.reciprocal_rank is None
    assert scores.ndcg_at_10 is None


def test_duplicate_retrieved_and_ground_truth_references_are_deduplicated() -> None:
    question = make_question([("DOC-A", "c01"), ("DOC-A", "c01")])
    retrieved = refs(("DOC-X", "c01"), ("DOC-A", "c01"), ("DOC-A", "c01"))

    assert recall_at_5(retrieved, question) == 1.0
    assert reciprocal_rank(retrieved, question) == 0.5


def test_mean_reciprocal_rank_averages_answerable_questions_only() -> None:
    first = make_question([("DOC-A", "c01")])
    second = make_question([("DOC-B", "c02")]).model_copy(
        update={"question_id": "metric-question-2"}
    )
    unanswerable = make_question([], answerability="unanswerable")

    assert mean_reciprocal_rank(
        [
            (refs(("DOC-A", "c01")), first),
            (refs(("DOC-X", "c01"), ("DOC-B", "c02")), second),
            ([], unanswerable),
        ]
    ) == pytest.approx((1.0 + 0.5) / 2)
    assert mean_reciprocal_rank([([], unanswerable)]) is None


@pytest.mark.parametrize("k", [-1, 1.5, True])
def test_recall_rejects_invalid_k(k: int | float | bool) -> None:
    with pytest.raises(ValueError, match="non-negative integer"):
        recall_at_k([], make_question([("DOC-A", "c01")]), k=k)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "reference",
    [
        ("DOC-A", ""),
        ("", "c01"),
        ("DOC-A", "c01", "extra"),
        {"document_ref": "DOC-A", "chunk_ref": ""},
    ],
)
def test_malformed_retrieval_references_are_rejected(reference: object) -> None:
    with pytest.raises(ValueError, match="Evidence|Malformed"):
        recall_at_5([reference], make_question([("DOC-A", "c01")]))  # type: ignore[list-item]


def test_metric_rejects_malformed_ground_truth_even_if_model_was_copied_unsafely() -> None:
    question = make_question([("DOC-A", "c01")]).model_copy(
        update={"relevant_evidence": [ExpectedEvidence.model_construct(document_ref="", chunk_ref="c01")]}
    )

    with pytest.raises(ValueError, match="non-empty"):
        recall_at_5([], question)


def test_latency_nearest_rank_percentiles_and_mean() -> None:
    durations = [10, 20, 30, 40, 100]

    assert percentile_latency(durations, 50) == 30
    assert percentile_latency(durations, 95) == 100
    assert p50_latency(durations) == 30
    assert p95_latency(durations) == 100
    assert mean_latency(durations) == 40
    assert latency_summary(durations).model_dump() == {
        "sample_count": 5,
        "p50_ms": 30.0,
        "p95_ms": 100.0,
        "mean_ms": 40.0,
    }


def test_empty_latency_input_has_no_measured_values() -> None:
    assert percentile_latency([], 50) is None
    assert mean_latency([]) is None
    assert latency_summary([]).model_dump() == {
        "sample_count": 0,
        "p50_ms": None,
        "p95_ms": None,
        "mean_ms": None,
    }


@pytest.mark.parametrize("durations", [[-1], [float("nan")], [float("inf")], [True]])
def test_invalid_latency_durations_are_rejected(durations: list[float]) -> None:
    with pytest.raises(ValueError, match="finite non-negative"):
        latency_summary(durations)


@pytest.mark.parametrize("percentile", [-1, 101, float("nan"), True])
def test_invalid_percentile_is_rejected(percentile: float) -> None:
    with pytest.raises(ValueError, match="between 0 and 100"):
        percentile_latency([10], percentile)


def test_quality_scores_are_explicitly_not_evaluated_by_default() -> None:
    request = AnswerQualityInput(
        question_id="q1",
        question="What is documented?",
        answer="The document says this.",
        answerability="answerable",
        retrieved_contexts=["Source text."],
    )
    result = AnswerQualityOutput(question_id=request.question_id)

    assert result.faithfulness.status == "not_evaluated"
    assert result.faithfulness.score is None
    assert result.answer_relevancy.status == "not_evaluated"
    assert result.answer_relevancy.score is None


def test_quality_metric_zero_is_distinct_from_unavailable() -> None:
    zero_score = QualityMetricResult(status="evaluated", score=0.0)
    unavailable = QualityMetricResult(status="unavailable")

    assert zero_score.score == 0.0
    assert zero_score.status == "evaluated"
    assert unavailable.score is None
    assert unavailable.status == "unavailable"
    with pytest.raises(ValidationError):
        QualityMetricResult(status="evaluated")


def test_ablation_defaults_match_existing_retrieval_modes() -> None:
    assert [config.mode for config in DEFAULT_ABLATIONS] == [
        RetrievalMode.DENSE,
        RetrievalMode.BM25,
        RetrievalMode.HYBRID,
        RetrievalMode.HYBRID_RERANK,
    ]


def test_ablation_configuration_validates_mode_specific_candidate_limit() -> None:
    assert AblationConfig(mode=RetrievalMode.HYBRID_RERANK, top_k=5, candidate_k=10)
    with pytest.raises(ValidationError, match="candidate_k"):
        AblationConfig(mode=RetrievalMode.HYBRID_RERANK, top_k=10, candidate_k=5)
    with pytest.raises(ValidationError, match="only supported"):
        AblationConfig(mode=RetrievalMode.HYBRID, candidate_k=20)
