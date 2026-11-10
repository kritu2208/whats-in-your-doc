"""Offline evaluation schemas, configuration, and pure metric utilities."""

from app.evaluation.config import AblationConfig, DEFAULT_ABLATIONS
from app.evaluation.dataset import (
    DatasetManifest,
    EvaluationDataset,
    EvaluationQuestion,
    load_evaluation_dataset,
)
from app.evaluation.metrics import (
    LatencySummary,
    RetrievalMetricScores,
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
    AnswerQualityEvaluator,
    AnswerQualityInput,
    AnswerQualityOutput,
    QualityMetricResult,
)
from app.evaluation.runner import (
    EvaluationRunReport,
    EvaluationRunner,
    StableReferenceMap,
    validate_reference_mapping,
    write_evaluation_report,
)

__all__ = [
    "AblationConfig",
    "AnswerQualityEvaluator",
    "AnswerQualityInput",
    "AnswerQualityOutput",
    "DEFAULT_ABLATIONS",
    "DatasetManifest",
    "EvaluationDataset",
    "EvaluationQuestion",
    "EvaluationRunReport",
    "EvaluationRunner",
    "LatencySummary",
    "QualityMetricResult",
    "RetrievalMetricScores",
    "StableReferenceMap",
    "latency_summary",
    "load_evaluation_dataset",
    "mean_latency",
    "mean_reciprocal_rank",
    "ndcg_at_10",
    "p50_latency",
    "p95_latency",
    "percentile_latency",
    "recall_at_5",
    "recall_at_10",
    "recall_at_k",
    "reciprocal_rank",
    "retrieval_metric_scores",
    "validate_reference_mapping",
    "write_evaluation_report",
]
