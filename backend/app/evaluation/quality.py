from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import date, datetime, timezone
from pathlib import Path
from statistics import fmean
from typing import Literal, Protocol

from pydantic import Field, ValidationError, model_validator

from app.evaluation.dataset import (
    Answerability,
    ExpectedEvidence,
    StrictModel,
)

QualityStatus = Literal["not_evaluated", "unavailable", "evaluated"]


class AnswerQualityInput(StrictModel):
    question_id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    answer: str = Field(min_length=1)
    answerability: Answerability
    retrieved_contexts: list[str]
    retrieved_evidence: list[ExpectedEvidence] = Field(default_factory=list)
    cited_evidence: list[ExpectedEvidence] = Field(default_factory=list)

    @model_validator(mode="after")
    def citations_are_retrieved(self) -> AnswerQualityInput:
        retrieved = {
            (item.document_ref, item.chunk_ref) for item in self.retrieved_evidence
        }
        cited = {(item.document_ref, item.chunk_ref) for item in self.cited_evidence}
        if not cited.issubset(retrieved):
            raise ValueError("Cited evidence must be part of the retrieved evidence.")
        return self


class QualityMetricResult(StrictModel):
    status: QualityStatus = "not_evaluated"
    score: float | None = Field(default=None, ge=0.0, le=1.0)
    explanation: str | None = None

    @model_validator(mode="after")
    def score_matches_status(self) -> QualityMetricResult:
        if self.status == "evaluated" and self.score is None:
            raise ValueError("An evaluated quality metric requires a score.")
        if self.status != "evaluated" and self.score is not None:
            raise ValueError("Unevaluated quality metrics must not include a score.")
        return self


class AnswerQualityOutput(StrictModel):
    question_id: str = Field(min_length=1)
    faithfulness: QualityMetricResult = Field(default_factory=QualityMetricResult)
    answer_relevancy: QualityMetricResult = Field(default_factory=QualityMetricResult)
    evaluator_name: str | None = None
    evaluator_model: str | None = None


class AnswerQualityEvaluator(Protocol):
    """Interface for answer-quality evaluators."""

    def evaluate(self, request: AnswerQualityInput) -> AnswerQualityOutput: ...


class ClaimFaithfulnessAssessment(StrictModel):
    claim: str = Field(min_length=1)
    supported: bool
    supporting_evidence: list[ExpectedEvidence] = Field(default_factory=list)
    cited_evidence: list[ExpectedEvidence] = Field(default_factory=list)

    @model_validator(mode="after")
    def support_has_evidence(self) -> ClaimFaithfulnessAssessment:
        if self.supported and not self.supporting_evidence:
            raise ValueError("Supported claims require at least one supporting source.")
        support_refs = {
            (item.document_ref, item.chunk_ref) for item in self.supporting_evidence
        }
        cited_refs = {(item.document_ref, item.chunk_ref) for item in self.cited_evidence}
        if not self.supported and (support_refs or cited_refs):
            raise ValueError("Unsupported claims cannot list supporting or cited sources.")
        if not cited_refs.issubset(support_refs):
            raise ValueError("Claim citations must be among its supporting sources.")
        return self


class FaithfulnessReview(StrictModel):
    reviewer: str = Field(min_length=1)
    reviewed_at: date
    all_factual_claims_assessed: bool
    claims: list[ClaimFaithfulnessAssessment] = Field(default_factory=list)


class FaithfulnessEvaluationRecord(StrictModel):
    request: AnswerQualityInput
    review: FaithfulnessReview


class FaithfulnessQuestionResult(StrictModel):
    question_id: str = Field(min_length=1)
    reviewer: str = Field(min_length=1)
    reviewed_at: date
    status: Literal["unavailable", "evaluated"]
    score: float | None = Field(default=None, ge=0.0, le=1.0)
    factual_claim_count: int = Field(ge=0)
    supported_claim_count: int = Field(ge=0)
    unsupported_claim_count: int = Field(ge=0)
    explanation: str

    @model_validator(mode="after")
    def score_matches_status(self) -> FaithfulnessQuestionResult:
        if self.status == "evaluated" and self.score is None:
            raise ValueError("An evaluated faithfulness result requires a score.")
        if self.status == "unavailable" and self.score is not None:
            raise ValueError("Unavailable faithfulness results must not include a score.")
        if (
            self.supported_claim_count + self.unsupported_claim_count
            != self.factual_claim_count
        ):
            raise ValueError("Faithfulness claim counts must add up.")
        return self


class FaithfulnessEvaluationReport(StrictModel):
    report_id: str = Field(min_length=1)
    generated_at: datetime
    dataset_id: str = Field(min_length=1)
    dataset_version: str = Field(min_length=1)
    mean_faithfulness: float | None = Field(default=None, ge=0.0, le=1.0)
    evaluated_questions: int = Field(ge=0)
    unavailable_questions: int = Field(ge=0)
    results: list[FaithfulnessQuestionResult]


def evaluate_faithfulness(
    request: AnswerQualityInput,
    review: FaithfulnessReview,
) -> AnswerQualityOutput:
    """Score supported factual claims using a completed human review."""
    retrieved = {
        (item.document_ref, item.chunk_ref) for item in request.retrieved_evidence
    }
    cited = {(item.document_ref, item.chunk_ref) for item in request.cited_evidence}

    supported_count = 0
    for assessment in review.claims:
        if assessment.supported:
            support_refs = {
                (item.document_ref, item.chunk_ref)
                for item in assessment.supporting_evidence
            }
            if not support_refs.issubset(retrieved):
                raise ValueError(
                    f"Supporting evidence for claim {assessment.claim!r} "
                    "was not in the retrieved context."
                )
            claim_citations = {
                (item.document_ref, item.chunk_ref)
                for item in assessment.cited_evidence
            }
            if not claim_citations.issubset(cited):
                raise ValueError(
                    f"Citations for claim {assessment.claim!r} "
                    "were not present in the answer citations."
                )
            supported_count += 1

    if not review.all_factual_claims_assessed:
        result = QualityMetricResult(
            status="unavailable",
            explanation="Review did not confirm that all factual claims were assessed.",
        )
    elif not review.claims:
        result = QualityMetricResult(
            status="unavailable",
            explanation="The answer contains no assessed factual claims.",
        )
    else:
        result = QualityMetricResult(
            status="evaluated",
            score=supported_count / len(review.claims),
            explanation=(
                f"{supported_count} of {len(review.claims)} factual claims "
                "were supported by retrieved context."
            ),
        )

    return AnswerQualityOutput(
        question_id=request.question_id,
        faithfulness=result,
        evaluator_name="manual-claim-review",
    )


def run_faithfulness_evaluation(
    *,
    dataset_id: str,
    dataset_version: str,
    records: Sequence[FaithfulnessEvaluationRecord],
) -> FaithfulnessEvaluationReport:
    if not records:
        raise ValueError("At least one faithfulness review record is required.")
    question_ids = [record.request.question_id for record in records]
    if len(question_ids) != len(set(question_ids)):
        raise ValueError("Faithfulness review question IDs must be unique.")

    results = []
    for record in records:
        output = evaluate_faithfulness(record.request, record.review)
        metric = output.faithfulness
        results.append(
            FaithfulnessQuestionResult(
                question_id=record.request.question_id,
                reviewer=record.review.reviewer,
                reviewed_at=record.review.reviewed_at,
                status=metric.status,
                score=metric.score,
                factual_claim_count=len(record.review.claims),
                supported_claim_count=sum(
                    claim.supported for claim in record.review.claims
                ),
                unsupported_claim_count=sum(
                    not claim.supported for claim in record.review.claims
                ),
                explanation=metric.explanation or "Faithfulness was not evaluated.",
            )
        )

    evaluated_scores = [
        result.score for result in results if result.status == "evaluated"
    ]
    now = datetime.now(timezone.utc)
    report_id = now.strftime("%Y%m%dT%H%M%S%fZ")
    return FaithfulnessEvaluationReport(
        report_id=report_id,
        generated_at=now,
        dataset_id=dataset_id,
        dataset_version=dataset_version,
        mean_faithfulness=(
            fmean(score for score in evaluated_scores if score is not None)
            if evaluated_scores
            else None
        ),
        evaluated_questions=len(evaluated_scores),
        unavailable_questions=len(results) - len(evaluated_scores),
        results=results,
    )


def load_faithfulness_records(path: Path) -> list[FaithfulnessEvaluationRecord]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise ValueError(f"Could not read faithfulness reviews: {path}") from exc

    records: list[FaithfulnessEvaluationRecord] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            records.append(
                FaithfulnessEvaluationRecord.model_validate(json.loads(line))
            )
        except (json.JSONDecodeError, ValidationError) as exc:
            raise ValueError(
                f"Invalid faithfulness review on line {line_number}: {exc}"
            ) from exc
    if not records:
        raise ValueError(f"No faithfulness review records found in {path}.")
    return records


def write_faithfulness_report(
    report: FaithfulnessEvaluationReport,
    destination: Path,
) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with destination.open("x", encoding="utf-8", newline="\n") as file:
            file.write(report.model_dump_json(indent=2))
            file.write("\n")
    except FileExistsError as exc:
        raise FileExistsError(
            f"Refusing to overwrite faithfulness report: {destination}"
        ) from exc
    return destination
