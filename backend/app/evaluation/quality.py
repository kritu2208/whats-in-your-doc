from __future__ import annotations

import json
import logging
import re
from collections.abc import Sequence
from datetime import date, datetime, timezone
from pathlib import Path
from statistics import fmean
from typing import Any, Callable, Literal, Protocol, TypeAlias

from pydantic import Field, ValidationError, model_validator

from app.core.config import get_settings
from app.evaluation.dataset import (
    Answerability,
    ExpectedEvidence,
    StrictModel,
)
from app.rag.workflow import UNANSWERABLE_ANSWER

logger = logging.getLogger(__name__)

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
    citation_accuracy: QualityMetricResult = Field(default_factory=QualityMetricResult)
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


# ---------------------------------------------------------------------------
# Automated LLM Judge, Answer Relevancy, Citation Accuracy & Abstention Models
# ---------------------------------------------------------------------------

def _safe_error_message(error: Exception | str) -> str:
    message = str(error).strip() or "No additional error detail."
    message = re.sub(r"gsk_[A-Za-z0-9_-]+", "[REDACTED]", message)
    message = re.sub(
        r"\bBearer\s+\S+",
        "Bearer [REDACTED]",
        message,
        flags=re.IGNORECASE,
    )
    message = re.sub(
        r"\b(password|passwd|api[_-]?key|token|secret)\s*[:=]\s*(['\"]?)[^\s,;'\"]+",
        r"\1=[REDACTED]",
        message,
        flags=re.IGNORECASE,
    )
    message = re.sub(
        r"\b(?:postgres(?:ql)?|mysql|mssql)(?:\+\w+)?://\S+",
        "[REDACTED_DATABASE_URL]",
        message,
        flags=re.IGNORECASE,
    )
    return message[:500]


def is_abstention_answer(answer: str, unanswerable_flag: bool | None = None) -> bool:
    """Return True if the answer represents a system abstention."""
    if unanswerable_flag is True:
        return True
    normalized = answer.strip().lower()
    if not normalized:
        return False
    if UNANSWERABLE_ANSWER.lower() in normalized:
        return True
    abstention_indicators = [
        "do not provide enough information",
        "does not provide enough information",
        "not enough information to answer",
        "no information provided",
        "cannot answer this question based on the",
        "the provided documents do not contain",
        "the uploaded documents do not contain",
        "information is not mentioned in the",
        "context does not mention",
        "context does not contain",
    ]
    return any(indicator in normalized for indicator in abstention_indicators)


class AbstentionClassification(StrictModel):
    question_id: str = Field(min_length=1)
    ground_truth_answerability: Answerability
    is_abstention: bool
    classification: Literal[
        "true_abstention",
        "false_abstention",
        "true_answer",
        "false_answer",
    ]
    explanation: str


class AbstentionMetricsSummary(StrictModel):
    total_questions: int = Field(ge=0)
    answerable_questions: int = Field(ge=0)
    unanswerable_questions: int = Field(ge=0)
    true_abstentions: int = Field(ge=0)
    false_abstentions: int = Field(ge=0)
    true_answers: int = Field(ge=0)
    false_answers: int = Field(ge=0)
    abstention_precision: float | None = Field(default=None, ge=0.0, le=1.0)
    abstention_recall: float | None = Field(default=None, ge=0.0, le=1.0)
    abstention_f1: float | None = Field(default=None, ge=0.0, le=1.0)
    abstention_accuracy: float = Field(ge=0.0, le=1.0)


def classify_abstention(
    question_id: str,
    ground_truth_answerability: Answerability,
    answer: str,
    unanswerable_flag: bool | None = None,
) -> AbstentionClassification:
    abstained = is_abstention_answer(answer, unanswerable_flag)
    if ground_truth_answerability == "unanswerable":
        if abstained:
            classification = "true_abstention"
            explanation = "Correctly abstained from answering an unanswerable question."
        else:
            classification = "false_answer"
            explanation = "Failed to abstain; generated an unsupported answer to an unanswerable question."
    else:
        if abstained:
            classification = "false_abstention"
            explanation = "Incorrectly abstained from answering an answerable question."
        else:
            classification = "true_answer"
            explanation = "Correctly attempted an answer to an answerable question."
    return AbstentionClassification(
        question_id=question_id,
        ground_truth_answerability=ground_truth_answerability,
        is_abstention=abstained,
        classification=classification,
        explanation=explanation,
    )


def summarize_abstention_metrics(
    classifications: Sequence[AbstentionClassification],
) -> AbstentionMetricsSummary:
    total = len(classifications)
    answerable = sum(
        1 for c in classifications if c.ground_truth_answerability == "answerable"
    )
    unanswerable = sum(
        1 for c in classifications if c.ground_truth_answerability == "unanswerable"
    )
    tp = sum(1 for c in classifications if c.classification == "true_abstention")
    fp = sum(1 for c in classifications if c.classification == "false_abstention")
    tn = sum(1 for c in classifications if c.classification == "true_answer")
    fn = sum(1 for c in classifications if c.classification == "false_answer")

    precision = (tp / (tp + fp)) if (tp + fp) > 0 else None
    recall = (tp / (tp + fn)) if (tp + fn) > 0 else None
    f1 = (
        (2 * precision * recall / (precision + recall))
        if (precision is not None and recall is not None and (precision + recall) > 0)
        else None
    )
    accuracy = ((tp + tn) / total) if total > 0 else 0.0

    return AbstentionMetricsSummary(
        total_questions=total,
        answerable_questions=answerable,
        unanswerable_questions=unanswerable,
        true_abstentions=tp,
        false_abstentions=fp,
        true_answers=tn,
        false_answers=fn,
        abstention_precision=precision,
        abstention_recall=recall,
        abstention_f1=f1,
        abstention_accuracy=accuracy,
    )


class ClaimJudgment(StrictModel):
    claim: str = Field(min_length=1)
    supported: bool
    explanation: str = Field(default="")


class FaithfulnessJudgeOutput(StrictModel):
    claims: list[ClaimJudgment] = Field(default_factory=list)
    reasoning: str = Field(default="")


class RelevancyJudgeOutput(StrictModel):
    score: float = Field(ge=0.0, le=1.0)
    explanation: str = Field(default="")


class CitationSupportJudgment(StrictModel):
    source_ref: str = Field(min_length=1)
    supports_claim: bool
    explanation: str = Field(default="")


class CitationJudgeOutput(StrictModel):
    citations: list[CitationSupportJudgment] = Field(default_factory=list)
    reasoning: str = Field(default="")


JudgeCaller: TypeAlias = Callable[[str, str], str]


def _parse_judge_json(raw: str) -> dict[str, Any]:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
        text = re.sub(r"```\s*$", "", text, flags=re.MULTILINE)
        text = text.strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        text = match.group(0)
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Judge returned non-JSON response: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("Judge output must be a JSON object.")
    return data


class LLMJudge:
    """Invokes an LLM judge provider with strict output parsing."""

    def __init__(
        self,
        *,
        provider: str = "mock",
        model: str = "mock-judge-v1",
        judge_caller: JudgeCaller | None = None,
        timeout_seconds: float = 30.0,
    ) -> None:
        self.provider = provider
        self.model = model
        self.judge_caller = judge_caller
        self.timeout_seconds = timeout_seconds

    def call_judge(self, system_prompt: str, user_prompt: str) -> str:
        if self.judge_caller is not None:
            return self.judge_caller(system_prompt, user_prompt)

        settings = get_settings()
        if self.provider == "ollama" or (
            self.provider == "auto" and settings.llm_provider == "ollama"
        ):
            import urllib.request

            url = f"{settings.ollama_base_url.rstrip('/')}/api/chat"
            payload = json.dumps(
                {
                    "model": (
                        self.model
                        if self.model != "mock-judge-v1"
                        else settings.ollama_model
                    ),
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "stream": False,
                    "format": "json",
                }
            ).encode("utf-8")
            req = urllib.request.Request(
                url,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data.get("message", {}).get("content", "")

        # Default: Groq
        api_key = settings.groq_api_key
        if not api_key:
            raise RuntimeError("GROQ_API_KEY is not set for automated LLM judge.")
        from importlib import import_module

        groq_module = import_module("groq")
        client = groq_module.Groq(api_key=api_key, timeout=self.timeout_seconds)
        model_name = (
            self.model if self.model != "mock-judge-v1" else settings.groq_model
        )
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        return response.choices[0].message.content or ""


class AutomatedAnswerQualityEvaluator:
    """Evaluates generated answer faithfulness, relevancy, and citation accuracy via LLM judge."""

    def __init__(self, judge: LLMJudge) -> None:
        self.judge = judge
        self.evaluator_name = f"llm-judge:{judge.provider}"
        self.evaluator_model = judge.model

    def evaluate_faithfulness(
        self,
        question: str,
        answer: str,
        retrieved_contexts: list[str],
        unanswerable_flag: bool | None = None,
    ) -> QualityMetricResult:
        if is_abstention_answer(answer, unanswerable_flag):
            return QualityMetricResult(
                status="unavailable",
                explanation="Answer is an abstention; no verifiable factual claims were asserted.",
            )
        if not retrieved_contexts or not any(c.strip() for c in retrieved_contexts):
            return QualityMetricResult(
                status="evaluated",
                score=0.0,
                explanation="No retrieved document context was provided; all factual claims are unsupported.",
            )

        system_prompt = (
            "You are an impartial evaluation judge assessing the faithfulness of a generated answer. "
            "Extract all discrete factual claims from the answer. "
            "For each claim, determine whether it is directly entailed and supported by the retrieved document context. "
            "Do not use outside knowledge. "
            "Return a JSON object with this exact shape: "
            '{"claims": [{"claim": "...", "supported": true/false, "explanation": "..."}], "reasoning": "..."}. '
            'If the answer makes no verifiable factual claims, return "claims": [].'
        )
        user_prompt = (
            f"Question:\n{question}\n\n"
            f"Retrieved Document Context:\n{chr(10).join(retrieved_contexts)}\n\n"
            f"Generated Answer:\n{answer}"
        )
        try:
            raw = self.judge.call_judge(system_prompt, user_prompt)
            parsed = _parse_judge_json(raw)
            result = FaithfulnessJudgeOutput.model_validate(parsed)
        except Exception as exc:
            return QualityMetricResult(
                status="unavailable",
                explanation=f"Judge evaluation failed: {_safe_error_message(exc)}",
            )

        if not result.claims:
            return QualityMetricResult(
                status="unavailable",
                explanation="The answer contains no verifiable factual claims.",
            )

        supported_count = sum(1 for c in result.claims if c.supported)
        score = supported_count / len(result.claims)
        return QualityMetricResult(
            status="evaluated",
            score=round(score, 4),
            explanation=(
                f"{supported_count} of {len(result.claims)} factual claims "
                "were supported by retrieved context."
            ),
        )

    def evaluate_answer_relevancy(
        self,
        question: str,
        answer: str,
        ground_truth_answerability: Answerability = "answerable",
        unanswerable_flag: bool | None = None,
    ) -> QualityMetricResult:
        if is_abstention_answer(answer, unanswerable_flag):
            if ground_truth_answerability == "unanswerable":
                return QualityMetricResult(
                    status="evaluated",
                    score=1.0,
                    explanation="Correctly and relevantly abstained on an unanswerable question.",
                )
            else:
                return QualityMetricResult(
                    status="evaluated",
                    score=0.0,
                    explanation="Abstained on an answerable question; failed to address user query.",
                )

        system_prompt = (
            "You are an impartial evaluation judge assessing answer relevancy. "
            "Evaluate whether the generated answer directly, clearly, and concisely addresses the user's question, "
            "without introducing off-topic tangents or failing to answer the inquiry. "
            "Return a JSON object with this exact shape: "
            '{"score": <float between 0.0 and 1.0>, "explanation": "..."}. '
            "1.0 means perfectly relevant and complete; 0.0 means completely irrelevant."
        )
        user_prompt = f"Question:\n{question}\n\nGenerated Answer:\n{answer}"
        try:
            raw = self.judge.call_judge(system_prompt, user_prompt)
            parsed = _parse_judge_json(raw)
            result = RelevancyJudgeOutput.model_validate(parsed)
        except Exception as exc:
            return QualityMetricResult(
                status="unavailable",
                explanation=f"Judge evaluation failed: {_safe_error_message(exc)}",
            )

        return QualityMetricResult(
            status="evaluated",
            score=round(result.score, 4),
            explanation=result.explanation or "Answer relevancy evaluated by judge.",
        )

    def evaluate_citation_accuracy(
        self,
        question: str,
        answer: str,
        cited_evidence: list[ExpectedEvidence],
        retrieved_contexts: list[str],
        unanswerable_flag: bool | None = None,
    ) -> QualityMetricResult:
        if is_abstention_answer(answer, unanswerable_flag):
            return QualityMetricResult(
                status="evaluated",
                score=1.0,
                explanation="No citations required for an abstention.",
            )
        if not cited_evidence:
            return QualityMetricResult(
                status="evaluated",
                score=0.0,
                explanation="Answer makes factual assertions but provides zero citations.",
            )

        citation_refs = [
            f"{item.document_ref}/{item.chunk_ref}" for item in cited_evidence
        ]
        system_prompt = (
            "You are an impartial evaluation judge assessing citation accuracy. "
            "Determine whether each cited source reference actually supports the statements made in the answer. "
            "Return a JSON object with this exact shape: "
            '{"citations": [{"source_ref": "...", "supports_claim": true/false, "explanation": "..."}], "reasoning": "..."}.'
        )
        user_prompt = (
            f"Question:\n{question}\n\n"
            f"Retrieved Document Context:\n{chr(10).join(retrieved_contexts)}\n\n"
            f"Generated Answer:\n{answer}\n\n"
            f"Cited Source References:\n{', '.join(citation_refs)}"
        )
        try:
            raw = self.judge.call_judge(system_prompt, user_prompt)
            parsed = _parse_judge_json(raw)
            result = CitationJudgeOutput.model_validate(parsed)
        except Exception as exc:
            return QualityMetricResult(
                status="unavailable",
                explanation=f"Judge evaluation failed: {_safe_error_message(exc)}",
            )

        if not result.citations:
            return QualityMetricResult(
                status="unavailable",
                explanation="Judge did not return citation assessments.",
            )

        supported_count = sum(1 for c in result.citations if c.supports_claim)
        score = supported_count / len(result.citations)
        return QualityMetricResult(
            status="evaluated",
            score=round(score, 4),
            explanation=(
                f"{supported_count} of {len(result.citations)} cited sources "
                "accurately support the statements made."
            ),
        )

    def evaluate(self, request: AnswerQualityInput) -> AnswerQualityOutput:
        faithfulness = self.evaluate_faithfulness(
            question=request.question,
            answer=request.answer,
            retrieved_contexts=request.retrieved_contexts,
        )
        relevancy = self.evaluate_answer_relevancy(
            question=request.question,
            answer=request.answer,
            ground_truth_answerability=request.answerability,
        )
        citation = self.evaluate_citation_accuracy(
            question=request.question,
            answer=request.answer,
            cited_evidence=request.cited_evidence,
            retrieved_contexts=request.retrieved_contexts,
        )
        return AnswerQualityOutput(
            question_id=request.question_id,
            faithfulness=faithfulness,
            answer_relevancy=relevancy,
            citation_accuracy=citation,
            evaluator_name=self.evaluator_name,
            evaluator_model=self.evaluator_model,
        )


class QualityEvaluationItem(StrictModel):
    question_id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    answer: str = Field(min_length=1)
    ground_truth_answerability: Answerability
    retrieved_contexts: list[str] = Field(default_factory=list)
    cited_evidence: list[ExpectedEvidence] = Field(default_factory=list)
    unanswerable_flag: bool | None = None


class AutomatedQualityQuestionResult(StrictModel):
    question_id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    answer: str = Field(min_length=1)
    ground_truth_answerability: Answerability
    faithfulness: QualityMetricResult
    answer_relevancy: QualityMetricResult
    citation_accuracy: QualityMetricResult
    abstention: AbstentionClassification
    evaluator_name: str
    evaluator_model: str


class QualityEvaluationReport(StrictModel):
    report_id: str = Field(min_length=1)
    generated_at: datetime
    dataset_id: str = Field(min_length=1)
    dataset_version: str = Field(min_length=1)
    evaluator_name: str = Field(min_length=1)
    evaluator_model: str = Field(min_length=1)
    total_questions: int = Field(ge=0)
    mean_faithfulness: float | None = Field(default=None, ge=0.0, le=1.0)
    mean_answer_relevancy: float | None = Field(default=None, ge=0.0, le=1.0)
    mean_citation_accuracy: float | None = Field(default=None, ge=0.0, le=1.0)
    abstention_summary: AbstentionMetricsSummary
    results: list[AutomatedQualityQuestionResult]
    limitations: list[str] = Field(default_factory=list)


def run_automated_quality_evaluation(
    *,
    evaluator: AutomatedAnswerQualityEvaluator,
    items: Sequence[QualityEvaluationItem],
    dataset_id: str,
    dataset_version: str,
    limitations: list[str] | None = None,
) -> QualityEvaluationReport:
    """Execute automated quality evaluation across items, producing aggregated results."""
    if not items:
        raise ValueError("At least one evaluation item is required.")
    question_ids = [item.question_id for item in items]
    if len(question_ids) != len(set(question_ids)):
        raise ValueError("Evaluation question IDs must be unique.")

    results: list[AutomatedQualityQuestionResult] = []
    classifications: list[AbstentionClassification] = []

    for item in items:
        faithfulness = evaluator.evaluate_faithfulness(
            question=item.question,
            answer=item.answer,
            retrieved_contexts=item.retrieved_contexts,
            unanswerable_flag=item.unanswerable_flag,
        )
        relevancy = evaluator.evaluate_answer_relevancy(
            question=item.question,
            answer=item.answer,
            ground_truth_answerability=item.ground_truth_answerability,
            unanswerable_flag=item.unanswerable_flag,
        )
        citation = evaluator.evaluate_citation_accuracy(
            question=item.question,
            answer=item.answer,
            cited_evidence=item.cited_evidence,
            retrieved_contexts=item.retrieved_contexts,
            unanswerable_flag=item.unanswerable_flag,
        )
        abstention = classify_abstention(
            question_id=item.question_id,
            ground_truth_answerability=item.ground_truth_answerability,
            answer=item.answer,
            unanswerable_flag=item.unanswerable_flag,
        )
        classifications.append(abstention)
        results.append(
            AutomatedQualityQuestionResult(
                question_id=item.question_id,
                question=item.question,
                answer=item.answer,
                ground_truth_answerability=item.ground_truth_answerability,
                faithfulness=faithfulness,
                answer_relevancy=relevancy,
                citation_accuracy=citation,
                abstention=abstention,
                evaluator_name=evaluator.evaluator_name,
                evaluator_model=evaluator.evaluator_model,
            )
        )

    abstention_summary = summarize_abstention_metrics(classifications)

    faithfulness_scores = [
        r.faithfulness.score
        for r in results
        if r.faithfulness.status == "evaluated" and r.faithfulness.score is not None
    ]
    relevancy_scores = [
        r.answer_relevancy.score
        for r in results
        if r.answer_relevancy.status == "evaluated"
        and r.answer_relevancy.score is not None
    ]
    citation_scores = [
        r.citation_accuracy.score
        for r in results
        if r.citation_accuracy.status == "evaluated"
        and r.citation_accuracy.score is not None
    ]

    now = datetime.now(timezone.utc)
    report_id = now.strftime("%Y%m%dT%H%M%S%fZ")

    default_limitations = [
        "Evaluations performed on synthetic development starter dataset.",
        "Automated LLM judges reflect heuristic alignment and model-specific reasoning.",
        "Human review remains the authoritative verification for real-world deployments.",
    ]

    return QualityEvaluationReport(
        report_id=report_id,
        generated_at=now,
        dataset_id=dataset_id,
        dataset_version=dataset_version,
        evaluator_name=evaluator.evaluator_name,
        evaluator_model=evaluator.evaluator_model,
        total_questions=len(results),
        mean_faithfulness=(
            fmean(faithfulness_scores) if faithfulness_scores else None
        ),
        mean_answer_relevancy=(
            fmean(relevancy_scores) if relevancy_scores else None
        ),
        mean_citation_accuracy=(
            fmean(citation_scores) if citation_scores else None
        ),
        abstention_summary=abstention_summary,
        results=results,
        limitations=limitations if limitations is not None else default_limitations,
    )


def generate_quality_markdown_summary(report: QualityEvaluationReport) -> str:
    """Format an automated quality evaluation report into clean GitHub Flavored Markdown."""
    lines = [
        f"# Answer Quality & Abstention Evaluation Report: {report.report_id}",
        "",
        "## 1. Overview",
        "",
        f"- **Generated At:** {report.generated_at.isoformat()}",
        f"- **Dataset ID:** `{report.dataset_id}` (v`{report.dataset_version}`)",
        f"- **Evaluator:** `{report.evaluator_name}`",
        f"- **Judge Model:** `{report.evaluator_model}`",
        f"- **Total Questions Evaluated:** {report.total_questions}",
        "",
        "## 2. Aggregate Quality Metrics",
        "",
        "| Metric | Mean Score | Status | Description |",
        "|---|---:|---|---|",
        (
            f"| **Faithfulness** | "
            f"{f'{report.mean_faithfulness:.4f}' if report.mean_faithfulness is not None else 'N/A'} | "
            f"{'Evaluated' if report.mean_faithfulness is not None else 'Unavailable'} | "
            "Fraction of factual claims supported by retrieved context |"
        ),
        (
            f"| **Answer Relevancy** | "
            f"{f'{report.mean_answer_relevancy:.4f}' if report.mean_answer_relevancy is not None else 'N/A'} | "
            f"{'Evaluated' if report.mean_answer_relevancy is not None else 'Unavailable'} | "
            "Semantic alignment with the user question |"
        ),
        (
            f"| **Citation Accuracy** | "
            f"{f'{report.mean_citation_accuracy:.4f}' if report.mean_citation_accuracy is not None else 'N/A'} | "
            f"{'Evaluated' if report.mean_citation_accuracy is not None else 'Unavailable'} | "
            "Fraction of citations providing genuine claim support |"
        ),
        "",
        "## 3. Abstention Evaluation Summary",
        "",
        "| Metric | Value | Meaning |",
        "|---|---:|---|",
        (
            f"| **Total Questions** | {report.abstention_summary.total_questions} | "
            f"Answerable ({report.abstention_summary.answerable_questions}), "
            f"Unanswerable ({report.abstention_summary.unanswerable_questions}) |"
        ),
        (
            f"| **Abstention Accuracy** | "
            f"{report.abstention_summary.abstention_accuracy:.4f} | "
            "Correct decisions (TP + TN) / total questions |"
        ),
        (
            f"| **Abstention Precision** | "
            f"{f'{report.abstention_summary.abstention_precision:.4f}' if report.abstention_summary.abstention_precision is not None else 'N/A'} | "
            "True abstentions / total abstentions |"
        ),
        (
            f"| **Abstention Recall** | "
            f"{f'{report.abstention_summary.abstention_recall:.4f}' if report.abstention_summary.abstention_recall is not None else 'N/A'} | "
            "Unanswerable questions correctly abstained |"
        ),
        (
            f"| **Abstention F1** | "
            f"{f'{report.abstention_summary.abstention_f1:.4f}' if report.abstention_summary.abstention_f1 is not None else 'N/A'} | "
            "Harmonic mean of precision and recall |"
        ),
        "",
        "### Confusion Matrix Breakdown",
        "",
        f"- **True Abstentions (TP):** {report.abstention_summary.true_abstentions} (correctly refused to answer unanswerable)",
        f"- **False Abstentions (FP):** {report.abstention_summary.false_abstentions} (incorrectly refused to answer answerable)",
        f"- **True Answers (TN):** {report.abstention_summary.true_answers} (correctly answered answerable)",
        f"- **False Answers (FN):** {report.abstention_summary.false_answers} (hallucinated / answered unanswerable)",
        "",
        "## 4. Limitations & Notes",
        "",
    ]
    for limitation in report.limitations:
        lines.append(f"- {limitation}")
    lines.append("")
    return "\n".join(lines)


def write_quality_evaluation_report(
    report: QualityEvaluationReport,
    destination_dir: Path,
) -> tuple[Path, Path]:
    """Write both JSON machine-readable data and a Markdown summary report."""
    destination_dir.mkdir(parents=True, exist_ok=True)
    json_path = destination_dir / f"quality-evaluation-{report.report_id}.json"
    md_path = destination_dir / f"quality-evaluation-{report.report_id}.md"

    if json_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing report: {json_path}")
    if md_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing summary: {md_path}")

    json_path.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    md_path.write_text(generate_quality_markdown_summary(report) + "\n", encoding="utf-8")
    return json_path, md_path
