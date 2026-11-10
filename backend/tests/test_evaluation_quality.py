from __future__ import annotations

from datetime import date

import pytest
from pydantic import ValidationError

from app.evaluation.dataset import ExpectedEvidence
from app.evaluation.quality import (
    AnswerQualityInput,
    ClaimFaithfulnessAssessment,
    FaithfulnessEvaluationRecord,
    FaithfulnessReview,
    evaluate_faithfulness,
    load_faithfulness_records,
    run_faithfulness_evaluation,
    write_faithfulness_report,
)

SOURCE = ExpectedEvidence(document_ref="DOC-A", chunk_ref="c01")


def make_request(
    *,
    question_id: str = "q001",
    retrieved: list[ExpectedEvidence] | None = None,
    cited: list[ExpectedEvidence] | None = None,
) -> AnswerQualityInput:
    return AnswerQualityInput(
        question_id=question_id,
        question="What does the document state?",
        answer="The document states a fact.",
        answerability="answerable",
        retrieved_contexts=["The document states a fact."],
        retrieved_evidence=retrieved if retrieved is not None else [SOURCE],
        cited_evidence=cited if cited is not None else [SOURCE],
    )


def make_review(
    claims: list[ClaimFaithfulnessAssessment],
    *,
    complete: bool = True,
) -> FaithfulnessReview:
    return FaithfulnessReview(
        reviewer="reviewer-1",
        reviewed_at=date(2025, 1, 2),
        all_factual_claims_assessed=complete,
        claims=claims,
    )


def make_record(
    question_id: str,
    claims: list[ClaimFaithfulnessAssessment],
    *,
    complete: bool = True,
) -> FaithfulnessEvaluationRecord:
    return FaithfulnessEvaluationRecord(
        request=make_request(question_id=question_id),
        review=make_review(claims, complete=complete),
    )


def test_faithfulness_scores_supported_claims_and_checks_citations() -> None:
    review = make_review(
        [
            ClaimFaithfulnessAssessment(
                claim="Supported and cited.",
                supported=True,
                supporting_evidence=[SOURCE],
                cited_evidence=[SOURCE],
            ),
            ClaimFaithfulnessAssessment(
                claim="Unsupported claim.",
                supported=False,
            ),
            ClaimFaithfulnessAssessment(
                claim="Supported but uncited.",
                supported=True,
                supporting_evidence=[SOURCE],
            ),
        ]
    )

    output = evaluate_faithfulness(make_request(cited=[SOURCE]), review)

    assert output.faithfulness.status == "evaluated"
    assert output.faithfulness.score == pytest.approx(2 / 3)
    assert output.evaluator_name == "manual-claim-review"
    assert output.answer_relevancy.status == "not_evaluated"


@pytest.mark.parametrize(
    ("claims", "complete", "explanation"),
    [
        (
            [ClaimFaithfulnessAssessment(claim="One claim.", supported=False)],
            False,
            "all factual claims",
        ),
        ([], True, "no assessed factual claims"),
    ],
)
def test_incomplete_or_empty_review_is_unavailable(
    claims: list[ClaimFaithfulnessAssessment],
    complete: bool,
    explanation: str,
) -> None:
    output = evaluate_faithfulness(
        make_request(),
        make_review(claims, complete=complete),
    )

    assert output.faithfulness.status == "unavailable"
    assert output.faithfulness.score is None
    assert explanation in (output.faithfulness.explanation or "")


def test_supported_claim_must_reference_retrieved_evidence() -> None:
    other_source = ExpectedEvidence(document_ref="DOC-B", chunk_ref="c02")
    review = make_review(
        [
            ClaimFaithfulnessAssessment(
                claim="A claim.",
                supported=True,
                supporting_evidence=[other_source],
            )
        ]
    )

    with pytest.raises(ValueError, match="not in the retrieved context"):
        evaluate_faithfulness(make_request(), review)


def test_claim_citations_must_be_present_in_the_answer_citations() -> None:
    review = make_review(
        [
            ClaimFaithfulnessAssessment(
                claim="A claim.",
                supported=True,
                supporting_evidence=[SOURCE],
                cited_evidence=[SOURCE],
            )
        ]
    )

    with pytest.raises(ValueError, match="not present in the answer citations"):
        evaluate_faithfulness(make_request(cited=[]), review)


def test_citations_must_be_in_the_retrieved_evidence() -> None:
    with pytest.raises(ValidationError, match="part of the retrieved evidence"):
        make_request(cited=[ExpectedEvidence(document_ref="DOC-B", chunk_ref="c02")])


def test_faithfulness_report_averages_only_evaluated_reviews() -> None:
    claims = [
        ClaimFaithfulnessAssessment(
            claim="Supported claim.",
            supported=True,
            supporting_evidence=[SOURCE],
            cited_evidence=[SOURCE],
        )
    ]
    report = run_faithfulness_evaluation(
        dataset_id="starter",
        dataset_version="1",
        records=[
            make_record("q001", claims),
            make_record("q002", [], complete=False),
        ],
    )

    assert report.mean_faithfulness == 1.0
    assert report.evaluated_questions == 1
    assert report.unavailable_questions == 1
    assert [result.status for result in report.results] == ["evaluated", "unavailable"]


def test_faithfulness_report_rejects_empty_or_duplicate_question_ids() -> None:
    with pytest.raises(ValueError, match="At least one"):
        run_faithfulness_evaluation(
            dataset_id="starter",
            dataset_version="1",
            records=[],
        )

    duplicate = make_record("q001", [])
    with pytest.raises(ValueError, match="must be unique"):
        run_faithfulness_evaluation(
            dataset_id="starter",
            dataset_version="1",
            records=[duplicate, duplicate],
        )


def test_load_and_write_faithfulness_report(tmp_path) -> None:
    record = make_record(
        "q001",
        [
            ClaimFaithfulnessAssessment(
                claim="Supported claim.",
                supported=True,
                supporting_evidence=[SOURCE],
                cited_evidence=[SOURCE],
            )
        ],
    )
    input_path = tmp_path / "reviews.jsonl"
    input_path.write_text(record.model_dump_json() + "\n", encoding="utf-8")

    loaded = load_faithfulness_records(input_path)
    report = run_faithfulness_evaluation(
        dataset_id="starter",
        dataset_version="1",
        records=loaded,
    )
    destination = tmp_path / "reports" / "faithfulness.json"

    assert write_faithfulness_report(report, destination) == destination
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        write_faithfulness_report(report, destination)


def test_load_faithfulness_records_reports_invalid_line(tmp_path) -> None:
    input_path = tmp_path / "reviews.jsonl"
    input_path.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="line 1"):
        load_faithfulness_records(input_path)
