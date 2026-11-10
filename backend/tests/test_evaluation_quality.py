from __future__ import annotations

from datetime import date

import pytest
from pydantic import ValidationError

from app.evaluation.dataset import ExpectedEvidence
from app.evaluation.quality import (
    AbstentionClassification,
    AbstentionMetricsSummary,
    AnswerQualityInput,
    AnswerQualityOutput,
    AutomatedAnswerQualityEvaluator,
    AutomatedQualityQuestionResult,
    ClaimFaithfulnessAssessment,
    FaithfulnessEvaluationRecord,
    FaithfulnessReview,
    LLMJudge,
    QualityEvaluationItem,
    QualityEvaluationReport,
    classify_abstention,
    evaluate_faithfulness,
    generate_quality_markdown_summary,
    is_abstention_answer,
    load_faithfulness_records,
    run_automated_quality_evaluation,
    run_faithfulness_evaluation,
    summarize_abstention_metrics,
    write_faithfulness_report,
    write_quality_evaluation_report,
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


# ---------------------------------------------------------------------------
# Automated LLM Judge, Relevancy, Citation Accuracy & Abstention Tests
# ---------------------------------------------------------------------------

def test_is_abstention_answer_handles_standard_and_variations() -> None:
    assert is_abstention_answer("The uploaded documents do not provide enough information to answer this question.")
    assert is_abstention_answer("The provided documents do not contain details regarding the budget.")
    assert is_abstention_answer("There is not enough information to answer this question.")
    assert is_abstention_answer("Anything", unanswerable_flag=True)
    assert not is_abstention_answer("The service desk is open from 08:00 to 18:00 UTC.", unanswerable_flag=False)
    assert not is_abstention_answer("")


def test_classify_abstention_and_summary_metrics() -> None:
    c1 = classify_abstention("q001", "unanswerable", "The uploaded documents do not provide enough information to answer this question.")
    assert c1.classification == "true_abstention"
    assert c1.is_abstention is True

    c2 = classify_abstention("q002", "unanswerable", "The CEO is Jane Doe.")
    assert c2.classification == "false_answer"
    assert c2.is_abstention is False

    c3 = classify_abstention("q003", "answerable", "The desk is open 08:00 to 18:00 UTC.")
    assert c3.classification == "true_answer"
    assert c3.is_abstention is False

    c4 = classify_abstention("q004", "answerable", "I cannot answer this question based on the documents.")
    assert c4.classification == "false_abstention"
    assert c4.is_abstention is True

    summary = summarize_abstention_metrics([c1, c2, c3, c4])
    assert summary.total_questions == 4
    assert summary.answerable_questions == 2
    assert summary.unanswerable_questions == 2
    assert summary.true_abstentions == 1
    assert summary.false_abstentions == 1
    assert summary.true_answers == 1
    assert summary.false_answers == 1
    assert summary.abstention_precision == pytest.approx(1 / (1 + 1))  # TP / (TP + FP) = 0.5
    assert summary.abstention_recall == pytest.approx(1 / (1 + 1))     # TP / (TP + FN) = 0.5
    assert summary.abstention_f1 == pytest.approx(0.5)
    assert summary.abstention_accuracy == pytest.approx((1 + 1) / 4)   # (TP + TN) / total = 0.5


def test_automated_faithfulness_evaluation_with_supported_and_unsupported_claims() -> None:
    import json

    def mock_caller(system_prompt: str, user_prompt: str) -> str:
        return json.dumps(
            {
                "claims": [
                    {"claim": "Claim 1", "supported": True, "explanation": "In context"},
                    {"claim": "Claim 2", "supported": False, "explanation": "Not in context"},
                    {"claim": "Claim 3", "supported": True, "explanation": "In context"},
                ],
                "reasoning": "2 of 3 supported.",
            }
        )

    judge = LLMJudge(provider="mock", model="test-judge", judge_caller=mock_caller)
    evaluator = AutomatedAnswerQualityEvaluator(judge)

    res = evaluator.evaluate_faithfulness(
        question="What are the hours?",
        answer="Hours are 08:00 to 18:00, and tickets close in 2 days, with 3 followups.",
        retrieved_contexts=["Staffed 08:00 to 18:00 UTC. 3 followups allowed."],
    )

    assert res.status == "evaluated"
    assert res.score == pytest.approx(2 / 3, abs=0.001)
    assert "2 of 3 factual claims were supported" in (res.explanation or "")


def test_automated_faithfulness_empty_claims_and_abstention() -> None:
    import json

    judge = LLMJudge(
        provider="mock",
        model="test-judge",
        judge_caller=lambda s, u: json.dumps({"claims": [], "reasoning": "no factual claims"}),
    )
    evaluator = AutomatedAnswerQualityEvaluator(judge)

    # When abstaining
    abstain_res = evaluator.evaluate_faithfulness(
        question="What is the CEO name?",
        answer="The uploaded documents do not provide enough information to answer this question.",
        retrieved_contexts=["Some text"],
    )
    assert abstain_res.status == "unavailable"
    assert abstain_res.score is None
    assert "abstention" in (abstain_res.explanation or "").lower()

    # When answer has no factual claims
    no_claims_res = evaluator.evaluate_faithfulness(
        question="Hello?",
        answer="I am here to help.",
        retrieved_contexts=["Some context"],
    )
    assert no_claims_res.status == "unavailable"
    assert no_claims_res.score is None
    assert "no verifiable factual claims" in (no_claims_res.explanation or "")


def test_automated_faithfulness_empty_context_scores_zero() -> None:
    judge = LLMJudge(provider="mock", model="test-judge", judge_caller=lambda s, u: "{}")
    evaluator = AutomatedAnswerQualityEvaluator(judge)

    res = evaluator.evaluate_faithfulness(
        question="What are the hours?",
        answer="Hours are 08:00 to 18:00 UTC.",
        retrieved_contexts=[],
    )
    assert res.status == "evaluated"
    assert res.score == 0.0
    assert "no retrieved document context" in (res.explanation or "").lower()


def test_automated_relevancy_evaluation() -> None:
    import json

    judge = LLMJudge(
        provider="mock",
        model="test-judge",
        judge_caller=lambda s, u: json.dumps({"score": 0.95, "explanation": "Highly responsive."}),
    )
    evaluator = AutomatedAnswerQualityEvaluator(judge)

    res = evaluator.evaluate_answer_relevancy(
        question="What are the hours?",
        answer="08:00 to 18:00 UTC weekdays.",
        ground_truth_answerability="answerable",
    )
    assert res.status == "evaluated"
    assert res.score == 0.95
    assert "Highly responsive." in (res.explanation or "")


def test_automated_relevancy_on_abstention() -> None:
    judge = LLMJudge(provider="mock", model="test-judge", judge_caller=lambda s, u: "{}")
    evaluator = AutomatedAnswerQualityEvaluator(judge)

    # Correct abstention on unanswerable question
    res_unans = evaluator.evaluate_answer_relevancy(
        question="Who is CEO?",
        answer="The uploaded documents do not provide enough information to answer this question.",
        ground_truth_answerability="unanswerable",
    )
    assert res_unans.status == "evaluated"
    assert res_unans.score == 1.0

    # Incorrect abstention on answerable question
    res_ans = evaluator.evaluate_answer_relevancy(
        question="What are the hours?",
        answer="The uploaded documents do not provide enough information to answer this question.",
        ground_truth_answerability="answerable",
    )
    assert res_ans.status == "evaluated"
    assert res_ans.score == 0.0


def test_automated_citation_accuracy_evaluation() -> None:
    import json

    def mock_caller(system_prompt: str, user_prompt: str) -> str:
        return json.dumps(
            {
                "citations": [
                    {"source_ref": "DOC-A/c01", "supports_claim": True, "explanation": "Direct match"},
                    {"source_ref": "DOC-A/c02", "supports_claim": False, "explanation": "Wrong page/topic"},
                ],
                "reasoning": "1 of 2 supported",
            }
        )

    judge = LLMJudge(provider="mock", model="test-judge", judge_caller=mock_caller)
    evaluator = AutomatedAnswerQualityEvaluator(judge)

    ev1 = ExpectedEvidence(document_ref="DOC-A", chunk_ref="c01")
    ev2 = ExpectedEvidence(document_ref="DOC-A", chunk_ref="c02")

    res = evaluator.evaluate_citation_accuracy(
        question="What is the policy?",
        answer="Policy statement citing both sources.",
        cited_evidence=[ev1, ev2],
        retrieved_contexts=["Text of chunk 1", "Text of chunk 2"],
    )
    assert res.status == "evaluated"
    assert res.score == 0.5
    assert "1 of 2 cited sources accurately support" in (res.explanation or "")


def test_automated_citation_accuracy_on_unsupported_or_missing_citations() -> None:
    judge = LLMJudge(provider="mock", model="test-judge", judge_caller=lambda s, u: "{}")
    evaluator = AutomatedAnswerQualityEvaluator(judge)

    # Factual answer with zero citations scores 0.0
    res_no_cites = evaluator.evaluate_citation_accuracy(
        question="What is the policy?",
        answer="The policy states that tickets must be answered in 4 hours.",
        cited_evidence=[],
        retrieved_contexts=["Some context"],
    )
    assert res_no_cites.status == "evaluated"
    assert res_no_cites.score == 0.0
    assert "zero citations" in (res_no_cites.explanation or "")

    # Abstention requires no citations
    res_abstain = evaluator.evaluate_citation_accuracy(
        question="Who is the CEO?",
        answer="The uploaded documents do not provide enough information to answer this question.",
        cited_evidence=[],
        retrieved_contexts=[],
    )
    assert res_abstain.status == "evaluated"
    assert res_abstain.score == 1.0


def test_llm_judge_malformed_json_surfaces_unavailable_status() -> None:
    judge = LLMJudge(
        provider="mock",
        model="test-judge",
        judge_caller=lambda s, u: "This is raw non-JSON text from the model.",
    )
    evaluator = AutomatedAnswerQualityEvaluator(judge)

    res = evaluator.evaluate_faithfulness(
        question="Question",
        answer="Answer with facts.",
        retrieved_contexts=["Context text."],
    )
    # Must NOT raise unhandled exception and must NOT set score to 0.0
    assert res.status == "unavailable"
    assert res.score is None
    assert "failed" in (res.explanation or "").lower()


def test_llm_judge_provider_failure_surfaces_unavailable_status() -> None:
    def failing_caller(s: str, u: str) -> str:
        raise TimeoutError("LLM provider timed out after 30s")

    judge = LLMJudge(provider="mock", model="test-judge", judge_caller=failing_caller)
    evaluator = AutomatedAnswerQualityEvaluator(judge)

    res = evaluator.evaluate_answer_relevancy(
        question="Question",
        answer="Answer.",
    )
    assert res.status == "unavailable"
    assert res.score is None
    assert "timed out" in (res.explanation or "").lower()


def test_run_automated_quality_evaluation_report_and_markdown(tmp_path) -> None:
    import json

    def mock_caller(s: str, u: str) -> str:
        if "faithfulness" in s.lower():
            return json.dumps({"claims": [{"claim": "c", "supported": True, "explanation": "ok"}], "reasoning": "good"})
        if "relevancy" in s.lower():
            return json.dumps({"score": 1.0, "explanation": "relevant"})
        if "citation" in s.lower():
            return json.dumps({"citations": [{"source_ref": "DOC/c1", "supports_claim": True, "explanation": "ok"}], "reasoning": "ok"})
        return "{}"

    judge = LLMJudge(provider="mock", model="test-judge-v1", judge_caller=mock_caller)
    evaluator = AutomatedAnswerQualityEvaluator(judge)

    ev = ExpectedEvidence(document_ref="DOC", chunk_ref="c1")
    items = [
        QualityEvaluationItem(
            question_id="q001",
            question="What is the rule?",
            answer="The rule is annual review.",
            ground_truth_answerability="answerable",
            retrieved_contexts=["The rule is annual review."],
            cited_evidence=[ev],
        ),
        QualityEvaluationItem(
            question_id="q002",
            question="What is the secret?",
            answer="The uploaded documents do not provide enough information to answer this question.",
            ground_truth_answerability="unanswerable",
            retrieved_contexts=[],
            cited_evidence=[],
            unanswerable_flag=True,
        ),
    ]

    report = run_automated_quality_evaluation(
        evaluator=evaluator,
        items=items,
        dataset_id="test-dataset",
        dataset_version="1.0",
    )

    assert report.total_questions == 2
    assert report.mean_faithfulness == 1.0
    assert report.mean_answer_relevancy == 1.0
    assert report.mean_citation_accuracy == 1.0
    assert report.abstention_summary.total_questions == 2
    assert report.abstention_summary.true_abstentions == 1
    assert report.abstention_summary.true_answers == 1
    assert report.abstention_summary.abstention_accuracy == 1.0

    # Test writing JSON and Markdown
    json_path, md_path = write_quality_evaluation_report(report, tmp_path / "reports")
    assert json_path.exists()
    assert md_path.exists()

    md_content = md_path.read_text(encoding="utf-8")
    assert "Answer Quality & Abstention Evaluation Report" in md_content
    assert "Faithfulness" in md_content
    assert "Abstention Accuracy" in md_content
