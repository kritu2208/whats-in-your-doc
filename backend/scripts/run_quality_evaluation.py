from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.evaluation.dataset import load_evaluation_dataset
from app.evaluation.quality import (
    AutomatedAnswerQualityEvaluator,
    LLMJudge,
    QualityEvaluationItem,
    run_automated_quality_evaluation,
    write_quality_evaluation_report,
)
from app.rag.workflow import UNANSWERABLE_ANSWER

BACKEND_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_DIR = BACKEND_ROOT / "evaluation_datasets" / "starter"
DEFAULT_OUTPUT_DIR = BACKEND_ROOT / "evaluation_outputs"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run automated answer quality evaluation (faithfulness, answer relevancy, "
            "citation accuracy, and abstention) using an LLM judge."
        ),
        epilog=(
            "Supports mock judge (for offline reproducible testing) as well as live "
            "Groq and Ollama providers."
        ),
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_DATASET_DIR,
        help=f"Dataset directory containing manifest.json and questions.jsonl (default: {DEFAULT_DATASET_DIR}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for timestamped JSON and Markdown reports (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--provider",
        choices=["mock", "groq", "ollama", "auto"],
        default="mock",
        help="LLM judge provider (default: mock).",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Judge model name override.",
    )
    parser.add_argument(
        "--input-answers",
        type=Path,
        default=None,
        help="Optional path to pre-generated answers JSON or JSONL file.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="Judge request timeout in seconds (default: 30.0).",
    )
    return parser


def _mock_judge_caller(system_prompt: str, user_prompt: str) -> str:
    """Deterministic mock judge for offline evaluation and reproducible verification."""
    if "faithfulness" in system_prompt.lower():
        # Check if user prompt claims are present in context
        return json.dumps(
            {
                "claims": [
                    {
                        "claim": "Reference fact stated in answer.",
                        "supported": True,
                        "explanation": "Directly supported by the supplied context.",
                    }
                ],
                "reasoning": "All factual statements are grounded in context.",
            }
        )
    if "relevancy" in system_prompt.lower():
        return json.dumps(
            {
                "score": 1.0,
                "explanation": "The answer directly and concisely addresses the question.",
            }
        )
    if "citation" in system_prompt.lower():
        return json.dumps(
            {
                "citations": [
                    {
                        "source_ref": "SYN-OPS-001/c01",
                        "supports_claim": True,
                        "explanation": "Source chunk provides the evidence cited.",
                    }
                ],
                "reasoning": "Citations are accurate.",
            }
        )
    return "{}"


def _load_corpus_contexts(dataset_dir: Path) -> dict[tuple[str, str], str]:
    """Map (document_ref, chunk_ref) to chunk text from starter corpus files."""
    contexts: dict[tuple[str, str], str] = {}
    ingestion_dir = dataset_dir / "corpus" / "ingestion"
    if ingestion_dir.exists():
        for file in ingestion_dir.glob("*.md"):
            name = file.stem  # e.g. SYN-OPS-001-c01
            parts = name.rsplit("-", 1)
            if len(parts) == 2:
                doc_ref, chunk_ref = parts[0], parts[1]
                try:
                    contexts[(doc_ref, chunk_ref)] = file.read_text(encoding="utf-8").strip()
                except OSError:
                    pass
    return contexts


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest, dataset = load_evaluation_dataset(args.dataset_dir)
    corpus_contexts = _load_corpus_contexts(args.dataset_dir)

    # Prepare judge
    if args.provider == "mock":
        judge = LLMJudge(
            provider="mock",
            model="mock-judge-v1",
            judge_caller=_mock_judge_caller,
            timeout_seconds=args.timeout,
        )
    else:
        judge = LLMJudge(
            provider=args.provider,
            model=args.model or ("qwen3-vl:2b" if args.provider == "ollama" else "openai/gpt-oss-20b"),
            timeout_seconds=args.timeout,
        )

    evaluator = AutomatedAnswerQualityEvaluator(judge)

    # Load items
    items: list[QualityEvaluationItem] = []
    if args.input_answers is not None:
        raw_text = args.input_answers.read_text(encoding="utf-8")
        if args.input_answers.suffix == ".jsonl":
            answer_records = [json.loads(line) for line in raw_text.splitlines() if line.strip()]
        else:
            answer_records = json.loads(raw_text)
        answer_by_id = {r["question_id"]: r for r in answer_records}
        for q in dataset.questions:
            rec = answer_by_id.get(q.question_id)
            if rec is None:
                continue
            items.append(
                QualityEvaluationItem(
                    question_id=q.question_id,
                    question=q.question,
                    answer=rec.get("answer", ""),
                    ground_truth_answerability=q.answerability,
                    retrieved_contexts=rec.get("retrieved_contexts", []),
                    cited_evidence=rec.get("cited_evidence", []),
                    unanswerable_flag=rec.get("unanswerable", None),
                )
            )
    else:
        # Evaluate baseline dataset reference answers & abstentions
        for q in dataset.questions:
            if q.answerability == "answerable":
                answer_text = q.reference_answer or ""
                chunk_texts = [
                    corpus_contexts.get((ev.document_ref, ev.chunk_ref), "")
                    for ev in q.relevant_evidence
                ]
                items.append(
                    QualityEvaluationItem(
                        question_id=q.question_id,
                        question=q.question,
                        answer=answer_text,
                        ground_truth_answerability="answerable",
                        retrieved_contexts=[c for c in chunk_texts if c],
                        cited_evidence=q.relevant_evidence,
                        unanswerable_flag=False,
                    )
                )
            else:
                items.append(
                    QualityEvaluationItem(
                        question_id=q.question_id,
                        question=q.question,
                        answer=UNANSWERABLE_ANSWER,
                        ground_truth_answerability="unanswerable",
                        retrieved_contexts=[],
                        cited_evidence=[],
                        unanswerable_flag=True,
                    )
                )

    try:
        report = run_automated_quality_evaluation(
            evaluator=evaluator,
            items=items,
            dataset_id=dataset.dataset_id,
            dataset_version=dataset.dataset_version,
        )
        json_path, md_path = write_quality_evaluation_report(report, args.output_dir)
    except Exception as exc:
        print(f"Quality evaluation failed: {exc}", file=sys.stderr)
        return 1

    print(
        f"Wrote quality evaluation report:\n"
        f"  JSON: {json_path}\n"
        f"  Markdown: {md_path}\n"
        f"Evaluated {report.total_questions} questions.\n"
        f"Mean Faithfulness: {report.mean_faithfulness}\n"
        f"Mean Answer Relevancy: {report.mean_answer_relevancy}\n"
        f"Mean Citation Accuracy: {report.mean_citation_accuracy}\n"
        f"Abstention Accuracy: {report.abstention_summary.abstention_accuracy:.4f} "
        f"(Precision: {report.abstention_summary.abstention_precision}, "
        f"Recall: {report.abstention_summary.abstention_recall}, "
        f"F1: {report.abstention_summary.abstention_f1})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
