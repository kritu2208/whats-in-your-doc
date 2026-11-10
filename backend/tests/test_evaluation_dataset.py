from __future__ import annotations

import json
import csv
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.evaluation.dataset import (
    EvaluationDataset,
    EvaluationQuestion,
    load_evaluation_dataset,
    validate_dataset_references,
)
from app.ingestion.chunking import chunk_pages
from app.ingestion.parsers import parse_document

DATASET_DIR = Path(__file__).resolve().parents[1] / "evaluation_datasets" / "starter"


def test_starter_dataset_is_valid_and_reviewable() -> None:
    manifest, dataset = load_evaluation_dataset(DATASET_DIR)

    assert manifest.corpus_label == "synthetic_starter"
    assert len(manifest.documents) == 8
    assert len(dataset.questions) == 60
    assert sum(question.answerability == "answerable" for question in dataset.questions) == 50
    assert sum(question.answerability == "unanswerable" for question in dataset.questions) == 10
    categories = {question.category for question in dataset.questions}
    assert categories == {
        "direct_fact",
        "multi_chunk",
        "cross_document",
        "paraphrased",
        "specific_detail",
        "unanswerable",
    }
    with (DATASET_DIR / "review_template.csv").open(encoding="utf-8", newline="") as file:
        review_rows = list(csv.DictReader(file))
    assert len(review_rows) == 60
    assert {row["question_id"] for row in review_rows} == {
        question.question_id for question in dataset.questions
    }


def test_starter_evidence_anchors_have_distinct_single_chunk_ingestion_units() -> None:
    manifest, dataset = load_evaluation_dataset(DATASET_DIR)
    stable_references: set[tuple[str, str]] = set()
    ingestion_files: set[str] = set()

    for document in manifest.documents:
        for stable_chunk in document.chunks:
            assert stable_chunk.ingestion_file is not None
            assert stable_chunk.ingestion_file not in ingestion_files
            ingestion_files.add(stable_chunk.ingestion_file)
            path = DATASET_DIR / stable_chunk.ingestion_file
            parsed = parse_document(path.read_bytes(), path.name)
            stored_chunks = chunk_pages(parsed.pages)
            assert len(stored_chunks) == 1
            stable_references.add((document.document_ref, stable_chunk.chunk_ref))

    assert len(ingestion_files) == 25
    for question in dataset.questions:
        for evidence in question.relevant_evidence:
            assert (evidence.document_ref, evidence.chunk_ref) in stable_references


def test_answerable_and_unanswerable_questions_have_distinct_ground_truth_shapes() -> None:
    answerable = EvaluationQuestion.model_validate(
        {
            "question_id": "valid-1",
            "question": "When does the service desk open?",
            "answerability": "answerable",
            "reference_answer": "At 08:00 UTC.",
            "relevant_evidence": [
                {"document_ref": "SYN-OPS-001", "chunk_ref": "c01"}
            ],
            "category": "direct_fact",
            "difficulty": "easy",
        }
    )
    unanswerable = EvaluationQuestion.model_validate(
        {
            "question_id": "valid-2",
            "question": "Which city hosts the data center?",
            "answerability": "unanswerable",
            "reference_answer": None,
            "relevant_evidence": [],
            "unanswerable_reason": "No location is specified in the corpus.",
            "category": "unanswerable",
            "difficulty": "easy",
        }
    )

    assert answerable.reference_answer
    assert answerable.relevant_evidence[0].document_ref == "SYN-OPS-001"
    assert unanswerable.reference_answer is None
    assert unanswerable.relevant_evidence == []
    assert unanswerable.unanswerable_reason


@pytest.mark.parametrize(
    "record",
    [
        {
            "question_id": "missing-answer",
            "question": "When does the desk open?",
            "answerability": "answerable",
            "reference_answer": None,
            "relevant_evidence": [{"document_ref": "SYN-OPS-001", "chunk_ref": "c01"}],
            "category": "direct_fact",
            "difficulty": "easy",
        },
        {
            "question_id": "missing-evidence",
            "question": "When does the desk open?",
            "answerability": "answerable",
            "reference_answer": "At 08:00 UTC.",
            "relevant_evidence": [],
            "category": "direct_fact",
            "difficulty": "easy",
        },
        {
            "question_id": "unsupported-answer",
            "question": "Which city hosts the data center?",
            "answerability": "unanswerable",
            "reference_answer": "Seattle.",
            "relevant_evidence": [],
            "unanswerable_reason": "No location is provided.",
            "category": "unanswerable",
            "difficulty": "easy",
        },
        {
            "question_id": "missing-unanswerable-reason",
            "question": "Which city hosts the data center?",
            "answerability": "unanswerable",
            "reference_answer": None,
            "relevant_evidence": [],
            "category": "unanswerable",
            "difficulty": "easy",
        },
    ],
)
def test_invalid_answerability_ground_truth_is_rejected(record: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        EvaluationQuestion.model_validate(record)


def test_duplicate_question_ids_are_rejected() -> None:
    question = {
        "question_id": "same-id",
        "question": "When does the desk open?",
        "answerability": "answerable",
        "reference_answer": "At 08:00 UTC.",
        "relevant_evidence": [{"document_ref": "SYN-OPS-001", "chunk_ref": "c01"}],
        "category": "direct_fact",
        "difficulty": "easy",
    }

    with pytest.raises(ValidationError, match="Question IDs must be unique"):
        EvaluationDataset.model_validate(
            {
                "dataset_id": "starter",
                "dataset_version": "1",
                "questions": [question, question],
            }
        )


def test_missing_document_or_chunk_references_are_rejected() -> None:
    manifest, dataset = load_evaluation_dataset(DATASET_DIR)
    invalid_question = dataset.questions[0].model_copy(
        update={
            "relevant_evidence": [
                dataset.questions[0].relevant_evidence[0].model_copy(
                    update={"chunk_ref": "does-not-exist"}
                )
            ]
        }
    )
    invalid_dataset = dataset.model_copy(update={"questions": [invalid_question]})

    with pytest.raises(ValueError, match="unknown chunk"):
        validate_dataset_references(invalid_dataset, manifest)

    invalid_question = dataset.questions[0].model_copy(
        update={
            "relevant_evidence": [
                dataset.questions[0].relevant_evidence[0].model_copy(
                    update={"document_ref": "UNKNOWN-DOC"}
                )
            ]
        }
    )
    invalid_dataset = dataset.model_copy(update={"questions": [invalid_question]})
    with pytest.raises(ValueError, match="unknown document"):
        validate_dataset_references(invalid_dataset, manifest)

    invalid_question = dataset.questions[0].model_copy(
        update={
            "relevant_evidence": [
                dataset.questions[0].relevant_evidence[0].model_copy(
                    update={"page_number": 1}
                )
            ]
        }
    )
    invalid_dataset = dataset.model_copy(update={"questions": [invalid_question]})
    with pytest.raises(ValueError, match="page number does not match"):
        validate_dataset_references(invalid_dataset, manifest)


def test_manifest_rejects_duplicate_chunks_and_missing_anchor(tmp_path: Path) -> None:
    manifest, dataset = load_evaluation_dataset(DATASET_DIR)
    first_document = manifest.documents[0]
    with pytest.raises(ValidationError, match="Duplicate stable chunk"):
        type(first_document).model_validate(
            {
                **first_document.model_dump(),
                "chunks": [
                    first_document.chunks[0].model_dump(),
                    first_document.chunks[0].model_dump(),
                ],
            }
        )

    missing_anchor_root = tmp_path / "dataset"
    missing_anchor_root.mkdir()
    (missing_anchor_root / "manifest.json").write_text(
        json.dumps(manifest.model_dump()), encoding="utf-8"
    )
    (missing_anchor_root / "questions.jsonl").write_text(
        "\n".join(question.model_dump_json() for question in dataset.questions),
        encoding="utf-8",
    )
    for document in manifest.documents:
        target = missing_anchor_root / document.content_file
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("# synthetic file without expected markers\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Chunk anchor"):
        load_evaluation_dataset(missing_anchor_root)


def test_starter_loader_does_not_initialize_retrieval_or_model_services() -> None:
    _, dataset = load_evaluation_dataset(DATASET_DIR)

    assert dataset.questions
