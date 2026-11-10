from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Answerability = Literal["answerable", "unanswerable"]
QuestionCategory = Literal[
    "direct_fact",
    "multi_chunk",
    "cross_document",
    "paraphrased",
    "specific_detail",
    "unanswerable",
]
Difficulty = Literal["easy", "medium", "hard"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class StableChunkReference(StrictModel):
    chunk_ref: str = Field(min_length=1, pattern=r"^[a-z0-9][a-z0-9_-]*$")
    anchor: str = Field(min_length=1)
    page_number: int | None = Field(default=None, ge=1)
    ingestion_file: str | None = Field(default=None, min_length=1)


class EvaluationDocument(StrictModel):
    document_ref: str = Field(min_length=1, pattern=r"^[A-Z0-9][A-Z0-9_-]*$")
    filename: str = Field(min_length=1)
    title: str = Field(min_length=1)
    content_file: str = Field(min_length=1)
    source_type: Literal["markdown", "text", "pdf"]
    chunks: list[StableChunkReference] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_chunk_refs(self) -> EvaluationDocument:
        refs = [chunk.chunk_ref for chunk in self.chunks]
        if len(refs) != len(set(refs)):
            raise ValueError(f"Duplicate stable chunk reference in {self.document_ref}.")
        return self


class DatasetManifest(StrictModel):
    dataset_id: str = Field(min_length=1)
    dataset_version: str = Field(min_length=1)
    corpus_label: Literal["synthetic_starter"]
    synthetic_notice: str = Field(min_length=1)
    documents: list[EvaluationDocument] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_document_refs(self) -> DatasetManifest:
        refs = [document.document_ref for document in self.documents]
        if len(refs) != len(set(refs)):
            raise ValueError("Document references must be unique.")
        return self


class ExpectedEvidence(StrictModel):
    document_ref: str = Field(min_length=1)
    chunk_ref: str = Field(min_length=1)
    page_number: int | None = Field(default=None, ge=1)


class EvaluationQuestion(StrictModel):
    question_id: str = Field(min_length=1, pattern=r"^[a-z0-9][a-z0-9_-]*$")
    question: str = Field(min_length=5)
    answerability: Answerability
    reference_answer: str | None = None
    relevant_evidence: list[ExpectedEvidence] = Field(default_factory=list)
    unanswerable_reason: str | None = None
    category: QuestionCategory
    difficulty: Difficulty

    @model_validator(mode="after")
    def answerability_is_explicit(self) -> EvaluationQuestion:
        if self.answerability == "answerable":
            if not self.reference_answer:
                raise ValueError("Answerable questions require a reference_answer.")
            if not self.relevant_evidence:
                raise ValueError("Answerable questions require relevant_evidence.")
            if self.unanswerable_reason is not None:
                raise ValueError("Answerable questions cannot include unanswerable_reason.")
            if self.category == "unanswerable":
                raise ValueError("Answerable questions cannot use the unanswerable category.")
        else:
            if self.reference_answer is not None:
                raise ValueError("Unanswerable questions must not have a reference_answer.")
            if self.relevant_evidence:
                raise ValueError("Unanswerable questions cannot claim supporting evidence.")
            if not self.unanswerable_reason:
                raise ValueError("Unanswerable questions require an explicit reason.")
            if self.category != "unanswerable":
                raise ValueError("Unanswerable questions must use the unanswerable category.")
        return self


class EvaluationDataset(StrictModel):
    dataset_id: str = Field(min_length=1)
    dataset_version: str = Field(min_length=1)
    questions: list[EvaluationQuestion] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_question_ids(self) -> EvaluationDataset:
        ids = [question.question_id for question in self.questions]
        if len(ids) != len(set(ids)):
            raise ValueError("Question IDs must be unique.")
        return self


def validate_dataset_references(
    dataset: EvaluationDataset,
    manifest: DatasetManifest,
    *,
    corpus_root: Path | None = None,
) -> None:
    if dataset.dataset_id != manifest.dataset_id:
        raise ValueError("Dataset ID does not match the corpus manifest.")
    if dataset.dataset_version != manifest.dataset_version:
        raise ValueError("Dataset version does not match the corpus manifest.")

    documents = {document.document_ref: document for document in manifest.documents}
    chunk_pages = {
        (document.document_ref, chunk.chunk_ref): chunk.page_number
        for document in manifest.documents
        for chunk in document.chunks
    }

    if corpus_root is not None:
        for document in manifest.documents:
            content_path = (corpus_root / document.content_file).resolve()
            if not content_path.is_relative_to(corpus_root.resolve()):
                raise ValueError(f"Corpus path escapes the dataset directory: {document.content_file}")
            if not content_path.is_file():
                raise ValueError(f"Corpus document is missing: {document.content_file}")
            source_text = content_path.read_text(encoding="utf-8")
            for chunk in document.chunks:
                if f"<!-- eval-chunk:{chunk.chunk_ref} -->" not in source_text:
                    raise ValueError(
                        f"Chunk anchor {chunk.chunk_ref!r} is missing from {document.content_file}."
                    )
                if chunk.ingestion_file is not None:
                    ingestion_path = (corpus_root / chunk.ingestion_file).resolve()
                    if not ingestion_path.is_relative_to(corpus_root.resolve()):
                        raise ValueError(
                            "Ingestion source path escapes the dataset directory: "
                            f"{chunk.ingestion_file}"
                        )
                    if not ingestion_path.is_file():
                        raise ValueError(
                            f"Ingestion source file is missing: {chunk.ingestion_file}"
                        )
                    source_section = _source_section(source_text, chunk.chunk_ref)
                    _validate_ingestion_unit(
                        source_section,
                        ingestion_path.read_bytes(),
                        ingestion_path.name,
                        chunk.chunk_ref,
                    )

    for question in dataset.questions:
        for evidence in question.relevant_evidence:
            if evidence.document_ref not in documents:
                raise ValueError(
                    f"{question.question_id} references unknown document "
                    f"{evidence.document_ref!r}."
                )
            ref = (evidence.document_ref, evidence.chunk_ref)
            if ref not in chunk_pages:
                raise ValueError(
                    f"{question.question_id} references unknown chunk "
                    f"{evidence.document_ref}/{evidence.chunk_ref}."
                )
            if evidence.page_number != chunk_pages[ref]:
                raise ValueError(
                    f"{question.question_id} page number does not match manifest for "
                    f"{evidence.document_ref}/{evidence.chunk_ref}."
                )


def load_evaluation_dataset(directory: Path) -> tuple[DatasetManifest, EvaluationDataset]:
    manifest_path = directory / "manifest.json"
    questions_path = directory / "questions.jsonl"
    manifest = DatasetManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    questions: list[EvaluationQuestion] = []
    for line_number, line in enumerate(
        questions_path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if line.strip():
            try:
                questions.append(EvaluationQuestion.model_validate(json.loads(line)))
            except (json.JSONDecodeError, ValueError) as exc:
                raise ValueError(f"Invalid question record on line {line_number}: {exc}") from exc

    dataset = EvaluationDataset(
        dataset_id=manifest.dataset_id,
        dataset_version=manifest.dataset_version,
        questions=questions,
    )
    validate_dataset_references(dataset, manifest, corpus_root=directory)
    return manifest, dataset


def _source_section(source_text: str, chunk_ref: str) -> str:
    lines = source_text.splitlines()
    marker = f"<!-- eval-chunk:{chunk_ref} -->"
    try:
        start = lines.index(marker) + 1
    except ValueError as exc:
        raise ValueError(f"Chunk anchor {chunk_ref!r} is missing from its source.") from exc
    end = next(
        (
            index
            for index in range(start, len(lines))
            if lines[index].startswith("<!-- eval-chunk:")
        ),
        len(lines),
    )
    return "\n".join(lines[start:end]).strip()


def _validate_ingestion_unit(
    source_section: str,
    ingestion_bytes: bytes,
    filename: str,
    chunk_ref: str,
) -> None:
    from app.ingestion.chunking import chunk_pages
    from app.ingestion.parsers import parse_document

    expected = parse_document(
        source_section.encode("utf-8"),
        filename,
    )
    actual = parse_document(ingestion_bytes, filename)
    if (
        expected.media_type != actual.media_type
        or expected.pages[0].text.split() != actual.pages[0].text.split()
    ):
        raise ValueError(
            f"Ingestion file for {chunk_ref!r} does not match its anchored source section."
        )

    chunks = chunk_pages(actual.pages)
    if len(chunks) != 1:
        raise ValueError(
            f"Ingestion file for {chunk_ref!r} must produce exactly one stored chunk; "
            f"the current parser/chunker produces {len(chunks)}."
        )
