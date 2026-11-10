from __future__ import annotations

import re
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import fmean
from time import perf_counter
from typing import Literal, Protocol

from pydantic import Field, ValidationError, model_validator
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.evaluation.config import AblationConfig, DEFAULT_ABLATIONS
from app.evaluation.dataset import (
    DatasetManifest,
    EvaluationDataset,
    EvaluationQuestion,
    StrictModel,
    validate_dataset_references,
)
from app.evaluation.metrics import (
    latency_summary,
    ndcg_at_10,
    recall_at_5,
    recall_at_10,
    reciprocal_rank,
)
from app.models import Chunk, Document
from app.retrieval.engine import RetrievalMode, RetrievalResult
from app.retrieval.reranker import CrossEncoderReranker


class MappedEvaluationChunk(StrictModel):
    chunk_ref: str = Field(min_length=1)
    document_id: uuid.UUID
    chunk_id: uuid.UUID


class MappedEvaluationDocument(StrictModel):
    document_ref: str = Field(min_length=1)
    chunks: list[MappedEvaluationChunk] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_stable_chunks(self) -> MappedEvaluationDocument:
        references = [chunk.chunk_ref for chunk in self.chunks]
        if len(references) != len(set(references)):
            raise ValueError(f"Duplicate stable chunk reference in mapping for {self.document_ref}.")
        return self


class StableReferenceMap(StrictModel):
    dataset_id: str = Field(min_length=1)
    dataset_version: str = Field(min_length=1)
    reviewed: Literal[True]
    documents: list[MappedEvaluationDocument] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_documents_and_database_ids(self) -> StableReferenceMap:
        refs = [document.document_ref for document in self.documents]
        ids = [
            chunk.document_id
            for document in self.documents
            for chunk in document.chunks
        ]
        chunk_ids = [
            chunk.chunk_id
            for document in self.documents
            for chunk in document.chunks
        ]
        if len(refs) != len(set(refs)):
            raise ValueError("Mapped document references must be unique.")
        if len(ids) != len(set(ids)):
            raise ValueError("Each stable evidence anchor must map to a distinct database document.")
        if len(chunk_ids) != len(set(chunk_ids)):
            raise ValueError("Each database chunk must map to exactly one stable evidence anchor.")
        return self

    @classmethod
    def load(cls, path: Path) -> StableReferenceMap:
        try:
            return cls.model_validate_json(path.read_text(encoding="utf-8"))
        except OSError as exc:
            raise ValueError(f"Could not read stable-reference mapping file: {path}") from exc
        except ValidationError as exc:
            raise ValueError(
                "Invalid stable-reference mapping. Check dataset identity, reviewed status, "
                "and document/chunk UUID fields."
            ) from exc


class EvaluationMappingError(ValueError):
    """Raised when stable dataset references cannot be verified against stored rows."""


class MappedDocumentRecord(Protocol):
    id: uuid.UUID
    filename: str
    status: str
    document_metadata: dict[str, object]


class MappedChunkRecord(Protocol):
    id: uuid.UUID
    document_id: uuid.UUID
    page_number: int | None


class ReferenceMappingStore(Protocol):
    def documents_by_ids(self, document_ids: Sequence[uuid.UUID]) -> Sequence[MappedDocumentRecord]: ...

    def chunks_by_document_ids(
        self,
        document_ids: Sequence[uuid.UUID],
    ) -> Sequence[MappedChunkRecord]: ...


@dataclass(frozen=True)
class DatabaseDocumentMappingRecord:
    id: uuid.UUID
    filename: str
    status: str
    document_metadata: dict[str, object]


@dataclass(frozen=True)
class DatabaseChunkMappingRecord:
    id: uuid.UUID
    document_id: uuid.UUID
    page_number: int | None


class SQLAlchemyReferenceMappingStore:
    def __init__(self, session: Session) -> None:
        self.session = session

    def documents_by_ids(
        self,
        document_ids: Sequence[uuid.UUID],
    ) -> Sequence[DatabaseDocumentMappingRecord]:
        if not document_ids:
            return []
        rows = self.session.execute(
            select(
                Document.id.label("id"),
                Document.filename.label("filename"),
                Document.status.label("status"),
                Document.document_metadata.label("document_metadata"),
            ).where(Document.id.in_(document_ids))
        ).mappings()
        return [
            DatabaseDocumentMappingRecord(
                id=row["id"],
                filename=row["filename"],
                status=row["status"],
                document_metadata=row["document_metadata"],
            )
            for row in rows
        ]

    def chunks_by_document_ids(
        self,
        document_ids: Sequence[uuid.UUID],
    ) -> Sequence[DatabaseChunkMappingRecord]:
        if not document_ids:
            return []
        rows = self.session.execute(
            select(
                Chunk.id.label("id"),
                Chunk.document_id.label("document_id"),
                Chunk.page_number.label("page_number"),
            ).where(Chunk.document_id.in_(document_ids))
        ).mappings()
        return [
            DatabaseChunkMappingRecord(
                id=row["id"],
                document_id=row["document_id"],
                page_number=row["page_number"],
            )
            for row in rows
        ]


def _expected_chunks(manifest: DatasetManifest) -> dict[str, dict[str, int | None]]:
    return {
        document.document_ref: {
            chunk.chunk_ref: chunk.page_number for chunk in document.chunks
        }
        for document in manifest.documents
    }


def validate_reference_mapping(
    mapping: StableReferenceMap,
    manifest: DatasetManifest,
    dataset: EvaluationDataset,
    store: ReferenceMappingStore,
) -> dict[uuid.UUID, tuple[str, str]]:
    """Verify complete stable-to-database mapping; return DB chunk ID to stable reference."""
    if mapping.reviewed is not True:
        raise EvaluationMappingError(
            "The stable-reference mapping must be manually reviewed before evaluation."
        )
    if mapping.dataset_id != dataset.dataset_id or mapping.dataset_id != manifest.dataset_id:
        raise EvaluationMappingError("Mapping dataset_id does not match the loaded dataset.")
    if (
        mapping.dataset_version != dataset.dataset_version
        or mapping.dataset_version != manifest.dataset_version
    ):
        raise EvaluationMappingError("Mapping dataset_version does not match the loaded dataset.")

    expected = _expected_chunks(manifest)
    provided_documents = {document.document_ref: document for document in mapping.documents}
    if set(provided_documents) != set(expected):
        missing = sorted(set(expected) - set(provided_documents))
        extra = sorted(set(provided_documents) - set(expected))
        raise EvaluationMappingError(
            f"Document mapping must exactly match the manifest (missing={missing}, extra={extra})."
        )

    manifest_documents = {document.document_ref: document for document in manifest.documents}
    mapped_chunks = {
        (document.document_ref, chunk.chunk_ref): chunk
        for document in mapping.documents
        for chunk in document.chunks
    }
    if set(mapped_chunks) != {
        (document_ref, chunk_ref)
        for document_ref, chunk_refs in expected.items()
        for chunk_ref in chunk_refs
    }:
        raise EvaluationMappingError(
            "The database mapping must exactly match every manifest evidence anchor."
        )
    mapped_database_document_ids = [
        chunk.document_id
        for document in mapping.documents
        for chunk in document.chunks
    ]
    if len(mapped_database_document_ids) != len(set(mapped_database_document_ids)):
        raise EvaluationMappingError(
            "Each stable evidence anchor must map to a distinct database document."
        )
    mapped_documents = {
        chunk.document_id: (document.document_ref, chunk.chunk_ref, chunk.chunk_id)
        for document in mapping.documents
        for chunk in document.chunks
    }
    mapped_ids = list(mapped_documents)
    db_documents = {record.id: record for record in store.documents_by_ids(mapped_ids)}
    if set(db_documents) != set(mapped_ids):
        missing_ids = sorted(str(value) for value in set(mapped_ids) - set(db_documents))
        raise EvaluationMappingError(f"Mapped database documents do not exist: {missing_ids}.")

    stable_by_database_chunk: dict[uuid.UUID, list[tuple[str, str]]] = {}
    expected_database_chunk_ids: set[uuid.UUID] = set()
    expected_document_by_chunk_id: dict[uuid.UUID, uuid.UUID] = {}
    for (document_ref, chunk_ref), mapped_chunk in mapped_chunks.items():
        manifest_document = manifest_documents[document_ref]
        stable_chunk = next(
            chunk for chunk in manifest_document.chunks if chunk.chunk_ref == chunk_ref
        )
        database_document = db_documents[mapped_chunk.document_id]
        if database_document.status != "ready":
            raise EvaluationMappingError(
                f"Mapped evidence {document_ref}/{chunk_ref} is not ready "
                f"(status={database_document.status!r})."
            )
        if stable_chunk.ingestion_file is None:
            raise EvaluationMappingError(
                f"Manifest evidence {document_ref}/{chunk_ref} has no ingestion_file."
            )
        expected_filename = Path(stable_chunk.ingestion_file).name
        if database_document.filename != expected_filename:
            raise EvaluationMappingError(
                f"Database filename does not match {expected_filename!r} for "
                f"{document_ref}/{chunk_ref}."
            )
        expected_format = {
            "markdown": "markdown",
            "text": "text",
            "pdf": "pdf",
        }[manifest_document.source_type]
        if database_document.document_metadata.get("source_format") != expected_format:
            raise EvaluationMappingError(
                f"Mapped source_format for {document_ref}/{chunk_ref} does not match "
                f"{expected_format!r}."
            )
        stable_by_database_chunk[mapped_chunk.chunk_id] = [(document_ref, chunk_ref)]
        expected_database_chunk_ids.add(mapped_chunk.chunk_id)
        expected_document_by_chunk_id[mapped_chunk.chunk_id] = mapped_chunk.document_id

    db_chunks = list(store.chunks_by_document_ids(mapped_ids))
    actual_chunk_ids = {chunk.id for chunk in db_chunks}
    if actual_chunk_ids != expected_database_chunk_ids:
        missing = sorted(str(value) for value in actual_chunk_ids - expected_database_chunk_ids)
        stale = sorted(str(value) for value in expected_database_chunk_ids - actual_chunk_ids)
        raise EvaluationMappingError(
            "Chunk mapping must cover every stored chunk exactly once "
            f"(unmapped_database_chunks={missing}, missing_or_wrong_owner={stale})."
        )
    chunks_by_id = {chunk.id: chunk for chunk in db_chunks}

    for chunk_id, expected_document_id in expected_document_by_chunk_id.items():
        chunk = chunks_by_id[chunk_id]
        if expected_document_id != chunk.document_id:
            raise EvaluationMappingError(
                f"Mapped chunk {chunk_id} belongs to a different database document."
            )
        for document_ref, chunk_ref in stable_by_database_chunk[chunk_id]:
            expected_page = expected[document_ref][chunk_ref]
            if chunk.page_number != expected_page:
                raise EvaluationMappingError(
                    f"Mapped chunk {chunk_id} page number does not match "
                    f"{document_ref}/{chunk_ref}."
                )
    return stable_by_database_chunk


class RetrievalExecutor(Protocol):
    def retrieve(
        self,
        query: str,
        *,
        mode: RetrievalMode | str = RetrievalMode.HYBRID,
        top_k: int = 5,
        document_ids: Sequence[uuid.UUID] | None = None,
    ) -> list[RetrievalResult]: ...


class EvaluationStableReference(StrictModel):
    document_ref: str
    chunk_ref: str


class EvaluationRankedChunk(StrictModel):
    stable_references: list[EvaluationStableReference] = Field(min_length=1)
    document_id: uuid.UUID
    chunk_id: uuid.UUID
    rank: int = Field(ge=1)
    relevance_score: float = Field(allow_inf_nan=False)


class EvaluationQuestionResult(StrictModel):
    question_id: str
    mode: RetrievalMode
    answerability: Literal["answerable", "unanswerable"]
    status: Literal["succeeded", "failed"]
    latency_ms: float = Field(ge=0, allow_inf_nan=False)
    retrieved: list[EvaluationRankedChunk] = Field(default_factory=list)
    recall_at_5: float | None = None
    recall_at_10: float | None = None
    reciprocal_rank: float | None = None
    ndcg_at_10: float | None = None
    error: str | None = None

    @model_validator(mode="after")
    def status_fields_match(self) -> EvaluationQuestionResult:
        metric_values = (
            self.recall_at_5,
            self.recall_at_10,
            self.reciprocal_rank,
            self.ndcg_at_10,
        )
        if self.status == "failed":
            if any(value is not None for value in metric_values):
                raise ValueError("Failed retrieval results cannot contain metric scores.")
            if not self.error:
                raise ValueError("Failed retrieval results require an error.")
        elif self.error is not None:
            raise ValueError("Successful retrieval results cannot contain an error.")
        if self.answerability == "unanswerable" and any(
            value is not None for value in metric_values
        ):
            raise ValueError("Unanswerable retrieval results must not contain retrieval scores.")
        return self


class ModeAggregateSummary(StrictModel):
    mode: RetrievalMode
    recall_at_5: float | None
    recall_at_10: float | None
    mean_reciprocal_rank: float | None
    ndcg_at_10: float | None
    p50_latency_ms: float | None
    p95_latency_ms: float | None
    mean_latency_ms: float | None
    successful_questions: int = Field(ge=0)
    failed_questions: int = Field(ge=0)
    answerable_questions: int = Field(ge=0)
    unanswerable_questions: int = Field(ge=0)


class EvaluationRunReport(StrictModel):
    run_id: str
    generated_at: datetime
    dataset_id: str
    dataset_version: str
    configurations: list[AblationConfig]
    summaries: list[ModeAggregateSummary]
    results: list[EvaluationQuestionResult]


class EvaluationRunner:
    def __init__(
        self,
        *,
        engine: RetrievalExecutor,
        reranker: CrossEncoderReranker,
        mapping_store: ReferenceMappingStore,
        configs: Sequence[AblationConfig] = DEFAULT_ABLATIONS,
    ) -> None:
        if not configs:
            raise ValueError("At least one ablation configuration is required.")
        modes = [config.mode for config in configs]
        if len(modes) != len(set(modes)):
            raise ValueError("Ablation configurations must have unique retrieval modes.")
        self.engine = engine
        self.reranker = reranker
        self.mapping_store = mapping_store
        self.configs = list(configs)

    def run(
        self,
        manifest: DatasetManifest,
        dataset: EvaluationDataset,
        mapping: StableReferenceMap,
    ) -> EvaluationRunReport:
        if not dataset.questions:
            raise ValueError("Evaluation dataset must contain at least one question.")
        if manifest.dataset_id != dataset.dataset_id:
            raise ValueError("Dataset ID does not match the manifest.")
        if manifest.dataset_version != dataset.dataset_version:
            raise ValueError("Dataset version does not match the manifest.")
        validate_dataset_references(dataset, manifest)
        stable_by_chunk_id = validate_reference_mapping(
            mapping,
            manifest,
            dataset,
            self.mapping_store,
        )
        corpus_document_ids = list(
            dict.fromkeys(
                chunk.document_id
                for document in mapping.documents
                for chunk in document.chunks
            )
        )
        document_ref_by_id = {
            chunk.document_id: document.document_ref
            for document in mapping.documents
            for chunk in document.chunks
        }

        results = [
            self._run_question(
                question,
                config,
                corpus_document_ids,
                stable_by_chunk_id,
                document_ref_by_id,
            )
            for question in dataset.questions
            for config in self.configs
        ]
        generated_at = datetime.now(UTC)
        return EvaluationRunReport(
            run_id=generated_at.strftime("%Y%m%dT%H%M%S%fZ"),
            generated_at=generated_at,
            dataset_id=dataset.dataset_id,
            dataset_version=dataset.dataset_version,
            configurations=self.configs,
            summaries=[
                aggregate_mode_results(config.mode, results) for config in self.configs
            ],
            results=results,
        )

    def _run_question(
        self,
        question: EvaluationQuestion,
        config: AblationConfig,
        document_ids: list[uuid.UUID],
        stable_by_chunk_id: dict[uuid.UUID, list[tuple[str, str]]],
        document_ref_by_id: dict[uuid.UUID, str],
    ) -> EvaluationQuestionResult:
        if not isinstance(question, EvaluationQuestion):
            raise TypeError("Evaluation dataset contains an invalid question.")
        started = perf_counter()
        retrieval_latency_ms: float | None = None
        try:
            if config.mode is RetrievalMode.HYBRID_RERANK:
                candidates = self.engine.retrieve(
                    question.question,
                    mode=RetrievalMode.HYBRID,
                    top_k=config.candidate_k or config.top_k,
                    document_ids=document_ids,
                )
                retrieved = self.reranker.rerank(
                    question.question,
                    candidates,
                    top_k=config.top_k,
                )
            else:
                retrieved = self.engine.retrieve(
                    question.question,
                    mode=config.mode,
                    top_k=config.top_k,
                    document_ids=document_ids,
                )

            retrieval_latency_ms = (perf_counter() - started) * 1000
            stable_references: list[tuple[str, str]] = []
            ranked_chunks: list[EvaluationRankedChunk] = []
            seen_chunk_ids: set[uuid.UUID] = set()
            for result_rank, result in enumerate(retrieved, start=1):
                if result.chunk_id in seen_chunk_ids:
                    raise EvaluationMappingError(
                        f"Retrieval returned duplicate database chunk {result.chunk_id}."
                    )
                seen_chunk_ids.add(result.chunk_id)
                stable_references_for_chunk = stable_by_chunk_id.get(result.chunk_id)
                if stable_references_for_chunk is None:
                    raise EvaluationMappingError(
                        f"Retrieved database chunk {result.chunk_id} has no stable-reference mapping."
                    )
                if any(
                    document_ref_by_id.get(result.document_id) != document_ref
                    for document_ref, _ in stable_references_for_chunk
                ):
                    raise EvaluationMappingError(
                        f"Retrieved chunk {result.chunk_id} document ownership does not match its mapping."
                    )
                stable_references.extend(stable_references_for_chunk)
                ranked_chunks.append(
                    EvaluationRankedChunk(
                        stable_references=[
                            EvaluationStableReference(
                                document_ref=document_ref,
                                chunk_ref=chunk_ref,
                            )
                            for document_ref, chunk_ref in stable_references_for_chunk
                        ],
                        document_id=result.document_id,
                        chunk_id=result.chunk_id,
                        rank=result_rank,
                        relevance_score=result.relevance_score,
                    )
                )
            return EvaluationQuestionResult(
                question_id=question.question_id,
                mode=config.mode,
                answerability=question.answerability,
                status="succeeded",
                latency_ms=retrieval_latency_ms,
                retrieved=ranked_chunks,
                recall_at_5=recall_at_5(stable_references, question),
                recall_at_10=recall_at_10(stable_references, question),
                reciprocal_rank=(
                    None
                    if question.answerability == "unanswerable"
                    else reciprocal_rank(stable_references, question)
                ),
                ndcg_at_10=ndcg_at_10(stable_references, question),
            )
        except Exception as exc:
            elapsed_ms = (
                retrieval_latency_ms
                if retrieval_latency_ms is not None
                else (perf_counter() - started) * 1000
            )
            return EvaluationQuestionResult(
                question_id=question.question_id,
                mode=config.mode,
                answerability=question.answerability,
                status="failed",
                latency_ms=elapsed_ms,
                error=_safe_error_message(exc),
            )


def _safe_error_message(error: Exception) -> str:
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
    return f"{type(error).__name__}: {message[:500]}"


def aggregate_mode_results(
    mode: RetrievalMode,
    results: Sequence[EvaluationQuestionResult],
) -> ModeAggregateSummary:
    selected = [result for result in results if result.mode is mode]
    successful = [result for result in selected if result.status == "succeeded"]
    answerable = [result for result in selected if result.answerability == "answerable"]
    unanswerable = [result for result in selected if result.answerability == "unanswerable"]

    def average(field: str) -> float | None:
        values = [
            value
            for result in successful
            if result.answerability == "answerable"
            if (value := getattr(result, field)) is not None
        ]
        return fmean(values) if values else None

    latencies = latency_summary([result.latency_ms for result in selected])
    return ModeAggregateSummary(
        mode=mode,
        recall_at_5=average("recall_at_5"),
        recall_at_10=average("recall_at_10"),
        mean_reciprocal_rank=average("reciprocal_rank"),
        ndcg_at_10=average("ndcg_at_10"),
        p50_latency_ms=latencies.p50_ms,
        p95_latency_ms=latencies.p95_ms,
        mean_latency_ms=latencies.mean_ms,
        successful_questions=len(successful),
        failed_questions=len(selected) - len(successful),
        answerable_questions=len(answerable),
        unanswerable_questions=len(unanswerable),
    )


def write_evaluation_report(report: EvaluationRunReport, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / f"evaluation-{report.run_id}.json"
    try:
        with target.open("x", encoding="utf-8", newline="\n") as file:
            file.write(report.model_dump_json(indent=2))
            file.write("\n")
    except FileExistsError as exc:
        raise FileExistsError(f"Evaluation report already exists: {target}") from exc
    return target
