from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import pytest
from pydantic import ValidationError

from app.evaluation.config import DEFAULT_ABLATIONS
from app.evaluation.dataset import (
    DatasetManifest,
    EvaluationDataset,
    EvaluationQuestion,
)
from app.evaluation.runner import (
    EvaluationMappingError,
    EvaluationRunner,
    MappedEvaluationChunk,
    MappedEvaluationDocument,
    SQLAlchemyReferenceMappingStore,
    StableReferenceMap,
    aggregate_mode_results,
    validate_reference_mapping,
    write_evaluation_report,
)
from app.retrieval.engine import RetrievalMode, RetrievalResult
from scripts.run_evaluation import (
    _ablation_configs,
    build_parser,
    write_mapping_template,
)

DOC_ID = uuid.UUID(int=10)
DOC_ID_2 = uuid.UUID(int=11)
CHUNK_1 = uuid.UUID(int=101)
CHUNK_2 = uuid.UUID(int=102)
UNKNOWN_CHUNK = uuid.UUID(int=999)


def fixture_manifest_dataset() -> tuple[DatasetManifest, EvaluationDataset]:
    manifest = DatasetManifest(
        dataset_id="runner-test",
        dataset_version="1",
        corpus_label="synthetic_starter",
        synthetic_notice="Synthetic test corpus.",
        documents=[
            {
                "document_ref": "SYN-TEST-001",
                "filename": "guide.md",
                "title": "Test guide",
                "content_file": "guide.md",
                "source_type": "markdown",
                "chunks": [
                    {
                        "chunk_ref": "c01",
                        "anchor": "first",
                        "ingestion_file": "ingestion/guide-c01.md",
                    },
                    {
                        "chunk_ref": "c02",
                        "anchor": "second",
                        "ingestion_file": "ingestion/guide-c02.md",
                    },
                ],
            }
        ],
    )
    questions = [
        EvaluationQuestion(
            question_id="q-answerable",
            question="What does the first section say?",
            answerability="answerable",
            reference_answer="The first section says hello.",
            relevant_evidence=[{"document_ref": "SYN-TEST-001", "chunk_ref": "c01"}],
            category="direct_fact",
            difficulty="easy",
        ),
        EvaluationQuestion(
            question_id="q-unanswerable",
            question="Which planet hosts the service?",
            answerability="unanswerable",
            reference_answer=None,
            relevant_evidence=[],
            unanswerable_reason="The corpus contains no planet information.",
            category="unanswerable",
            difficulty="easy",
        ),
    ]
    return manifest, EvaluationDataset(
        dataset_id=manifest.dataset_id,
        dataset_version=manifest.dataset_version,
        questions=questions,
    )


def fixture_mapping(*, reviewed: bool = True) -> StableReferenceMap:
    return StableReferenceMap(
        dataset_id="runner-test",
        dataset_version="1",
        reviewed=reviewed,  # type: ignore[arg-type]
        documents=[
            MappedEvaluationDocument(
                document_ref="SYN-TEST-001",
                chunks=[
                    MappedEvaluationChunk(
                        chunk_ref="c01",
                        document_id=DOC_ID,
                        chunk_id=CHUNK_1,
                    ),
                    MappedEvaluationChunk(
                        chunk_ref="c02",
                        document_id=DOC_ID_2,
                        chunk_id=CHUNK_2,
                    ),
                ],
            )
        ],
    )


@dataclass
class FakeDocument:
    id: uuid.UUID
    filename: str = "guide.md"
    status: str = "ready"
    document_metadata: dict[str, object] | None = None

    def __post_init__(self) -> None:
        if self.document_metadata is None:
            self.document_metadata = {"source_format": "markdown"}


@dataclass
class FakeChunk:
    id: uuid.UUID
    document_id: uuid.UUID
    page_number: int | None = None


class FakeMappingStore:
    def __init__(
        self,
        documents: list[FakeDocument] | None = None,
        chunks: list[FakeChunk] | None = None,
    ) -> None:
        self.documents = documents or [
            FakeDocument(DOC_ID, filename="guide-c01.md"),
            FakeDocument(DOC_ID_2, filename="guide-c02.md"),
        ]
        self.chunks = chunks or [
            FakeChunk(CHUNK_1, DOC_ID),
            FakeChunk(CHUNK_2, DOC_ID_2),
        ]

    def documents_by_ids(self, document_ids: Sequence[uuid.UUID]) -> Sequence[FakeDocument]:
        return [document for document in self.documents if document.id in document_ids]

    def chunks_by_document_ids(self, document_ids: Sequence[uuid.UUID]) -> Sequence[FakeChunk]:
        return [chunk for chunk in self.chunks if chunk.document_id in document_ids]


def fake_result(chunk_id: uuid.UUID, *, document_id: uuid.UUID | None = None) -> RetrievalResult:
    if document_id is None:
        document_id = DOC_ID if chunk_id in {CHUNK_1, UNKNOWN_CHUNK} else DOC_ID_2
    return RetrievalResult(
        document_id=document_id,
        document_name="guide-c01.md" if document_id == DOC_ID else "guide-c02.md",
        chunk_id=chunk_id,
        chunk_text="synthetic test result",
        page_number=None,
        chunk_index=chunk_id.int - 101,
        relevance_score=0.75,
    )


class FakeEngine:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self.fail_modes: set[RetrievalMode] = set()
        self.unknown_result = False
        self.failure_message = "retrieval failed"

    def retrieve(
        self,
        query: str,
        *,
        mode: RetrievalMode | str = RetrievalMode.HYBRID,
        top_k: int = 5,
        document_ids: Sequence[uuid.UUID] | None = None,
    ) -> list[RetrievalResult]:
        retrieval_mode = RetrievalMode(mode)
        self.calls.append(
            {
                "query": query,
                "mode": retrieval_mode,
                "top_k": top_k,
                "document_ids": document_ids,
            }
        )
        if retrieval_mode in self.fail_modes:
            raise RuntimeError(f"{self.failure_message} for {retrieval_mode.value}")
        if self.unknown_result:
            return [fake_result(UNKNOWN_CHUNK)]
        return [fake_result(CHUNK_1), fake_result(CHUNK_2)][:top_k]


class FakeReranker:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def rerank(
        self,
        query: str,
        candidates: list[RetrievalResult],
        *,
        top_k: int,
    ) -> list[RetrievalResult]:
        self.calls.append({"query": query, "candidates": candidates, "top_k": top_k})
        return list(reversed(candidates))[:top_k]


def make_runner(
    *,
    configs: Sequence = DEFAULT_ABLATIONS,
    store: FakeMappingStore | None = None,
    engine: FakeEngine | None = None,
) -> tuple[EvaluationRunner, FakeEngine, FakeReranker]:
    fake_engine = engine or FakeEngine()
    reranker = FakeReranker()
    return (
        EvaluationRunner(
            engine=fake_engine,  # type: ignore[arg-type]
            reranker=reranker,  # type: ignore[arg-type]
            mapping_store=store or FakeMappingStore(),
            configs=configs,  # type: ignore[arg-type]
        ),
        fake_engine,
        reranker,
    )


def test_reference_mapping_resolves_all_stable_chunk_references() -> None:
    manifest, dataset = fixture_manifest_dataset()

    mapping = validate_reference_mapping(
        fixture_mapping(),
        manifest,
        dataset,
        FakeMappingStore(),
    )

    assert mapping == {
        CHUNK_1: [("SYN-TEST-001", "c01")],
        CHUNK_2: [("SYN-TEST-001", "c02")],
    }


def test_mapping_validation_queries_do_not_load_embedding_vectors() -> None:
    class FakeResult:
        def __init__(self, rows: list[dict[str, object]]) -> None:
            self.rows = rows

        def mappings(self) -> list[dict[str, object]]:
            return self.rows

    class FakeSession:
        def __init__(self) -> None:
            self.statements = []

        def execute(self, statement: object) -> FakeResult:
            self.statements.append(statement)
            if len(self.statements) == 1:
                return FakeResult(
                    [
                        {
                            "id": DOC_ID,
                            "filename": "guide.md",
                            "status": "ready",
                            "document_metadata": {"source_format": "markdown"},
                        }
                    ]
                )
            return FakeResult(
                [
                    {"id": CHUNK_1, "document_id": DOC_ID, "page_number": None},
                    {"id": CHUNK_2, "document_id": DOC_ID, "page_number": None},
                ]
            )

    session = FakeSession()
    store = SQLAlchemyReferenceMappingStore(session)  # type: ignore[arg-type]

    documents = store.documents_by_ids([DOC_ID])
    chunks = store.chunks_by_document_ids([DOC_ID])

    assert documents[0].filename == "guide.md"
    assert len(chunks) == 2
    assert all("embedding" not in str(statement).lower() for statement in session.statements)


def test_stable_anchors_cannot_share_database_document_or_chunk() -> None:
    manifest, dataset = fixture_manifest_dataset()
    mapping = fixture_mapping().model_copy(
        update={
            "documents": [
                fixture_mapping().documents[0].model_copy(
                    update={
                        "chunks": [
                            MappedEvaluationChunk(
                                chunk_ref="c01",
                                document_id=DOC_ID,
                                chunk_id=CHUNK_1,
                            ),
                            MappedEvaluationChunk(
                                chunk_ref="c02",
                                document_id=DOC_ID,
                                chunk_id=CHUNK_2,
                            ),
                        ]
                    }
                )
            ]
        }
    )

    with pytest.raises(EvaluationMappingError, match="distinct database document"):
        validate_reference_mapping(mapping, manifest, dataset, FakeMappingStore())


def test_reference_mapping_rejects_missing_stable_references() -> None:
    manifest, dataset = fixture_manifest_dataset()
    mapping = fixture_mapping().model_copy(
        update={
            "documents": [
                fixture_mapping().documents[0].model_copy(
                    update={"chunks": [fixture_mapping().documents[0].chunks[0]]}
                )
            ]
        }
    )

    with pytest.raises(EvaluationMappingError, match="exactly match every manifest"):
        validate_reference_mapping(mapping, manifest, dataset, FakeMappingStore())


@pytest.mark.parametrize(
    ("store", "error"),
    [
        (
            FakeMappingStore(
                documents=[
                    FakeDocument(DOC_ID, filename="guide-c01.md", status="pending"),
                    FakeDocument(DOC_ID_2, filename="guide-c02.md"),
                ]
            ),
            "not ready",
        ),
        (
            FakeMappingStore(
                documents=[
                    FakeDocument(DOC_ID, filename="other.md"),
                    FakeDocument(DOC_ID_2, filename="guide-c02.md"),
                ]
            ),
            "filename does not match",
        ),
        (
            FakeMappingStore(
                chunks=[
                    FakeChunk(CHUNK_1, DOC_ID),
                    FakeChunk(CHUNK_2, DOC_ID_2),
                    FakeChunk(uuid.UUID(int=99), DOC_ID),
                ]
            ),
            "cover every stored chunk",
        ),
        (
            FakeMappingStore(
                chunks=[
                    FakeChunk(CHUNK_1, DOC_ID, page_number=2),
                    FakeChunk(CHUNK_2, DOC_ID_2),
                ]
            ),
            "page number does not match",
        ),
    ],
)
def test_reference_mapping_rejects_stale_or_inconsistent_database_rows(
    store: FakeMappingStore,
    error: str,
) -> None:
    manifest, dataset = fixture_manifest_dataset()
    with pytest.raises(EvaluationMappingError, match=error):
        validate_reference_mapping(fixture_mapping(), manifest, dataset, store)


def test_runner_executes_all_four_modes_with_existing_engine_contract() -> None:
    manifest, dataset = fixture_manifest_dataset()
    runner, engine, reranker = make_runner()

    report = runner.run(manifest, dataset, fixture_mapping())

    assert len(report.results) == 8
    assert {result.mode for result in report.results} == {
        RetrievalMode.DENSE,
        RetrievalMode.BM25,
        RetrievalMode.HYBRID,
        RetrievalMode.HYBRID_RERANK,
    }
    assert len(reranker.calls) == 2
    assert [call["mode"] for call in engine.calls].count(RetrievalMode.HYBRID) == 4
    assert all(call["document_ids"] == [DOC_ID, DOC_ID_2] for call in engine.calls)
    rerank_call = next(call for call in engine.calls if call["top_k"] == 50)
    assert rerank_call["mode"] is RetrievalMode.HYBRID
    assert len(report.summaries) == 4
    assert report.results[0].retrieved[0].stable_references[0].document_ref == "SYN-TEST-001"


def test_unanswerable_questions_have_no_retrieval_scores() -> None:
    manifest, dataset = fixture_manifest_dataset()
    runner, _, _ = make_runner()

    report = runner.run(manifest, dataset, fixture_mapping())
    unanswerable = [result for result in report.results if result.answerability == "unanswerable"]

    assert len(unanswerable) == 4
    assert all(result.status == "succeeded" for result in unanswerable)
    assert all(result.recall_at_5 is None for result in unanswerable)
    assert all(result.recall_at_10 is None for result in unanswerable)
    assert all(result.reciprocal_rank is None for result in unanswerable)
    assert all(result.ndcg_at_10 is None for result in unanswerable)


def test_runner_records_failed_modes_without_turning_failures_into_zero_scores() -> None:
    manifest, dataset = fixture_manifest_dataset()
    engine = FakeEngine()
    engine.fail_modes.add(RetrievalMode.BM25)
    runner, _, _ = make_runner(engine=engine)

    report = runner.run(manifest, dataset, fixture_mapping())
    bm25_results = [result for result in report.results if result.mode is RetrievalMode.BM25]
    bm25_summary = next(
        summary for summary in report.summaries if summary.mode is RetrievalMode.BM25
    )

    assert all(result.status == "failed" for result in bm25_results)
    assert all(result.recall_at_5 is None and result.error for result in bm25_results)
    assert bm25_summary.successful_questions == 0
    assert bm25_summary.failed_questions == 2
    assert bm25_summary.recall_at_5 is None


def test_report_failure_messages_redact_secret_shaped_values() -> None:
    manifest, dataset = fixture_manifest_dataset()
    engine = FakeEngine()
    engine.fail_modes.add(RetrievalMode.BM25)
    engine.failure_message = "gsk_sensitive password=secret-value"
    runner, _, _ = make_runner(engine=engine)

    report = runner.run(manifest, dataset, fixture_mapping())
    report_json = report.model_dump_json()
    bm25_failure = next(
        result for result in report.results if result.mode is RetrievalMode.BM25
    )

    assert "gsk_sensitive" not in report_json
    assert "secret-value" not in report_json
    assert "[REDACTED]" in bm25_failure.error


def test_unmapped_retrieved_chunk_fails_instead_of_guessing_reference() -> None:
    manifest, dataset = fixture_manifest_dataset()
    engine = FakeEngine()
    engine.unknown_result = True
    runner, _, _ = make_runner(engine=engine)

    report = runner.run(manifest, dataset, fixture_mapping())

    assert all(result.status == "failed" for result in report.results)
    assert all("no stable-reference mapping" in result.error for result in report.results)
    assert all(result.reciprocal_rank is None for result in report.results)


def test_mode_aggregation_counts_answerability_and_summarizes_successful_values() -> None:
    manifest, dataset = fixture_manifest_dataset()
    runner, _, _ = make_runner()
    report = runner.run(manifest, dataset, fixture_mapping())
    dense_summary = next(
        summary for summary in report.summaries if summary.mode is RetrievalMode.DENSE
    )

    assert dense_summary.successful_questions == 2
    assert dense_summary.failed_questions == 0
    assert dense_summary.answerable_questions == 1
    assert dense_summary.unanswerable_questions == 1
    assert dense_summary.recall_at_5 == 1.0
    assert dense_summary.mean_reciprocal_rank == 1.0
    assert dense_summary.p50_latency_ms is not None
    assert dense_summary.mean_latency_ms is not None


def test_empty_aggregation_has_no_fabricated_metric_or_latency_values() -> None:
    summary = aggregate_mode_results(RetrievalMode.DENSE, [])
    assert summary.successful_questions == 0
    assert summary.failed_questions == 0
    assert summary.answerable_questions == 0
    assert summary.unanswerable_questions == 0
    assert summary.recall_at_5 is None
    assert summary.mean_reciprocal_rank is None
    assert summary.p50_latency_ms is None


def test_report_serializes_to_timestamped_json_without_credentials(tmp_path: Path) -> None:
    manifest, dataset = fixture_manifest_dataset()
    runner, _, _ = make_runner()
    report = runner.run(manifest, dataset, fixture_mapping())

    output = write_evaluation_report(report, tmp_path / "evaluation_outputs")
    serialized = json.loads(output.read_text(encoding="utf-8"))

    assert output.name == f"evaluation-{report.run_id}.json"
    assert serialized["dataset_id"] == dataset.dataset_id
    assert len(serialized["results"]) == 8
    assert "gsk_" not in output.read_text(encoding="utf-8")
    with pytest.raises(FileExistsError, match="already exists"):
        write_evaluation_report(report, output.parent)


def test_invalid_mapping_and_runner_configurations_are_rejected() -> None:
    with pytest.raises(ValidationError):
        fixture_mapping(reviewed=False)
    with pytest.raises(ValueError, match="At least one"):
        make_runner(configs=[])
    with pytest.raises(ValueError, match="unique retrieval modes"):
        make_runner(configs=[DEFAULT_ABLATIONS[0], DEFAULT_ABLATIONS[0]])


def test_dataset_cannot_be_empty() -> None:
    with pytest.raises(ValidationError):
        EvaluationDataset(dataset_id="empty", dataset_version="1", questions=[])


def test_mapping_template_is_explicitly_incomplete_and_never_overwrites(tmp_path: Path) -> None:
    dataset_dir = Path(__file__).resolve().parents[1] / "evaluation_datasets" / "starter"
    destination = tmp_path / "evaluation_outputs" / "mapping.json"

    written = write_mapping_template(dataset_dir, destination)
    template = json.loads(written.read_text(encoding="utf-8"))

    assert template["reviewed"] is False
    assert len(template["documents"]) == 8
    assert all(
        chunk["document_id"] is None
        for document in template["documents"]
        for chunk in document["chunks"]
    )
    assert all(
        chunk["chunk_id"] is None
        for document in template["documents"]
        for chunk in document["chunks"]
    )
    assert all(
        chunk["ingestion_file"]
        for document in template["documents"]
        for chunk in document["chunks"]
    )
    with pytest.raises(ValueError, match="Refusing to overwrite"):
        write_mapping_template(dataset_dir, destination)


def test_cli_help_and_ablation_depth_validation() -> None:
    parser = build_parser()
    help_text = parser.format_help()

    assert "--mapping" in help_text
    assert "--write-mapping-template" in help_text
    assert "does not generate answers or call Groq" in parser.description
    assert len(_ablation_configs(top_k=5, candidate_k=8)) == 4
    with pytest.raises(ValueError, match="greater than or equal"):
        _ablation_configs(top_k=10, candidate_k=5)
