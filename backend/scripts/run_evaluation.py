from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from sqlalchemy.exc import SQLAlchemyError

from app.db.session import get_session_factory
from app.evaluation.config import AblationConfig, DEFAULT_ABLATIONS
from app.evaluation.dataset import load_evaluation_dataset
from app.evaluation.runner import (
    EvaluationRunner,
    SQLAlchemyReferenceMappingStore,
    StableReferenceMap,
    write_evaluation_report,
)
from app.ingestion.embeddings import get_embedding_service
from app.retrieval.engine import RetrievalEngine, RetrievalMode, SQLAlchemyRetrievalStore
from app.retrieval.reranker import get_reranker_service

BACKEND_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_DIR = BACKEND_ROOT / "evaluation_datasets" / "starter"
DEFAULT_OUTPUT_DIR = BACKEND_ROOT / "evaluation_outputs"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run offline retrieval evaluation across dense, BM25, hybrid, "
            "and hybrid_rerank modes. This performs retrieval only; it does not "
            "generate answers or call Groq."
        ),
        epilog=(
            "Before running, create and review a complete stable-reference mapping. "
            "Use --write-mapping-template to generate its starting structure."
        ),
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_DATASET_DIR,
        help=f"Dataset directory containing manifest.json and questions.jsonl (default: {DEFAULT_DATASET_DIR}).",
    )
    mapping_group = parser.add_mutually_exclusive_group(required=True)
    mapping_group.add_argument(
        "--mapping",
        type=Path,
        help="Reviewed JSON mapping from dataset references to database document/chunk UUIDs.",
    )
    mapping_group.add_argument(
        "--write-mapping-template",
        type=Path,
        help="Write an incomplete, intentionally non-runnable mapping template and exit.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for timestamped JSON reports (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument("--top-k", type=int, default=10, help="Final retrieval depth (1-100).")
    parser.add_argument(
        "--candidate-k",
        type=int,
        default=50,
        help="Hybrid candidate depth before reranking (1-100; must be >= top-k).",
    )
    return parser


def write_mapping_template(dataset_dir: Path, destination: Path) -> Path:
    manifest, dataset = load_evaluation_dataset(dataset_dir)
    template = {
        "dataset_id": dataset.dataset_id,
        "dataset_version": dataset.dataset_version,
        "reviewed": False,
        "documents": [
            {
                "document_ref": document.document_ref,
                "chunks": [
                    {
                        "chunk_ref": chunk.chunk_ref,
                        "ingestion_file": chunk.ingestion_file,
                        "document_id": None,
                        "chunk_id": None,
                    }
                    for chunk in document.chunks
                ],
            }
            for document in manifest.documents
        ],
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with destination.open("x", encoding="utf-8", newline="\n") as file:
            json.dump(template, file, indent=2)
            file.write("\n")
    except FileExistsError as exc:
        raise ValueError(f"Refusing to overwrite existing mapping template: {destination}") from exc
    return destination


def _ablation_configs(top_k: int, candidate_k: int) -> list[AblationConfig]:
    if not 1 <= top_k <= 100:
        raise ValueError("--top-k must be between 1 and 100.")
    if not 1 <= candidate_k <= 100:
        raise ValueError("--candidate-k must be between 1 and 100.")
    if candidate_k < top_k:
        raise ValueError("--candidate-k must be greater than or equal to --top-k.")
    return [
        AblationConfig(
            mode=config.mode,
            top_k=top_k,
            candidate_k=(
                candidate_k
                if config.mode is RetrievalMode.HYBRID_RERANK
                else None
            ),
        )
        for config in DEFAULT_ABLATIONS
    ]


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.write_mapping_template is not None:
            path = write_mapping_template(args.dataset_dir, args.write_mapping_template)
            print(
                f"Created incomplete mapping template at {path}. "
                "Fill every database UUID, manually verify each source-anchor mapping, "
                'and set "reviewed" to true before running.'
            )
            return 0

        if args.mapping is None:
            parser.error("Provide --mapping or --write-mapping-template.")
        configs = _ablation_configs(args.top_k, args.candidate_k)
        manifest, dataset = load_evaluation_dataset(args.dataset_dir)
        mapping = StableReferenceMap.load(args.mapping)
        with get_session_factory()() as session:
            runner = EvaluationRunner(
                engine=RetrievalEngine(SQLAlchemyRetrievalStore(session), get_embedding_service()),
                reranker=get_reranker_service(),
                mapping_store=SQLAlchemyReferenceMappingStore(session),
                configs=configs,
            )
            report = runner.run(manifest, dataset, mapping)
        report_path = write_evaluation_report(report, args.output_dir)
    except SQLAlchemyError:
        print(
            "Database access failed. Verify DATABASE_URL, that PostgreSQL is available, "
            "and that migrations have been applied.",
            file=sys.stderr,
        )
        return 2
    except (OSError, ValueError) as exc:
        print(f"Evaluation could not run: {exc}", file=sys.stderr)
        return 2

    failed = sum(result.status == "failed" for result in report.results)
    print(
        f"Evaluation run {report.run_id} wrote {report_path} "
        f"({len(report.results)} question-mode records, {failed} failed)."
    )
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
