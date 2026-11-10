from __future__ import annotations

import argparse
import sys
from pathlib import Path

from app.evaluation.quality import (
    load_faithfulness_records,
    run_faithfulness_evaluation,
    write_faithfulness_report,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Score manually reviewed answer claims against retrieved evidence "
            "and write a faithfulness report."
        )
    )
    parser.add_argument("--input", type=Path, required=True, help="Reviewed JSONL file.")
    parser.add_argument("--output", type=Path, required=True, help="New report JSON path.")
    parser.add_argument("--dataset-id", required=True, help="Reviewed dataset identifier.")
    parser.add_argument("--dataset-version", required=True, help="Reviewed dataset version.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        records = load_faithfulness_records(args.input)
        report = run_faithfulness_evaluation(
            dataset_id=args.dataset_id,
            dataset_version=args.dataset_version,
            records=records,
        )
        destination = write_faithfulness_report(report, args.output)
    except (FileExistsError, ValueError, OSError) as exc:
        print(f"Faithfulness evaluation failed: {exc}", file=sys.stderr)
        return 1

    print(
        f"Wrote {destination}: {report.evaluated_questions} evaluated, "
        f"{report.unavailable_questions} unavailable; "
        f"mean faithfulness={report.mean_faithfulness}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
