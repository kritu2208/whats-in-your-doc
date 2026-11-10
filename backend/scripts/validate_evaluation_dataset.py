from collections import Counter
from pathlib import Path

from app.evaluation.dataset import load_evaluation_dataset

DATASET_DIR = Path(__file__).resolve().parents[1] / "evaluation_datasets" / "starter"


def main() -> None:
    manifest, dataset = load_evaluation_dataset(DATASET_DIR)
    answerable = sum(question.answerability == "answerable" for question in dataset.questions)
    unanswerable = len(dataset.questions) - answerable
    print(
        f"Validated {dataset.dataset_id} v{dataset.dataset_version}: "
        f"{len(manifest.documents)} documents, {len(dataset.questions)} questions "
        f"({answerable} answerable, {unanswerable} unanswerable)."
    )
    counts = Counter(question.category for question in dataset.questions)
    print("Categories: " + ", ".join(f"{key}={counts[key]}" for key in sorted(counts)))


if __name__ == "__main__":
    main()
