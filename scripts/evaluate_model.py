"""Run leakage-free model evaluation and save reproducible metrics."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "model"
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from training import cross_validate_model, load_dataset, save_metrics, train_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the TurPilot ML pipeline.")
    parser.add_argument(
        "--folds",
        type=int,
        default=5,
        help="K-Fold cross-validation split count.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "model" / "metrics.json",
        help="Metrics JSON output path.",
    )
    return parser.parse_args()


def format_percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def main() -> None:
    args = parse_args()
    df = load_dataset(MODEL_DIR)
    _, holdout_metrics = train_model(df)
    cv_metrics = cross_validate_model(df, n_splits=args.folds)

    metrics = {
        "dataset_rows_used": len(df),
        "holdout": holdout_metrics,
        "cross_validation": cv_metrics,
    }
    save_metrics(metrics, args.output)

    print(f"Rows used: {len(df)}")
    print("Hold-out exact match:", format_percent(holdout_metrics["exact_match_accuracy"]))
    print("Hold-out mean category accuracy:", format_percent(holdout_metrics["mean_category_accuracy"]))
    print(
        f"{args.folds}-Fold exact match:",
        format_percent(cv_metrics["exact_match_accuracy_mean"]),
        f"+/- {format_percent(cv_metrics['exact_match_accuracy_std'])}",
    )
    print(
        f"{args.folds}-Fold mean category accuracy:",
        format_percent(cv_metrics["mean_category_accuracy_mean"]),
        f"+/- {format_percent(cv_metrics['mean_category_accuracy_std'])}",
    )
    print(f"Saved metrics: {args.output}")


if __name__ == "__main__":
    main()
