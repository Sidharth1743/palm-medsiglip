from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from helpers.plotting import (
    save_confusion_matrix_png,
    save_roc_curve_png,
    save_score_histogram_png,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate baseline zero-shot predictions for anemia classification."
    )
    parser.add_argument(
        "--predictions-csv",
        default="results/baseline_zero_shot/baseline_predictions.csv",
        help="CSV output from baseline-model-running.py",
    )
    parser.add_argument(
        "--output-dir",
        default="results/baseline_zero_shot",
        help="Directory to save evaluation metrics.",
    )
    return parser.parse_args()


def resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def read_predictions(csv_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y_true: list[int] = []
    y_pred: list[int] = []
    y_prob_anemia: list[float] = []

    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"label", "pred_label", "prob_anemia"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError(
                f"{csv_path} must contain columns: label,pred_label,prob_anemia"
            )
        for row in reader:
            y_true.append(int(row["label"]))
            y_pred.append(int(row["pred_label"]))
            y_prob_anemia.append(float(row["prob_anemia"]))

    if not y_true:
        raise ValueError(f"No rows found in {csv_path}")
    return (
        np.asarray(y_true, dtype=np.int64),
        np.asarray(y_pred, dtype=np.int64),
        np.asarray(y_prob_anemia, dtype=np.float32),
    )


def confusion_matrix_binary(y_true: np.ndarray, y_pred: np.ndarray) -> list[list[int]]:
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    return [[tn, fp], [fn, tp]]


def roc_auc_binary(y_true: np.ndarray, y_score: np.ndarray) -> float:
    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(-y_score)
    y_sorted = y_true[order]
    tp = np.cumsum(y_sorted == 1)
    fp = np.cumsum(y_sorted == 0)
    tpr = tp / n_pos
    fpr = fp / n_neg
    tpr = np.concatenate(([0.0], tpr, [1.0]))
    fpr = np.concatenate(([0.0], fpr, [1.0]))
    return float(np.trapezoid(tpr, fpr))


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict[str, Any]:
    cm = confusion_matrix_binary(y_true, y_pred)
    tn, fp = cm[0]
    fn, tp = cm[1]
    total = tn + fp + fn + tp
    accuracy = (tn + tp) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    auc = roc_auc_binary(y_true, y_prob)
    return {
        "num_examples": int(total),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "auc": float(auc),
        "f1": float(f1),
        "confusion_matrix": cm,
    }


def main() -> None:
    args = parse_args()
    pred_csv = resolve_path(args.predictions_csv)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    y_true, y_pred, y_prob = read_predictions(pred_csv)
    metrics = compute_metrics(y_true, y_pred, y_prob)

    metrics_path = output_dir / "baseline_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    cm_plot = output_dir / "baseline_confusion_matrix.png"
    roc_plot = output_dir / "baseline_roc_curve.png"
    hist_plot = output_dir / "baseline_score_histogram.png"
    save_confusion_matrix_png(metrics["confusion_matrix"], cm_plot, title="Zero-Shot Baseline Confusion Matrix")
    save_roc_curve_png(y_true, y_prob, roc_plot, title="Zero-Shot Baseline ROC Curve")
    save_score_histogram_png(
        y_true,
        y_prob,
        hist_plot,
        title="Zero-Shot Baseline Predicted Probability Histogram",
    )

    print("Baseline metrics:")
    print(f"  num_examples: {metrics['num_examples']}")
    print(f"  accuracy: {metrics['accuracy']:.6f}")
    print(f"  precision: {metrics['precision']:.6f}")
    print(f"  recall: {metrics['recall']:.6f}")
    print(f"  auc: {metrics['auc']:.6f}")
    print(f"  f1: {metrics['f1']:.6f}")
    print(f"  confusion_matrix: {metrics['confusion_matrix']}")
    print(f"Saved metrics: {metrics_path}")
    print(f"Saved plots: {cm_plot}, {roc_plot}, {hist_plot}")


if __name__ == "__main__":
    main()
