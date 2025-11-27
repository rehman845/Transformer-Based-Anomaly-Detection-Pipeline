import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve

from src.eval.metrics import compute_metrics
from src.eval.visualize import (
    plot_precision_recall_curve,
    plot_roc_curve,
    plot_score_distribution,
    plot_scores,
    plot_threshold_curve,
)


def evaluate_threshold(scores: np.ndarray, labels: np.ndarray, threshold: float) -> dict:
    return compute_metrics(scores, labels, percentile=None, explicit_threshold=threshold)


def evaluate_baseline(scores: np.ndarray, labels: np.ndarray, percentile: float, output_root: Path) -> dict:
    metrics = compute_metrics(scores, labels, percentile=percentile)
    with open(output_root / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    return metrics


def sweep_thresholds(
    scores: np.ndarray,
    labels: np.ndarray,
    min_percentile: float,
    max_percentile: float,
    steps: int,
    output_root: Path,
) -> dict:
    percentiles = np.linspace(min_percentile, max_percentile, steps)
    thresholds = np.percentile(scores, percentiles)
    best_metrics = None
    best_f1 = -1.0
    f1_values = []

    with open(output_root / "threshold_sweep.csv", "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["percentile", "threshold", "precision", "recall", "f1"])
        for p, threshold in zip(percentiles, thresholds):
            metrics = evaluate_threshold(scores, labels, threshold)
            metrics["percentile"] = p
            writer.writerow([p, threshold, metrics["precision"], metrics["recall"], metrics["f1"]])
            f1_values.append(metrics["f1"])
            if metrics["f1"] > best_f1:
                best_f1 = metrics["f1"]
                best_metrics = metrics

    if best_metrics is not None:
        with open(output_root / "metrics_best.json", "w", encoding="utf-8") as f:
            json.dump(best_metrics, f, indent=2)
        plot_threshold_curve(
            thresholds,
            np.array(f1_values),
            ylabel="F1-score",
            path=output_root / "f1_vs_threshold.png",
        )
        print(
            "Best F1 {:.4f} at percentile {:.2f} (threshold {:.4f})".format(
                best_metrics["f1"], best_metrics["percentile"], best_metrics["threshold"]
            )
        )
    else:
        print("Threshold sweep produced no valid metrics.")

    return best_metrics


def main(
    scores_path: str,
    labels_path: str,
    output_dir: str,
    percentile: float,
    sweep: bool,
    min_percentile: float,
    max_percentile: float,
    steps: int,
) -> None:
    scores = np.load(scores_path)
    labels = np.load(labels_path)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    baseline_metrics = evaluate_baseline(scores, labels, percentile, output_root)
    print("Baseline metrics saved to", output_root / "metrics.json")

    if sweep:
        sweep_thresholds(
            scores,
            labels,
            min_percentile=min_percentile,
            max_percentile=max_percentile,
            steps=steps,
            output_root=output_root,
        )

    plot_scores(scores, labels, baseline_metrics["threshold"], output_root / "scores.png")
    plot_score_distribution(scores, labels, output_root / "score_distribution.png")

    if len(np.unique(labels)) > 1:
        fpr, tpr, _ = roc_curve(labels, scores)
        precision_curve, recall_curve, _ = precision_recall_curve(labels, scores)
        plot_roc_curve(fpr, tpr, output_root / "roc.png")
        plot_precision_recall_curve(recall_curve, precision_curve, output_root / "precision_recall.png")
    else:
        print("Skipping ROC/PR plots: labels contain a single class.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate anomaly scores.")
    parser.add_argument("--scores", type=str, required=True, help="Path to npy file with scores")
    parser.add_argument("--labels", type=str, required=True, help="Path to npy file with labels")
    parser.add_argument("--output", type=str, default="outputs/eval", help="Directory to store evaluation artifacts")
    parser.add_argument("--percentile", type=float, default=95.0, help="Baseline threshold percentile")
    parser.add_argument("--sweep", action="store_true", help="Run threshold sweep to maximize F1 score")
    parser.add_argument("--min_percentile", type=float, default=80.0, help="Minimum percentile for sweep")
    parser.add_argument("--max_percentile", type=float, default=99.9, help="Maximum percentile for sweep")
    parser.add_argument("--steps", type=int, default=80, help="Number of thresholds to evaluate in sweep")
    args = parser.parse_args()

    main(
        args.scores,
        args.labels,
        args.output,
        args.percentile,
        args.sweep,
        args.min_percentile,
        args.max_percentile,
        args.steps,
    )

