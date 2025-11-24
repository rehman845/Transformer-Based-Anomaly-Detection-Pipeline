import argparse
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
)


def main(scores_path: str, labels_path: str, output_dir: str, percentile: float) -> None:
    scores = np.load(scores_path)
    labels = np.load(labels_path)
    metrics = compute_metrics(scores, labels, percentile=percentile)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    with open(output_root / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    plot_scores(scores, labels, metrics["threshold"], output_root / "scores.png")
    plot_score_distribution(scores, labels, output_root / "score_distribution.png")

    if len(np.unique(labels)) > 1:
        fpr, tpr, _ = roc_curve(labels, scores)
        precision_curve, recall_curve, _ = precision_recall_curve(labels, scores)
        plot_roc_curve(fpr, tpr, output_root / "roc.png")
        plot_precision_recall_curve(recall_curve, precision_curve, output_root / "precision_recall.png")
    else:
        print("Skipping ROC/PR plots: labels contain a single class.")
    print("Saved metrics and plot to", output_root)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate anomaly scores.")
    parser.add_argument("--scores", type=str, required=True, help="Path to npy file with scores")
    parser.add_argument("--labels", type=str, required=True, help="Path to npy file with labels")
    parser.add_argument("--output", type=str, default="outputs/eval", help="Directory to store evaluation artifacts")
    parser.add_argument("--percentile", type=float, default=95.0, help="Threshold percentile")
    args = parser.parse_args()
    main(args.scores, args.labels, args.output, args.percentile)

