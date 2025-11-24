from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np


def _prepare_path(path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def plot_scores(scores: np.ndarray, labels: np.ndarray, threshold: float, path: str | Path) -> None:
    plt.figure(figsize=(12, 4))
    plt.plot(scores, label="Anomaly Score")
    plt.axhline(threshold, color="red", linestyle="--", label="Threshold")
    anomaly_indices = np.where(labels == 1)[0]
    plt.scatter(anomaly_indices, scores[anomaly_indices], color="orange", s=10, label="Anomalies")
    plt.legend()
    plt.tight_layout()
    path = _prepare_path(path)
    plt.savefig(path)
    plt.close()


def plot_roc_curve(fpr: np.ndarray, tpr: np.ndarray, path: str | Path) -> None:
    plt.figure(figsize=(5, 5))
    plt.plot(fpr, tpr, label="ROC Curve")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    path = _prepare_path(path)
    plt.savefig(path)
    plt.close()


def plot_precision_recall_curve(recall: np.ndarray, precision: np.ndarray, path: str | Path) -> None:
    plt.figure(figsize=(5, 5))
    plt.plot(recall, precision, label="Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.tight_layout()
    path = _prepare_path(path)
    plt.savefig(path)
    plt.close()


def plot_score_distribution(scores: np.ndarray, labels: np.ndarray, path: str | Path) -> None:
    plt.figure(figsize=(6, 4))
    normal_scores = scores[labels == 0]
    anomaly_scores = scores[labels == 1]
    plt.hist(
        normal_scores,
        bins=50,
        alpha=0.6,
        label="Normal",
        density=True,
        color="steelblue",
    )
    if len(anomaly_scores) > 0:
        plt.hist(
            anomaly_scores,
            bins=50,
            alpha=0.6,
            label="Anomaly",
            density=True,
            color="orange",
        )
    plt.xlabel("Anomaly Score")
    plt.ylabel("Density")
    plt.title("Anomaly Score Distribution")
    plt.legend()
    plt.tight_layout()
    path = _prepare_path(path)
    plt.savefig(path)
    plt.close()


def plot_score_distribution(scores: np.ndarray, labels: np.ndarray, path: str | Path) -> None:
    plt.figure(figsize=(6, 4))
    normal_scores = scores[labels == 0]
    anomaly_scores = scores[labels == 1]

    plt.hist(
        normal_scores,
        bins=50,
        alpha=0.6,
        label=f"Normal ({len(normal_scores)})",
        color="blue",
    )
    if len(anomaly_scores) > 0:
        plt.hist(
            anomaly_scores,
            bins=50,
            alpha=0.6,
            label=f"Anomaly ({len(anomaly_scores)})",
            color="orange",
        )

    plt.xlabel("Anomaly Score")
    plt.ylabel("Frequency")
    plt.title("Anomaly Score Distribution")
    plt.legend()
    plt.tight_layout()
    path = _prepare_path(path)
    plt.savefig(path)
    plt.close()

