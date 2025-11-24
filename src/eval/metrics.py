from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score


def compute_metrics(scores: np.ndarray, labels: np.ndarray, percentile: float = 95.0) -> Dict[str, float]:
    threshold = np.percentile(scores, percentile)
    preds = (scores >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)

    actual_ratio = float(labels.mean()) if len(labels) > 0 else 0.0
    predicted_ratio = float(preds.mean()) if len(preds) > 0 else 0.0

    metrics = {
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "anomaly_ratio": actual_ratio,
        "predicted_anomaly_ratio": predicted_ratio,
        "roc_auc": float(roc_auc_score(labels, scores)) if len(np.unique(labels)) > 1 else 0.0,
        "pr_auc": float(average_precision_score(labels, scores)),
    }
    return metrics

