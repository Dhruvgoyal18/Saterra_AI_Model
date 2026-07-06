"""Metrics for the imbalanced clean/artifact classification problem.

Accuracy is deliberately not reported: with artifact tiles making up a small
minority (see docs for the actual per-fold counts), a trivial "always clean"
model scores >90% accuracy while being useless. Precision/recall/F1/PR-AUC at
the tile level, plus IoU on the stitched full-image mask, are what actually
distinguish a working detector from a useless one.
"""

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_fscore_support


def compute_classification_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    pr_auc = float(average_precision_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else float("nan")
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "pr_auc": pr_auc,
    }


def compute_iou(pred_mask: np.ndarray, true_mask: np.ndarray) -> float:
    """IoU between the stitched tile-prediction mask and the ground-truth
    artifact mask -- the metric that matches how an annotator actually thinks
    (outlining a region), rather than per-tile accuracy.
    """
    intersection = np.logical_and(pred_mask, true_mask).sum()
    union = np.logical_or(pred_mask, true_mask).sum()
    return float(intersection / union) if union > 0 else float("nan")
