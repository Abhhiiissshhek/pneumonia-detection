"""Metric computation and evaluation loop shared by training and evaluation."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader

from .config import POSITIVE_CLASS


@torch.no_grad()
def predict_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Run the model over a loader.

    Returns:
        labels, positive-class probabilities, mean loss (0.0 without criterion).
    """
    model.eval()
    all_labels, all_probs = [], []
    total_loss, n = 0.0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        if criterion is not None:
            total_loss += criterion(logits, labels).item() * labels.size(0)
        n += labels.size(0)
        all_probs.append(torch.softmax(logits, dim=1)[:, POSITIVE_CLASS].cpu().numpy())
        all_labels.append(labels.cpu().numpy())
    return (
        np.concatenate(all_labels),
        np.concatenate(all_probs),
        total_loss / max(n, 1),
    )


def compute_metrics(labels: np.ndarray, probs: np.ndarray, threshold: float = 0.5) -> dict:
    """Binary classification metrics with PNEUMONIA as the positive class."""
    preds = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    try:
        auc = float(roc_auc_score(labels, probs))
    except ValueError:  # only one class present
        auc = float("nan")
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(labels, preds)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),  # sensitivity
        "specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "roc_auc": auc,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "support": int(len(labels)),
    }


def threshold_for_recall(labels: np.ndarray, probs: np.ndarray, target_recall: float) -> float:
    """Highest threshold whose recall on these labels is still >= target_recall."""
    pos = np.sort(probs[labels == POSITIVE_CLASS])
    if len(pos) == 0:
        return 0.5
    # Allow at most k positives below the threshold.
    k = int(np.floor((1 - target_recall) * len(pos)))
    return float(pos[k])
