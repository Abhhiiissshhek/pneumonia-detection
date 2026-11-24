"""Evaluate a trained checkpoint on the held-out test set.

Writes metrics.json, a confusion matrix and a ROC curve to ``reports/``.

Example:
    python -m pneumonia.evaluate --checkpoint models/pneumonia_resnet18.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402

from .config import CLASS_NAMES, DATA_DIR, DEFAULT_CHECKPOINT, DISPLAY_NAMES, REPORTS_DIR  # noqa: E402
from .data import get_test_loader  # noqa: E402
from .metrics import compute_metrics, predict_loader  # noqa: E402
from .predict import load_model  # noqa: E402
from .utils import get_device, write_json  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, default=DATA_DIR)
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    p.add_argument("--reports-dir", type=Path, default=REPORTS_DIR)
    p.add_argument("--split", default="test")
    p.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="decision threshold (default: the one stored in the checkpoint)",
    )
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--device", default="auto")
    return p.parse_args(argv)


def plot_confusion_matrix(cm: dict, path: Path) -> None:
    matrix = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
    labels = [DISPLAY_NAMES[c] for c in CLASS_NAMES]
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(matrix, cmap="Blues")
    for i in range(2):
        for j in range(2):
            color = "white" if matrix[i, j] > matrix.max() / 2 else "black"
            ax.text(j, i, f"{matrix[i, j]}", ha="center", va="center", color=color, fontsize=14)
    ax.set_xticks([0, 1], labels)
    ax.set_yticks([0, 1], labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion matrix (test set)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_roc(labels: np.ndarray, probs: np.ndarray, auc: float, path: Path) -> None:
    fpr, tpr, _ = roc_curve(labels, probs)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.plot(fpr, tpr, label=f"ResNet18 (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", linewidth=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate (recall)")
    ax.set_title("ROC curve (test set)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def evaluate(args: argparse.Namespace) -> dict:
    device = get_device(args.device)
    model, ckpt = load_model(args.checkpoint, device)
    threshold = args.threshold if args.threshold is not None else ckpt["threshold"]

    loader = get_test_loader(
        args.data_dir,
        batch_size=args.batch_size,
        img_size=ckpt["img_size"],
        num_workers=args.num_workers,
        split=args.split,
    )
    labels, probs, _ = predict_loader(model, loader, device)

    results = {
        "split": args.split,
        "checkpoint": str(args.checkpoint),
        "at_threshold": compute_metrics(labels, probs, threshold),
        "at_0.5": compute_metrics(labels, probs, 0.5),
    }

    reports = Path(args.reports_dir)
    reports.mkdir(parents=True, exist_ok=True)
    write_json(reports / "metrics.json", results)
    plot_confusion_matrix(results["at_threshold"]["confusion_matrix"], reports / "confusion_matrix.png")
    if results["at_threshold"]["roc_auc"] == results["at_threshold"]["roc_auc"]:
        plot_roc(labels, probs, results["at_threshold"]["roc_auc"], reports / "roc_curve.png")

    m = results["at_threshold"]
    print(f"\nTest set ({m['support']} images), threshold = {threshold:.3f}")
    print(f"  Accuracy    {m['accuracy']:.4f}")
    print(f"  Recall      {m['recall']:.4f}  (pneumonia sensitivity)")
    print(f"  Specificity {m['specificity']:.4f}")
    print(f"  Precision   {m['precision']:.4f}")
    print(f"  F1          {m['f1']:.4f}")
    print(f"  ROC-AUC     {m['roc_auc']:.4f}")
    print(f"  Confusion   {m['confusion_matrix']}")
    print(f"\nReports written to {reports}/")
    return results


def main(argv: list[str] | None = None) -> None:
    evaluate(parse_args(argv))


if __name__ == "__main__":
    main()
