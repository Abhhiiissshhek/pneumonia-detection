"""Train the pneumonia classifier.

Example:
    python -m pneumonia.train --data-dir data/chest_xray --epochs 10
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn

from .config import DATA_DIR, DEFAULT_CHECKPOINT, IMG_SIZE, REPORTS_DIR, SEED
from .data import get_train_val_loaders
from .metrics import compute_metrics, predict_loader, threshold_for_recall
from .model import build_model, count_parameters
from .utils import get_device, save_checkpoint, set_seed, write_json


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, default=DATA_DIR)
    p.add_argument("--output", type=Path, default=DEFAULT_CHECKPOINT, help="checkpoint path")
    p.add_argument("--reports-dir", type=Path, default=REPORTS_DIR)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--finetune", choices=["head", "layer4", "all"], default="layer4")
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--patience", type=int, default=3, help="early-stopping patience (epochs)")
    p.add_argument(
        "--target-recall",
        type=float,
        default=0.95,
        help="lower the decision threshold (from 0.5) until validation pneumonia recall reaches this",
    )
    p.add_argument("--img-size", type=int, default=IMG_SIZE)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--no-pretrained", action="store_true", help="train from random init")
    return p.parse_args(argv)


def train(args: argparse.Namespace) -> dict:
    set_seed(args.seed)
    device = get_device(args.device)

    data = get_train_val_loaders(
        args.data_dir,
        batch_size=args.batch_size,
        val_fraction=args.val_fraction,
        img_size=args.img_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    print(f"Device: {device}")
    print(f"Train images: {len(data.train.dataset)} | Val images: {len(data.val.dataset)}")
    print(f"Class counts (train) NORMAL={data.class_counts[0]} PNEUMONIA={data.class_counts[1]}")

    model = build_model(pretrained=not args.no_pretrained, finetune=args.finetune).to(device)
    trainable, total = count_parameters(model)
    print(f"Parameters: {trainable:,} trainable / {total:,} total")

    # The dataset is ~3:1 PNEUMONIA:NORMAL, so weight the loss by inverse frequency.
    counts = torch.tensor(data.class_counts, dtype=torch.float)
    class_weights = (counts.sum() / (len(counts) * counts.clamp(min=1))).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    history: list[dict] = []
    best_auc, best_epoch, epochs_without_improvement = -1.0, 0, 0

    for epoch in range(1, args.epochs + 1):
        start = time.time()
        model.train()
        running_loss, correct, seen = 0.0, 0, 0
        for images, labels in data.train:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            seen += labels.size(0)
        scheduler.step()

        val_labels, val_probs, val_loss = predict_loader(model, data.val, device, criterion)
        val = compute_metrics(val_labels, val_probs)
        row = {
            "epoch": epoch,
            "train_loss": running_loss / seen,
            "train_acc": correct / seen,
            "val_loss": val_loss,
            "val_acc": val["accuracy"],
            "val_recall": val["recall"],
            "val_auc": val["roc_auc"],
            "lr": optimizer.param_groups[0]["lr"],
            "seconds": round(time.time() - start, 1),
        }
        history.append(row)
        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train loss {row['train_loss']:.4f} acc {row['train_acc']:.3f} | "
            f"val loss {val_loss:.4f} acc {val['accuracy']:.3f} "
            f"recall {val['recall']:.3f} AUC {val['roc_auc']:.3f} | {row['seconds']}s"
        )

        auc = val["roc_auc"] if val["roc_auc"] == val["roc_auc"] else val["accuracy"]  # NaN-safe
        if auc > best_auc:
            best_auc, best_epoch, epochs_without_improvement = auc, epoch, 0
            # Only ever *lower* the threshold below 0.5 to reach the target recall;
            # raising it to fit a small validation set would overfit and hurt recall.
            threshold = min(0.5, threshold_for_recall(val_labels, val_probs, args.target_recall))
            val_at_t = compute_metrics(val_labels, val_probs, threshold)
            save_checkpoint(
                args.output,
                model,
                img_size=args.img_size,
                threshold=threshold,
                metrics={"epoch": epoch, "val": val_at_t},
            )
            print(f"  ↳ saved best model to {args.output} (threshold={threshold:.3f})")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"Early stopping: no val AUC improvement for {args.patience} epochs.")
                break

    summary = {"best_epoch": best_epoch, "best_val_auc": best_auc, "history": history}
    write_json(Path(args.reports_dir) / "training_history.json", summary)
    print(f"\nDone. Best epoch {best_epoch} (val AUC {best_auc:.4f}).")
    return summary


def main(argv: list[str] | None = None) -> None:
    train(parse_args(argv))


if __name__ == "__main__":
    main()
