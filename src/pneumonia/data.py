"""Datasets, transforms and data loaders.

Expected layout (Kaggle "Chest X-Ray Images (Pneumonia)" dataset)::

    data/chest_xray/
        train/{NORMAL,PNEUMONIA}/*.jpeg
        test/{NORMAL,PNEUMONIA}/*.jpeg
        val/...            # ignored: only 16 images, too small to be useful

A stratified, seeded validation split is carved out of ``train`` so that the
validation set is the same on every run and the ``test`` set is never touched
during training or model selection.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from .config import CLASS_NAMES, IMAGENET_MEAN, IMAGENET_STD, IMG_SIZE, SEED


def train_transform(img_size: int = IMG_SIZE) -> transforms.Compose:
    """Light augmentation suited to X-rays (no vertical flips or colour jitter)."""
    return transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.95, 1.05)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def eval_transform(img_size: int = IMG_SIZE) -> transforms.Compose:
    """Deterministic preprocessing used for validation, test and inference."""
    return transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def _check_split_dir(path: Path) -> None:
    if not path.is_dir():
        raise FileNotFoundError(
            f"Dataset folder not found: {path}\n"
            "Download the Kaggle 'chest-xray-pneumonia' dataset "
            "(see scripts/download_data.sh) or pass --data-dir."
        )


def _check_classes(ds: datasets.ImageFolder) -> None:
    if tuple(ds.classes) != CLASS_NAMES:
        raise ValueError(f"Expected class folders {CLASS_NAMES}, found {tuple(ds.classes)}")


@dataclass
class DataBundle:
    train: DataLoader
    val: DataLoader
    class_counts: np.ndarray  # number of training images per class


def get_train_val_loaders(
    data_dir: str | Path,
    *,
    batch_size: int = 32,
    val_fraction: float = 0.1,
    img_size: int = IMG_SIZE,
    num_workers: int = 2,
    seed: int = SEED,
) -> DataBundle:
    """Build train/val loaders from ``data_dir/train`` with a stratified split."""
    train_dir = Path(data_dir) / "train"
    _check_split_dir(train_dir)

    # Two views of the same folder so train and val get different transforms.
    train_view = datasets.ImageFolder(train_dir, transform=train_transform(img_size))
    val_view = datasets.ImageFolder(train_dir, transform=eval_transform(img_size))
    _check_classes(train_view)

    targets = np.array(train_view.targets)
    train_idx, val_idx = train_test_split(
        np.arange(len(targets)),
        test_size=val_fraction,
        stratify=targets,
        random_state=seed,
    )

    generator = torch.Generator().manual_seed(seed)
    pin = torch.cuda.is_available()
    train_loader = DataLoader(
        Subset(train_view, train_idx.tolist()),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin,
        generator=generator,
    )
    val_loader = DataLoader(
        Subset(val_view, val_idx.tolist()),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
    )
    counts = np.bincount(targets[train_idx], minlength=len(CLASS_NAMES))
    return DataBundle(train=train_loader, val=val_loader, class_counts=counts)


def get_test_loader(
    data_dir: str | Path,
    *,
    batch_size: int = 32,
    img_size: int = IMG_SIZE,
    num_workers: int = 2,
    split: str = "test",
) -> DataLoader:
    """Loader for the held-out test split (never used during training)."""
    split_dir = Path(data_dir) / split
    _check_split_dir(split_dir)
    ds = datasets.ImageFolder(split_dir, transform=eval_transform(img_size))
    _check_classes(ds)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
