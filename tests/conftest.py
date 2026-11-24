"""Shared fixtures: a tiny synthetic chest-X-ray-shaped dataset and a trained checkpoint.

The real dataset is ~1.2 GB, so tests use generated grayscale images where
"PNEUMONIA" images contain bright blobs. This exercises the whole pipeline
(data → train → evaluate → predict → app) in a few seconds on CPU.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from pneumonia import train as train_mod

IMG = 64


def _make_image(rng: np.random.Generator, pneumonia: bool) -> Image.Image:
    arr = rng.normal(70, 15, (IMG, IMG))
    if pneumonia:
        for _ in range(3):
            cx, cy = rng.integers(10, IMG - 10, size=2)
            yy, xx = np.ogrid[:IMG, :IMG]
            arr[(xx - cx) ** 2 + (yy - cy) ** 2 < 49] += 120
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode="L")


def make_dataset(root: Path, n_train=(24, 48), n_test=(8, 8), seed: int = 0) -> Path:
    """Create root/{train,test}/{NORMAL,PNEUMONIA}/*.jpeg (imbalanced like the real data)."""
    rng = np.random.default_rng(seed)
    for split, (n_norm, n_pneu) in {"train": n_train, "test": n_test}.items():
        for cls, n in (("NORMAL", n_norm), ("PNEUMONIA", n_pneu)):
            d = root / split / cls
            d.mkdir(parents=True, exist_ok=True)
            for i in range(n):
                _make_image(rng, cls == "PNEUMONIA").save(d / f"{cls.lower()}_{i}.jpeg")
    return root


@pytest.fixture(scope="session")
def dataset_dir(tmp_path_factory) -> Path:
    return make_dataset(tmp_path_factory.mktemp("chest_xray"))


@pytest.fixture(scope="session")
def trained(tmp_path_factory, dataset_dir) -> dict:
    """Train a small model once per test session and return paths + summary."""
    out = tmp_path_factory.mktemp("run")
    args = train_mod.parse_args(
        [
            "--data-dir",
            str(dataset_dir),
            "--output",
            str(out / "model.pt"),
            "--reports-dir",
            str(out / "reports"),
            "--epochs",
            "3",
            "--batch-size",
            "16",
            "--lr",
            "1e-3",
            "--finetune",
            "all",
            "--img-size",
            str(IMG),
            "--num-workers",
            "0",
            "--device",
            "cpu",
            "--val-fraction",
            "0.2",
            "--no-pretrained",
        ]
    )
    summary = train_mod.train(args)
    return {"checkpoint": out / "model.pt", "reports": out / "reports", "summary": summary, "dir": out}


@pytest.fixture
def sample_image(dataset_dir) -> Path:
    return next((dataset_dir / "test" / "PNEUMONIA").glob("*.jpeg"))
