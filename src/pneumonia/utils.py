"""Shared helpers: seeding, device selection, checkpoint I/O."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .config import CLASS_NAMES, IMG_SIZE


def set_seed(seed: int) -> None:
    """Make runs reproducible across python, numpy and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(preferred: str = "auto") -> torch.device:
    """Pick CUDA, then Apple MPS, then CPU, unless a device is given."""
    if preferred != "auto":
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    *,
    arch: str = "resnet18",
    class_names: tuple[str, ...] = CLASS_NAMES,
    img_size: int = IMG_SIZE,
    threshold: float = 0.5,
    metrics: dict[str, Any] | None = None,
) -> Path:
    """Save weights together with everything needed to run inference."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "arch": arch,
            "class_names": list(class_names),
            "img_size": img_size,
            "threshold": threshold,
            "metrics": metrics or {},
        },
        path,
    )
    return path


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict:
    """Load a checkpoint, accepting both the new dict format and a raw state_dict.

    Raw state_dicts were produced by earlier versions of this repo
    (``model/pneumonia_model.pth``); they are wrapped with default metadata.
    """
    obj = torch.load(path, map_location=map_location, weights_only=True)
    if isinstance(obj, dict) and "state_dict" in obj:
        obj.setdefault("arch", "resnet18")
        obj.setdefault("class_names", list(CLASS_NAMES))
        obj.setdefault("img_size", IMG_SIZE)
        obj.setdefault("threshold", 0.5)
        obj.setdefault("metrics", {})
        return obj
    return {
        "state_dict": obj,
        "arch": "resnet18",
        "class_names": list(CLASS_NAMES),
        "img_size": IMG_SIZE,
        "threshold": 0.5,
        "metrics": {},
    }


def write_json(path: str | Path, data: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))
