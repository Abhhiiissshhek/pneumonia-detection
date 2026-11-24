"""Inference on single images, from Python or the command line.

Example:
    python -m pneumonia.predict path/to/xray.jpeg --gradcam out.png
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image

from .config import DEFAULT_CHECKPOINT, DISPLAY_NAMES, LEGACY_CHECKPOINT, POSITIVE_CLASS
from .data import eval_transform
from .gradcam import GradCAM, overlay_heatmap
from .model import build_model, load_state_dict_compat
from .utils import get_device, load_checkpoint


def load_model(checkpoint: str | Path, device: torch.device | str = "cpu"):
    """Load a checkpoint into a ResNet18 in eval mode. Returns (model, checkpoint_dict)."""
    ckpt = load_checkpoint(checkpoint, map_location=device)
    model = build_model(num_classes=len(ckpt["class_names"]), pretrained=False, finetune="head")
    load_state_dict_compat(model, ckpt["state_dict"])
    model.to(device).eval()
    return model, ckpt


def resolve_checkpoint(path: str | Path | None = None) -> Path | None:
    """First existing checkpoint among: the given path, the default, the legacy path."""
    for candidate in (path, DEFAULT_CHECKPOINT, LEGACY_CHECKPOINT):
        if candidate and Path(candidate).is_file():
            return Path(candidate)
    return None


@dataclass
class Prediction:
    label: str  # "Normal" or "Pneumonia"
    pneumonia_probability: float
    threshold: float

    @property
    def is_pneumonia(self) -> bool:
        return self.pneumonia_probability >= self.threshold

    @property
    def confidence(self) -> float:
        """Probability of the predicted label."""
        p = self.pneumonia_probability
        return p if self.is_pneumonia else 1 - p


class Predictor:
    """Wraps a trained model for image-level predictions and Grad-CAM."""

    def __init__(self, checkpoint: str | Path, device: str = "auto", threshold: float | None = None):
        self.device = get_device(device)
        self.model, ckpt = load_model(checkpoint, self.device)
        self.class_names = ckpt["class_names"]
        self.threshold = threshold if threshold is not None else float(ckpt["threshold"])
        self.metrics = ckpt.get("metrics", {})
        self.transform = eval_transform(ckpt["img_size"])

    def _prepare(self, image: Image.Image) -> torch.Tensor:
        return self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)

    @torch.no_grad()
    def predict(self, image: Image.Image, threshold: float | None = None) -> Prediction:
        """Classify one image. ``threshold`` overrides the checkpoint's threshold."""
        t = self.threshold if threshold is None else threshold
        probs = torch.softmax(self.model(self._prepare(image)), dim=1)[0]
        p = float(probs[POSITIVE_CLASS])
        label_key = self.class_names[POSITIVE_CLASS] if p >= t else self.class_names[1 - POSITIVE_CLASS]
        return Prediction(DISPLAY_NAMES.get(label_key, label_key), p, t)

    def gradcam(self, image: Image.Image, class_idx: int = POSITIVE_CLASS) -> Image.Image:
        """Heatmap overlay showing which regions push the model towards ``class_idx``."""
        with GradCAM(self.model, self.model.layer4) as cam:
            heatmap = cam(self._prepare(image), class_idx=class_idx)
        return overlay_heatmap(image, heatmap)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("images", nargs="+", type=Path)
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--threshold", type=float, default=None)
    p.add_argument("--gradcam", type=Path, default=None, help="save Grad-CAM overlay (single image only)")
    p.add_argument("--device", default="auto")
    args = p.parse_args(argv)

    ckpt = resolve_checkpoint(args.checkpoint)
    if ckpt is None:
        raise SystemExit(
            "No checkpoint found. Train one with `python -m pneumonia.train` or pass --checkpoint."
        )
    predictor = Predictor(ckpt, device=args.device, threshold=args.threshold)

    for path in args.images:
        image = Image.open(path)
        pred = predictor.predict(image)
        print(
            f"{path}: {pred.label} (P(pneumonia)={pred.pneumonia_probability:.3f}, "
            f"threshold={pred.threshold:.3f})"
        )
    if args.gradcam:
        predictor.gradcam(Image.open(args.images[0])).save(args.gradcam)
        print(f"Grad-CAM saved to {args.gradcam}")


if __name__ == "__main__":
    main()
