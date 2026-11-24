"""Grad-CAM: highlight the regions of the X-ray that drove the prediction.

Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks via
Gradient-based Localization", ICCV 2017.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import colormaps
from PIL import Image


class GradCAM:
    """Minimal Grad-CAM for any CNN layer, implemented with forward/backward hooks."""

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self._activations: torch.Tensor | None = None
        self._gradients: torch.Tensor | None = None
        self._handles = [
            target_layer.register_forward_hook(self._save_activation),
            target_layer.register_full_backward_hook(self._save_gradient),
        ]

    def _save_activation(self, _module, _inp, out):
        self._activations = out.detach()

    def _save_gradient(self, _module, _grad_in, grad_out):
        self._gradients = grad_out[0].detach()

    def __call__(self, x: torch.Tensor, class_idx: int | None = None) -> np.ndarray:
        """Return a heatmap in [0, 1] with the same H×W as ``x``.

        Args:
            x: a single preprocessed image, shape (1, 3, H, W).
            class_idx: class to explain; defaults to the predicted class.
        """
        self.model.eval()
        x = x.clone().requires_grad_(True)  # lets gradients flow through frozen layers
        with torch.enable_grad():
            logits = self.model(x)
            if class_idx is None:
                class_idx = int(logits.argmax(1).item())
            self.model.zero_grad(set_to_none=True)
            logits[0, class_idx].backward()

        assert self._activations is not None and self._gradients is not None
        weights = self._gradients.mean(dim=(2, 3), keepdim=True)  # GAP over H, W
        cam = F.relu((weights * self._activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam[0, 0].cpu().numpy()
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()
        return cam

    def remove(self) -> None:
        for h in self._handles:
            h.remove()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.remove()


def overlay_heatmap(image: Image.Image, cam: np.ndarray, alpha: float = 0.45) -> Image.Image:
    """Blend a [0, 1] heatmap (jet colormap) over the original image."""
    base = image.convert("RGB")
    heat = Image.fromarray(np.uint8(cam * 255)).resize(base.size, Image.BILINEAR)
    colored = colormaps["jet"](np.asarray(heat) / 255.0)[..., :3]
    colored = Image.fromarray(np.uint8(colored * 255))
    return Image.blend(base, colored, alpha)
