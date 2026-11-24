"""Model definition: ImageNet-pretrained ResNet18 with a 2-class head."""

from __future__ import annotations

from typing import Literal

import torch.nn as nn
import torchvision.models as models

FinetuneMode = Literal["head", "layer4", "all"]


def build_model(
    num_classes: int = 2,
    *,
    pretrained: bool = True,
    finetune: FinetuneMode = "layer4",
    dropout: float = 0.2,
) -> nn.Module:
    """Create a ResNet18 classifier.

    Args:
        num_classes: number of output classes.
        pretrained: load ImageNet weights (needs internet the first time).
        finetune: which part of the network is trainable.
            ``head`` = only the new classifier, ``layer4`` = last residual
            block + classifier (good default), ``all`` = whole network.
        dropout: dropout before the final linear layer.
    """
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)

    for param in model.parameters():
        param.requires_grad = finetune == "all"
    if finetune == "layer4":
        for param in model.layer4.parameters():
            param.requires_grad = True

    in_features = model.fc.in_features
    model.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, num_classes))
    return model


def load_state_dict_compat(model: nn.Module, state_dict: dict) -> None:
    """Load weights, also accepting old checkpoints whose head was a bare Linear.

    Older versions saved ``fc.weight``/``fc.bias``; the current head is
    ``fc = Sequential(Dropout, Linear)`` whose keys are ``fc.1.*``.
    """
    if "fc.weight" in state_dict and "fc.1.weight" not in state_dict:
        state_dict = dict(state_dict)
        state_dict["fc.1.weight"] = state_dict.pop("fc.weight")
        state_dict["fc.1.bias"] = state_dict.pop("fc.bias")
    model.load_state_dict(state_dict)


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """Return (trainable, total) parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total
