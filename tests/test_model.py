import numpy as np
import torch
import torch.nn as nn

from pneumonia.metrics import compute_metrics, threshold_for_recall
from pneumonia.model import build_model, count_parameters, load_state_dict_compat


def test_forward_shape():
    model = build_model(pretrained=False).eval()
    assert model(torch.randn(2, 3, 224, 224)).shape == (2, 2)


def test_finetune_modes_freeze_correct_layers():
    head, _ = count_parameters(build_model(pretrained=False, finetune="head"))
    l4, _ = count_parameters(build_model(pretrained=False, finetune="layer4"))
    full, total = count_parameters(build_model(pretrained=False, finetune="all"))
    assert head < l4 < full == total
    assert head == 512 * 2 + 2


def test_loads_legacy_state_dict():
    """Checkpoints from the old version had fc = Linear(512, 2)."""
    import torchvision.models as models

    old = models.resnet18(weights=None)
    old.fc = nn.Linear(512, 2)
    new = build_model(pretrained=False)
    load_state_dict_compat(new, old.state_dict())
    assert torch.equal(new.fc[1].weight, old.fc.weight)


def test_metrics_perfect_and_confusion():
    labels = np.array([0, 0, 1, 1])
    m = compute_metrics(labels, np.array([0.1, 0.2, 0.8, 0.9]))
    assert m["accuracy"] == m["recall"] == m["specificity"] == m["roc_auc"] == 1.0
    m = compute_metrics(labels, np.array([0.6, 0.2, 0.4, 0.9]))
    assert m["confusion_matrix"] == {"tn": 1, "fp": 1, "fn": 1, "tp": 1}


def test_threshold_for_recall_reaches_target():
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 2, 500)
    probs = np.clip(labels * 0.4 + rng.random(500) * 0.6, 0, 1)
    for target in (0.8, 0.95, 0.99):
        t = threshold_for_recall(labels, probs, target)
        assert compute_metrics(labels, probs, t)["recall"] >= target
