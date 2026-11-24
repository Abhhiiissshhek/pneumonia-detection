import numpy as np
import pytest
import torch

from pneumonia.data import eval_transform, get_test_loader, get_train_val_loaders


def test_split_is_stratified_and_disjoint(dataset_dir):
    bundle = get_train_val_loaders(dataset_dir, batch_size=8, val_fraction=0.25, img_size=64, num_workers=0)
    train_idx = set(bundle.train.dataset.indices)
    val_idx = set(bundle.val.dataset.indices)
    assert train_idx.isdisjoint(val_idx), "train and val must not overlap"
    assert len(train_idx) + len(val_idx) == 72

    targets = np.array(bundle.val.dataset.dataset.targets)[list(val_idx)]
    assert targets.sum() / len(targets) == pytest.approx(48 / 72, abs=0.06)  # class ratio preserved


def test_split_is_reproducible(dataset_dir):
    a = get_train_val_loaders(dataset_dir, img_size=64, num_workers=0, seed=1)
    b = get_train_val_loaders(dataset_dir, img_size=64, num_workers=0, seed=1)
    assert a.val.dataset.indices == b.val.dataset.indices


def test_val_uses_deterministic_transform(dataset_dir):
    bundle = get_train_val_loaders(dataset_dir, img_size=64, num_workers=0)
    x1, _ = bundle.val.dataset[0]
    x2, _ = bundle.val.dataset[0]
    assert torch.equal(x1, x2), "validation images must not be augmented"


def test_test_loader_shapes(dataset_dir):
    images, labels = next(iter(get_test_loader(dataset_dir, batch_size=4, img_size=64, num_workers=0)))
    assert images.shape == (4, 3, 64, 64)
    assert set(labels.tolist()) <= {0, 1}


def test_grayscale_and_rgb_give_same_shape():
    from PIL import Image

    t = eval_transform(32)
    assert t(Image.new("L", (50, 40))).shape == t(Image.new("RGB", (50, 40))).shape == (3, 32, 32)


def test_missing_data_dir_has_helpful_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="download_data"):
        get_train_val_loaders(tmp_path / "nope")
