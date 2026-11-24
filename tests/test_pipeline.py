"""End-to-end: train → evaluate → predict → Grad-CAM on the synthetic dataset."""

import json

import numpy as np
from PIL import Image

from pneumonia import evaluate as eval_mod
from pneumonia.gradcam import GradCAM
from pneumonia.predict import Predictor
from pneumonia.predict import main as predict_main
from pneumonia.utils import load_checkpoint


def test_training_saves_checkpoint_with_metadata(trained):
    ckpt = load_checkpoint(trained["checkpoint"])
    assert ckpt["class_names"] == ["NORMAL", "PNEUMONIA"]
    assert ckpt["img_size"] == 64
    assert 0.0 <= ckpt["threshold"] <= 1.0
    assert "val" in ckpt["metrics"]
    history = json.loads((trained["reports"] / "training_history.json").read_text())
    assert len(history["history"]) >= 1


def test_model_learns_synthetic_task(trained):
    # Blobs vs. no blobs is easy; anything near chance means training is broken.
    assert trained["summary"]["best_val_auc"] > 0.8


def test_evaluate_writes_reports(trained, dataset_dir):
    reports = trained["dir"] / "eval"
    results = eval_mod.evaluate(
        eval_mod.parse_args(
            [
                "--data-dir",
                str(dataset_dir),
                "--checkpoint",
                str(trained["checkpoint"]),
                "--reports-dir",
                str(reports),
                "--num-workers",
                "0",
                "--device",
                "cpu",
            ]
        )
    )
    assert results["at_threshold"]["support"] == 16
    for f in ("metrics.json", "confusion_matrix.png", "roc_curve.png"):
        assert (reports / f).is_file()


def test_predictor_outputs(trained, sample_image):
    predictor = Predictor(trained["checkpoint"], device="cpu")
    pred = predictor.predict(Image.open(sample_image))
    assert pred.label in {"Normal", "Pneumonia"}
    assert 0.0 <= pred.pneumonia_probability <= 1.0
    assert 0.5 <= pred.confidence <= 1.0 or pred.threshold != 0.5
    # threshold override
    assert predictor.predict(Image.open(sample_image), threshold=0.0).label == "Pneumonia"
    assert predictor.predict(Image.open(sample_image), threshold=1.01).label == "Normal"


def test_gradcam_heatmap(trained, sample_image):
    predictor = Predictor(trained["checkpoint"], device="cpu")
    x = predictor._prepare(Image.open(sample_image))
    with GradCAM(predictor.model, predictor.model.layer4) as cam:
        heat = cam(x, class_idx=1)
    assert heat.shape == (64, 64)
    assert np.isfinite(heat).all() and heat.min() >= 0 and heat.max() <= 1
    overlay = predictor.gradcam(Image.open(sample_image))
    assert overlay.size == Image.open(sample_image).size
    # hooks must be removed so the model is reusable
    assert not predictor.model.layer4._forward_hooks


def test_predict_cli(trained, sample_image, tmp_path, capsys):
    out = tmp_path / "cam.png"
    predict_main([str(sample_image), "--checkpoint", str(trained["checkpoint"]), "--gradcam", str(out)])
    assert "P(pneumonia)=" in capsys.readouterr().out
    assert out.is_file()
