"""Streamlit web app: upload a chest X-ray, get a prediction with Grad-CAM.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import streamlit as st
from PIL import Image, UnidentifiedImageError

# Allow `streamlit run app.py` without `pip install -e .`
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from pneumonia.config import DEFAULT_CHECKPOINT, DEFAULT_MODEL_URL  # noqa: E402
from pneumonia.predict import Predictor, resolve_checkpoint  # noqa: E402

st.set_page_config(page_title="Pneumonia Detection", page_icon="🫁", layout="wide")


def _download_weights(url: str, dest: Path) -> Path | None:
    """Fetch weights from a URL (e.g. a GitHub Release asset). Returns None on failure."""
    import torch

    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        torch.hub.download_url_to_file(url, str(dest), progress=False)
        return dest
    except Exception:  # noqa: BLE001 - any network/HTTP error means "not available"
        if dest.exists():
            dest.unlink()
        return None


@st.cache_resource(show_spinner="Loading model…")
def get_predictor() -> Predictor | None:
    ckpt = resolve_checkpoint(os.environ.get("MODEL_PATH"))
    if ckpt is None and os.environ.get("MODEL_URL", DEFAULT_MODEL_URL):
        ckpt = _download_weights(os.environ.get("MODEL_URL", DEFAULT_MODEL_URL), DEFAULT_CHECKPOINT)
    return Predictor(ckpt) if ckpt else None


st.title("🫁 Pneumonia Detection from Chest X-Rays")
st.caption(
    "ResNet18 fine-tuned on the Kaggle Chest X-Ray (Pneumonia) dataset, "
    "with Grad-CAM to show where the model is looking."
)

predictor = get_predictor()

with st.sidebar:
    st.header("About")
    st.markdown(
        "- **Model:** ResNet18 (ImageNet-pretrained, fine-tuned)\n"
        "- **Classes:** Normal / Pneumonia\n"
        "- **Explainability:** Grad-CAM on the last conv block\n"
        "- [Source code](https://github.com/Abhhiiissshhek/pneumonia-detection)"
    )
    if predictor is not None:
        threshold = st.slider(
            "Decision threshold",
            0.05,
            0.95,
            float(round(predictor.threshold, 2)),
            0.01,
            help="Lower values catch more pneumonia cases (higher recall) at the cost of more false alarms.",
        )
        val = predictor.metrics.get("val")
        if val:
            st.subheader("Validation metrics")
            st.metric("Recall", f"{val['recall']:.1%}")
            st.metric("ROC-AUC", f"{val['roc_auc']:.3f}")
    st.warning("For educational purposes only. Not a medical device and not a substitute for a doctor.")

if predictor is None:
    st.error(
        "No trained model found.\n\n"
        "Train one with `python -m pneumonia.train`, which saves it to "
        f"`{DEFAULT_CHECKPOINT.relative_to(Path(__file__).resolve().parent)}`, "
        "or set `MODEL_PATH` / `MODEL_URL`."
    )
    st.stop()

uploaded = st.file_uploader("Upload a chest X-ray", type=["jpg", "jpeg", "png"])
if uploaded is None:
    st.info("Upload a frontal chest X-ray (JPEG or PNG) to get a prediction.")
    st.stop()

try:
    image = Image.open(uploaded).convert("RGB")
except UnidentifiedImageError:
    st.error("That file doesn't look like a valid image.")
    st.stop()

pred = predictor.predict(image, threshold=threshold)

col1, col2 = st.columns(2)
with col1:
    st.image(image, caption="Uploaded X-ray", use_container_width=True)
with col2:
    with st.spinner("Computing Grad-CAM…"):
        st.image(
            predictor.gradcam(image),
            caption="Grad-CAM: red regions push the model towards 'Pneumonia'",
            use_container_width=True,
        )

st.subheader("Result")
if pred.is_pneumonia:
    st.error(f"**{pred.label}** detected (confidence {pred.confidence:.1%})")
else:
    st.success(f"**{pred.label}** (confidence {pred.confidence:.1%})")
st.progress(pred.pneumonia_probability, text=f"P(pneumonia) = {pred.pneumonia_probability:.3f}")
st.caption(f"Decision threshold: {pred.threshold:.2f}")
