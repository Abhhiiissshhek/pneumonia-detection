"""Smoke tests for the Streamlit app using streamlit.testing.AppTest."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app.py")


@pytest.fixture(autouse=True)
def _clear_cache():
    import streamlit as st

    st.cache_resource.clear()
    yield
    st.cache_resource.clear()


def test_app_without_model_shows_helpful_error(monkeypatch, tmp_path):
    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "missing.pt"))
    monkeypatch.setenv("MODEL_URL", "")  # don't try to download
    monkeypatch.setattr("pneumonia.predict.DEFAULT_CHECKPOINT", tmp_path / "none.pt")
    monkeypatch.setattr("pneumonia.predict.LEGACY_CHECKPOINT", tmp_path / "none.pth")
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    assert "No trained model found" in at.error[0].value


def test_app_loads_model(monkeypatch, trained):
    monkeypatch.setenv("MODEL_PATH", str(trained["checkpoint"]))
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    assert len(at.slider) == 1  # threshold slider rendered => model loaded
    assert "Upload a frontal chest X-ray" in at.info[0].value
