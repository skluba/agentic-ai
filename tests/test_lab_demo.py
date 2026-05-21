"""Traced lab smoke path."""

from __future__ import annotations

from app.rag.lab_demo import run_streamlit_lab


def test_run_streamlit_lab_returns_metrics():
    out = run_streamlit_lab(32, doc_label="unit-doc")
    assert out["indexed_total"] == 1
    assert out["hit_count"] >= 1
    assert out["nearest_l2"] is not None
    assert isinstance(out["nearest_l2"], float)
