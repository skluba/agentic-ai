"""Streamlit façade for prototyping the retrieval stack."""

from __future__ import annotations

import numpy as np
import streamlit as st
from app.config import Settings, clear_settings_cache, get_settings
from app.observability import langfuse_enabled
from app.rag import FaissFlatIndex

st.set_page_config(page_title="Agentic AI — RAG lab", layout="wide")

clear_settings_cache()
settings = get_settings()

st.title("Agentic AI — retrieval lab")


def summarize(settings_obj: Settings) -> None:
    st.subheader("Active configuration")
    st.json(
        {
            "project": settings_obj.gcp_project_id,
            "vertex_region": settings_obj.vertex_location,
            "gemini_model": settings_obj.gemini_model,
            "langfuse_connected": langfuse_enabled(settings_obj),
            "faiss_dimension": settings_obj.faiss_dimension,
        }
    )


summarize(settings)

st.subheader("Smoke test · FAISS")
dim = settings.faiss_dimension
index = FaissFlatIndex(dim)
vec = np.zeros(dim, dtype=np.float32)
vec[0] = 1.0
index.add_vectors(vec.reshape(1, -1), payloads=["dummy-doc"])
hits = index.search(vec, k=3)
nearest = hits[0][0].distance if hits else None
nearest_txt = f"{nearest:.6f}" if nearest is not None else "n/a"
st.success(f"indexed={len(index)} nearest_L2={nearest_txt}")

st.caption("Instrument ADK + Vertex Gemini from app code; use Langfuse SDK where traces add value.")
