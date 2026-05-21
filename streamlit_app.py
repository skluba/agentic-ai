"""Streamlit façade for prototyping the retrieval stack."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv

# Load `.env` before Langfuse/Google SDK singletons consume process env (best practice).
load_dotenv(Path(__file__).resolve().parent / ".env", override=False)

import streamlit as st  # noqa: E402
from app.config import Settings, clear_settings_cache, get_settings  # noqa: E402
from app.observability import flush_langfuse, langfuse_enabled  # noqa: E402
from app.rag.lab_demo import run_streamlit_lab  # noqa: E402
from langfuse import propagate_attributes  # noqa: E402

st.set_page_config(page_title="Agentic AI — RAG lab", layout="wide")

clear_settings_cache()
settings = get_settings()

if "langfuse_lab_session_id" not in st.session_state:
    st.session_state.langfuse_lab_session_id = str(uuid4())

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
            "langfuse_lab_session_hint": (
                "Sessions view filters on `session_id` when Langfuse receives events."
                if langfuse_enabled(settings_obj)
                else "Set LANGFUSE_* in `.env` to stream hierarchical traces."
            ),
        },
    )


summarize(settings)

st.subheader("Smoke test · FAISS (+ Langfuse trace)")
with propagate_attributes(
    session_id=st.session_state.langfuse_lab_session_id,
    tags=["surface:streamlit", "workflow:research-lab"],
    trace_name="streamlit-research-lab",
    metadata={
        "gcp_project": settings.gcp_project_id,
        "vertex_region": settings.vertex_location,
    },
):
    demo = run_streamlit_lab(settings.faiss_dimension)

flush_langfuse()

nearest = demo["nearest_l2"]
nearest_txt = f"{nearest:.6f}" if nearest is not None else "n/a"
st.success(
    f"indexed={demo['indexed_total']} nearest_L2={nearest_txt} hits={demo['hit_count']}",
)

if langfuse_enabled(settings):
    st.info(
        "Open Langfuse → Traces — look for **`streamlit-research-lab`** with nested "
        "**`rag-lab-*`** retriever spans. Session id matches this Streamlit runtime "
        "(see Sessions when enabled).",
    )

st.json(demo)

st.caption(
    "Hierarchical `@observe` spans are defined in `app/rag/lab_demo.py` "
    "(Langfuse instrumentation guidance)."
)
