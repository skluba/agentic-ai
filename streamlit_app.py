"""Streamlit façade for prototyping the retrieval stack + Phase 1 RAG."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv


# Load `.env` before Langfuse/Google SDK singletons consume process env (best practice).
load_dotenv(Path(__file__).resolve().parent / ".env", override=False)

import streamlit as st  # noqa: E402
from app.agents.session_runner import concatenate_agent_text, run_core_rag_turn_sync  # noqa: E402
from app.config import Settings, clear_settings_cache, get_settings  # noqa: E402
from app.knowledge import KnowledgeCorpus, build_embedder_from_settings  # noqa: E402
from app.observability import flush_langfuse, langfuse_enabled  # noqa: E402
from app.rag.lab_demo import run_streamlit_lab  # noqa: E402
from langfuse import propagate_attributes  # noqa: E402

st.set_page_config(page_title="Agentic AI — Phase 1 RAG", layout="wide")

clear_settings_cache()
settings = get_settings()

if "langfuse_lab_session_id" not in st.session_state:
    st.session_state.langfuse_lab_session_id = str(uuid4())
if "core_rag_session_id" not in st.session_state:
    st.session_state.core_rag_session_id = str(uuid4())

st.title("Agentic AI — retrieval + Phase 1 RAG")


def summarize(settings_obj: Settings) -> None:
    st.subheader("Active configuration")
    st.json(
        {
            "project": settings_obj.gcp_project_id,
            "vertex_region": settings_obj.vertex_location,
            "gemini_model": settings_obj.gemini_model,
            "gemini_developer_api_key_configured": bool(settings_obj.gemini_api_key),
            "embedding_model": settings_obj.embedding_model,
            "embedding_dimension": settings_obj.embedding_dimension,
            "langfuse_connected": langfuse_enabled(settings_obj),
            "langfuse_lab_session_hint": (
                "Sessions view filters on `session_id` when Langfuse receives events."
                if langfuse_enabled(settings_obj)
                else "Set LANGFUSE_* in `.env` to stream hierarchical traces."
            ),
        },
    )


summarize(settings)

tab_lab, tab_rag = st.tabs(["Instrumentation smoke · FAISS", "Phase 1 · Core RAG (ADK)"])

with tab_lab:
    st.markdown(
        "Validates Langfuse decorators + deterministic FAISS plumbing without touching Gemini chat "
        "(dimension here is orthogonal to corpus embeddings)."
    )

    demo_dim = st.slider(
        "Smoke-test vector dimension",
        min_value=8,
        max_value=256,
        value=32,
        step=8,
    )
    st.subheader("Langfuse-backed smoke run")
    with propagate_attributes(
        session_id=st.session_state.langfuse_lab_session_id,
        tags=["surface:streamlit", "workflow:research-lab"],
        trace_name="streamlit-research-lab",
        metadata={
            "gcp_project": settings.gcp_project_id,
            "vertex_region": settings.vertex_location,
        },
    ):
        demo = run_streamlit_lab(demo_dim)

    flush_langfuse()

    nearest = demo["nearest_l2"]
    nearest_txt = f"{nearest:.6f}" if nearest is not None else "n/a"
    st.success(
        f"indexed={demo['indexed_total']} nearest_L2={nearest_txt} hits={demo['hit_count']}",
    )
    if langfuse_enabled(settings):
        st.info(
            "Inspect Langfuse trace **`streamlit-research-lab`** — nested spans use `rag-lab-*` "
            "names.",
        )
    st.json(demo)

with tab_rag:
    st.subheader("Corpus ingestion")
    st.caption(
        "Phase 1 binds `search_private_knowledge` to an in-memory corpus. Chat uses Gemini via "
        "ADK — set GEMINI_API_KEY (or GOOGLE_API_KEY in `.env`) for the AI Studio path, "
        "or rely on Vertex + `gcloud auth application-default login` without an API key."
    )

    offline_embeddings = st.toggle(
        "Use deterministic embeddings (recommended for demos without Vertex embedding quota)",
        value=True,
    )

    corpus = st.session_state.get("phase1_corpus")
    mode_flag = st.session_state.get("phase1_offline_emb")
    if (
        corpus is None
        or mode_flag != offline_embeddings
        or getattr(corpus, "embedding_dim", None) != settings.embedding_dimension
    ):
        corpus = KnowledgeCorpus(
            embedder=build_embedder_from_settings(settings, offline=offline_embeddings),
        )
        st.session_state.phase1_corpus = corpus
        st.session_state.phase1_offline_emb = offline_embeddings

    doc_label = st.text_input("Human-readable corpus label", value="phase1-notes")
    raw_docs = st.text_area(
        "Paste knowledge-base text",
        height=280,
        help="Large blobs are chunked automatically (≈750 characters with overlap).",
    )

    if st.button("Ingest corpus", type="primary"):
        if not raw_docs.strip():
            st.warning("Nothing to ingest.")
        else:
            added = corpus.ingest_text(doc_id=doc_label, raw_text=raw_docs)
            st.success(
                "Ingest complete — "
                f"{added} chunk(s) indexed (embedding dim {corpus.embedding_dim}).",
            )

    st.metric("Chunk count", corpus.chunk_count)

    st.subheader("Ask the agent")
    question = st.text_input("Question grounded in corpus", "")
    ask = st.button("Run Plan → Execute → Synthesize")

    if ask:
        if corpus.chunk_count == 0:
            st.error("Ingest documents before asking questions.")
        elif not question.strip():
            st.warning("Provide a question.")
        else:
            with st.spinner("Running Gemini + document search …"):
                try:
                    reply, raw_events = run_core_rag_turn_sync(
                        settings=settings,
                        corpus=corpus,
                        question=question,
                        user_id=st.session_state.get("core_user", "streamlit-operator"),
                        session_id=st.session_state.core_rag_session_id,
                    )
                except Exception as exc:
                    st.error(f"Invocation failed — {exc!s}")
                else:
                    if not reply:
                        reply = concatenate_agent_text(raw_events)
                    st.markdown(reply or "_No textual reply emitted — inspect events/logs._")


st.caption(
    "`app/agents/core_rag.py` defines the MVP agent · `app/knowledge/store.py` wires "
    "chunking/embeddings/FAISS · `session_runner.run_core_rag_turn_sync` runs ADK Runner."
)
