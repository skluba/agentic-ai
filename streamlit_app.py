"""Streamlit façade for retrieval lab, Phase 1 RAG, and Phase 2 hybrid research."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv


# Load `.env` before Langfuse/Google SDK singletons consume process env (best practice).
load_dotenv(Path(__file__).resolve().parent / ".env", override=False)

import streamlit as st  # noqa: E402
from app.agents.session_runner import (  # noqa: E402
    concatenate_agent_text,
    run_core_rag_turn_sync,
    run_phase2_external_turn_sync,
)
from app.config import Settings, clear_settings_cache, get_settings  # noqa: E402
from app.knowledge import KnowledgeCorpus, build_embedder_from_settings  # noqa: E402
from app.observability import flush_langfuse, langfuse_enabled  # noqa: E402
from app.rag.lab_demo import run_streamlit_lab  # noqa: E402
from langfuse import propagate_attributes  # noqa: E402

st.set_page_config(page_title="Agentic AI — RAG + hybrid research", layout="wide")

clear_settings_cache()
settings = get_settings()

if "langfuse_lab_session_id" not in st.session_state:
    st.session_state.langfuse_lab_session_id = str(uuid4())
if "core_rag_session_id" not in st.session_state:
    st.session_state.core_rag_session_id = str(uuid4())
if "phase2_session_id" not in st.session_state:
    st.session_state.phase2_session_id = str(uuid4())

st.title("Agentic AI — retrieval + Phase 1 & 2 RAG")


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

st.markdown("### Shared knowledge corpus — Phase 1 & Phase 2")
st.caption(
    "One corpus feeds Phase 1 private answers and Phase 2 hybrid grounding. "
    "Configure ADC or GEMINI_API_KEY."
)

offline_embeddings = st.toggle(
    "Use deterministic embeddings (recommended without Vertex embedding quota)",
    value=True,
)

corpus_shared = st.session_state.get("shared_rag_corpus")
mode_shared = st.session_state.get("shared_rag_offline_emb")
legacy_corpus = st.session_state.get("phase1_corpus")
legacy_offline = st.session_state.get("phase1_offline_emb")
legacy_dim_match = getattr(legacy_corpus, "embedding_dim", None) == settings.embedding_dimension
if corpus_shared is None and legacy_corpus is not None and legacy_dim_match:
    corpus_shared = legacy_corpus
    st.session_state.shared_rag_corpus = corpus_shared
    if legacy_offline is not None:
        st.session_state.shared_rag_offline_emb = legacy_offline

if (
    corpus_shared is None
    or mode_shared != offline_embeddings
    or getattr(corpus_shared, "embedding_dim", None) != settings.embedding_dimension
):
    corpus_shared = KnowledgeCorpus(
        embedder=build_embedder_from_settings(settings, offline=offline_embeddings),
    )
    st.session_state.shared_rag_corpus = corpus_shared
    st.session_state.shared_rag_offline_emb = offline_embeddings

doc_label = st.text_input("Human-readable corpus label", value="shared-notes")
raw_docs = st.text_area(
    "Paste knowledge-base text",
    height=240,
    help="Large blobs are chunked automatically (≈750 characters with overlap).",
)

if st.button("Ingest corpus", type="primary"):
    if not raw_docs.strip():
        st.warning("Nothing to ingest.")
    else:
        added = corpus_shared.ingest_text(doc_id=doc_label, raw_text=raw_docs)
        st.success(
            "Ingest complete — "
            f"{added} chunk(s) indexed (embedding dim {corpus_shared.embedding_dim}).",
        )

st.metric("Chunk count", corpus_shared.chunk_count)

tab_lab, tab_p1, tab_p2 = st.tabs(
    [
        "Instrumentation smoke · FAISS",
        "Phase 1 · Core RAG (ADK)",
        "Phase 2 · External knowledge (Hybrid)",
    ],
)


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


with tab_p1:
    st.markdown(
        "Phase 1 binds **`search_private_knowledge`** only. Ingest data above — without chunks the "
        "agent refuses to spin up."
    )
    question = st.text_input("Question grounded in corpus", "", key="phase1_q")
    ask = st.button("Run Phase 1 · Plan → Execute → Synthesize", key="phase1_go")

    if ask:
        if corpus_shared.chunk_count == 0:
            st.error("Ingest documents before asking questions.")
        elif not question.strip():
            st.warning("Provide a question.")
        else:
            with st.spinner("Running Gemini + document search …"):
                try:
                    reply, raw_events = run_core_rag_turn_sync(
                        settings=settings,
                        corpus=corpus_shared,
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


with tab_p2:
    st.markdown(
        "**Phase 2** treats uploads as canonical evidence.\n"
        "Host Search fills gaps/recency/external-only facts.\n\n"
        "With chunks loaded, **`search_private_knowledge`** precedes web."
    )
    st.caption(
        "Hosted search may depend on SKU/entitlements. Hybrid mode installs grounding even without "
        "uploads."
    )
    question2 = st.text_input("Hybrid question", "", key="phase2_q")
    ask2 = st.button(
        "Run Phase 2 · Hybrid Plan → Execute → Synthesize",
        key="phase2_go",
    )

    if ask2:
        if not question2.strip():
            st.warning("Provide a question.")
        else:
            with st.spinner("Running Gemini + corpus + hosted Google Search …"):
                try:
                    reply2, raw_events2 = run_phase2_external_turn_sync(
                        settings=settings,
                        corpus=corpus_shared,
                        question=question2,
                        user_id=st.session_state.get("core_user", "streamlit-operator"),
                        session_id=st.session_state.phase2_session_id,
                    )
                except Exception as exc:
                    st.error(f"Invocation failed — {exc!s}")
                else:
                    if not reply2:
                        reply2 = concatenate_agent_text(raw_events2)
                    st.markdown(reply2 or "_No textual reply emitted — inspect events/logs._")


st.caption(
    "Phase 1 (`app/agents/core_rag.py`), Phase 2 (`app/agents/external_knowledge.py`), "
    "shared tooling under `app/tools/`, corpus in `app/knowledge/store.py`."
)
