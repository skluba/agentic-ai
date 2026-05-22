"""Streamlit façade for retrieval lab through Phase 4 refinement loop."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv


# Load `.env` before Langfuse/Google SDK singletons consume process env (best practice).
load_dotenv(Path(__file__).resolve().parent / ".env", override=False)

import streamlit as st  # noqa: E402
from app.agents.refinement_loop import run_phase4_refinement_loop_sync  # noqa: E402
from app.agents.session_runner import (  # noqa: E402
    concatenate_agent_text,
    run_core_rag_turn_sync,
    run_phase2_external_turn_sync,
    run_phase3_mcp_turn_sync,
)
from app.config import Settings, clear_settings_cache, get_settings  # noqa: E402
from app.knowledge import KnowledgeCorpus, build_embedder_from_settings  # noqa: E402
from app.observability import flush_langfuse, langfuse_enabled  # noqa: E402
from app.rag.lab_demo import run_streamlit_lab  # noqa: E402
from langfuse import propagate_attributes  # noqa: E402

_MSG_PROVIDE_QUESTION = "Provide a question."
_REPLY_FALLBACK_MARKDOWN = "_No textual reply emitted — inspect events/logs._"

st.set_page_config(page_title="Agentic AI — RAG + MCP + refinement loop", layout="wide")

clear_settings_cache()
settings = get_settings()

if "langfuse_lab_session_id" not in st.session_state:
    st.session_state.langfuse_lab_session_id = str(uuid4())
if "core_rag_session_id" not in st.session_state:
    st.session_state.core_rag_session_id = str(uuid4())
if "phase2_session_id" not in st.session_state:
    st.session_state.phase2_session_id = str(uuid4())
if "phase3_session_id" not in st.session_state:
    st.session_state.phase3_session_id = str(uuid4())
if "phase4_session_id" not in st.session_state:
    st.session_state.phase4_session_id = str(uuid4())

st.title("Agentic AI — retrieval + Phase 1–4 agents")


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
            "mcp_financial_fetch_transport": settings_obj.mcp_financial_fetch_transport,
            "mcp_financial_docker_image": settings_obj.mcp_financial_docker_image,
        },
    )


summarize(settings)

st.markdown("### Shared knowledge corpus — Phases 1–4")
st.caption(
    "Corpus feeds Phases 1–4 (Phase 4 stacks critique-driven refinements atop Phase 3)."
    " Set ADC or GEMINI_API_KEY. Phase 3 needs MCP fetch; see MCP_FINANCIAL_* in `.env.example`."
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

tab_lab, tab_p1, tab_p2, tab_p3, tab_p4 = st.tabs(
    [
        "Instrumentation smoke · FAISS",
        "Phase 1 · Core RAG (ADK)",
        "Phase 2 · External knowledge (Hybrid)",
        "Phase 3 · MCP Yahoo finance",
        "Phase 4 · Autonomous refinement",
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
            st.warning(_MSG_PROVIDE_QUESTION)
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
                    st.markdown(reply or _REPLY_FALLBACK_MARKDOWN)


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
            st.warning(_MSG_PROVIDE_QUESTION)
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
                    st.markdown(reply2 or _REPLY_FALLBACK_MARKDOWN)


with tab_p3:
    st.markdown(
        "**Phase 3** adds **`fetch_yahoo_finance_markets_via_mcp`**: MCP **fetch** against "
        "fixed Yahoo listings (most-active US equities, crypto, currencies). Hosted Google Search "
        "covers non-finance intents and backs up when MCP yields nothing usable."
    )
    st.caption(
        "`docker pull mcp/fetch` (recommended per upstream docs) **or** `uvx mcp-server-fetch`; "
        "override with `MCP_FINANCIAL_FETCH_TRANSPORT` / image in `.env`."
    )
    question3 = st.text_input("Hybrid + finance question", "", key="phase3_q")
    ask3 = st.button(
        "Run Phase 3 · Corpus + MCP markets + hybrid search",
        key="phase3_go",
    )

    if ask3:
        if not question3.strip():
            st.warning(_MSG_PROVIDE_QUESTION)
        else:
            with st.spinner("Running Gemini + corpus + MCP fetch + hosted search …"):
                try:
                    reply3, raw_events3 = run_phase3_mcp_turn_sync(
                        settings=settings,
                        corpus=corpus_shared,
                        question=question3,
                        user_id=st.session_state.get("core_user", "streamlit-operator"),
                        session_id=st.session_state.phase3_session_id,
                    )
                except Exception as exc:
                    st.error(f"Invocation failed — {exc!s}")
                else:
                    if not reply3:
                        reply3 = concatenate_agent_text(raw_events3)
                    st.markdown(reply3 or _REPLY_FALLBACK_MARKDOWN)


with tab_p4:
    st.markdown(
        "**Phase 4** runs **Phase‑3 hybrid research**, then asks Gemini — in a reviewer role — "
        "to emit critique JSON (**gaps** + **follow_up_queries**). Unsatisfied passes launch "
        "**additional research rounds** (same tools) toward a tightened answer."
    )
    st.caption(
        "`max_critique_iterations` bounds how many **post-draft refinement cycles** may run "
        "(each critiques the latest draft before optionally spinning another Phase‑3 invocation)."
    )
    question4 = st.text_input("Research question (with critique refinement)", "", key="phase4_q")
    max_crit = st.number_input(
        "Max critique-driven refinement iterations",
        min_value=0,
        max_value=6,
        value=2,
        step=1,
        key="phase4_max_crit",
        help="Each iteration critiques the newest draft then, if needed, researches again.",
    )
    ask4 = st.button(
        "Run Phase 4 · Research + critique loops",
        key="phase4_go",
    )

    if ask4:
        if not question4.strip():
            st.warning(_MSG_PROVIDE_QUESTION)
        else:
            with st.spinner("Running refinement loop (research + critiques) …"):
                try:
                    refined = run_phase4_refinement_loop_sync(
                        settings=settings,
                        corpus=corpus_shared,
                        question=question4,
                        max_critique_iterations=int(max_crit),
                        user_id=st.session_state.get("core_user", "streamlit-operator"),
                        session_id=st.session_state.phase4_session_id,
                    )
                except Exception as exc:
                    st.error(f"Invocation failed — {exc!s}")
                else:
                    status = (
                        "Critique satisfied reviewer."
                        if refined.terminated_because_satisfied
                        else "Stopped at iteration ceiling (still unsatisfied — inspect critiques)."
                    )
                    st.caption(status)
                    st.markdown(refined.final_answer or _REPLY_FALLBACK_MARKDOWN)

                    expand = st.expander(
                        "Refinement trace (research rounds + critiques)", expanded=False
                    )
                    with expand:
                        rounds = []
                        for rec in refined.research_history:
                            rounds.append(
                                {"round_index": rec.round_index, "prompt_preview": rec.prompt[:400]}
                            )
                        st.json({"research_rounds": rounds})
                        critiques_payload = []
                        for cblock in refined.critiques:
                            critiques_payload.append(
                                {
                                    "critique_index": cblock.critique_index,
                                    "decision": cblock.critique.model_dump(mode="python"),
                                    "raw_snippet": cblock.raw_critique_response[:2000],
                                }
                            )
                        st.json({"critiques": critiques_payload})


st.caption(
    "Phase 1 `app/agents/core_rag.py`; Phase 2 `app/agents/external_knowledge.py`; "
    "Phase 3 `app/agents/phase3_mcp.py`; Phase 4 `app/agents/refinement_loop.py`; tools "
    "`app/tools/` + `app/mcp/` MCP client; corpus `app/knowledge/store.py`; runner facades "
    "`app/agents/session_runner.py`."
)
