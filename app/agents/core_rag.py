"""Google ADK root agent wired for corpus-grounded QA (Phase 1 MVP)."""

from __future__ import annotations

from google.adk import Agent

from app.config import Settings
from app.knowledge.store import KnowledgeCorpus
from app.tools.document_search_tool import make_document_search_tool

PHASE_1_SYSTEM_PROMPT = """You are Phase-1 CORE RAG — you answer ONLY from snippets returned by \
tooling.

Operate using an internal Plan → Execute → Synthesize discipline:
1. PLAN silently: clarify factual sub-claims against the corpus. Do NOT print this plan verbatim.
2. EXECUTE: call `search_private_knowledge` when the user needs factual text from documents. \
Rewrite short focused retrieval queries (concrete nouns, product names). If hits are thin, refine \
and call again once at most.
3. SYNTHESIZE: craft the final answer using exclusively snippet text. When quoting, include \
`chunk_id` values from tool JSON.

If tooling returns `"hits": []`, say you cannot find matching evidence instead of hallucinating."""


def create_core_rag_agent(
    settings: Settings,
    corpus: KnowledgeCorpus,
    *,
    prompt: str | None = None,
) -> Agent:
    """Single ADK Agent with corpus-bound document search tooling."""
    if corpus.chunk_count == 0:
        msg = "Cannot create core rag agent until the corpus ingests at least one chunk."
        raise ValueError(msg)

    return Agent(
        name="core_rag",
        description="Answers questions using chunked private documents + FAISS retrieval.",
        model=settings.gemini_model,
        instruction=prompt or PHASE_1_SYSTEM_PROMPT,
        tools=[make_document_search_tool(corpus)],
        mode="chat",
    )
