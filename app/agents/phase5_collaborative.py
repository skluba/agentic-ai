"""Phase 5 — collaborative orchestrator with A2A News Agent delegation."""

from __future__ import annotations

from google.adk import Agent

from app.agents.phase3_mcp import PHASE_3_SYSTEM_PROMPT
from app.config import Settings, configure_google_genai_api_key_environment
from app.knowledge.store import KnowledgeCorpus
from app.tools.document_search_tool import make_document_search_tool
from app.tools.financial_markets_mcp_tool import make_financial_markets_mcp_tool
from app.tools.google_search_tool import make_google_web_search_tool
from app.tools.news_agent_a2a_tool import make_news_agent_a2a_tool

PHASE_5_COLLAB_PROMPT_TAIL = """

**Phase-5 collaborator (A2A News specialist)** — **`delegate_to_news_kb_specialist_via_a2a`**
- Prefer this tool when questions clearly ask for **topical/newsroom-style briefings**, **coverage\
 digests**, or **what happened recently / breaking updates** spanning sources — not when the\
 user wants canonical Yahoo mover tables (**§2** MCP path still owns those intents).
- Pass a terse **focused topic line** (`topic` argument); widen `time_horizon_hours` if the caller\
 cites **days/weeks** of history (tool caps hours).
- If the tool returns JSON **`{"ok": false` ... `hint`}** signalling routing is unavailable —\
 continue with MCP + **`WEB`** as in Phase 3 guidance.
- In synthesis, cite remote answers as **`REMOTE_NEWS_AGENT`** and reconcile with MCP / corpus /
 **`WEB`**.
"""


PHASE_5_SYSTEM_PROMPT = PHASE_3_SYSTEM_PROMPT + PHASE_5_COLLAB_PROMPT_TAIL


def create_phase5_collaborative_agent(
    settings: Settings,
    corpus: KnowledgeCorpus,
    *,
    prompt: str | None = None,
) -> Agent:
    """Phase 3 tooling plus optional A2A News delegation."""
    configure_google_genai_api_key_environment(settings)

    tools: list = []
    if corpus.chunk_count > 0:
        tools.append(make_document_search_tool(corpus))
    tools.append(make_financial_markets_mcp_tool(settings))
    tools.append(make_google_web_search_tool(model_name=settings.gemini_model))
    if settings.news_agent_a2a_base_url.strip():
        tools.append(make_news_agent_a2a_tool(settings))

    return Agent(
        name="phase5_collaborative_hybrid",
        description=(
            "Phase-3 MCP + corpus + grounded search orchestrator delegating topical news bursts "
            "to a standalone A2A News Agent when reachable."
        ),
        model=settings.gemini_model,
        instruction=prompt or PHASE_5_SYSTEM_PROMPT,
        tools=tools,
        mode="chat",
    )
