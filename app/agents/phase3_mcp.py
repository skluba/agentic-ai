"""Phase 3 — corpus + MCP Yahoo finance snapshots + optional hosted Google Search."""

from __future__ import annotations

from google.adk import Agent

from app.config import Settings, configure_google_genai_api_key_environment
from app.knowledge.store import KnowledgeCorpus
from app.tools.document_search_tool import make_document_search_tool
from app.tools.financial_markets_mcp_tool import make_financial_markets_mcp_tool
from app.tools.google_search_tool import make_google_web_search_tool

PHASE_3_SYSTEM_PROMPT = """You are Phase-3 research: **corpus grounding**, **MCP Yahoo finance**,\
 and **hosted Google Search** — one cohesive reply.

**Non-negotiable order (Plan → Execute silently)**
1. **Local-first.** If **`search_private_knowledge`** exists, call it before remote tools whenever\
 uploads could matter.
2. **Financial routing (planner)** — If the question is about **listed stocks**, **crypto/token\
 markets**, or **foreign exchange / currency pairs** (symbols, movers, benchmarks, Yahoo-style\
 listings):
   - Call **`fetch_yahoo_finance_markets_via_mcp`** with the narrowest **`segments`** that covers\
 the ask (`stocks` | `crypto` | `currencies` | `all`).
   - **Do not** launch hosted Google Search for those intents while MCP JSON shows usable\
 `markdown`. Treat MCP output as **primary market context**.
   - If MCP returns **`ok:false`** or empty payloads, say so briefly, then fall back (see §4).

3. **General / non-financial / broad research** — use **hosted Google Search** (**`WEB`**) instead\
 of MCP. Examples: news digests, docs, unrelated domains, troubleshooting.

4. **Fallbacks**
   - After failed MCP finance fetch, **`WEB`** is allowed as backup.
   - After empty corpus hits, rely on MCP or **`WEB`** as appropriate.

**Synthesis hygiene**
- When local snippets apply, say what uploads contain (**`PRIVATE_KB`**, **`chunk_id`** on quotes)\
 before external layers.
- Label MCP-derived prose **MCP / Yahoo tables**; label search **From web:** / **`WEB`**.
- Never invent tickers or prices missing from tool output.

Operational guardrails:
- Never hallucinate tool payloads; summarise only grounded tool outputs.
- If every layer is weak, state uncertainty and what evidence is missing.
"""


def create_phase3_mcp_agent(
    settings: Settings,
    corpus: KnowledgeCorpus,
    *,
    prompt: str | None = None,
) -> Agent:
    """ADK agent: optional corpus, Yahoo finance via MCP fetch, plus Google Search."""
    configure_google_genai_api_key_environment(settings)

    tools: list = []
    if corpus.chunk_count > 0:
        tools.append(make_document_search_tool(corpus))
    tools.append(make_financial_markets_mcp_tool(settings))
    tools.append(make_google_web_search_tool(model_name=settings.gemini_model))

    return Agent(
        name="phase3_mcp_hybrid",
        description=(
            "Routes finance questions to MCP Yahoo snapshots; general questions to hosted search; "
            "optional corpus first."
        ),
        model=settings.gemini_model,
        instruction=prompt or PHASE_3_SYSTEM_PROMPT,
        tools=tools,
        mode="chat",
    )
