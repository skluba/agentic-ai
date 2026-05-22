"""Phase 6 — Phase-5 collaborator plus **`produce_structured_canvas`** artefacts."""

from __future__ import annotations

from google.adk import Agent

from app.agents.phase5_collaborative import PHASE_5_SYSTEM_PROMPT
from app.config import Settings, configure_google_genai_api_key_environment
from app.knowledge.store import KnowledgeCorpus
from app.tools.canvas_tool import make_canvas_delivery_tool
from app.tools.document_search_tool import make_document_search_tool
from app.tools.financial_markets_mcp_tool import make_financial_markets_mcp_tool
from app.tools.google_search_tool import make_google_web_search_tool
from app.tools.news_agent_a2a_tool import make_news_agent_a2a_tool

PHASE_6_CANVAS_PROMPT_TAIL = """

**Phase-6 Canvas (`produce_structured_canvas`) — planner routing**
1. Prefer research tools **first** (**`search_private_knowledge`**, **MCP finance**, **`WEB`**,\
 **`REMOTE_NEWS_AGENT`** as in Phase 5) whenever external facts materially answer the brief.
2. Call **`produce_structured_canvas`** when the user expressly wants a deliverable (**report**,\
 **brief**, **memo**, **HTML/HTML-friendly deck**, reproducible **`code`** sample) summarising\
 gathered insights — synthesise factual bullets into `markdown_body`, then invoke the tool\
 **once per** artefact variant (repeat only when the user asks for distinct formats such as\
 Markdown **and** HTML).
3. Parameters:
   - `output_kind` ∈ `markdown_report` | `html_report` | `code_snippet`
   - `title` terse human-readable heading
   - `markdown_body` fully structured prose (Markdown `## Sections`, grounded bullets)\
     **or executable code listing** when `code_snippet`
   - `programming_language` **required non-empty string** whenever `output_kind='code_snippet'`
4. In your natural-language reply briefly mention that the Canvas artefact JSON is authoritative\
 (`artifact` payload) alongside any narrative highlights.
"""


PHASE_6_SYSTEM_PROMPT = PHASE_5_SYSTEM_PROMPT + PHASE_6_CANVAS_PROMPT_TAIL


def create_phase6_canvas_agent(
    settings: Settings,
    corpus: KnowledgeCorpus,
    *,
    prompt: str | None = None,
) -> Agent:
    """Phase-5 tool belt plus Canvas delivery callable."""
    configure_google_genai_api_key_environment(settings)

    tools: list = []
    if corpus.chunk_count > 0:
        tools.append(make_document_search_tool(corpus))
    tools.append(make_financial_markets_mcp_tool(settings))
    tools.append(make_google_web_search_tool(model_name=settings.gemini_model))
    if settings.news_agent_a2a_base_url.strip():
        tools.append(make_news_agent_a2a_tool(settings))
    tools.append(make_canvas_delivery_tool())

    return Agent(
        name="phase6_canvas_hybrid",
        description=(
            "Research stack from Phase 5 with structured Canvas artefacts (Markdown/HTML/code) "
            "for stakeholder-ready outputs."
        ),
        model=settings.gemini_model,
        instruction=prompt or PHASE_6_SYSTEM_PROMPT,
        tools=tools,
        mode="chat",
    )
