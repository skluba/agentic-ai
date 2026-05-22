"""Minimal News specialist — corpus-first retrieval plus hosted Google Search.

This agent backs the standalone **News Agent** A2A endpoint (Phase 5). It deliberately
scopes instructions to topical **news briefing** workloads so the orchestrator agent
knows what to delegate remotely.
"""

from __future__ import annotations

from google.adk import Agent
from google.adk.apps.app import App
from google.adk.events.event import Event
from google.adk.runners import Runner
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.genai import types

from app.config import Settings, configure_google_genai_api_key_environment
from app.knowledge.store import KnowledgeCorpus
from app.tools.document_search_tool import make_document_search_tool
from app.tools.google_search_tool import make_google_web_search_tool

NEWS_FOCUS_SYSTEM_PROMPT = """You are a **focused news-retrieval analyst** answering in one reply.

Mission
- Produce a **latest-news style briefing** grounded in uploads on this agent's knowledge\
 base (when available) plus **hosted Google Search** for recency/breadth.

Order
1. If **`search_private_knowledge`** is available → query it once with sharp keywords (**news**,\
 **timeline**, headline nouns) whenever uploads could conceivably cover the user's topic.
2. Use **`google_search`** to gather **recent** corroborating coverage and fill gaps;\
 prefer clearly dated items for "what changed recently" queries.
3. If the corpus is empty or yielded no snippets, acknowledge that succinctly — then rely\
 wholly on **`WEB`** for the briefing.

Formatting
- Start with **`NEWS_AGENT · KB`** bullets when uploads helped; **`NEWS_AGENT · WEB`**\
 for web-only augmentation.
- If evidence is thin, spell out what remains unknown.

Operational guardrails
- Quote or paraphrase only from tool payloads; invent no outlets, dates, or quotes.
"""


def _user_turn_news(text: str) -> types.Content:
    trimmed = text.strip()
    return types.Content(role="user", parts=[types.Part(text=trimmed)])


def create_news_kb_agent(
    settings: Settings, corpus: KnowledgeCorpus, *, prompt: str | None = None
) -> Agent:
    """ADK agent specialising in corpus + web news summaries for the A2A News server."""
    configure_google_genai_api_key_environment(settings)
    tools: list = []
    if corpus.chunk_count > 0:
        tools.append(make_document_search_tool(corpus))
    tools.append(make_google_web_search_tool(model_name=settings.gemini_model))

    return Agent(
        name="news_kb_standalone",
        description=(
            "Remote News Agent: condensed topical briefings grounded in corpus hits "
            "(when ingested server-side) and hosted Google Search."
        ),
        model=settings.gemini_model,
        instruction=prompt or NEWS_FOCUS_SYSTEM_PROMPT,
        tools=tools,
        mode="chat",
    )


async def run_news_kb_turn(
    *,
    settings: Settings,
    corpus: KnowledgeCorpus,
    payload: str,
    user_id: str = "news-agent-user",
    session_id: str = "news-kb-session",
    app_label: str = "news_kb_a2a",
) -> tuple[str, list[Event]]:
    """Single ADK Runner turn producing the textual answer for the News A2A executor."""
    from app.agents.session_runner import concatenate_agent_text  # defer import

    agent = create_news_kb_agent(settings, corpus)
    app_wrapper = App(name=app_label, root_agent=agent)
    runner = Runner(
        app=app_wrapper,
        session_service=InMemorySessionService(),
        auto_create_session=True,
    )

    events: list[Event] = []
    async for event in runner.run_async(
        user_id=user_id,
        session_id=session_id,
        new_message=_user_turn_news(payload),
        yield_user_message=False,
    ):
        events.append(event)

    return concatenate_agent_text(events), events
