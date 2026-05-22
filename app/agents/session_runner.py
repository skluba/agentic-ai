"""Utility helpers around ADK Runner for quick CLI/Streamlit turns."""

from __future__ import annotations

from typing import Any

from google.adk.apps.app import App
from google.adk.events.event import Event
from google.adk.runners import Runner
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.genai import types

from app.agents.core_rag import create_core_rag_agent
from app.agents.external_knowledge import create_external_knowledge_agent
from app.agents.phase3_mcp import create_phase3_mcp_agent
from app.agents.phase5_collaborative import create_phase5_collaborative_agent
from app.agents.phase6_canvas import create_phase6_canvas_agent
from app.config import Settings
from app.knowledge.store import KnowledgeCorpus


def _user_turn(text: str) -> types.Content:
    trimmed = text.strip()
    return types.Content(role="user", parts=[types.Part(text=trimmed)])


def concatenate_agent_text(events: list[Event]) -> str:
    """Collect textual model parts authored by orchestrated agents."""
    pieces: list[str] = []
    for event in events:
        author = getattr(event, "author", "")
        if not author or author == "user":
            continue
        content = getattr(event, "content", None)
        if not content:
            continue
        for part in content.parts:
            if part.text:
                pieces.append(part.text)
    return "\n".join(pieces).strip()


async def run_core_rag_turn(
    *,
    settings: Settings,
    corpus: KnowledgeCorpus,
    question: str,
    user_id: str = "local-user",
    session_id: str = "core-session",
    app_label: str = "core_rag_mvp",
) -> tuple[str, list[Event]]:
    """Execute a single conversational turn returning assistant text plus raw events."""
    agent = create_core_rag_agent(settings, corpus)
    app = App(name=app_label, root_agent=agent)
    runner = Runner(
        app=app,
        session_service=InMemorySessionService(),
        auto_create_session=True,
    )

    events: list[Event] = []
    async for event in runner.run_async(
        user_id=user_id,
        session_id=session_id,
        new_message=_user_turn(question),
        yield_user_message=False,
    ):
        events.append(event)

    return concatenate_agent_text(events), events


def run_core_rag_turn_sync(
    **kwargs: Any,
) -> tuple[str, list[Event]]:
    """Blocking helper for notebooks / Streamlit."""
    import asyncio

    return asyncio.run(run_core_rag_turn(**kwargs))


async def run_phase2_external_turn(
    *,
    settings: Settings,
    corpus: KnowledgeCorpus,
    question: str,
    user_id: str = "local-user",
    session_id: str = "phase2-session",
    app_label: str = "phase2_external",
) -> tuple[str, list[Event]]:
    """Phase-2 conversational turn — private corpus (optional) plus Google Search grounding."""
    agent = create_external_knowledge_agent(settings, corpus)
    app = App(name=app_label, root_agent=agent)
    runner = Runner(
        app=app,
        session_service=InMemorySessionService(),
        auto_create_session=True,
    )

    events: list[Event] = []
    async for event in runner.run_async(
        user_id=user_id,
        session_id=session_id,
        new_message=_user_turn(question),
        yield_user_message=False,
    ):
        events.append(event)

    return concatenate_agent_text(events), events


def run_phase2_external_turn_sync(**kwargs: Any) -> tuple[str, list[Event]]:
    """Blocking facade for notebooks / Streamlit."""
    import asyncio

    return asyncio.run(run_phase2_external_turn(**kwargs))


async def run_phase3_mcp_turn(
    *,
    settings: Settings,
    corpus: KnowledgeCorpus,
    question: str,
    user_id: str = "local-user",
    session_id: str = "phase3-session",
    app_label: str = "phase3_mcp",
) -> tuple[str, list[Event]]:
    """Phase-3 turn — corpus (optional), MCP Yahoo finance snapshots, plus Google Search."""
    agent = create_phase3_mcp_agent(settings, corpus)
    app = App(name=app_label, root_agent=agent)
    runner = Runner(
        app=app,
        session_service=InMemorySessionService(),
        auto_create_session=True,
    )

    events: list[Event] = []
    async for event in runner.run_async(
        user_id=user_id,
        session_id=session_id,
        new_message=_user_turn(question),
        yield_user_message=False,
    ):
        events.append(event)

    return concatenate_agent_text(events), events


def run_phase3_mcp_turn_sync(**kwargs: Any) -> tuple[str, list[Event]]:
    """Blocking façade for notebooks / Streamlit."""
    import asyncio

    return asyncio.run(run_phase3_mcp_turn(**kwargs))


async def run_phase5_collaborative_turn(
    *,
    settings: Settings,
    corpus: KnowledgeCorpus,
    question: str,
    user_id: str = "local-user",
    session_id: str = "phase5-session",
    app_label: str = "phase5_collab",
) -> tuple[str, list[Event]]:
    """Phase-5 turn — Phase-3 toolkit plus delegated A2A News briefing tool when configured."""
    agent = create_phase5_collaborative_agent(settings, corpus)
    app = App(name=app_label, root_agent=agent)
    runner = Runner(
        app=app,
        session_service=InMemorySessionService(),
        auto_create_session=True,
    )

    events: list[Event] = []
    async for event in runner.run_async(
        user_id=user_id,
        session_id=session_id,
        new_message=_user_turn(question),
        yield_user_message=False,
    ):
        events.append(event)

    return concatenate_agent_text(events), events


def run_phase5_collaborative_turn_sync(**kwargs: Any) -> tuple[str, list[Event]]:
    """Blocking façade for notebooks / Streamlit."""
    import asyncio

    return asyncio.run(run_phase5_collaborative_turn(**kwargs))


async def run_phase6_canvas_turn(
    *,
    settings: Settings,
    corpus: KnowledgeCorpus,
    question: str,
    user_id: str = "local-user",
    session_id: str = "phase6-session",
    app_label: str = "phase6_canvas",
) -> tuple[str, list[Event]]:
    """Phase 6 — Phase-5 collaborator plus **`produce_structured_canvas`** artefacts."""
    agent = create_phase6_canvas_agent(settings, corpus)
    app = App(name=app_label, root_agent=agent)
    runner = Runner(
        app=app,
        session_service=InMemorySessionService(),
        auto_create_session=True,
    )

    events: list[Event] = []
    async for event in runner.run_async(
        user_id=user_id,
        session_id=session_id,
        new_message=_user_turn(question),
        yield_user_message=False,
    ):
        events.append(event)

    return concatenate_agent_text(events), events


def run_phase6_canvas_turn_sync(**kwargs: Any) -> tuple[str, list[Event]]:
    """Blocking façade for notebooks / Streamlit."""
    import asyncio

    return asyncio.run(run_phase6_canvas_turn(**kwargs))
