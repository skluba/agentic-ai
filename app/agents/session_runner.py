"""Utility helpers around ADK Runner for quick CLI/Streamlit turns."""

from __future__ import annotations

from typing import Any

from google.adk.apps.app import App
from google.adk.events.event import Event
from google.adk.runners import Runner
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.genai import types

from app.agents.core_rag import create_core_rag_agent
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
