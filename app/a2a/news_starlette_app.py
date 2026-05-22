"""Starlette wiring for News Agent A2A (agent card + REST routes)."""

from __future__ import annotations

from a2a.server.request_handlers.default_request_handler_v2 import DefaultRequestHandlerV2
from a2a.server.routes.agent_card_routes import create_agent_card_routes
from a2a.server.routes.rest_routes import create_rest_routes
from a2a.server.tasks import InMemoryTaskStore
from starlette.applications import Starlette

from app.a2a.news_agent_card import build_news_kb_agent_card
from app.a2a.news_executor import NewsKbA2AExecutor
from app.config import Settings
from app.knowledge.store import KnowledgeCorpus


def build_news_a2a_starlette_app(
    *,
    settings: Settings,
    corpus: KnowledgeCorpus,
    public_base_url: str,
) -> Starlette:
    """Return a Starlette app exposing /.well-known/agent-card.json and /message:send REST."""
    base = public_base_url.strip().rstrip("/")
    agent_card = build_news_kb_agent_card(public_base_url=base)
    handler = DefaultRequestHandlerV2(
        agent_executor=NewsKbA2AExecutor(settings, corpus),
        task_store=InMemoryTaskStore(),
        agent_card=agent_card,
    )

    routes: list = [*create_agent_card_routes(agent_card), *create_rest_routes(handler)]

    return Starlette(routes=routes)
