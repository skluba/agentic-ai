"""Phase 5 — News Agent card, collaborator wiring, executor + ASGI agent-card smoke."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from a2a.types.a2a_pb2 import AgentCard, Message
from app.a2a.news_agent_card import build_news_kb_agent_card
from app.a2a.news_executor import NewsKbA2AExecutor
from app.a2a.news_starlette_app import build_news_a2a_starlette_app
from app.agents.phase3_mcp import create_phase3_mcp_agent
from app.agents.phase5_collaborative import create_phase5_collaborative_agent
from app.config import Settings
from app.knowledge import FakeEmbeddingBackend, KnowledgeCorpus
from google.adk.tools.google_search_tool import GoogleSearchTool
from google.protobuf.json_format import ParseDict
from httpx import ASGITransport, AsyncClient


def test_news_kb_agent_card_rest_interface():
    card = build_news_kb_agent_card(public_base_url="http://localhost:8090/")
    assert card.name == "news-knowledge-agent"
    assert card.supported_interfaces
    assert "HTTP" in card.supported_interfaces[0].protocol_binding


def test_phase5_adds_news_tool_when_url_configured():
    settings = Settings(
        gcp_project_id="phase5-wire",
        embedding_dimension=8,
        news_agent_a2a_base_url=" http://news-agent:8090 ",
    )
    empty_corpus = KnowledgeCorpus(FakeEmbeddingBackend(8))
    orchestrator = create_phase5_collaborative_agent(settings, empty_corpus)
    finance_tool, builtin, third = orchestrator.tools
    assert callable(finance_tool)
    assert isinstance(builtin, GoogleSearchTool)
    assert callable(third)


def test_phase5_matches_phase3_when_news_url_blank():
    base = Settings(gcp_project_id="phase5-wire2", embedding_dimension=16)
    corpus = KnowledgeCorpus(FakeEmbeddingBackend(16))
    assert corpus.ingest_text(doc_id="d", raw_text=("economy " * 40)) >= 1

    orch = create_phase5_collaborative_agent(base, corpus)
    ref = create_phase3_mcp_agent(base, corpus)
    assert len(orch.tools) == len(ref.tools)
    orch_types = [type(t).__name__ for t in orch.tools]
    ref_types = [type(t).__name__ for t in ref.tools]
    assert orch_types == ref_types


@pytest.mark.asyncio
async def test_news_executor_queues_reply(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = Settings(gcp_project_id="phase5-exec", embedding_dimension=8)
    corpus = KnowledgeCorpus(FakeEmbeddingBackend(8))
    executor = NewsKbA2AExecutor(settings, corpus)
    event_queue = AsyncMock()

    monkeypatch.setattr(
        "app.a2a.news_executor.run_news_kb_turn",
        AsyncMock(return_value=("SYNTHESIS_TEXT", [])),
    )

    ctx = MagicMock()
    ctx.task_id = "tid-test"
    ctx.context_id = "ctx-test"
    ctx.get_user_input = MagicMock(return_value="   solar tariffs news   ")

    await executor.execute(ctx, event_queue)

    event_queue.enqueue_event.assert_awaited_once()
    (posted,) = event_queue.enqueue_event.call_args[0]
    assert isinstance(posted, Message)


@pytest.mark.asyncio
async def test_a2a_agent_card_well_known(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = Settings(gcp_project_id="phase5-http", embedding_dimension=8)
    corpus = KnowledgeCorpus(FakeEmbeddingBackend(8))
    app_asgi = build_news_a2a_starlette_app(
        settings=settings,
        corpus=corpus,
        public_base_url="http://testserver",
    )

    monkeypatch.setattr(
        "app.a2a.news_executor.run_news_kb_turn",
        AsyncMock(return_value=("unused", [])),
    )

    transport = ASGITransport(app=app_asgi)
    async with AsyncClient(transport=transport, base_url="http://testserver") as ac:
        card_resp = await ac.get("/.well-known/agent-card.json", headers={"A2A-Version": "1.0"})
        assert card_resp.status_code == 200
        card = ParseDict(card_resp.json(), AgentCard(), ignore_unknown_fields=True)
        assert card.skills
