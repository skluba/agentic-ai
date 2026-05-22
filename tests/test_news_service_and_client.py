"""Unit coverage for standalone News uvicorn wiring and outbound A2A client."""

from __future__ import annotations

import asyncio

import pytest
from app.a2a import news_service
from app.config import Settings, clear_settings_cache
from app.tools import news_agent_client


@pytest.mark.asyncio
async def test_delegate_latest_news_short_circuits_empty_topic() -> None:
    out = await news_agent_client.delegate_latest_news_via_a2a(
        base_url="http://example.test",
        topic="   ",
    )
    assert "empty topic" in out.lower()


@pytest.mark.asyncio
async def test_delegate_latest_news_streams_message(monkeypatch: pytest.MonkeyPatch) -> None:
    from a2a.helpers.proto_helpers import new_text_message
    from a2a.types.a2a_pb2 import Role, StreamResponse

    class FakeClient:
        async def send_message(self, _req):  # noqa: ANN001
            await asyncio.sleep(0)
            msg = new_text_message("remote digest", role=Role.ROLE_AGENT)
            chunk = StreamResponse()
            chunk.message.CopyFrom(msg)
            yield chunk

        async def close(self) -> None:
            """Async close hook exercised by delegates; noop for stubs."""
            await asyncio.sleep(0)

    class FakeFactory:
        """Minimal stand-in for ClientFactory in unit tests."""

        def __init__(self, _cfg=None) -> None:
            _ = _cfg  # constructor mirrors production arity only

        async def create_from_url(self, _url: str) -> FakeClient:
            await asyncio.sleep(0)
            return FakeClient()

    monkeypatch.setattr(news_agent_client, "ClientFactory", FakeFactory)

    out = await news_agent_client.delegate_latest_news_via_a2a(
        base_url="http://unittest.local",
        topic="AI regs",
        http_timeout_seconds=30.0,
    )
    assert "remote digest" in out


def test_news_service_builds_starlette_when_public_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NEWS_AGENT_PUBLIC_BASE_URL", "http://standalone-news:8099")
    clear_settings_cache()
    try:
        app = news_service.build_news_starlette_application()
        assert app is not None
    finally:
        monkeypatch.delenv("NEWS_AGENT_PUBLIC_BASE_URL", raising=False)
        clear_settings_cache()


def test_uvicorn_factory_is_build_alias(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NEWS_AGENT_PUBLIC_BASE_URL", "http://alias-test:8090")
    clear_settings_cache()
    try:
        assert news_service.uvicorn_news_application_factory() is not None
    finally:
        monkeypatch.delenv("NEWS_AGENT_PUBLIC_BASE_URL", raising=False)
        clear_settings_cache()


def test_news_service_requires_public_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        news_service,
        "get_settings",
        lambda: Settings(gcp_project_id="t", embedding_dimension=8, news_agent_public_base_url=""),
    )
    with pytest.raises(RuntimeError, match="NEWS_AGENT_PUBLIC_BASE_URL"):
        news_service.build_news_starlette_application()


@pytest.mark.asyncio
async def test_news_a2a_tool_without_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NEWS_AGENT_A2A_BASE_URL", raising=False)

    settings = Settings(
        gcp_project_id="phase5-tools",
        embedding_dimension=8,
        news_agent_a2a_base_url="",
    )
    from app.tools.news_agent_a2a_tool import make_news_agent_a2a_tool

    fn = make_news_agent_a2a_tool(settings)
    raw = await fn(topic="anything")
    assert "false" in raw.lower()
