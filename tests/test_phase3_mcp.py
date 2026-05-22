"""Phase 3 agent wiring and MCP financial tool mocks."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import patch

from app.agents.phase3_mcp import create_phase3_mcp_agent
from app.config import Settings
from app.knowledge import FakeEmbeddingBackend, KnowledgeCorpus
from app.mcp.fetch_client import call_tool_result_to_payload
from app.tools.financial_markets_mcp_tool import (
    urls_for_segments,
)
from google.adk.tools.google_search_tool import GoogleSearchTool
from mcp.types import CallToolResult, TextContent


def test_urls_for_segments_subsets_full_map():
    assert set(urls_for_segments("all").keys()) == {
        "us_stocks_most_active",
        "crypto_all",
        "currencies",
    }
    assert list(urls_for_segments("stocks").keys()) == ["us_stocks_most_active"]
    assert list(urls_for_segments("crypto").keys()) == ["crypto_all"]
    assert list(urls_for_segments("currencies").keys()) == ["currencies"]


def test_call_tool_result_payload_ok_and_error():
    ok_res = CallToolResult(
        content=[TextContent(type="text", text=" markdown body ")], isError=False
    )
    pay = call_tool_result_to_payload(ok_res)
    assert pay["ok"] is True
    assert "markdown body" in pay["markdown"]

    err_res = CallToolResult(content=[TextContent(type="text", text="boom")], isError=True)
    err_pay = call_tool_result_to_payload(err_res)
    assert err_pay["ok"] is False
    assert err_pay["is_error"] is True


async def fake_fetch_bundle(*_args, **_kwargs):  # noqa: ANN001
    await asyncio.sleep(0)
    return {
        "us_stocks_most_active": {"ok": True, "markdown": "# mocked tickers"},
    }


def test_financial_tool_happy_json():
    settings = Settings(gcp_project_id="phase3-fin")

    async def run_tool() -> str:
        from app.tools.financial_markets_mcp_tool import make_financial_markets_mcp_tool

        tool_fn = make_financial_markets_mcp_tool(settings)
        with patch(
            "app.tools.financial_markets_mcp_tool.fetch_keyed_urls_via_mcp",
            side_effect=fake_fetch_bundle,
        ):
            return await tool_fn(segments="stocks", max_length_per_url=4000)

    raw = asyncio.run(run_tool())
    decoded = json.loads(raw)
    assert decoded["segments_requested"] == "stocks"
    assert decoded["mcp_fetch_by_source"]["us_stocks_most_active"]["markdown"]


def test_phase3_tools_finance_then_search_always():
    settings = Settings(gcp_project_id="phase3-wire", embedding_dimension=8)
    empty = KnowledgeCorpus(FakeEmbeddingBackend(8))

    agent = create_phase3_mcp_agent(settings, empty)
    assert len(agent.tools) == 2
    finance_tool, builtin = agent.tools
    assert callable(finance_tool)
    assert isinstance(builtin, GoogleSearchTool)


def test_phase3_with_corpus_adds_documents():
    settings = Settings(gcp_project_id="phase3-wire3", embedding_dimension=16)
    corpus = KnowledgeCorpus(FakeEmbeddingBackend(16))
    assert corpus.ingest_text(doc_id="d", raw_text=("finance note " * 40)) >= 1

    agent = create_phase3_mcp_agent(settings, corpus)
    assert len(agent.tools) == 3
    doc_tool, finance_tool, builtin = agent.tools
    assert callable(doc_tool)
    assert callable(finance_tool)
    assert isinstance(builtin, GoogleSearchTool)


def test_financial_tool_serializes_spawn_oserror_json():
    settings = Settings(gcp_project_id="phase3-os")

    async def boom(*_args, **_kwargs):  # noqa: ANN001
        raise OSError(2, "no such executable")

    async def run_tool() -> str:
        from app.tools.financial_markets_mcp_tool import make_financial_markets_mcp_tool

        fn = make_financial_markets_mcp_tool(settings)
        with patch(
            "app.tools.financial_markets_mcp_tool.fetch_keyed_urls_via_mcp",
            side_effect=boom,
        ):
            return await fn(segments="currencies")

    decoded = json.loads(asyncio.run(run_tool()))
    assert decoded["ok"] is False
    assert "spawn_failed:" in decoded["error"]


async def exploding_fetch(*_args, **_kwargs):  # noqa: ANN001
    raise RuntimeError("mcp spawn failed")


def test_financial_tool_serializes_unexpected_errors():
    settings = Settings(gcp_project_id="phase3-err")

    async def run_tool() -> str:
        from app.tools.financial_markets_mcp_tool import make_financial_markets_mcp_tool

        fn = make_financial_markets_mcp_tool(settings)
        with patch(
            "app.tools.financial_markets_mcp_tool.fetch_keyed_urls_via_mcp",
            side_effect=exploding_fetch,
        ):
            return await fn(segments="crypto", max_length_per_url=1000)

    raw = asyncio.run(run_tool())
    decoded = json.loads(raw)
    assert decoded["ok"] is False
    assert "mcp spawn failed" in decoded["error"]
