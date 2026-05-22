"""Unit coverage for MCP stdio wiring without spawning Docker."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from unittest.mock import MagicMock, patch

from app.mcp.fetch_client import call_tool_result_to_payload, fetch_keyed_urls_via_mcp
from mcp.client.stdio import StdioServerParameters
from mcp.types import CallToolResult, ImageContent, TextContent


def test_payload_serializes_non_text_block():
    res = CallToolResult(
        content=[ImageContent(type="image", data="eHh4", mimeType="image/png")],
        isError=False,
    )
    pay = call_tool_result_to_payload(res)
    json_chunk = pay["markdown"]
    assert "image/png" in json_chunk or "eHh4" in json_chunk


def test_fetch_keyed_urls_batches_calls():
    urls = {"a": "https://example.com/a", "b": "https://example.com/b"}
    invoked: list[tuple[str, dict]] = []

    class FakeSession:
        async def initialize(self) -> None:
            await asyncio.sleep(0)

        async def __aenter__(self) -> FakeSession:
            return self

        async def __aexit__(self, *_exc):  # noqa: ANN004
            return False

        async def call_tool(self, name: str, arguments: dict | None = None):  # noqa: ANN201
            await asyncio.sleep(0)
            invoked.append((name, arguments or {}))
            return CallToolResult(content=[TextContent(type="text", text="snippet")], isError=False)

    @asynccontextmanager
    async def fake_stdio(*_args, **_kwargs):
        await asyncio.sleep(0)
        yield (MagicMock(), MagicMock())

    session_cm = FakeSession()

    async def run() -> dict:
        with (
            patch("app.mcp.fetch_client.stdio_client", fake_stdio),
            patch("app.mcp.fetch_client.ClientSession", return_value=session_cm),
        ):
            return await fetch_keyed_urls_via_mcp(
                StdioServerParameters(command="noop", args=[]),
                urls,
                max_length_per_url=1234,
            )

    out = asyncio.run(run())

    assert [entry[1]["url"] for entry in invoked] == [urls["a"], urls["b"]]
    assert invoked[0][0] == "fetch"
    assert out["a"]["ok"] is True
