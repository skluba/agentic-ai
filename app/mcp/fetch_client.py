"""Call the MCP ``fetch`` tool over stdio — one subprocess session per batch."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import mcp.types as mcp_types
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client


def call_tool_result_to_payload(result: mcp_types.CallToolResult) -> dict[str, Any]:
    """Flatten ``CallToolResult`` into JSON-friendly fields."""
    texts: list[str] = []
    for block in result.content:
        btype = getattr(block, "type", None)
        if btype == "text":
            texts.append(str(getattr(block, "text", "")))
        else:
            texts.append(block.model_dump_json())
    joined = "\n".join(t for t in texts if t).strip()
    if result.isError:
        return {"ok": False, "error": joined or "mcp_fetch_error", "is_error": True}
    return {"ok": True, "markdown": joined, "is_error": False}


async def fetch_keyed_urls_via_mcp(
    server: StdioServerParameters,
    keyed_urls: Mapping[str, str],
    *,
    max_length_per_url: int = 6000,
    start_index: int = 0,
) -> dict[str, Any]:
    """Run ``fetch`` for each ``key → url`` inside a single stdio MCP session."""
    out: dict[str, Any] = {}
    async with stdio_client(server) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            for key, url in keyed_urls.items():
                result = await session.call_tool(
                    "fetch",
                    {
                        "url": url,
                        "max_length": max_length_per_url,
                        "start_index": start_index,
                        "raw": False,
                    },
                )
                out[key] = call_tool_result_to_payload(result)
    return out
