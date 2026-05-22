"""Map :class:`~app.config.Settings` to MCP stdio launch parameters."""

from __future__ import annotations

import sys

from mcp.client.stdio import StdioServerParameters

from app.config import Settings


def stdio_parameters_for_fetch_server(settings: Settings) -> StdioServerParameters:
    """How to spawn the reference **mcp-server-fetch** (Docker, uvx, or ``python -m``).

    Defaults match Anthropic docs: ``docker run -i --rm mcp/fetch``.
    """
    transport = settings.mcp_financial_fetch_transport
    if transport == "docker":
        image = settings.mcp_financial_docker_image.strip() or "mcp/fetch"
        return StdioServerParameters(command="docker", args=["run", "-i", "--rm", image])
    if transport == "uvx":
        return StdioServerParameters(command="uvx", args=["mcp-server-fetch"])
    # python module (requires optional ``mcp-server-fetch`` pip package)
    return StdioServerParameters(command=sys.executable, args=["-m", "mcp_server_fetch"])
