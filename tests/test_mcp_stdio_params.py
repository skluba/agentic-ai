"""stdio launch presets for MCP fetch server."""

from __future__ import annotations

import sys

from app.config import Settings
from app.mcp.stdio_params import stdio_parameters_for_fetch_server, stdio_python_mcp_fetch_server


def test_stdio_docker_launch_params():
    s = Settings(
        gcp_project_id="t-docker",
        mcp_financial_fetch_transport="docker",
        mcp_financial_docker_image="mcp/fetch",
    )
    p = stdio_parameters_for_fetch_server(s)
    assert p.command == "docker"
    assert p.args == ["run", "-i", "--rm", "mcp/fetch"]


def test_stdio_uvx_launch_params():
    s = Settings(gcp_project_id="t-uv", mcp_financial_fetch_transport="uvx")
    p = stdio_parameters_for_fetch_server(s)
    assert p.command == "uvx"
    assert p.args == ["mcp-server-fetch"]


def test_stdio_python_launch_uses_executable():
    s = Settings(gcp_project_id="t-py", mcp_financial_fetch_transport="python")
    p = stdio_parameters_for_fetch_server(s)
    assert p.command == sys.executable
    assert p.args == ["-m", "mcp_server_fetch"]


def test_stdio_python_mcp_fetch_server_matches_settings_python_mode():
    p = stdio_python_mcp_fetch_server()
    assert p.command == sys.executable
    assert p.args == ["-m", "mcp_server_fetch"]
