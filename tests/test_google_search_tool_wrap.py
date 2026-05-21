"""Factory for Gemini Google Search tool wrapper."""

from __future__ import annotations

from app.tools.google_search_tool import make_google_web_search_tool
from google.adk.tools.google_search_tool import GoogleSearchTool


def test_google_web_tool_respects_multi_tool_bypass():
    tool = make_google_web_search_tool(model_name="gemini-test", bypass_multi_tools_limit=True)
    assert isinstance(tool, GoogleSearchTool)
    assert tool.bypass_multi_tools_limit is True
    assert tool.model == "gemini-test"


def test_google_web_tool_can_disable_multi_tool_bypass():
    tool = make_google_web_search_tool(model_name="gemini-other", bypass_multi_tools_limit=False)
    assert tool.bypass_multi_tools_limit is False
