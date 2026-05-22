"""Gemini/Google Search grounding for ADK hybrid agents."""

from __future__ import annotations

from google.adk.tools.google_search_tool import GoogleSearchTool


def make_google_web_search_tool(
    *, model_name: str, bypass_multi_tools_limit: bool = True
) -> GoogleSearchTool:
    """Gemini-hosted Google Search for hybrid ADK stacks.

    When ``bypass_multi_tools_limit`` is ``True``, ADK can pair grounding with corpus\
    **`search_private_knowledge`** in one agent.
    """
    return GoogleSearchTool(
        bypass_multi_tools_limit=bypass_multi_tools_limit,
        model=model_name,
    )
