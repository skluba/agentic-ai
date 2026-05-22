"""Phase 5 — orchestrator-side tool invoking the standalone News Agent over A2A."""

from __future__ import annotations

import json

from app.config import Settings
from app.tools.news_agent_client import delegate_latest_news_via_a2a


def make_news_agent_a2a_tool(settings: Settings):
    """Expose an ADK-async callable wired to REST A2A (HTTP+JSON) News Agent."""

    async def delegate_to_news_kb_specialist_via_a2a(
        topic: str,
        time_horizon_hours: int = 72,
    ) -> str:
        """Delegate to the remote News Agent for **latest-news** briefings (A2A).

        Prefer this specialist when users want **recent developments**, breaking **headlines**,
        **coverage digests**, or **journalistic timelines** sourced from uploads on the News
        Agent host plus grounded web retrieval — avoiding duplicating Phase 3 MCP finance work.

        Provide a terse yet specific ``topic``. Control recency loosely with ``time_horizon_hours``
        when the timeframe matters (still advisory for the downstream model).

        Leave stock/crypto/FX table pulls to MCP; use WEB or this delegation for narrative news.
        """
        base_url = settings.news_agent_a2a_base_url.strip()
        if not base_url:
            return json.dumps(
                {"ok": False, "hint": "Set NEWS_AGENT_A2A_BASE_URL to reach the specialist."},
                ensure_ascii=False,
            )
        bounded = max(1, min(int(time_horizon_hours), 720))
        return await delegate_latest_news_via_a2a(
            base_url=base_url,
            topic=topic,
            time_horizon_hours=bounded,
            http_timeout_seconds=settings.news_agent_http_timeout_seconds,
        )

    return delegate_to_news_kb_specialist_via_a2a
