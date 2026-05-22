"""Agent-to-Agent helpers (standalone News Agent over the A2A HTTP surface)."""

from app.a2a.news_starlette_app import build_news_a2a_starlette_app

__all__ = ["build_news_a2a_starlette_app"]
