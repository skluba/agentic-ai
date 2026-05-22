"""Uvicorn entrypoint wiring for `docker compose`/local runs of the News Agent."""

from __future__ import annotations

from app.a2a.news_starlette_app import build_news_a2a_starlette_app
from app.config import get_settings
from app.knowledge import KnowledgeCorpus, build_embedder_from_settings


def build_news_starlette_application():  # noqa: ANN201 — ASGI callable for uvicorn
    """Compose Settings + Corpus + advertised public URL into the News A2A Starlette app."""
    settings = get_settings()
    public = settings.news_agent_public_base_url.strip()
    if not public:
        msg = (
            "Set NEWS_AGENT_PUBLIC_BASE_URL to the URL other agents/clients reach "
            "(e.g. http://news-agent:8090 in Compose)"
        )
        raise RuntimeError(msg)
    corpus = KnowledgeCorpus(
        embedder=build_embedder_from_settings(
            settings,
            offline=settings.news_agent_use_offline_embeddings,
        ),
    )
    return build_news_a2a_starlette_app(
        settings=settings,
        corpus=corpus,
        public_base_url=public,
    )


def uvicorn_news_application_factory():  # noqa: ANN201
    """``uvicorn app.a2a.news_service:uvicorn_news_application_factory --factory``"""
    return build_news_starlette_application()
