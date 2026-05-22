"""Runtime configuration sourced from environment (GCP, Gemini, Langfuse)."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central settings for GCP Vertex Gemini, retrieval, observability."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    gcp_project_id: str = Field(
        default="gd-gcp-gridu-genai",
        validation_alias=AliasChoices("GOOGLE_CLOUD_PROJECT", "GCP_PROJECT_ID"),
    )
    vertex_location: str = Field(default="europe-west4", validation_alias="VERTEX_LOCATION")
    gemini_model: str = Field(default="gemini-2.0-flash", validation_alias="GEMINI_MODEL")

    gemini_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("GEMINI_API_KEY", "GOOGLE_API_KEY"),
        description=(
            "AI Studio Gemini API key. Copied into GOOGLE_API_KEY for google-genai (Developer API)."
        ),
    )

    embedding_model: str = Field(
        default="text-embedding-004",
        validation_alias="EMBEDDING_MODEL",
    )
    embedding_dimension: int = Field(
        default=768,
        ge=8,
        validation_alias=AliasChoices("EMBEDDING_DIMENSION", "FAISS_EMBEDDING_DIMENSION"),
    )

    langfuse_public_key: str | None = Field(default=None, validation_alias="LANGFUSE_PUBLIC_KEY")
    langfuse_secret_key: str | None = Field(default=None, validation_alias="LANGFUSE_SECRET_KEY")
    langfuse_host: str | None = Field(default=None, validation_alias="LANGFUSE_HOST")

    mcp_financial_fetch_transport: Literal["docker", "uvx", "python"] = Field(
        default="docker",
        validation_alias=AliasChoices(
            "MCP_FINANCIAL_FETCH_TRANSPORT",
            "PHASE3_MCP_FETCH_TRANSPORT",
        ),
        description=(
            "Launch mcp-server-fetch via Docker, uvx, or python -m (requires pip "
            "`mcp-server-fetch` when using python)."
        ),
    )
    mcp_financial_docker_image: str = Field(
        default="mcp/fetch",
        validation_alias="MCP_FINANCIAL_DOCKER_IMAGE",
        description="Docker image for MCP fetch when transport=docker.",
    )

    news_agent_a2a_base_url: str = Field(
        default="",
        validation_alias=AliasChoices("NEWS_AGENT_A2A_BASE_URL", "NEWS_AGENT_URL"),
        description=(
            "Base HTTP URL reachable by the orchestrator for the standalone News Agent A2A card."
        ),
    )
    news_agent_public_base_url: str = Field(
        default="",
        validation_alias=AliasChoices("NEWS_AGENT_PUBLIC_BASE_URL"),
        description=(
            "Public URL echoed in /.well-known; must equal the REST base callers use to reach it."
        ),
    )
    news_agent_use_offline_embeddings: bool = Field(
        default=True,
        validation_alias=AliasChoices("NEWS_AGENT_USE_OFFLINE_EMBEDDINGS"),
        description="Deterministic embeddings for the News Agent process (offline toggle).",
    )
    news_agent_http_timeout_seconds: float = Field(
        default=120.0,
        ge=5.0,
        le=900.0,
        validation_alias=AliasChoices("NEWS_AGENT_HTTP_TIMEOUT_SECONDS"),
        description="HTTP timeout seconds for outbound A2A tooling from the orchestrator.",
    )


@lru_cache
def get_settings(**overrides: Any) -> Settings:
    """Cached settings; optional overrides simplify tests without env coupling."""
    if not overrides:
        return Settings()
    return Settings(**overrides)


def clear_settings_cache() -> None:
    """Drop cached Settings (tests)."""
    get_settings.cache_clear()


def news_agent_a2a_url_host_resolution_hint(settings: Settings) -> str | None:
    """Explain common DNS failure when NEWS_AGENT_* uses Compose service hostname on host.

    Errno (-2 name not known / getaddrinfo) happens when Streamlit runs on the laptop while
    ``NEWS_AGENT_A2A_BASE_URL`` points at ``http://news-agent:8090`` — that hostname exists only on
    Docker's embedded DNS for sibling containers.

    Detection uses ``/.dockerenv`` — enough for Compose / typical cloud images; obscure container
    runtimes might omit it, in which case we simply skip this hint.

    Args:
        settings: Active Settings for the orchestrator process.

    Returns:
        Markdown-safe message for Streamlit/help text, or None if no heuristic applies.
    """
    raw = (settings.news_agent_a2a_base_url or "").strip()
    if not raw:
        return None
    try:
        hostname = (urlparse(raw).hostname or "").lower()
    except ValueError:
        return None
    if hostname != "news-agent":
        return None
    if Path("/.dockerenv").exists():
        return None
    return (
        "`NEWS_AGENT_A2A_BASE_URL` uses hostname `news-agent`, which resolves only "
        "**inside Docker Compose** (the internal DNS name for the `news-agent` service). "
        "This Streamlit process is running on the **host** — use "
        "**`NEWS_AGENT_A2A_BASE_URL=http://localhost:8090`** in `.env` (ensure the Compose "
        "service publishes **`8090:8090`**), save, and restart Streamlit."
    )


def configure_google_genai_api_key_environment(settings: Settings) -> None:
    """If a Gemini Developer API key is configured, expose it via ``GOOGLE_API_KEY``.

    ``pydantic-settings`` loads values into ``Settings`` but does not mutate the process
    environment. ``google.genai.Client`` (used by ADK ``Agent`` chat) relies on that env var
    when not using Vertex with ``projects/`` model resource names.

    Prefer Vertex ADC in production and use this path for AI Studio-style keys locally or in demos.
    """
    if settings.gemini_api_key and settings.gemini_api_key.strip():
        os.environ["GOOGLE_API_KEY"] = settings.gemini_api_key.strip()
