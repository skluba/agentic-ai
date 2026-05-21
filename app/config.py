"""Runtime configuration sourced from environment (GCP, Gemini, Langfuse)."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

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

    faiss_dimension: int = Field(default=128, ge=8, validation_alias="FAISS_EMBEDDING_DIMENSION")

    langfuse_public_key: str | None = Field(default=None, validation_alias="LANGFUSE_PUBLIC_KEY")
    langfuse_secret_key: str | None = Field(default=None, validation_alias="LANGFUSE_SECRET_KEY")
    langfuse_host: str | None = Field(default=None, validation_alias="LANGFUSE_HOST")


@lru_cache
def get_settings(**overrides: Any) -> Settings:
    """Cached settings; optional overrides simplify tests without env coupling."""
    if not overrides:
        return Settings()
    return Settings(**overrides)


def clear_settings_cache() -> None:
    """Drop cached Settings (tests)."""
    get_settings.cache_clear()
