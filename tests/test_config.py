"""Settings / configuration behaviour."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from app.config import (
    Settings,
    clear_settings_cache,
    configure_google_genai_api_key_environment,
    get_settings,
    news_agent_a2a_url_host_resolution_hint,
)


@pytest.fixture(autouse=True)
def _reset_cached_settings():
    clear_settings_cache()
    yield
    clear_settings_cache()


def test_defaults_match_gcp_preset():
    cfg = Settings()
    assert cfg.gcp_project_id == "gd-gcp-gridu-genai"
    assert "gemini" in cfg.gemini_model.lower()
    assert cfg.embedding_model == "text-embedding-004"
    assert cfg.embedding_dimension == 768
    assert cfg.gemini_api_key is None


def test_cached_get_settings(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    first = get_settings()
    second = get_settings()
    assert first is second


def test_overrides_via_constructor():
    cfg = Settings(
        gemini_model="gemini-custom",
        gcp_project_id="other",
        vertex_location="us-central1",
        embedding_model="custom-embed",
        embedding_dimension=256,
    )
    assert cfg.gemini_model == "gemini-custom"
    assert cfg.gcp_project_id == "other"
    assert cfg.vertex_location == "us-central1"
    assert cfg.embedding_model == "custom-embed"
    assert cfg.embedding_dimension == 256


def test_gemini_api_key_aliases():
    cfg = Settings(gemini_api_key="from-field")
    assert cfg.gemini_api_key == "from-field"


def test_configure_google_genai_api_key_sets_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    configure_google_genai_api_key_environment(Settings(gemini_api_key=" trimmed "))
    assert os.environ["GOOGLE_API_KEY"] == "trimmed"


def test_configure_google_genai_skips_when_no_key(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    configure_google_genai_api_key_environment(Settings(gemini_api_key=None))
    assert "GOOGLE_API_KEY" not in os.environ


def test_get_settings_kw_overrides_cached():
    try:
        custom = get_settings(gcp_project_id="tmp-local")
        assert custom.gcp_project_id == "tmp-local"
    finally:
        clear_settings_cache()


def test_news_agent_a2a_host_resolution_hint_blank_url():
    assert news_agent_a2a_url_host_resolution_hint(Settings(news_agent_a2a_base_url="")) is None


def test_news_agent_a2a_host_resolution_hint_localhost_ok():
    assert news_agent_a2a_url_host_resolution_hint(
        Settings(news_agent_a2a_base_url="http://localhost:8090/")
    ) is None


def test_news_agent_a2a_host_resolution_warns_compose_dns_on_host(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(Path, "exists", lambda self: False)
    cfg = Settings(news_agent_a2a_base_url="http://news-agent:8090/")
    hint = news_agent_a2a_url_host_resolution_hint(cfg)
    assert hint is not None
    assert "localhost:8090" in hint


def test_news_agent_a2a_host_resolution_suppressed_inside_container(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(Path, "exists", lambda self: str(self) == "/.dockerenv")
    cfg = Settings(news_agent_a2a_base_url="http://news-agent:8090/")
    assert news_agent_a2a_url_host_resolution_hint(cfg) is None
