"""Langfuse wiring without live network."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from app.config import Settings, clear_settings_cache
from app.observability.langfuse_client import (
    LangfuseUnavailable,
    get_langfuse,
    langfuse_enabled,
    start_optional_span,
)


@pytest.fixture(autouse=True)
def _reset_settings():
    clear_settings_cache()
    yield
    clear_settings_cache()


def test_langfuse_disabled_without_env():
    assert langfuse_enabled() is False
    client = get_langfuse()
    assert client is None


def test_strict_mode_raises_when_incomplete(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("LANGFUSE_HOST", raising=False)
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)

    clear_settings_cache()
    with pytest.raises(LangfuseUnavailable):
        get_langfuse(strict=True)


@patch("langfuse.Langfuse", autospec=False)
def test_client_materializes_when_env_present(
    MockLangfuse: MagicMock, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("LANGFUSE_HOST", "http://localhost:3010/")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk")

    MockLangfuse.return_value = MagicMock(name="client")
    clear_settings_cache()
    client = get_langfuse()
    assert client is MockLangfuse.return_value
    MockLangfuse.assert_called_once()
    kw = MockLangfuse.call_args.kwargs
    assert kw["host"] == "http://localhost:3010"


def test_start_optional_span_noops_without_credentials(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    clear_settings_cache()
    assert start_optional_span(name="x", settings=Settings()) is None


@patch("langfuse.Langfuse", autospec=False)
def test_start_optional_span_returns_handle(
    MockLangfuse: MagicMock, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("LANGFUSE_HOST", "http://langfuse/")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk")

    span = MagicMock()
    lf = MagicMock()
    lf.start_span.return_value = span
    MockLangfuse.return_value = lf

    clear_settings_cache()
    handle = start_optional_span(name="demo", input={"k": "v"})
    lf.start_span.assert_called_once_with(name="demo", input={"k": "v"})
    assert handle is span
