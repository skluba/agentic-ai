"""Synthetic coverage for session runner helpers."""

from __future__ import annotations

from types import SimpleNamespace

from app.agents.session_runner import concatenate_agent_text
from google.genai import types


def test_concatenate_collects_agent_text():
    evt = SimpleNamespace(
        author="core_rag",
        content=types.Content(
            role="model",
            parts=[types.Part(text="partial"), types.Part(text=" answer")],
        ),
    )
    assert concatenate_agent_text([evt]) == "partial\n answer"


def test_concatenate_skips_users():
    user_evt = SimpleNamespace(author="user", content=None)
    assert concatenate_agent_text([user_evt]) == ""
