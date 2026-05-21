"""Runner / asyncio helpers for Phase 1 session execution."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

from app.agents.session_runner import (
    concatenate_agent_text,
    run_core_rag_turn,
    run_core_rag_turn_sync,
)
from app.config import Settings
from app.knowledge import FakeEmbeddingBackend, KnowledgeCorpus
from google.genai import types


def test_concatenate_skips_blank_text_parts():
    evt = MagicMock()
    evt.author = "model"
    evt.content = types.Content(role="model", parts=[types.Part(text=None)])
    assert concatenate_agent_text([evt]) == ""


def test_concatenate_when_content_missing():
    evt = MagicMock()
    evt.author = "bot"
    evt.content = None
    assert concatenate_agent_text([evt]) == ""


def test_concatenate_when_author_blank():
    evt = MagicMock()
    evt.author = ""
    evt.content = types.Content(role="model", parts=[types.Part(text="x")])
    assert concatenate_agent_text([evt]) == ""


def test_run_core_rag_turn_mocked_runner_yields_concatenated_text():
    corpus = KnowledgeCorpus(FakeEmbeddingBackend(embedding_dim=16))
    assert corpus.ingest_text(doc_id="doc", raw_text=("word " * 80)) >= 1
    settings = Settings(gcp_project_id="pytest")

    evt = MagicMock()
    evt.author = "core_rag"
    evt.content = types.Content(role="model", parts=[types.Part(text="synthetic answer")])

    mock_runner_inst = MagicMock()

    async def fake_run(**_kwargs):  # noqa: ANN003
        yield evt

    mock_runner_inst.run_async = fake_run

    async def body() -> tuple[str, list]:
        with patch("app.agents.session_runner.Runner", return_value=mock_runner_inst):
            return await run_core_rag_turn(settings=settings, corpus=corpus, question="  q  ")

    text_out, collected = asyncio.run(body())
    assert text_out == "synthetic answer"
    assert collected == [evt]


def test_run_core_rag_turn_sync_delegates_to_asyncio_run():
    sentinel: tuple[str, list] = ("ok", [])
    kwargs = {
        "settings": Settings(gcp_project_id="x"),
        "corpus": KnowledgeCorpus(FakeEmbeddingBackend(8)),
        "question": "?",
    }

    with patch("asyncio.run", return_value=sentinel) as mocked:
        out = run_core_rag_turn_sync(**kwargs)

    mocked.assert_called_once()
    assert out == sentinel
