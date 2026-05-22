"""Runner / asyncio helpers for Phase 1 session execution."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

from app.agents import session_runner
from app.agents.session_runner import (
    concatenate_agent_text,
    run_core_rag_turn,
    run_core_rag_turn_sync,
    run_phase2_external_turn,
    run_phase2_external_turn_sync,
    run_phase3_mcp_turn,
    run_phase3_mcp_turn_sync,
    run_phase5_collaborative_turn,
    run_phase5_collaborative_turn_sync,
)
from app.config import Settings
from app.knowledge import FakeEmbeddingBackend, KnowledgeCorpus
from google.genai import types


def test_user_turn_strips_whitespace():
    content = session_runner._user_turn("  hello world  \n")
    assert content.role == "user"
    assert content.parts[0].text == "hello world"


def test_user_turn_blank_becomes_empty_string():
    blank = session_runner._user_turn("   \t  ")
    assert blank.parts[0].text == ""

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


def test_run_phase2_external_turn_mocked_runner_yields_text():
    empty_corpus = KnowledgeCorpus(FakeEmbeddingBackend(embedding_dim=12))
    settings = Settings(gcp_project_id="phase2-runner-test")

    evt = MagicMock()
    evt.author = "hybrid_external_research"
    evt.content = types.Content(role="model", parts=[types.Part(text="phase2 synthesized")])

    mock_runner_inst = MagicMock()

    async def fake_run_phase2(**_kwargs):  # noqa: ANN003
        yield evt

    mock_runner_inst.run_async = fake_run_phase2

    async def body() -> tuple[str, list]:
        with patch("app.agents.session_runner.Runner", return_value=mock_runner_inst):
            return await run_phase2_external_turn(
                settings=settings,
                corpus=empty_corpus,
                question="  q2  ",
            )

    text_out, collected = asyncio.run(body())
    assert text_out == "phase2 synthesized"
    assert collected == [evt]


def test_run_phase2_external_turn_sync_delegates():
    sentinel: tuple[str, list] = ("p2-ok", [])
    kwargs = {
        "settings": Settings(gcp_project_id="phase2-sync"),
        "corpus": KnowledgeCorpus(FakeEmbeddingBackend(8)),
        "question": "hybrid?",
    }

    with patch("asyncio.run", return_value=sentinel) as mocked:
        out = run_phase2_external_turn_sync(**kwargs)

    mocked.assert_called_once()
    assert out == sentinel


def test_run_phase3_mcp_turn_mocked_runner_yields_text():
    empty_corpus = KnowledgeCorpus(FakeEmbeddingBackend(embedding_dim=12))
    settings = Settings(gcp_project_id="phase3-runner-test")

    evt = MagicMock()
    evt.author = "phase3_mcp_hybrid"
    evt.content = types.Content(role="model", parts=[types.Part(text="phase3 synthesized")])

    mock_runner_inst = MagicMock()

    async def fake_run_phase3(**_kwargs):  # noqa: ANN003
        yield evt

    mock_runner_inst.run_async = fake_run_phase3

    async def body() -> tuple[str, list]:
        with patch("app.agents.session_runner.Runner", return_value=mock_runner_inst):
            return await run_phase3_mcp_turn(
                settings=settings,
                corpus=empty_corpus,
                question="  EUR USD  ",
            )

    text_out, collected = asyncio.run(body())
    assert text_out == "phase3 synthesized"
    assert collected == [evt]


def test_run_phase3_mcp_turn_sync_delegates():
    sentinel: tuple[str, list] = ("p3-ok", [])
    kwargs = {
        "settings": Settings(gcp_project_id="phase3-sync"),
        "corpus": KnowledgeCorpus(FakeEmbeddingBackend(8)),
        "question": "btc movers?",
    }

    with patch("asyncio.run", return_value=sentinel) as mocked:
        out = run_phase3_mcp_turn_sync(**kwargs)

    mocked.assert_called_once()
    assert out == sentinel


def test_run_phase5_collaborative_turn_mocked_runner():
    empty_corpus = KnowledgeCorpus(FakeEmbeddingBackend(embedding_dim=12))
    settings = Settings(
        gcp_project_id="phase5-runner-test",
        news_agent_a2a_base_url="http://stub",
    )

    evt = MagicMock()
    evt.author = "phase5_collaborative_hybrid"
    evt.content = types.Content(role="model", parts=[types.Part(text="phase5 collaborative")])

    mock_runner_inst = MagicMock()

    async def fake_run(**_kwargs):  # noqa: ANN003
        yield evt

    mock_runner_inst.run_async = fake_run

    async def body() -> tuple[str, list]:
        with patch("app.agents.session_runner.Runner", return_value=mock_runner_inst):
            return await run_phase5_collaborative_turn(
                settings=settings,
                corpus=empty_corpus,
                question="recent policy news",
            )

    text_out, collected = asyncio.run(body())
    assert text_out == "phase5 collaborative"
    assert collected == [evt]


def test_run_phase5_collaborative_turn_sync_delegates():
    sentinel: tuple[str, list] = ("p5-ok", [])
    kwargs = {
        "settings": Settings(gcp_project_id="phase5-sync"),
        "corpus": KnowledgeCorpus(FakeEmbeddingBackend(8)),
        "question": "headlines?",
    }

    with patch("asyncio.run", return_value=sentinel) as mocked:
        out = run_phase5_collaborative_turn_sync(**kwargs)

    mocked.assert_called_once()
    assert out == sentinel
