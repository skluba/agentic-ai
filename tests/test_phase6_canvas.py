"""Phase 6 Canvas models, tooling, wiring."""

from __future__ import annotations

import json

import pytest
from app.agents.phase6_canvas import create_phase6_canvas_agent
from app.canvas.models import CanvasProduceInput
from app.config import Settings
from app.knowledge import FakeEmbeddingBackend, KnowledgeCorpus
from app.tools.canvas_tool import make_canvas_delivery_tool
from pydantic import ValidationError


def test_canvas_produce_input_requires_code_language_for_snippet():
    with pytest.raises(ValidationError):
        CanvasProduceInput(
            output_kind="code_snippet",
            title="x",
            markdown_body='print("hi")',
        )


@pytest.mark.parametrize("kind", ["markdown_report", "html_report"])
def test_canvas_produce_input_accepts_reports(kind: str):
    inp = CanvasProduceInput(
        output_kind=kind,  # type: ignore[arg-type]
        title=" Q ",
        markdown_body="# body ",
    )
    assert inp.title == "Q"
    assert inp.output_kind == kind


@pytest.mark.parametrize("invalid", ["pdf", "", "wat"])
def test_canvas_validation_rejects_bad_output_kind_strings(invalid: str):
    with pytest.raises(ValidationError):
        CanvasProduceInput(output_kind=invalid, title="t", markdown_body="b")  # type: ignore[arg-type]


def test_canvas_tool_markdown_report_json():
    tool = make_canvas_delivery_tool()
    raw = tool(
        output_kind="markdown_report",
        title="Status",
        markdown_body="- One\n- Two",
        programming_language="",
    )
    payload = json.loads(raw)
    assert payload["ok"] is True
    assert payload["mime"] == "text/markdown"
    assert "# Status" in payload["artifact"]
    assert "One" in payload["artifact"]


def test_canvas_tool_html_report_contains_article():
    tool = make_canvas_delivery_tool()
    raw = tool(
        output_kind="html_report",
        title="<Safe>",
        markdown_body="_italic_ paragraph",
        programming_language="",
    )
    payload = json.loads(raw)
    assert payload["ok"] is True
    assert payload["mime"] == "text/html"
    html = payload["artifact"]
    assert "<!DOCTYPE html>" in html
    assert "<article>" in html
    assert "<em>italic</em>" in html


def test_canvas_tool_code_snippet_fences_python():
    tool = make_canvas_delivery_tool()
    raw = tool(
        output_kind="code_snippet",
        title="Tiny example",
        markdown_body='print("ok")',
        programming_language="python",
    )
    payload = json.loads(raw)
    assert payload["ok"] is True
    assert "```python" in payload["artifact"]


def test_canvas_tool_reports_validation_issues():
    tool = make_canvas_delivery_tool()
    raw = tool(
        output_kind="code_snippet",
        title="Bad",
        markdown_body='print("")',
        programming_language=" ",
    )
    payload = json.loads(raw)
    assert payload["ok"] is False


def test_phase6_tool_inventory_snapshot():
    """Corpus-less / no NEWS URL → MCP + grounded search + canvas."""
    cfg = Settings(gcp_project_id="phase6-inventory-test", embedding_dimension=8)
    empty = KnowledgeCorpus(FakeEmbeddingBackend(embedding_dim=8))
    agent = create_phase6_canvas_agent(cfg, empty)
    assert len(agent.tools) == 3

    corpus = KnowledgeCorpus(FakeEmbeddingBackend(embedding_dim=8))
    assert corpus.ingest_text(doc_id="n", raw_text=("line " * 40)) >= 1
    agent_docs = create_phase6_canvas_agent(cfg, corpus)
    assert len(agent_docs.tools) == 4


def test_phase6_adds_news_tool_when_env_url():
    cfg = Settings(
        gcp_project_id="phase6-inv",
        embedding_dimension=8,
        news_agent_a2a_base_url=" http://stub:8090 ",
    )
    empty = KnowledgeCorpus(FakeEmbeddingBackend(embedding_dim=8))
    agent = create_phase6_canvas_agent(cfg, empty)
    assert len(agent.tools) == 4


def test_phase6_corpus_news_and_canvas_stack_height():
    cfg = Settings(
        gcp_project_id="phase6-inv",
        embedding_dimension=8,
        news_agent_a2a_base_url="http://stub:8090",
    )
    corpus = KnowledgeCorpus(FakeEmbeddingBackend(embedding_dim=8))
    assert corpus.ingest_text(doc_id="n", raw_text=("line " * 40)) >= 1
    agent = create_phase6_canvas_agent(cfg, corpus)
    assert len(agent.tools) == 5
