"""Phase 2 hybrid agent factory wiring."""

from __future__ import annotations

from app.agents.external_knowledge import create_external_knowledge_agent
from app.config import Settings
from app.knowledge import FakeEmbeddingBackend, KnowledgeCorpus
from google.adk.tools.google_search_tool import GoogleSearchTool


def _tiny_corpus() -> KnowledgeCorpus:
    corpus = KnowledgeCorpus(FakeEmbeddingBackend(embedding_dim=16))
    assert corpus.ingest_text(doc_id="doc", raw_text=("alpha bravo " * 80)) >= 1
    return corpus


def test_phase2_tools_include_documents_and_web_when_corpus_loaded():
    settings = Settings(gcp_project_id="unit-phase2-test", embedding_dimension=16)
    agent = create_external_knowledge_agent(settings, _tiny_corpus())
    assert len(agent.tools) == 2
    func_tool, builtin = agent.tools
    assert callable(func_tool)
    assert isinstance(builtin, GoogleSearchTool)


def test_phase2_tools_web_only_when_corpus_empty():
    settings = Settings(gcp_project_id="unit-phase2-test", embedding_dimension=16)
    empty_corpus = KnowledgeCorpus(FakeEmbeddingBackend(16))
    assert empty_corpus.chunk_count == 0
    agent = create_external_knowledge_agent(settings, empty_corpus)
    assert len(agent.tools) == 1
    assert isinstance(agent.tools[0], GoogleSearchTool)
