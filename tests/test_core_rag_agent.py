"""Factory validation for CORE RAG ADK Agent."""

from __future__ import annotations

import pytest
from app.agents.core_rag import create_core_rag_agent
from app.config import Settings
from app.knowledge import FakeEmbeddingBackend, KnowledgeCorpus


@pytest.fixture()
def tiny_corpus() -> KnowledgeCorpus:
    corpus = KnowledgeCorpus(FakeEmbeddingBackend(embedding_dim=24))
    assert corpus.ingest_text(doc_id="test", raw_text=("alpha beta " * 120)) >= 1
    return corpus


def test_raises_without_chunks():
    settings = Settings(gcp_project_id="unit-local", embedding_dimension=24)
    empty_corpus = KnowledgeCorpus(FakeEmbeddingBackend(24))
    with pytest.raises(ValueError):
        create_core_rag_agent(settings, empty_corpus)


def test_creates_when_chunks_present(tiny_corpus):
    settings = Settings(gcp_project_id="unit-local", embedding_dimension=24)
    agent = create_core_rag_agent(settings, tiny_corpus)
    assert agent.name == "core_rag"
    assert agent.tools
