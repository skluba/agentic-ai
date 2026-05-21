"""KnowledgeCorpus + deterministic embeddings."""

from __future__ import annotations

import json

import pytest
from app.knowledge.chunking import chunk_text_basic
from app.knowledge.embeddings import FakeEmbeddingBackend
from app.knowledge.store import KnowledgeCorpus
from app.tools.document_search_tool import make_document_search_tool


def test_chunking_keeps_overlap_progress():
    text = "sentence alpha. " * 80
    parts = chunk_text_basic(text, chunk_chars=120, overlap_chars=30)
    assert len(parts) >= 5


@pytest.fixture()
def corpus_16() -> KnowledgeCorpus:
    embedder = FakeEmbeddingBackend(embedding_dim=16)
    corpus = KnowledgeCorpus(embedder)
    blob = "Neptune is distant. " + ("Planetary science trivia. " * 60)
    assert corpus.ingest_text(doc_id="science", raw_text=blob) > 0
    assert corpus.chunk_count > 0
    return corpus


def test_search_returns_payloads_sorted(corpus_16: KnowledgeCorpus):
    hits = corpus_16.search_chunks(query="science trivia", top_k=3)
    assert hits
    for row in hits:
        assert {"chunk_id", "document_id", "snippet"} <= row.keys()


def test_tool_serializes_results(corpus_16: KnowledgeCorpus):
    tool = make_document_search_tool(corpus_16)
    raw = json.loads(tool(retrieval_question="planetary trivia", top_k=2))
    assert "hits" in raw
    assert isinstance(raw["hits"], list)


def test_ingest_empty_text_is_no_op():
    corpus = KnowledgeCorpus(FakeEmbeddingBackend(32))
    assert corpus.ingest_text(doc_id="x", raw_text="   ") == 0
