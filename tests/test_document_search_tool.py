"""Document search tool JSON contract."""

from __future__ import annotations

import json

from app.knowledge.embeddings import FakeEmbeddingBackend
from app.knowledge.store import KnowledgeCorpus
from app.tools.document_search_tool import make_document_search_tool


def test_empty_query_returns_payload():
    corpus = KnowledgeCorpus(FakeEmbeddingBackend(embedding_dim=8))
    tool = make_document_search_tool(corpus)
    payload = json.loads(tool(retrieval_question="   ", top_k=3))
    assert payload["hits"] == []
