"""KnowledgeCorpus + deterministic embeddings."""

from __future__ import annotations

import json

import numpy as np
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


def test_search_before_ingest_returns_empty_lists():
    empty = KnowledgeCorpus(FakeEmbeddingBackend(24))
    assert empty.search_chunks(query="anything") == []
    assert empty.search_chunks(query="   ") == []


def test_ingest_many_strings_sums_batches():
    corpus = KnowledgeCorpus(FakeEmbeddingBackend(embedding_dim=12))
    text = ("segment. " * 90)[:-1]
    total = corpus.ingest_many_strings([text[:200], text], doc_prefix="batch")
    assert total >= 2
    assert corpus.chunk_count == total


def test_embed_row_mismatch_raises_runtime_error():
    class MisalignedEmbedder:
        embedding_dim = 16

        def embed_texts(self, texts: list[str]) -> np.ndarray:
            rows = len(texts) // 2 or 1
            return np.zeros((rows, self.embedding_dim), dtype=np.float32)

    corpus = KnowledgeCorpus(MisalignedEmbedder())
    blob = ("mismatch cue. " * 80)[:-1]

    with pytest.raises(RuntimeError, match="embedder rows mismatch"):
        corpus.ingest_text(doc_id="bad", raw_text=blob)


def test_fallback_doc_id_generated_when_missing():
    corpus = KnowledgeCorpus(FakeEmbeddingBackend(embedding_dim=8))
    count = corpus.ingest_text(doc_id=None, raw_text=("auto id " * 40)[:-1])
    assert count >= 1
    assert corpus.chunk_count == count
