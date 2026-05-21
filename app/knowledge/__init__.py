"""Knowledge corpus + chunk utilities for retrieval."""

from app.knowledge.chunking import chunk_text_basic
from app.knowledge.embeddings import (
    EmbeddingBackend,
    FakeEmbeddingBackend,
    VertexTextEmbeddingBackend,
    build_embedder_from_settings,
)
from app.knowledge.store import ChunkRecord, KnowledgeCorpus

__all__ = [
    "ChunkRecord",
    "EmbeddingBackend",
    "FakeEmbeddingBackend",
    "KnowledgeCorpus",
    "VertexTextEmbeddingBackend",
    "build_embedder_from_settings",
    "chunk_text_basic",
]
