"""FAISS-backed corpus with chunk metadata."""

from __future__ import annotations

from dataclasses import dataclass
from uuid import uuid4

import numpy as np

from app.knowledge.chunking import chunk_text_basic
from app.knowledge.embeddings import EmbeddingBackend, FakeEmbeddingBackend
from app.rag.faiss_store import FaissFlatIndex


@dataclass(frozen=True, slots=True)
class ChunkRecord:
    chunk_id: str
    doc_id: str
    text: str


class KnowledgeCorpus:
    """Ingest raw documents → chunks → embeddings → FAISS (thread-safe index)."""

    __slots__ = ("_chunks", "_embedder", "_index")

    def __init__(self, embedder: EmbeddingBackend | None = None) -> None:
        self._embedder = embedder or FakeEmbeddingBackend()
        self._index: FaissFlatIndex | None = None
        self._chunks: dict[str, ChunkRecord] = {}

    def _ensure_index_dim(self, dim: int) -> None:
        if self._index is None:
            self._index = FaissFlatIndex(dim)

    @staticmethod
    def _l2_normalize_rows(matrix: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms = np.clip(norms, 1e-12, None)
        out = matrix / norms.astype(np.float32)
        return out.astype(np.float32)

    @staticmethod
    def _l2_normalize_vector(vector: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vector)
        if norm <= 1e-12:
            return vector.astype(np.float32)
        out = vector / norm
        return out.astype(np.float32)

    @property
    def chunk_count(self) -> int:
        return len(self._chunks)

    @property
    def embedding_dim(self) -> int:
        return self._embedder.embedding_dim

    def ingest_text(
        self,
        *,
        doc_id: str | None,
        raw_text: str,
        chunk_chars: int = 750,
        overlap_chars: int = 150,
    ) -> int:
        """Chunk + embed ``raw_text``. Returns added chunk rows."""
        did = doc_id or f"doc-{uuid4()}"
        parts = chunk_text_basic(raw_text, chunk_chars=chunk_chars, overlap_chars=overlap_chars)
        if not parts:
            return 0
        matrix = self._embedder.embed_texts(parts)
        rows, cols = matrix.shape
        self._ensure_index_dim(cols)
        if rows != len(parts):
            msg = "embedder rows mismatch"
            raise RuntimeError(msg)

        payloads: list[str] = []
        for offset, snippet in enumerate(parts):
            cid = f"{did}::chunk-{offset}"
            record = ChunkRecord(chunk_id=cid, doc_id=did, text=snippet)
            self._chunks[cid] = record
            payloads.append(cid)

        if self._index is None:
            msg = "index not initialized"
            raise RuntimeError(msg)
        matrix = self._l2_normalize_rows(matrix)
        self._index.add_vectors(matrix, payloads=payloads)
        return rows

    def ingest_many_strings(
        self,
        payloads: list[str],
        *,
        doc_prefix: str = "memo",
        chunk_chars: int = 750,
        overlap_chars: int = 150,
    ) -> int:
        total = 0
        for i, blob in enumerate(payloads):
            total += self.ingest_text(
                doc_id=f"{doc_prefix}-{i}",
                raw_text=blob,
                chunk_chars=chunk_chars,
                overlap_chars=overlap_chars,
            )
        return total

    def search_chunks(
        self,
        *,
        query: str,
        top_k: int = 6,
    ) -> list[dict[str, str | float]]:
        """Retrieve chunk metadata + heuristic relevance derived from Euclidean distance."""
        if self._index is None:
            return []
        trimmed = query.strip()
        if not trimmed:
            return []
        vectors = self._embedder.embed_texts([trimmed])
        if vectors.size == 0:
            return []

        vector = self._l2_normalize_vector(vectors[0])
        usable = len(self._index)
        hits = self._index.search(vector, k=min(top_k, max(usable, 1)))

        formatted: list[dict[str, str | float]] = []
        for hit, cid in hits:
            rec = self._chunks.get(cid)
            if rec is None:
                continue
            distance = hit.distance
            score = float(1.0 / (1.0 + distance))
            formatted.append(
                {
                    "chunk_id": rec.chunk_id,
                    "document_id": rec.doc_id,
                    "snippet": rec.text[:2000],
                    "relevance_approx": score,
                    "distance_l2": float(distance),
                },
            )
        return formatted[:top_k]
