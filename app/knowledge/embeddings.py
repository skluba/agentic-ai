"""Text embedding backends (Vertex / deterministic fake)."""

from __future__ import annotations

import hashlib
import logging
from typing import Protocol

import numpy as np
from google.genai.types import EmbedContentConfig

from app.config import Settings

logger = logging.getLogger(__name__)


class EmbeddingBackend(Protocol):
    """Embeds batches of trimmed strings."""

    embedding_dim: int

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Return float32 ndarray shaped (batch, embedding_dim)."""

        ...


class FakeEmbeddingBackend:
    """Deterministic, offline-friendly embeddings for CI (not semantically faithful)."""

    def __init__(self, embedding_dim: int = 32) -> None:
        if embedding_dim < 8:
            msg = "embedding_dim must be >= 8"
            raise ValueError(msg)
        self.embedding_dim = embedding_dim

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.embedding_dim), dtype=np.float32)
        vectors = np.zeros((len(texts), self.embedding_dim), dtype=np.float32)
        for i, txt in enumerate(texts):
            seed = int(hashlib.sha256(txt.encode()).hexdigest(), 16) % (2**32)
            rng = np.random.default_rng(seed)
            v = rng.standard_normal(self.embedding_dim, dtype=np.float32)
            norm = np.linalg.norm(v)
            vectors[i] = v / norm if norm > 1e-6 else v
        return vectors


class VertexTextEmbeddingBackend:
    """Vertex-backed embeddings via google-genai."""

    def __init__(
        self,
        *,
        project: str,
        location: str,
        model_name: str = "text-embedding-004",
        output_dimensionality: int = 768,
    ) -> None:
        try:
            from google import genai  # noqa: PLC0415
        except ImportError as err:  # pragma: no cover
            raise RuntimeError("google-genai is required for VertexTextEmbeddingBackend") from err

        self._client = genai.Client(vertexai=True, project=project, location=location)
        self._model = model_name
        self.embedding_dim = output_dimensionality
        self._config = EmbedContentConfig(output_dimensionality=output_dimensionality)

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.embedding_dim), dtype=np.float32)
        resp = self._client.models.embed_content(
            model=self._model,
            contents=list(texts),
            config=self._config,
        )
        if not resp.embeddings:
            logger.warning("Embedding response missing vectors; filling zeros.")
            return np.zeros((len(texts), self.embedding_dim), dtype=np.float32)

        stacked: list[np.ndarray] = []
        for embedding in resp.embeddings:
            if embedding is None or not getattr(embedding, "values", None):
                stacked.append(np.zeros(self.embedding_dim, dtype=np.float32))
                continue
            vec = np.asarray(embedding.values, dtype=np.float32)
            if vec.size != self.embedding_dim:
                msg = (
                    f"Embedding length {vec.size} != configured {self.embedding_dim}. "
                    "Adjust embedding_dimension or embedding_model in Settings."
                )
                raise ValueError(msg)
            stacked.append(vec)
        return np.stack(stacked, axis=0)


def build_embedder_from_settings(
    settings: Settings,
    *,
    offline: bool = False,
) -> EmbeddingBackend:
    """Prefer Vertex embeddings locally; deterministic fake embeddings for pytest/offline."""
    if offline:
        return FakeEmbeddingBackend(settings.embedding_dimension)
    return VertexTextEmbeddingBackend(
        project=settings.gcp_project_id,
        location=settings.vertex_location,
        model_name=settings.embedding_model,
        output_dimensionality=settings.embedding_dimension,
    )
