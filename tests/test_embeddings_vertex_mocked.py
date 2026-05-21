"""VertexTextEmbeddingBackend with mocked google-genai client."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from app.config import Settings
from app.knowledge.embeddings import (
    FakeEmbeddingBackend,
    VertexTextEmbeddingBackend,
    build_embedder_from_settings,
)


def _ev(values: list[float]) -> MagicMock:
    m = MagicMock()
    m.values = values
    return m


def test_fake_backend_rejects_tiny_dimensions() -> None:
    with pytest.raises(ValueError, match="embedding_dim"):
        FakeEmbeddingBackend(embedding_dim=4)


def test_fake_backend_empty_documents_matrix_shape() -> None:
    fe = FakeEmbeddingBackend(embedding_dim=12)
    out = fe.embed_texts([])
    assert out.shape == (0, 12)


@patch("google.genai.Client")
def test_vertex_embed_texts_returns_stacked_arrays(mock_cls: MagicMock) -> None:
    client = MagicMock()
    mock_cls.return_value = client
    client.models.embed_content.return_value = MagicMock(
        embeddings=[_ev([0.0, 1.0, 0.0, 1.0]), _ev([0.25] * 4)],
    )

    vb = VertexTextEmbeddingBackend(project="p", location="eu", output_dimensionality=4)
    out = vb.embed_texts(["alpha", "beta"])
    assert out.shape == (2, 4)
    assert vb.embedding_dim == 4


@patch("google.genai.Client")
def test_vertex_empty_batch_returns_empty_matrix(mock_cls: MagicMock) -> None:
    client = MagicMock()
    mock_cls.return_value = client
    vb = VertexTextEmbeddingBackend(project="p", location="eu", output_dimensionality=4)
    out = vb.embed_texts([])
    assert out.shape == (0, 4)
    client.models.embed_content.assert_not_called()


@patch("google.genai.Client")
def test_vertex_missing_embeddings_fallback_zeros(mock_cls: MagicMock) -> None:
    client = MagicMock()
    mock_cls.return_value = client
    client.models.embed_content.return_value = MagicMock(embeddings=None)

    vb = VertexTextEmbeddingBackend(project="p", location="eu", output_dimensionality=8)
    out = vb.embed_texts(["only"])
    assert out.shape == (1, 8)
    np.testing.assert_array_equal(out, np.zeros((1, 8), dtype=np.float32))


@patch("google.genai.Client")
def test_vertex_sparse_embedding_positions_become_zeros(mock_cls: MagicMock) -> None:
    client = MagicMock()
    mock_cls.return_value = client
    bad = MagicMock()
    bad.values = None
    ok = MagicMock()
    ok.values = [1.0, 0.0, 0.0]
    client.models.embed_content.return_value = MagicMock(embeddings=[bad, ok])

    vb = VertexTextEmbeddingBackend(project="p", location="eu", output_dimensionality=3)
    out = vb.embed_texts(["a", "b"])
    assert out.shape == (2, 3)


@patch("google.genai.Client")
def test_vertex_wrong_dimensionality_raises(mock_cls: MagicMock) -> None:
    client = MagicMock()
    mock_cls.return_value = client
    client.models.embed_content.return_value = MagicMock(embeddings=[_ev([1.0, 2.0])])

    vb = VertexTextEmbeddingBackend(project="p", location="eu", output_dimensionality=4)
    with pytest.raises(ValueError, match="Embedding length"):
        vb.embed_texts(["x"])


@patch("google.genai.Client")
def test_build_embedder_offline_returns_fake_otherwise_vertex(mock_cls: MagicMock) -> None:
    client = MagicMock()
    mock_cls.return_value = client

    offline = build_embedder_from_settings(
        Settings(gcp_project_id="unit", embedding_dimension=64),
        offline=True,
    )
    offline.embed_texts(["z"])
    mock_cls.assert_not_called()

    build_embedder_from_settings(
        Settings(gcp_project_id="unit", embedding_dimension=64),
        offline=False,
    )
    mock_cls.assert_called_once()
