"""FAISS wrapper coverage."""

from __future__ import annotations

import numpy as np
import pytest
from app.rag.faiss_store import FaissFlatIndex


def test_requires_reasonable_dimension():
    with pytest.raises(ValueError):
        FaissFlatIndex(4)


def test_add_and_roundtrip_hit():
    index = FaissFlatIndex(8)
    a = np.eye(8, dtype=np.float32)
    index.add_vectors(a[0].reshape(1, -1), payloads=("row0",))
    hits = index.search(a[0], k=3)
    assert len(hits) == 1
    hit = hits[0]
    assert hit[1] == "row0"
    assert hit[0].distance < 1e-4


def test_multi_vector_returns_sorted_candidates():
    index = FaissFlatIndex(8)
    vecs = np.zeros((3, 8), dtype=np.float32)
    vecs[0, 0] = 1.0
    vecs[1, 0] = 5.0
    vecs[2, 0] = -1.0
    index.add_vectors(vecs, payloads=["a", "b", "c"])
    query = np.zeros(8, dtype=np.float32)
    query[0] = 4.9
    hits = index.search(query, k=3)
    assert [text for _, text in hits][:2] == ["b", "a"]


def test_shape_validation():
    idx = FaissFlatIndex(16)
    with pytest.raises(ValueError):
        idx.add_vectors(np.zeros((2, 2), dtype=np.float32), payloads=["only-one"])
    with pytest.raises(ValueError):
        idx.search(np.zeros(2, dtype=np.float32), k=5)


def test_invalid_k_raises():
    idx = FaissFlatIndex(16)
    with pytest.raises(ValueError):
        idx.search(np.ones(16, dtype=np.float32), k=0)
