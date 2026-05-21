"""Chunking helpers."""

from __future__ import annotations

import pytest
from app.knowledge.chunking import chunk_text_basic


def test_chunk_text_basic_empty():
    assert chunk_text_basic("") == []
    assert chunk_text_basic("   ") == []


def test_chunk_splits_with_overlap():
    text = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" * 15
    chunks = chunk_text_basic(text, chunk_chars=60, overlap_chars=10)
    assert len(chunks) >= 5
    merged = set("".join(chunks))
    base = set(text.replace(" ", ""))
    assert merged <= base


def test_chunk_text_invalid_params():
    with pytest.raises(ValueError):
        chunk_text_basic("abc", chunk_chars=10)
    with pytest.raises(ValueError):
        chunk_text_basic("abc", chunk_chars=400, overlap_chars=400)
