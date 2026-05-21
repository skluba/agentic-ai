"""Retrieval backends (in-memory FAISS, future persistent stores)."""

from app.rag.faiss_store import FaissFlatIndex

__all__ = ["FaissFlatIndex"]
