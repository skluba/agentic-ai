"""Streamlit-facing RAG lab operations with Langfuse hierarchical tracing.

Instrumentation follows Langfuse observability guidelines: descriptive observation names,
proper ``retriever`` types for vector steps, bounded trace inputs via ``capture_input=False``
plus explicit ``update_current_span``, and graceful no-ops when credentials are absent.

See https://langfuse.com/docs/sdk/python and https://langfuse.com/docs/tracing
"""

from __future__ import annotations

from typing import Any

import numpy as np
from langfuse import get_client, observe

from app.rag import FaissFlatIndex


def _sanitized_lab_input(dim: int, doc_label: str) -> dict[str, Any]:
    """Avoid dumping raw vectors/config into Langfuse traces."""
    return {
        "embedding_dim": dim,
        "document_label_chars": len(doc_label),
        "vectors_added": 1,
    }


@observe(
    name="rag-lab-build-index",
    as_type="retriever",
    capture_input=False,
    capture_output=False,
)
def _build_flat_index(dim: int) -> FaissFlatIndex:
    get_client().update_current_span(
        input={"embedding_dim": dim, "phase": "create_faiss_index"},
    )
    return FaissFlatIndex(dim)


@observe(
    name="rag-lab-index-document",
    as_type="retriever",
    capture_input=False,
    capture_output=False,
)
def _index_unit_vector(idx: FaissFlatIndex, dim: int, doc_label: str) -> None:
    vec = np.zeros(dim, dtype=np.float32)
    vec[0] = 1.0
    get_client().update_current_span(
        input=_sanitized_lab_input(dim, doc_label),
    )
    idx.add_vectors(vec.reshape(1, -1), payloads=[doc_label])


@observe(
    name="rag-lab-similarity-search",
    as_type="retriever",
    capture_input=False,
    capture_output=True,
)
def _search_same_vector(idx: FaissFlatIndex, dim: int, k: int = 3) -> dict[str, Any]:
    query = np.zeros(dim, dtype=np.float32)
    query[0] = 1.0
    get_client().update_current_span(input={"embedding_dim": dim, "top_k": k})
    hits = idx.search(query, k=k)
    nearest = hits[0][0].distance if hits else None
    return {
        "hit_count": len(hits),
        "nearest_l2": nearest,
        "indexed_total": len(idx),
    }


@observe(
    name="rag-lab-pipeline-demo",
    as_type="chain",
    capture_input=False,
    capture_output=True,
)
def run_streamlit_lab(embedding_dim: int, *, doc_label: str = "dummy-doc") -> dict[str, Any]:
    """Chains ingest + search spans for hierarchical traces in Langfuse."""
    get_client().update_current_span(input=_sanitized_lab_input(embedding_dim, doc_label))
    idx = _build_flat_index(embedding_dim)
    _index_unit_vector(idx, embedding_dim, doc_label)
    return _search_same_vector(idx, embedding_dim)
