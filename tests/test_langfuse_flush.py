"""Langfuse flush must never raise during local/dev runs."""

from __future__ import annotations

import langfuse
import pytest
from app.observability.langfuse_flush import flush_langfuse


def test_flush_swallows_upstream_errors(monkeypatch: pytest.MonkeyPatch):
    def boom_public_client():
        class _Broken:
            def flush(self):  # noqa: ANN001
                raise RuntimeError("simulated telemetry outage")

        return _Broken()

    monkeypatch.setattr(langfuse, "get_client", boom_public_client)
    flush_langfuse()
