"""Flush queued Langfuse events (critical for Streamlit + short-lived scripts).

See Langfuse instrumentation guidance: forgetting ``flush()`` drops buffered spans.
"""

from __future__ import annotations


def flush_langfuse() -> None:
    """Best-effort flush; skips work when tracing is unavailable."""
    try:
        from langfuse import get_client  # noqa: PLC0415 — defer import until callers need flush

        get_client().flush()
    except Exception:  # noqa: BLE001 — never break callers on telemetry faults
        return
