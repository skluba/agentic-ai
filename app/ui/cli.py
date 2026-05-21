"""Streamlit entry shim for ``pip install -e .`` console script."""

from __future__ import annotations

import sys
from pathlib import Path


def run_streamlit() -> None:
    """Launch the Streamlit UI defined at the repository root."""
    from streamlit.web import cli as stcli  # noqa: PLC0415

    repo_root = Path(__file__).resolve().parents[2]
    dashboard = repo_root / "streamlit_app.py"
    sys.argv = [
        "streamlit",
        "run",
        str(dashboard),
        "--server.address",
        "0.0.0.0",
        "--server.port",
        "8501",
    ]
    sys.exit(stcli.main())
