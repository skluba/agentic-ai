"""Console entrypoints."""

from __future__ import annotations

from unittest.mock import MagicMock

from app.ui import cli


def test_streamlit_launcher_invokes_cli(monkeypatch):
    mocked_exit = MagicMock()
    mocked_main = MagicMock(return_value=None)
    monkeypatch.setattr(cli.sys, "exit", mocked_exit)
    monkeypatch.setattr(
        "streamlit.web.cli.main",
        mocked_main,
    )

    cli.run_streamlit()

    assert cli.sys.argv[:2] == ["streamlit", "run"]
    assert cli.sys.argv[2].endswith("streamlit_app.py")

    mocked_main.assert_called_once()
    mocked_exit.assert_called_once_with(None)
