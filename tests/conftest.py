"""Pytest defaults.

- Isolate working directory so `Settings(env_file=".env")` does not pick up repo `.env`.
- Blank Langfuse env vars so shells with exported keys remain deterministic under CI.
"""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _pytest_workdir_no_dotenv(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Avoid reading developer ``.env`` during unit tests."""
    wd = tmp_path / "proj"
    wd.mkdir()
    monkeypatch.chdir(wd)


@pytest.fixture(autouse=True)
def _neutralize_langfuse_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LANGFUSE_HOST", "")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "")
    monkeypatch.setenv("LANGFUSE_TRACING_ENABLED", "false")
