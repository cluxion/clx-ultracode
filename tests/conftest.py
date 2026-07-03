from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def isolated_ultracode_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CLUXION_EFFORT_ULTRACODE_HOME", str(tmp_path / "ultracode-home"))
