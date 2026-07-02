from __future__ import annotations

import json
import tomllib
from pathlib import Path


def test_root_plugin_artifacts_are_version_synced() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    version = pyproject["project"]["version"]
    lockfile = tomllib.loads(Path("uv.lock").read_text(encoding="utf-8"))

    claude = json.loads(Path(".claude-plugin/plugin.json").read_text(encoding="utf-8"))
    codex = json.loads(Path(".codex-plugin/plugin.json").read_text(encoding="utf-8"))

    assert claude["version"] == version
    assert codex["version"] == version
    assert lockfile["package"][0]["version"] == version
    assert claude == codex
    assert Path("commands/cluxion-consensus.md").is_file()
    assert Path("commands/ultracode-doctor.md").is_file()
    assert Path("skills/ultracode/SKILL.md").is_file()


def test_no_root_adapter_forks_or_fictional_codex_snippets() -> None:
    assert not Path("adapters").exists()
    assert not Path("adapters/codex/config-snippet.toml").exists()
def test_marketplace_manifest_is_version_synced() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    version = pyproject["project"]["version"]

    marketplace = json.loads(Path(".claude-plugin/marketplace.json").read_text(encoding="utf-8"))
    assert marketplace["plugins"][0]["version"] == version
    assert marketplace["plugins"][0]["source"] == "./"
