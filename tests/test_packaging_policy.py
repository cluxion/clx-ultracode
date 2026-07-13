from __future__ import annotations

import json
import tomllib
from pathlib import Path

CANONICAL_PLUGIN_ID = "clx-ultracode"
# Python distribution / Hermes entry-point identity stays on the legacy name.
PYTHON_DIST_NAME = "cluxion-agentplugin-effort-ultracode"
PUBLIC_REPO_URL = "https://github.com/cluxion/clx-ultracode.git"


def test_root_plugin_artifacts_are_version_synced() -> None:
    from cluxion_effort_ultracode import __version__

    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    version = pyproject["project"]["version"]
    lockfile = tomllib.loads(Path("uv.lock").read_text(encoding="utf-8"))

    claude = json.loads(Path(".claude-plugin/plugin.json").read_text(encoding="utf-8"))
    codex = json.loads(Path(".codex-plugin/plugin.json").read_text(encoding="utf-8"))

    assert __version__ == version
    assert claude["version"] == version
    assert codex["version"] == version
    assert lockfile["package"][0]["version"] == version
    assert claude == codex
    assert claude["name"] == CANONICAL_PLUGIN_ID
    assert codex["name"] == CANONICAL_PLUGIN_ID
    assert Path("commands/clx-consensus.md").is_file()
    assert not Path("commands/cluxion-consensus.md").exists()
    assert Path("commands/ultracode-doctor.md").is_file()
    skill = Path("skills/clx-ultracode/SKILL.md")
    assert skill.is_file()
    assert "name: clx-ultracode" in skill.read_text(encoding="utf-8")
    assert not Path("skills/ultracode").exists()


def test_no_root_adapter_forks_or_fictional_codex_snippets() -> None:
    assert not Path("adapters").exists()
    assert not Path("adapters/codex/config-snippet.toml").exists()


def test_marketplace_manifest_is_version_synced() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    version = pyproject["project"]["version"]

    marketplace = json.loads(Path(".claude-plugin/marketplace.json").read_text(encoding="utf-8"))
    assert marketplace["name"] == CANONICAL_PLUGIN_ID
    assert marketplace["plugins"][0]["name"] == CANONICAL_PLUGIN_ID
    assert marketplace["plugins"][0]["version"] == version
    assert marketplace["plugins"][0]["source"] == "./"


def test_discovery_identity_vs_python_distribution_compat() -> None:
    """Public/discovery names may move; Python dist/entry-point names must not."""
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    project = pyproject["project"]

    assert project["name"] == PYTHON_DIST_NAME
    assert project["urls"]["Repository"] == PUBLIC_REPO_URL
    hermes_eps = project["entry-points"]["hermes_agent.plugins"]
    assert PYTHON_DIST_NAME in hermes_eps
    assert "cluxion-ultracode" in project["scripts"]
