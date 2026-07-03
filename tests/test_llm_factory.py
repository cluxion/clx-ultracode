from __future__ import annotations

import pytest

from cluxion_effort_ultracode.adapters.codex_llm import CodexSubprocessLlm
from cluxion_effort_ultracode.adapters.hermes_llm import HermesSubprocessLlm
from cluxion_effort_ultracode.llm_factory import default_llm, timeout_from_env


def test_default_llm_stays_hermes_first(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CLUXION_EFFORT_ULTRACODE_HERMES_BINARY", raising=False)
    monkeypatch.delenv("CLUXION_EFFORT_ULTRACODE_HERMES_MODEL", raising=False)
    monkeypatch.delenv("CLUXION_EFFORT_ULTRACODE_HERMES_TIMEOUT", raising=False)

    llm = default_llm()

    assert isinstance(llm, HermesSubprocessLlm)
    assert llm.binary == "hermes"


def test_default_llm_builds_codex_adapter_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CLUXION_EFFORT_ULTRACODE_CODEX_BINARY", "/opt/codex")
    monkeypatch.setenv("CLUXION_EFFORT_ULTRACODE_CODEX_MODEL", "gpt-test")
    monkeypatch.setenv("CLUXION_EFFORT_ULTRACODE_HERMES_TIMEOUT", "7")

    llm = default_llm("codex")

    assert isinstance(llm, CodexSubprocessLlm)
    assert llm.binary == "/opt/codex"
    assert llm.model == "gpt-test"
    assert llm.timeout_seconds == 7


def test_default_llm_rejects_unknown_adapter() -> None:
    with pytest.raises(ValueError, match="unknown adapter"):
        default_llm("bogus")


def test_timeout_from_env_remains_shared_agent_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CLUXION_EFFORT_ULTRACODE_HERMES_TIMEOUT", "9")

    assert timeout_from_env() == 9
