from __future__ import annotations

import pytest

from cluxion_effort_ultracode.adapters.codex_llm import CodexSubprocessLlm
from cluxion_effort_ultracode.adapters.hermes_llm import HermesSubprocessLlm
from cluxion_effort_ultracode.core.errors import require_positive_finite, validation_error_code
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


def test_validation_error_code_handles_timeout_env_name() -> None:
    assert (
        validation_error_code(ValueError("CLUXION_EFFORT_ULTRACODE_HERMES_TIMEOUT must be numeric"))
        == "invalid_timeout"
    )


@pytest.mark.parametrize("raw", ["nan", "inf", "-inf", "0", "-1", "0.0"])
def test_timeout_from_env_rejects_non_finite_or_non_positive(raw: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CLUXION_EFFORT_ULTRACODE_HERMES_TIMEOUT", raw)

    with pytest.raises(ValueError, match=r"greater than zero|must be"):
        timeout_from_env()


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf"), 0.0, -1.0, 10**400])
def test_default_llm_rejects_non_finite_or_non_positive_timeout(value: float | int) -> None:
    with pytest.raises(ValueError, match=r"timeout_seconds|greater than zero|must be"):
        default_llm(timeout_seconds=value)


def test_timeout_from_env_keeps_positive_finite(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CLUXION_EFFORT_ULTRACODE_HERMES_TIMEOUT", "12.5")

    assert timeout_from_env() == 12.5


def test_require_positive_finite_accepts_int_and_float() -> None:
    assert require_positive_finite(12, "timeout_seconds") == 12.0
    assert require_positive_finite(12.5, "timeout_seconds") == 12.5


@pytest.mark.parametrize("value", [True, False, "12.5", "1", None, object()])
def test_require_positive_finite_rejects_non_numeric_types(value: object) -> None:
    with pytest.raises(ValueError, match="timeout_seconds"):
        require_positive_finite(value, "timeout_seconds")


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf"), 0, -1.0, 10**400])
def test_require_positive_finite_rejects_non_finite_non_positive_or_overflow(value: object) -> None:
    with pytest.raises(ValueError, match="timeout_seconds") as caught:
        require_positive_finite(value, "timeout_seconds")
    assert not isinstance(caught.value, OverflowError)
