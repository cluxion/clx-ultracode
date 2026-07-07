"""JSONL call-record serialization for debate journals."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any


def call_record(
    seq: int,
    prompt: str,
    prompt_sha: str,
    model: str | None,
    raw: Mapping[str, Any] | str,
    usage: object,
    duration_ms: int,
) -> dict[str, object]:
    seat, round_index, phase = _prompt_meta(prompt)
    text, response_type = _response_text(raw)
    return {
        "type": "call",
        "seq": seq,
        "agent_seat": seat,
        "round": round_index,
        "phase": phase,
        "prompt_sha256": prompt_sha,
        "model": model,
        "response_text": text,
        "response_type": response_type,
        "tokens": _token_dict(prompt, raw, usage),
        "duration_ms": duration_ms,
    }


def decode_response(record: Mapping[str, Any]) -> Mapping[str, Any] | str:
    text = str(record.get("response_text", ""))
    if record.get("response_type") != "json":
        return text
    if not text:
        return text
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return text
    return parsed if isinstance(parsed, Mapping) else text


def total_tokens(value: object) -> int:
    return int(value.get("total_tokens", 0)) if isinstance(value, Mapping) else 0


def _prompt_meta(prompt: str) -> tuple[str, int, str]:
    seat = ""
    round_index = 0
    phase = "independent"
    for line in prompt.splitlines():
        if line.startswith("Agent: "):
            seat = line.removeprefix("Agent: ").strip()
        elif line.startswith("Round: "):
            value = line.removeprefix("Round: ").strip()
            first = value.split(" ", 1)[0]
            round_index = int(first) if first.isdigit() else 0
            phase = "independent" if "independent" in value else "debate"
    return seat, round_index, phase


def _response_text(raw: Mapping[str, Any] | str) -> tuple[str, str]:
    if isinstance(raw, str):
        return raw, "text"
    return json.dumps(raw, ensure_ascii=False, sort_keys=True, separators=(",", ":")), "json"


def _token_dict(prompt: str, raw: Mapping[str, Any] | str, usage: object) -> dict[str, int | bool]:
    real = _real_usage(usage) or _real_usage(raw.get("usage") if isinstance(raw, Mapping) else None)
    if real is not None:
        return real
    input_tokens = _estimate_tokens(prompt)
    output_tokens = _estimate_tokens(_response_text(raw)[0])
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "estimated": True,
    }


def _real_usage(value: object) -> dict[str, int | bool] | None:
    if not isinstance(value, Mapping):
        return None
    usage = value.get("usage")
    data = usage if isinstance(usage, Mapping) else value
    total = _int_token(data, "total_tokens", "tokens")
    input_tokens = _int_token(data, "input_tokens", "prompt_tokens") or 0
    output_tokens = _int_token(data, "output_tokens", "completion_tokens")
    if total is None and output_tokens is not None:
        total = input_tokens + output_tokens
    if total is None:
        return None
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens if output_tokens is not None else max(0, total - input_tokens),
        "total_tokens": total,
        "estimated": False,
    }


def _int_token(value: Mapping[str, Any], *keys: str) -> int | None:
    for key in keys:
        raw = value.get(key)
        if isinstance(raw, bool):
            continue
        if isinstance(raw, int):
            return raw
        if isinstance(raw, float) and raw.is_integer():
            return int(raw)
    return None


def _estimate_tokens(value: str) -> int:
    return max(1, (len(value) + 3) // 4)
