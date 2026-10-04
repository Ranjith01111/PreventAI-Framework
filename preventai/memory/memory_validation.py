"""Validation helpers for deterministic in-process memory."""

from __future__ import annotations

from collections.abc import Mapping
import json
import math
from typing import Any


def text(value: Any, field: str, *, max_length: int = 256) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > max_length:
        raise ValueError(f"{field} must be a non-empty string")
    if any(ord(char) < 32 and char not in "\t\n\r" for char in value):
        raise ValueError(f"{field} contains control characters")
    return value


def json_value(value: Any, field: str = "value") -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{field} contains a non-finite number")
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            json_value(item, field)
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            text(key, f"{field} key")
            json_value(item, field)
        return
    raise ValueError(f"{field} contains an unsupported value")


def encoded_size(value: Any) -> int:
    json_value(value)
    return len(json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode())
