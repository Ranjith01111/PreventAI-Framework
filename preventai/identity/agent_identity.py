"""Registered identities without credential material."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping
from uuid import UUID


def _identity_id(value: str, field: str = "identity_id") -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 256:
        raise ValueError(f"{field} must be a non-empty string")
    if any(ord(char) < 32 for char in value):
        raise ValueError(f"{field} contains control characters")
    return value


@dataclass(frozen=True, slots=True)
class AgentIdentity:
    agent_id: str
    name: str | None = None
    enabled: bool = True
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        _identity_id(self.agent_id, "agent_id")
        if self.name is not None:
            _identity_id(self.name, "name")
        if not isinstance(self.enabled, bool):
            raise ValueError("enabled must be boolean")
        metadata = {} if self.metadata is None else dict(self.metadata)
        if not isinstance(self.metadata, (Mapping, type(None))):
            raise ValueError("metadata must be an object")
        object.__setattr__(self, "metadata", MappingProxyType(metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "enabled": self.enabled,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class ToolIdentity:
    tool_id: str
    agent_id: str
    enabled: bool = True

    def __post_init__(self) -> None:
        _identity_id(self.tool_id, "tool_id")
        _identity_id(self.agent_id, "agent_id")
        if not isinstance(self.enabled, bool):
            raise ValueError("enabled must be boolean")
