"""Strict, deterministic Phase 1 security request contract."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
import json
import math
from typing import Any, ClassVar, Mapping


class ContractValidationError(ValueError):
    """Raised when a security contract is malformed or unsafe."""


class DataClassification(str, Enum):
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"
    PRIVATE = "private"
    SENSITIVE = "sensitive"
    SECRET = "secret"


class SourceTrust(str, Enum):
    UNTRUSTED = "untrusted"
    UNKNOWN = "unknown"
    TRUSTED = "trusted"
    VERIFIED = "verified"


class RequestedBy(str, Enum):
    AGENT = "agent"
    USER = "user"
    SYSTEM = "system"
    DELEGATED_AGENT = "delegated_agent"


def _text(value: Any, field: str, *, max_length: int = 256) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > max_length:
        raise ContractValidationError(f"{field} must be a non-empty string")
    if any(ord(char) < 32 and char not in "\t\n\r" for char in value):
        raise ContractValidationError(f"{field} contains control characters")
    return value


def _enum(value: Any, enum_type: type[Enum], field: str) -> Enum:
    if not isinstance(value, enum_type):
        raise ContractValidationError(f"{field} must be a {enum_type.__name__}")
    return value


def _json_value(value: Any, field: str) -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if math.isfinite(value):
            return
        raise ContractValidationError(f"{field} contains a non-finite number")
    if isinstance(value, list):
        for item in value:
            _json_value(item, field)
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise ContractValidationError(f"{field} keys must be non-empty strings")
            _json_value(item, field)
        return
    raise ContractValidationError(f"{field} contains a non-JSON value")


def _mapping(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ContractValidationError(f"{field} must be an object")
    result = dict(value)
    _json_value(result, field)
    return result


@dataclass(frozen=True, slots=True)
class AgentIdentity:
    agent_id: str
    identity_type: str

    def __post_init__(self) -> None:
        _text(self.agent_id, "agent_id")
        _text(self.identity_type, "identity_type")

    def to_dict(self) -> dict[str, str]:
        return {"agent_id": self.agent_id, "identity_type": self.identity_type}


@dataclass(frozen=True, slots=True)
class SessionIdentity:
    session_id: str

    def __post_init__(self) -> None:
        _text(self.session_id, "session_id")

    def to_dict(self) -> dict[str, str]:
        return {"session_id": self.session_id}


@dataclass(frozen=True, slots=True)
class Action:
    name: str
    parameters: Mapping[str, Any]

    def __post_init__(self) -> None:
        _text(self.name, "action.name")
        object.__setattr__(self, "parameters", _mapping(self.parameters, "action.parameters"))

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "parameters": dict(self.parameters)}


@dataclass(frozen=True, slots=True)
class Resource:
    resource_id: str
    resource_type: str

    def __post_init__(self) -> None:
        _text(self.resource_id, "resource.resource_id")
        _text(self.resource_type, "resource.resource_type")

    def to_dict(self) -> dict[str, str]:
        return {"resource_id": self.resource_id, "resource_type": self.resource_type}


@dataclass(frozen=True, slots=True)
class Tool:
    name: str
    version: str

    def __post_init__(self) -> None:
        _text(self.name, "tool.name")
        _text(self.version, "tool.version")

    def to_dict(self) -> dict[str, str]:
        return {"name": self.name, "version": self.version}


@dataclass(frozen=True, slots=True)
class DataMetadata:
    classification: DataClassification
    source_trust: SourceTrust

    def __post_init__(self) -> None:
        _enum(self.classification, DataClassification, "data.classification")
        _enum(self.source_trust, SourceTrust, "data.source_trust")

    def to_dict(self) -> dict[str, str]:
        return {
            "classification": self.classification.value,
            "source_trust": self.source_trust.value,
        }


@dataclass(frozen=True, slots=True)
class SecurityRequest:
    """A security-sensitive action request; this class performs no evaluation."""

    SUPPORTED_PROTOCOL_VERSIONS: ClassVar[frozenset[str]] = frozenset({"1.0"})

    protocol_version: str
    request_id: str
    timestamp: datetime
    agent: AgentIdentity
    session: SessionIdentity
    requested_by: RequestedBy
    action: Action
    resource: Resource
    tool: Tool
    data: DataMetadata
    context: Mapping[str, Any]
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.protocol_version not in self.SUPPORTED_PROTOCOL_VERSIONS:
            raise ContractValidationError("unsupported protocol_version")
        _text(self.request_id, "request_id")
        if not isinstance(self.timestamp, datetime) or self.timestamp.tzinfo is None:
            raise ContractValidationError("timestamp must be timezone-aware")
        if self.timestamp.utcoffset() is None:
            raise ContractValidationError("timestamp must have a valid timezone")
        _enum(self.requested_by, RequestedBy, "requested_by")
        for field, value, expected in (
            ("agent", self.agent, AgentIdentity),
            ("session", self.session, SessionIdentity),
            ("action", self.action, Action),
            ("resource", self.resource, Resource),
            ("tool", self.tool, Tool),
            ("data", self.data, DataMetadata),
        ):
            if not isinstance(value, expected):
                raise ContractValidationError(f"{field} must be a {expected.__name__}")
        object.__setattr__(self, "context", _mapping(self.context, "context"))
        object.__setattr__(self, "metadata", _mapping(self.metadata, "metadata"))

    def to_dict(self) -> dict[str, Any]:
        timestamp = self.timestamp.astimezone(timezone.utc).isoformat(
            timespec="microseconds"
        ).replace("+00:00", "Z")
        return {
            "protocol_version": self.protocol_version,
            "request_id": self.request_id,
            "timestamp": timestamp,
            "agent": self.agent.to_dict(),
            "session": self.session.to_dict(),
            "requested_by": self.requested_by.value,
            "action": self.action.to_dict(),
            "resource": self.resource.to_dict(),
            "tool": self.tool.to_dict(),
            "data": self.data.to_dict(),
            "context": dict(self.context),
            "metadata": dict(self.metadata),
        }

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(), ensure_ascii=True, sort_keys=True, separators=(",", ":")
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> SecurityRequest:
        if not isinstance(value, Mapping):
            raise ContractValidationError("request must be an object")
        expected = {
            "protocol_version", "request_id", "timestamp", "agent", "session",
            "requested_by", "action", "resource", "tool", "data", "context", "metadata",
        }
        if set(value) != expected:
            raise ContractValidationError("request fields do not exactly match the contract")
        try:
            timestamp = datetime.fromisoformat(value["timestamp"].replace("Z", "+00:00"))
            agent = value["agent"]
            session = value["session"]
            action = value["action"]
            resource = value["resource"]
            tool = value["tool"]
            data = value["data"]
            if not all(isinstance(item, Mapping) for item in (agent, session, action, resource, tool, data)):
                raise ContractValidationError("nested contract fields must be objects")
            if set(agent) != {"agent_id", "identity_type"} or set(session) != {"session_id"}:
                raise ContractValidationError("invalid identity fields")
            if set(action) != {"name", "parameters"} or set(resource) != {"resource_id", "resource_type"}:
                raise ContractValidationError("invalid action or resource fields")
            if set(tool) != {"name", "version"} or set(data) != {"classification", "source_trust"}:
                raise ContractValidationError("invalid tool or data fields")
            return cls(
                protocol_version=value["protocol_version"],
                request_id=value["request_id"],
                timestamp=timestamp,
                agent=AgentIdentity(**agent),
                session=SessionIdentity(**session),
                requested_by=RequestedBy(value["requested_by"]),
                action=Action(**action),
                resource=Resource(**resource),
                tool=Tool(**tool),
                data=DataMetadata(
                    classification=DataClassification(data["classification"]),
                    source_trust=SourceTrust(data["source_trust"]),
                ),
                context=value["context"],
                metadata=value["metadata"],
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ContractValidationError("invalid security request") from exc

    @classmethod
    def from_json(cls, value: str) -> SecurityRequest:
        if not isinstance(value, str):
            raise ContractValidationError("JSON input must be a string")
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ContractValidationError("invalid JSON") from exc
        return cls.from_dict(decoded)
