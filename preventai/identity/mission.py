"""Deterministic mission and action-scope enforcement."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import hashlib
import unicodedata
import threading
from types import MappingProxyType
from typing import Any, Iterator, Mapping

from ..audit.event import AuditEventType
from ..audit.logger import AuditLogger

_APPLICATION_CONTEXT_TOKEN = object()
from ..core.decision import Decision, DecisionReason, DecisionType, Severity


class MissionState(str, Enum):
    ACTIVE = "active"
    REVOKED = "revoked"
    EXPIRED = "expired"


class MissionError(ValueError):
    """Raised for invalid mission administration operations."""


def _canonical_identifier(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 256:
        raise MissionError(f"{field_name} must be a non-empty string")
    normalized = unicodedata.normalize("NFKC", value).strip().casefold()
    if any(ord(char) < 32 for char in normalized):
        raise MissionError(f"{field_name} contains control characters")
    segments = normalized.replace("\\", "/").split("/")
    if any(segment in {".", ".."} for segment in segments):
        raise MissionError(f"{field_name} contains a traversal segment")
    return normalized


def _freeze(value: Any, field_name: str = "context_constraints") -> Any:
    if isinstance(value, str):
        return (
            value
            if field_name == "metadata"
            else _canonical_identifier(value, field_name)
        )
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            raise MissionError(f"{field_name} contains an invalid number")
        return value
    if isinstance(value, Mapping):
        frozen = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise MissionError(f"{field_name} keys must be non-empty strings")
            frozen[_canonical_identifier(key, field_name)] = _freeze(item, field_name)
        return MappingProxyType(frozen)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item, field_name) for item in value)
    raise MissionError(f"{field_name} contains an unsupported value")


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def _scope(values: tuple[str, ...], fallback: str | None, field_name: str) -> tuple[str, ...]:
    if not isinstance(values, tuple):
        raise MissionError(f"{field_name} must be a tuple of strings")
    candidates = values or ((fallback,) if fallback is not None else ())
    if not candidates:
        raise MissionError(f"{field_name} must not be empty")
    normalized_values = []
    for value in candidates:
        if not isinstance(value, str) or not value.strip():
            raise MissionError(f"{field_name} must contain non-empty strings")
        if "*" in value and value != "*":
            raise MissionError(f"{field_name} supports only the full-scope wildcard")
        normalized_values.append(
            "*" if value == "*" else _canonical_identifier(value, field_name)
        )
    normalized = tuple(normalized_values)
    if "*" in normalized and len(normalized) != 1:
        raise MissionError(f"{field_name} cannot combine wildcard and explicit scopes")
    return tuple(dict.fromkeys(normalized))


@dataclass(frozen=True, slots=True)
class MissionBinding:
    agent_id: str
    mission_id: str
    mission_version: int
    context: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "agent_id", _canonical_identifier(self.agent_id, "agent_id"))
        object.__setattr__(self, "mission_id", _canonical_identifier(self.mission_id, "mission_id"))
        if (
            not isinstance(self.mission_version, int)
            or isinstance(self.mission_version, bool)
            or self.mission_version < 1
        ):
            raise MissionError("mission_version must be a positive integer")
        if not isinstance(self.context, Mapping):
            raise MissionError("mission binding context must be a mapping")
        object.__setattr__(self, "context", _freeze(self.context, "binding_context"))


@dataclass(frozen=True, slots=True)
class Mission:
    mission_id: str
    agent_id: str
    action: str = ""
    resource: str = ""
    tool: str = ""
    destination: str | None = None
    data_scope: tuple[str, ...] = ()
    version: int = 1
    issued_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    revoked: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)
    allowed_actions: tuple[str, ...] = ()
    allowed_resources: tuple[str, ...] = ()
    allowed_tools: tuple[str, ...] = ()
    allowed_destinations: tuple[str, ...] = ()
    allowed_data_classes: tuple[str, ...] = ()
    context_constraints: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "mission_id", _canonical_identifier(self.mission_id, "mission_id"))
        object.__setattr__(self, "agent_id", _canonical_identifier(self.agent_id, "agent_id"))
        if not isinstance(self.version, int) or isinstance(self.version, bool) or self.version < 1:
            raise MissionError("mission version must be a positive integer")
        if not isinstance(self.issued_at, datetime) or self.issued_at.tzinfo is None:
            raise MissionError("issued_at must be timezone-aware")
        if not isinstance(self.expires_at, datetime) or self.expires_at.tzinfo is None:
            raise MissionError("expires_at must be timezone-aware")
        if self.expires_at <= self.issued_at:
            raise MissionError("expires_at must be later than issued_at")
        if not isinstance(self.revoked, bool):
            raise MissionError("revoked must be boolean")
        if not isinstance(self.data_scope, tuple) or any(
            not isinstance(item, str) or not item.strip() for item in self.data_scope
        ):
            raise MissionError("data_scope must be a tuple of non-empty strings")
        if not isinstance(self.metadata, Mapping):
            raise MissionError("metadata must be a mapping")
        if not isinstance(self.context_constraints, Mapping):
            raise MissionError("context_constraints must be a mapping")
        actions = _scope(self.allowed_actions, self.action, "allowed_actions")
        resources = _scope(self.allowed_resources, self.resource, "allowed_resources")
        tools = _scope(self.allowed_tools, self.tool, "allowed_tools")
        destinations = _scope(
            self.allowed_destinations,
            self.destination or "*",
            "allowed_destinations",
        )
        raw_data_classes = self.allowed_data_classes or self.data_scope
        if raw_data_classes:
            data_classes = _scope(tuple(raw_data_classes), None, "allowed_data_classes")
        else:
            data_classes = ()
        object.__setattr__(self, "allowed_actions", actions)
        object.__setattr__(self, "allowed_resources", resources)
        object.__setattr__(self, "allowed_tools", tools)
        object.__setattr__(self, "allowed_destinations", destinations)
        object.__setattr__(self, "allowed_data_classes", data_classes)
        object.__setattr__(self, "action", actions[0] if len(actions) == 1 else "*")
        object.__setattr__(self, "resource", resources[0] if len(resources) == 1 else "*")
        object.__setattr__(self, "tool", tools[0] if len(tools) == 1 else "*")
        object.__setattr__(
            self,
            "destination",
            destinations[0] if len(destinations) == 1 and destinations[0] != "*" else None,
        )
        object.__setattr__(self, "data_scope", data_classes)
        object.__setattr__(self, "metadata", _freeze(self.metadata, "metadata"))
        object.__setattr__(
            self,
            "context_constraints",
            _freeze(self.context_constraints, "context_constraints"),
        )

    @property
    def state(self) -> MissionState:
        now = datetime.now(timezone.utc)
        if self.revoked:
            return MissionState.REVOKED
        if self.expires_at <= now:
            return MissionState.EXPIRED
        return MissionState.ACTIVE

    def matches(
        self,
        *,
        action: str | None = None,
        resource: str | None = None,
        tool: str | None = None,
        destination: str | None = None,
        data: Mapping[str, Any] | str | None = None,
        context: Mapping[str, Any] | None = None,
        defer_destination: bool = False,
        defer_data: bool = False,
        defer_context: bool = False,
    ) -> bool:
        if action is not None and not _scope_matches(self.allowed_actions, action):
            return False
        if resource is not None and not _scope_matches(self.allowed_resources, resource):
            return False
        if tool is not None and not _scope_matches(self.allowed_tools, tool):
            return False
        if destination is not None:
            if not _scope_matches(self.allowed_destinations, destination):
                return False
        elif self.allowed_destinations != ("*",) and not defer_destination:
            return False
        if self.allowed_data_classes and not defer_data:
            if data is None:
                return False
            if isinstance(data, str):
                scope_values = (_canonical_identifier(data, "data_class"),)
            elif isinstance(data, Mapping):
                value = (
                    data.get("classification")
                    or data.get("class")
                    or data.get("data")
                    or data.get("data_class")
                    or data.get("classification_name")
                )
                if isinstance(value, Enum):
                    value = value.value
                scope_values = (
                    (_canonical_identifier(value, "data_class"),)
                    if isinstance(value, str)
                    else ()
                )
            elif isinstance(data, (list, tuple)):
                scope_values = tuple(
                    _canonical_identifier(item.value if isinstance(item, Enum) else item, "data_class")
                    for item in data
                    if isinstance(item, (str, Enum))
                )
            else:
                if isinstance(data, Enum):
                    scope_values = (_canonical_identifier(data.value, "data_class"),)
                else:
                    scope_values = ()
            if not scope_values or not any(
                _scope_matches(self.allowed_data_classes, item) for item in scope_values
            ):
                return False
        if self.context_constraints and not defer_context:
            if not isinstance(context, Mapping):
                return False
            normalized_context = {
                _canonical_identifier(key, "context"): _freeze(value, "context")
                for key, value in context.items()
            }
            if any(
                key not in normalized_context or normalized_context[key] != expected
                for key, expected in self.context_constraints.items()
            ):
                return False
        return True

    def allows_capability(self, capability: Any) -> bool:
        """Return whether a delegation capability stays within this mission."""
        if not hasattr(capability, "action"):
            return False
        scopes = (
            (self.allowed_actions, capability.action),
            (self.allowed_resources, capability.resource),
            (self.allowed_tools, capability.tool),
        )
        for mission_scope, requested in scopes:
            if requested == "*":
                if mission_scope != ("*",):
                    return False
            elif not _scope_matches(mission_scope, requested):
                return False
        for mission_scope, requested_values in (
            (self.allowed_destinations, capability.destinations),
            (self.allowed_data_classes or ("*",), capability.data_classes),
        ):
            if requested_values == ("*",):
                if mission_scope != ("*",):
                    return False
            elif mission_scope != ("*",) and not set(requested_values).issubset(mission_scope):
                return False
        return all(
            key in capability.context_constraints
            and capability.context_constraints[key] == value
            for key, value in self.context_constraints.items()
        )

    @property
    def mission_version(self) -> int:
        return self.version

    def to_dict(self) -> dict[str, Any]:
        return {
            "mission_id": self.mission_id,
            "agent_id": self.agent_id,
            "action": self.action,
            "resource": self.resource,
            "tool": self.tool,
            "destination": self.destination,
            "data_scope": list(self.data_scope),
            "version": self.version,
            "issued_at": self.issued_at.isoformat().replace("+00:00", "Z"),
            "expires_at": self.expires_at.isoformat().replace("+00:00", "Z"),
            "revoked": self.revoked,
            "metadata": _thaw(self.metadata),
            "allowed_actions": list(self.allowed_actions),
            "allowed_resources": list(self.allowed_resources),
            "allowed_tools": list(self.allowed_tools),
            "allowed_destinations": list(self.allowed_destinations),
            "allowed_data_classes": list(self.allowed_data_classes),
            "context_constraints": _thaw(self.context_constraints),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "Mission":
        if not isinstance(value, Mapping):
            raise MissionError("mission must be a mapping")
        required = {
            "mission_id", "agent_id", "action", "resource", "tool", "version",
            "issued_at", "expires_at", "revoked",
        }
        if not required.issubset(value):
            raise MissionError("mission fields do not exactly match the contract")
        optional = {
            "destination", "data_scope", "metadata", "allowed_actions",
            "allowed_resources", "allowed_tools", "allowed_destinations",
            "allowed_data_classes", "context_constraints",
        }
        if set(value) - required - optional:
            raise MissionError("mission fields do not exactly match the contract")
        if not isinstance(value["revoked"], bool):
            raise MissionError("revoked must be boolean")
        try:
            return cls(
                mission_id=value["mission_id"],
                agent_id=value["agent_id"],
                action=value.get("action", "*"),
                resource=value.get("resource", "*"),
                tool=value.get("tool", "*"),
                destination=value.get("destination"),
                data_scope=tuple(value.get("data_scope", ())),
                version=value["version"],
                issued_at=datetime.fromisoformat(str(value["issued_at"]).replace("Z", "+00:00")),
                expires_at=datetime.fromisoformat(str(value["expires_at"]).replace("Z", "+00:00")),
                revoked=value["revoked"],
                metadata=value.get("metadata", {}),
                allowed_actions=tuple(value.get("allowed_actions", ())),
                allowed_resources=tuple(value.get("allowed_resources", ())),
                allowed_tools=tuple(value.get("allowed_tools", ())),
                allowed_destinations=tuple(value.get("allowed_destinations", ())),
                allowed_data_classes=tuple(value.get("allowed_data_classes", ())),
                context_constraints=value.get("context_constraints", {}),
            )
        except (TypeError, ValueError) as exc:
            raise MissionError("invalid mission") from exc


def _scope_matches(scope: tuple[str, ...], candidate: str) -> bool:
    canonical = _canonical_identifier(candidate, "requested_scope")
    return scope == ("*",) or canonical in scope


def _scope_expands(previous: tuple[str, ...], current: tuple[str, ...]) -> bool:
    if previous == ("*",):
        return False
    return current == ("*",) or not set(current).issubset(previous)


def _mission_expands(previous: Mission, current: Mission) -> bool:
    dimensions = (
        (previous.allowed_actions, current.allowed_actions),
        (previous.allowed_resources, current.allowed_resources),
        (previous.allowed_tools, current.allowed_tools),
        (previous.allowed_destinations, current.allowed_destinations),
        (previous.allowed_data_classes or ("*",), current.allowed_data_classes or ("*",)),
    )
    if any(_scope_expands(before, after) for before, after in dimensions):
        return True
    return not all(
        key in current.context_constraints and current.context_constraints[key] == value
        for key, value in previous.context_constraints.items()
    )


class MissionRegistry:
    """Thread-safe registry for deterministic mission scoping."""

    __slots__ = (
        "_missions", "_by_agent", "_bindings", "_revoked", "_lock",
        "_execution_lock", "_execution_depth", "_audit", "_audit_failed",
        "_application_owned", "require_missions",
    )

    def __init__(
        self,
        *,
        audit_logger: AuditLogger | None = None,
        require_missions: bool = True,
        _owner_token: object | None = None,
    ) -> None:
        if audit_logger is not None and not isinstance(audit_logger, AuditLogger):
            raise TypeError("audit_logger must be an AuditLogger")
        if require_missions is not True:
            raise MissionError("mission enforcement is mandatory")
        self._missions: dict[tuple[str, int], Mission] = {}
        self._by_agent: dict[str, set[str]] = {}
        self._bindings: dict[str, MissionBinding] = {}
        self._revoked: set[tuple[str, int]] = set()
        self._lock = threading.RLock()
        self._execution_lock = threading.RLock()
        self._execution_depth = threading.local()
        self._audit = audit_logger or AuditLogger()
        self._audit_failed = False
        self._application_owned = _owner_token is _APPLICATION_CONTEXT_TOKEN
        self.require_missions = True

    @property
    def application_owned(self) -> bool:
        return self._application_owned

    def register(self, mission: Mission) -> Mission:
        if not isinstance(mission, Mission):
            raise TypeError("mission must be a Mission")
        with self._execution_lock:
            self._ensure_not_reentrant_update()
            with self._lock:
                key = (mission.mission_id, mission.version)
                if key in self._missions:
                    raise MissionError("mission version already registered")
                versions = [
                    version for mid, version in self._missions if mid == mission.mission_id
                ]
                if mission.version != max(versions, default=0) + 1:
                    raise MissionError("mission versions must be registered sequentially")
                if versions and self._missions[(mission.mission_id, max(versions))].agent_id != mission.agent_id:
                    raise MissionError("mission agent binding cannot change between versions")
                expands_scope = bool(
                    versions
                    and _mission_expands(
                        self._missions[(mission.mission_id, max(versions))], mission
                    )
                )
                self._missions[key] = mission
                self._by_agent.setdefault(mission.agent_id, set()).add(mission.mission_id)
                self._record_transition(
                    mission.mission_id,
                    mission.agent_id,
                    mission.version,
                    MissionState.ACTIVE,
                    "scope_escalation" if expands_scope else "register",
                )
                return mission

    def get(self, mission_id: str, version: int | None = None) -> Mission | None:
        mission_id = _canonical_identifier(mission_id, "mission_id")
        if version is not None and (
            not isinstance(version, int) or isinstance(version, bool) or version < 1
        ):
            return None
        with self._lock:
            if version is None:
                version = max(
                    (item_version for mid, item_version in self._missions if mid == mission_id),
                    default=0,
                )
            return self._missions.get((mission_id, version))

    def bind(
        self,
        agent_id: str,
        mission_id: str,
        version: int,
        *,
        context: Mapping[str, Any] | None = None,
    ) -> MissionBinding:
        binding = MissionBinding(
            agent_id,
            mission_id,
            version,
            context if context is not None else {},
        )
        with self._execution_lock:
            self._ensure_not_reentrant_update()
            with self._lock:
                mission = self._missions.get((binding.mission_id, binding.mission_version))
                if mission is None or mission.agent_id != binding.agent_id:
                    raise MissionError("mission binding does not match a registered version")
                self._bindings[binding.agent_id] = binding
                return binding

    def binding_for(self, agent_id: str) -> MissionBinding | None:
        agent_id = _canonical_identifier(agent_id, "agent_id")
        with self._lock:
            return self._bindings.get(agent_id)

    def state_of(self, mission_id: str, version: int) -> MissionState:
        mission = self.get(mission_id, version)
        if mission is None:
            return MissionState.REVOKED
        if (mission.mission_id, mission.version) in self._revoked:
            return MissionState.REVOKED
        return mission.state

    def list_for_agent(self, agent_id: str) -> tuple[Mission, ...]:
        agent_id = _canonical_identifier(agent_id, "agent_id")
        with self._lock:
            mission_ids = self._by_agent.get(agent_id, set())
            return tuple(
                self._missions[(mid, version)]
                for mid in sorted(mission_ids)
                for version in sorted(
                    item_version for item_mid, item_version in self._missions if item_mid == mid
                )
            )

    def revoke(self, mission_id: str, version: int | None = None) -> Mission:
        mission_id = _canonical_identifier(mission_id, "mission_id")
        if version is not None and (
            not isinstance(version, int) or isinstance(version, bool) or version < 1
        ):
            raise MissionError("mission version must be a positive integer")
        with self._execution_lock:
            self._ensure_not_reentrant_update()
            with self._lock:
                versions = sorted(
                    item_version
                    for item_mid, item_version in self._missions
                    if item_mid == mission_id
                    and (version is None or item_version == version)
                )
                if not versions:
                    raise MissionError("unknown mission")
                for item_version in versions:
                    mission = self._missions[(mission_id, item_version)]
                    self._revoked.add((mission.mission_id, mission.version))
                    self._record_transition(
                        mission.mission_id,
                        mission.agent_id,
                        mission.version,
                        MissionState.REVOKED,
                        "revoke",
                    )
                return self._missions[(mission_id, version or versions[-1])]

    def check(
        self,
        agent_id: str,
        *,
        action: str,
        resource: str,
        tool: str,
        destination: str | None = None,
        data: Mapping[str, Any] | str | None = None,
        operation: str = "privileged_operation",
        context: Mapping[str, Any] | None = None,
        mission_id: str | None = None,
        mission_version: int | None = None,
        defer_destination: bool = False,
        defer_data: bool = False,
        defer_context: bool = False,
    ) -> Decision:
        with self._lock:
            if self._audit_failed:
                return self._decision(
                    DecisionReason.MISSION_INVALID, agent_id=None, mission_id=None,
                    mission_version=None, action=action, resource=resource,
                    operation=operation, scope_violation_type="audit_failure",
                )
            try:
                canonical_agent = _canonical_identifier(agent_id, "agent_id")
                binding = self._bindings.get(canonical_agent)
                if mission_version is not None and (
                    not isinstance(mission_version, int)
                    or isinstance(mission_version, bool)
                    or mission_version < 1
                ):
                    binding = None
                if (
                    binding is None
                    or (mission_id is not None and _canonical_identifier(mission_id, "mission_id") != binding.mission_id)
                    or (mission_version is not None and mission_version != binding.mission_version)
                ):
                    return self._decision(
                        DecisionReason.MISSION_VERSION_INVALID, agent_id=canonical_agent,
                        mission_id=mission_id, mission_version=mission_version,
                        action=action, resource=resource, operation=operation,
                        scope_violation_type="mission_binding_invalid",
                    )
                mission = self._missions.get((binding.mission_id, binding.mission_version))
                if mission is None or mission.agent_id != canonical_agent:
                    return self._decision(
                        DecisionReason.MISSION_VERSION_INVALID, agent_id=canonical_agent,
                        mission_id=binding.mission_id, mission_version=binding.mission_version,
                        action=action, resource=resource, operation=operation,
                        scope_violation_type="mission_version_invalid",
                    )
                key = (mission.mission_id, mission.version)
                if key in self._revoked or mission.revoked:
                    return self._decision(
                        DecisionReason.MISSION_REVOKED, agent_id=canonical_agent,
                        mission_id=mission.mission_id, mission_version=mission.version,
                        action=action, resource=resource, operation=operation,
                        scope_violation_type="mission_revoked",
                    )
                if mission.expires_at <= datetime.now(timezone.utc):
                    return self._decision(
                        DecisionReason.MISSION_EXPIRED, agent_id=canonical_agent,
                        mission_id=mission.mission_id, mission_version=mission.version,
                        action=action, resource=resource, operation=operation,
                        scope_violation_type="mission_expired",
                    )
                try:
                    effective_context = dict(binding.context)
                    if context is not None:
                        if not isinstance(context, Mapping):
                            raise MissionError("request context must be a mapping")
                        request_context = {
                            _canonical_identifier(key, "context"): _freeze(value, "context")
                            for key, value in context.items()
                        }
                        if any(
                            key in effective_context and effective_context[key] != value
                            for key, value in request_context.items()
                        ):
                            request_context = {}
                            effective_context = {}
                            matches = False
                        else:
                            effective_context.update(request_context)
                            matches = mission.matches(
                                action=action,
                                resource=resource,
                                tool=tool,
                                destination=destination,
                                data=data,
                                context=effective_context,
                                defer_destination=defer_destination,
                                defer_data=defer_data,
                                defer_context=defer_context,
                            )
                    else:
                        matches = mission.matches(
                            action=action,
                            resource=resource,
                            tool=tool,
                            destination=destination,
                            data=data,
                            context=effective_context,
                            defer_destination=defer_destination,
                            defer_data=defer_data,
                            defer_context=defer_context,
                        )
                except Exception:
                    return self._decision(
                        DecisionReason.MISSION_INVALID, agent_id=canonical_agent,
                        mission_id=mission.mission_id, mission_version=mission.version,
                        action=action, resource=resource, operation=operation,
                        scope_violation_type="scope_parse_failure",
                    )

                if not matches:
                    return self._decision(
                        DecisionReason.MISSION_SCOPE_DENIED, agent_id=canonical_agent,
                        mission_id=mission.mission_id, mission_version=mission.version,
                        action=action, resource=resource, operation=operation,
                        scope_violation_type="scope_mismatch",
                    )
                return self._decision(
                    None, agent_id=canonical_agent, mission_id=mission.mission_id,
                    mission_version=mission.version, action=action, resource=resource,
                    operation=operation, scope_violation_type=None,
                )
            except Exception:
                return self._decision(
                    DecisionReason.MISSION_INVALID, agent_id=None, mission_id=None,
                    mission_version=None, action=action, resource=resource,
                    operation=operation, scope_violation_type="canonicalization_failure",
                )

    def allows_capability(self, agent_id: str, capability: Any) -> bool:
        with self._lock:
            binding = self._bindings.get(_canonical_identifier(agent_id, "agent_id"))
            if binding is None:
                return False
            mission = self._missions.get((binding.mission_id, binding.mission_version))
            if mission is None or mission.state is not MissionState.ACTIVE:
                return False
            return mission.allows_capability(capability)

    @contextmanager
    def execution_guard(self, agent_id: str, **request: Any) -> Iterator[Decision]:
        with self._execution_lock:
            depth = getattr(self._execution_depth, "value", 0)
            self._execution_depth.value = depth + 1
            try:
                yield self.check(agent_id, **request)
            finally:
                self._execution_depth.value = depth

    def _ensure_not_reentrant_update(self) -> None:
        if getattr(self._execution_depth, "value", 0):
            raise MissionError("mission administration is forbidden during guarded execution")

    def _decision(
        self,
        reason: DecisionReason | None,
        *,
        agent_id: str | None,
        mission_id: str | None,
        mission_version: int | None,
        action: str,
        resource: str,
        operation: str,
        scope_violation_type: str | None,
    ) -> Decision:
        allowed = reason is None
        selected_reason = DecisionReason.REQUEST_ALLOWED if allowed else reason
        event_type = (
            AuditEventType.MISSION_ALLOWED if allowed else
            AuditEventType.MISSION_SCOPE_DENIED if reason is DecisionReason.MISSION_SCOPE_DENIED else
            AuditEventType.MISSION_REVOKED if reason is DecisionReason.MISSION_REVOKED else
            AuditEventType.MISSION_EXPIRED if reason is DecisionReason.MISSION_EXPIRED else
            AuditEventType.MISSION_VERSION_INVALID if reason is DecisionReason.MISSION_VERSION_INVALID else
            AuditEventType.MISSION_INVALID
        )
        try:
            action_digest = hashlib.sha256(
                unicodedata.normalize("NFKC", action).encode("utf-8")
            ).hexdigest()
            resource_digest = hashlib.sha256(
                unicodedata.normalize("NFKC", resource).encode("utf-8")
            ).hexdigest()
            event = self._audit.record(
                event_type,
                agent_id=agent_id,
                action=f"sha256:{action_digest}",
                resource=f"sha256:{resource_digest}",
                decision=DecisionType.ALLOW.value if allowed else DecisionType.BLOCK.value,
                reason=selected_reason.value,
                metadata={
                    "mission_id": mission_id or "unknown",
                    "mission_version": mission_version or 0,
                    "operation": operation,
                    "scope_violation_type": scope_violation_type or "none",
                },
            )
            if event is None:
                raise RuntimeError("mandatory mission audit event was not recorded")
        except Exception:
            self._audit_failed = True
            return Decision(
                DecisionType.BLOCK,
                DecisionReason.MISSION_INVALID,
                severity=Severity.CRITICAL,
                metadata={"operation": operation, "stage": "mission_audit"},
            )
        return Decision(
            DecisionType.ALLOW if allowed else DecisionType.BLOCK,
            selected_reason,
            severity=Severity.INFO if allowed else Severity.HIGH,
            metadata={
                "operation": operation,
                "agent_id": agent_id,
                "mission_id": mission_id,
                "mission_version": mission_version,
                "scope_violation_type": scope_violation_type,
            },
        )

    def _record_transition(
        self,
        mission_id: str,
        agent_id: str,
        mission_version: int,
        state: MissionState,
        reason: str,
    ) -> None:
        if self._audit is None:
            return
        try:
            event = self._audit.record(
                (
                    AuditEventType.MISSION_SCOPE_ESCALATION
                    if reason == "scope_escalation"
                    else AuditEventType.MISSION_VALIDATION
                ),
                agent_id=agent_id,
                decision=DecisionType.ALLOW.value,
                reason=reason,
                metadata={
                    "mission_id": mission_id,
                    "mission_version": mission_version,
                    "state": state.value,
                },
            )
            if event is None:
                raise RuntimeError("mandatory mission audit event was not recorded")
        except Exception:
            self._audit_failed = True
            raise
