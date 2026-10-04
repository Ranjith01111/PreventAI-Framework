"""Deterministic security boundary for inter-agent messages."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import hmac
import json
import math
import secrets
import threading
import time
from typing import Any, Callable, Mapping, Protocol

from .security_request import ContractValidationError
from ..core.decision import Decision, DecisionType
from ..core.data_release import (
    DataReleaseAuthorization, DataReleaseRequest, DestinationTrust,
    FlowClassification, verify_release_authorization,
)
from ..identity.delegation import Capability, DelegationGrant, DelegationRegistry
from ..identity.lifecycle import AgentLifecycleRegistry
from ..runtime.context import RuntimeController
from ..runtime.loop_detection import ExecutionSignature
from ..audit.event import AuditEventType
from ..audit.logger import AuditLogger

_MESSAGE_TYPES = frozenset({
    "request", "response", "event", "capability_request", "capability_grant",
    "delegation", "error",
})
_IDENTITY_LIMIT = 256


def _json(value: Any, field: str, depth: int = 0) -> None:
    if depth > 16:
        raise ContractValidationError(f"{field} nesting limit exceeded")
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            raise ContractValidationError(f"{field} contains non-finite data")
        return
    if isinstance(value, list):
        for item in value:
            _json(item, field, depth + 1)
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise ContractValidationError(f"{field} has invalid key")
            _json(item, field, depth + 1)
        return
    raise ContractValidationError(f"{field} contains unsupported data")


def _text(value: Any, field: str, limit: int = _IDENTITY_LIMIT) -> str:
    if not isinstance(value, str) or not value or len(value) > limit or any(ord(c) < 32 for c in value):
        raise ContractValidationError(f"{field} is invalid")
    return value


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


@dataclass(frozen=True, slots=True)
class AgentMessagePolicy:
    allowed_message_types: tuple[str, ...] = tuple(sorted(_MESSAGE_TYPES))
    allowed_senders: tuple[str, ...] = ()
    allowed_receivers: tuple[str, ...] = ()
    allowed_tenants: tuple[str, ...] = ()
    allowed_capabilities: tuple[str, ...] = ()
    max_payload_bytes: int = 256 * 1024
    max_message_bytes: int = 512 * 1024
    max_chain_depth: int = 8
    max_concurrent: int = 1
    ttl_seconds: float = 30.0
    one_time: bool = True

    def __post_init__(self) -> None:
        for field in ("allowed_message_types", "allowed_senders", "allowed_receivers",
                      "allowed_tenants", "allowed_capabilities"):
            values = getattr(self, field)
            if not isinstance(values, tuple) or any(not isinstance(v, str) or not v for v in values):
                raise ValueError(f"{field} is invalid")
        if any(v not in _MESSAGE_TYPES for v in self.allowed_message_types):
            raise ValueError("unsupported message type")
        for field in ("max_payload_bytes", "max_message_bytes", "max_chain_depth", "max_concurrent"):
            value = getattr(self, field)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{field} must be positive")
        if (
            not isinstance(self.ttl_seconds, (int, float))
            or not math.isfinite(self.ttl_seconds)
            or self.ttl_seconds <= 0
        ):
            raise ValueError("ttl_seconds must be positive")
        if not isinstance(self.one_time, bool):
            raise ValueError("one_time must be boolean")


@dataclass(frozen=True, slots=True)
class AgentMessage:
    message_id: str
    message_type: str
    sender_agent_id: str
    receiver_agent_id: str
    sender_session_id: str
    receiver_session_id: str | None
    conversation_id: str
    parent_message_id: str | None
    timestamp: datetime
    expires_at: float
    payload: Mapping[str, Any]
    capability_scope: tuple[str, ...] = ()
    delegation_scope: tuple[str, ...] = ()
    classification: str = "internal"
    correlation_id: str | None = None
    protocol_version: str = "1.0"
    tenant_id: str | None = None
    chain_depth: int = 0
    parent_authority: DelegationGrant | None = None
    delegation_authority: DelegationGrant | None = None

    def __post_init__(self) -> None:
        for field in ("message_id", "sender_agent_id", "receiver_agent_id",
                      "sender_session_id", "conversation_id", "classification", "protocol_version"):
            _text(getattr(self, field), field)
        if self.receiver_session_id is not None:
            _text(self.receiver_session_id, "receiver_session_id")
        if self.parent_message_id is not None:
            _text(self.parent_message_id, "parent_message_id")
        if self.correlation_id is not None:
            _text(self.correlation_id, "correlation_id")
        if self.tenant_id is not None:
            _text(self.tenant_id, "tenant_id")
        if self.message_type not in _MESSAGE_TYPES:
            raise ContractValidationError("unsupported message type")
        if not isinstance(self.timestamp, datetime) or self.timestamp.tzinfo is None:
            raise ContractValidationError("timestamp must be timezone-aware")
        if (
            not isinstance(self.expires_at, (int, float))
            or not math.isfinite(self.expires_at)
            or self.expires_at <= 0
        ):
            raise ContractValidationError("expires_at is invalid")
        if not isinstance(self.payload, Mapping):
            raise ContractValidationError("payload must be an object")
        _json(dict(self.payload), "payload")
        if not isinstance(self.chain_depth, int) or self.chain_depth < 0:
            raise ContractValidationError("chain_depth is invalid")
        if self.parent_authority is not None and not isinstance(
            self.parent_authority, DelegationGrant
        ):
            raise ContractValidationError("parent_authority is invalid")
        if self.delegation_authority is not None and not isinstance(
            self.delegation_authority, DelegationGrant
        ):
            raise ContractValidationError("delegation_authority is invalid")
        object.__setattr__(self, "capability_scope", tuple(self.capability_scope))
        object.__setattr__(self, "delegation_scope", tuple(self.delegation_scope))

    def canonical(self) -> dict[str, Any]:
        return {
            "message_id": self.message_id, "message_type": self.message_type,
            "sender_agent_id": self.sender_agent_id, "receiver_agent_id": self.receiver_agent_id,
            "sender_session_id": self.sender_session_id, "receiver_session_id": self.receiver_session_id,
            "conversation_id": self.conversation_id, "parent_message_id": self.parent_message_id,
            "timestamp": self.timestamp.astimezone(timezone.utc).isoformat(),
            "expires_at": self.expires_at, "payload": dict(self.payload),
            "capability_scope": self.capability_scope, "delegation_scope": self.delegation_scope,
            "classification": self.classification, "correlation_id": self.correlation_id,
            "protocol_version": self.protocol_version, "tenant_id": self.tenant_id,
            "chain_depth": self.chain_depth,
            "parent_authority": (
                self.parent_authority.to_dict() if self.parent_authority is not None else None
            ),
            "delegation_authority": (
                self.delegation_authority.to_dict()
                if self.delegation_authority is not None else None
            ),
        }

    def digest(self) -> str:
        return hashlib.sha256(_canonical(self.canonical()).encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class AgentMessageResponse:
    message_id: str
    accepted: bool
    payload: Mapping[str, Any] | None = None
    error: str | None = None


@dataclass(frozen=True, slots=True)
class AgentMessageAuthorization:
    message_digest: str
    sender_agent_id: str
    receiver_agent_id: str
    sender_session_id: str
    receiver_session_id: str | None
    message_type: str
    tenant_id: str | None
    sender_mission_id: str
    sender_mission_version: int
    receiver_mission_id: str
    receiver_mission_version: int
    expires_at: float
    nonce: str
    signature: str


@dataclass(frozen=True, slots=True)
class AgentSecurityContext:
    agent_id: str
    session_id: str
    tenant_id: str | None = None
    authenticated: bool = True
    capabilities: tuple[str, ...] = ()
    authentication_proof: bytes = field(
        default_factory=lambda: secrets.token_bytes(32), init=False, repr=False,
        compare=True,
    )


class AgentCommunicationDenied(PermissionError):
    def __init__(self, category: str):
        self.category = category
        super().__init__(category)


class ManagedBoundaryAdapter(Protocol):
    """Narrow adapter to an existing PreventAI-managed execution boundary."""

    boundary_id: str
    boundary: object

    def authorize(self, message: AgentMessage) -> object:
        """Return the tool-specific authorization or raise its denial."""

    def dispatch(self, message: AgentMessage, authorization: object) -> Mapping[str, Any]:
        """Dispatch only through the managed boundary."""


@dataclass(frozen=True, slots=True)
class ManagedBoundaryRegistration:
    boundary_type: str
    boundary_id: str
    adapter: ManagedBoundaryAdapter

    def __post_init__(self) -> None:
        if not self.boundary_type or not self.boundary_id:
            raise ValueError("managed boundary identity is required")
        if getattr(self.adapter, "boundary_id", None) != self.boundary_id:
            raise ValueError("managed boundary adapter identity mismatch")
        if not getattr(self.adapter, "boundary", None):
            raise ValueError("managed boundary instance is required")
        if not callable(getattr(self.adapter, "authorize", None)) or not callable(
            getattr(self.adapter, "dispatch", None)
        ):
            raise ValueError("managed boundary adapter contract is invalid")


class ManagedBoundaryRegistry:
    """Registry of explicitly typed, already-authorized PreventAI boundaries."""

    _ALLOWED_TYPES = frozenset({"mcp", "sandbox", "network", "filesystem", "database"})

    def __init__(self) -> None:
        self._registrations: dict[str, ManagedBoundaryRegistration] = {}

    def register(self, registration: ManagedBoundaryRegistration) -> None:
        if not isinstance(registration, ManagedBoundaryRegistration):
            raise TypeError("registration is required")
        if registration.boundary_type not in self._ALLOWED_TYPES:
            raise ValueError("unsupported managed boundary type")
        expected_modules = {
            "mcp": ("preventai.protocol.mcp", "MCPBoundary"),
            "sandbox": ("preventai.tools.sandbox", "Sandbox"),
            "network": ("preventai.tools.network", "Network"),
            "filesystem": ("preventai.tools.filesystem", "FileSystem"),
            "database": ("preventai.tools.database", "Database"),
        }
        module_name, class_name = expected_modules[registration.boundary_type]
        boundary_type = getattr(__import__(module_name, fromlist=[class_name]), class_name)
        if not isinstance(registration.adapter.boundary, boundary_type):
            raise ValueError("managed boundary instance type mismatch")
        adapter_types = {
            "mcp": MCPBoundaryAdapter,
            "sandbox": SandboxBoundaryAdapter,
            "network": NetworkBoundaryAdapter,
            "filesystem": FileSystemBoundaryAdapter,
            "database": DatabaseBoundaryAdapter,
        }
        if not isinstance(registration.adapter, adapter_types[registration.boundary_type]):
            raise ValueError("managed boundary adapter type mismatch")
        if registration.boundary_id in self._registrations:
            raise ValueError("managed boundary already registered")
        self._registrations[registration.boundary_id] = registration

    def get(self, boundary_id: str, boundary_type: str) -> ManagedBoundaryAdapter:
        registration = self._registrations.get(boundary_id)
        if registration is None or registration.boundary_type != boundary_type:
            raise AgentCommunicationDenied("managed_boundary_denied")
        return registration.adapter


def _require_active(
    lifecycle: AgentLifecycleRegistry | None, agent_id: str,
) -> None:
    if not isinstance(lifecycle, AgentLifecycleRegistry):
        raise AgentCommunicationDenied("lifecycle_denied")
    try:
        if lifecycle.check(agent_id, operation="agent_communication").decision_type.value != "allow":
            raise AgentCommunicationDenied("lifecycle_denied")
    except AgentCommunicationDenied:
        raise
    except Exception as exc:
        raise AgentCommunicationDenied("lifecycle_denied") from exc


class _LifecycleBoundAdapter:
    _lifecycle: AgentLifecycleRegistry | None
    _mission_registry: object | None
    _mission_agent_id: str | None

    def _bind_mission(self, mission_registry: object, agent_id: str) -> None:
        from ..identity.mission import MissionRegistry
        if (
            not isinstance(mission_registry, MissionRegistry)
            or not mission_registry.application_owned
            or not isinstance(agent_id, str)
        ):
            raise AgentCommunicationDenied("mission_context_missing")
        if (
            self._mission_registry is not None
            and self._mission_registry is not mission_registry
        ):
            raise AgentCommunicationDenied("mission_context_mismatch")
        bind_boundary = getattr(self.boundary, "bind_mission_context", None)
        if not callable(bind_boundary):
            raise AgentCommunicationDenied("mission_context_missing")
        try:
            bind_boundary(mission_registry, agent_id, tool=self.boundary_id)
        except TypeError:
            bind_boundary(mission_registry, agent_id)
        self._mission_registry = mission_registry
        self._mission_agent_id = agent_id

    def _require_mission(self) -> None:
        if self._mission_registry is None or self._mission_agent_id is None:
            raise AgentCommunicationDenied("mission_context_missing")

    def _bind_lifecycle(self, lifecycle: AgentLifecycleRegistry) -> None:
        if not isinstance(lifecycle, AgentLifecycleRegistry):
            raise AgentCommunicationDenied("lifecycle_denied")
        if self._lifecycle is not None and self._lifecycle is not lifecycle:
            raise AgentCommunicationDenied("lifecycle_context_mismatch")
        self._lifecycle = lifecycle

    def _check_message_lifecycle(self, message: AgentMessage) -> None:
        _require_active(self._lifecycle, message.sender_agent_id)
        _require_active(self._lifecycle, message.receiver_agent_id)


class MCPBoundaryAdapter(_LifecycleBoundAdapter):
    boundary_type = "mcp"

    def __init__(
        self, boundary: object, *, boundary_id: str,
    ) -> None:
        from .mcp import MCPBoundary
        if not isinstance(boundary, MCPBoundary):
            raise TypeError("boundary must be an MCPBoundary")
        self.boundary = boundary
        self.boundary_id = boundary_id
        self._lifecycle = None
        self._mission_registry = None
        self._mission_agent_id = None

    def authorize(self, message: AgentMessage) -> object:
        self._check_message_lifecycle(message)
        from .mcp import MCPRequest
        payload = dict(message.payload)
        request = self.boundary.parse(
            payload["message"], protocol_version=payload["protocol_version"],
            client_id=payload["client_id"], session_id=payload["session_id"],
        )
        return request, self.boundary.issue(request)

    def dispatch(self, message: AgentMessage, authorization: object) -> Mapping[str, Any]:
        self._check_message_lifecycle(message)
        request, token = authorization
        response = self.boundary.dispatch(request, token)
        return {"request_id": response.request_id, "result": dict(response.result or {})}


class SandboxBoundaryAdapter(_LifecycleBoundAdapter):
    boundary_type = "sandbox"

    def __init__(
        self, boundary: object, *, boundary_id: str,
    ) -> None:
        from ..tools.sandbox import Sandbox
        if not isinstance(boundary, Sandbox):
            raise TypeError("boundary must be a Sandbox")
        self.boundary = boundary
        self.boundary_id = boundary_id
        self._lifecycle = None
        self._mission_registry = None
        self._mission_agent_id = None

    def authorize(self, message: AgentMessage) -> object:
        self._check_message_lifecycle(message)
        from ..tools.sandbox import SandboxCommand, _FRAMEWORK_ISSUER
        payload = dict(message.payload)
        command = SandboxCommand(
            payload["executable"], tuple(payload.get("args", ())),
            payload.get("cwd"), payload.get("environment"), False,
        )
        token = self.boundary._issue_authorization(
            agent_id=message.sender_agent_id, session_id=message.sender_session_id,
            tool_id=self.boundary_id, action=payload["action"],
            resource=payload["resource"], command=command, _issuer=_FRAMEWORK_ISSUER,
        )
        return command, token

    def dispatch(self, message: AgentMessage, authorization: object) -> Mapping[str, Any]:
        self._check_message_lifecycle(message)
        command, token = authorization
        result = self.boundary.execute(
            command, authorization=token, agent_id=message.sender_agent_id,
            session_id=message.sender_session_id, tool_id=self.boundary_id,
            action=dict(message.payload)["action"], resource=dict(message.payload)["resource"],
        )
        return {"decision": result.decision.decision_type.value, "stdout": result.stdout,
                "stderr": result.stderr, "returncode": result.returncode}


class NetworkBoundaryAdapter(_LifecycleBoundAdapter):
    boundary_type = "network"

    def __init__(
        self, boundary: object, *, boundary_id: str,
    ) -> None:
        from ..tools.network import Network
        if not isinstance(boundary, Network):
            raise TypeError("boundary must be a Network")
        self.boundary = boundary
        self.boundary_id = boundary_id
        self._lifecycle = None
        self._mission_registry = None
        self._mission_agent_id = None

    def authorize(self, message: AgentMessage) -> object:
        self._check_message_lifecycle(message)
        from ..tools.network import NetworkRequest, _FRAMEWORK_ISSUER
        payload = dict(message.payload)
        request = NetworkRequest(
            payload["url"], payload.get("method", "GET"),
            payload.get("headers"), str(payload.get("body", "")).encode("utf-8"),
            agent_id=message.sender_agent_id, session_id=message.sender_session_id,
            tool_id=self.boundary_id,
        )
        plan = self.boundary.prepare(request)
        token = self.boundary._issue_authorization(
            request, resolved_addresses=plan.addresses, _issuer=_FRAMEWORK_ISSUER,
        )
        return request, token

    def dispatch(self, message: AgentMessage, authorization: object) -> Mapping[str, Any]:
        self._check_message_lifecycle(message)
        request, token = authorization
        return self.boundary.execute(request, authorization=token).output_value()


class FileSystemBoundaryAdapter(_LifecycleBoundAdapter):
    boundary_type = "filesystem"

    def __init__(
        self, boundary: object, *, boundary_id: str,
    ) -> None:
        from ..tools.filesystem import FileSystem
        if not isinstance(boundary, FileSystem):
            raise TypeError("boundary must be a FileSystem")
        self.boundary = boundary
        self.boundary_id = boundary_id
        self._lifecycle = None
        self._mission_registry = None
        self._mission_agent_id = None

    def authorize(self, message: AgentMessage) -> object:
        self._check_message_lifecycle(message)
        from ..tools.filesystem import FileSystemAuthorization, FileSystemRequest
        payload = dict(message.payload)
        request = FileSystemRequest(
            payload["operation"], payload["path"],
            data=str(payload.get("data", "")).encode("utf-8"),
            overwrite=bool(payload.get("overwrite", False)),
            agent_id=message.sender_agent_id, session_id=message.sender_session_id,
            tool_id=self.boundary_id,
        )
        plan = self.boundary._prepare(request)
        expiry = time.time() + 30.0
        nonce = secrets.token_urlsafe(24)
        signature = self.boundary._sign(self.boundary._authorization_body(request, plan, expiry, nonce))
        return request, FileSystemAuthorization(
            request.digest(), str(plan.path),
            str(plan.destination) if plan.destination else None,
            expiry, nonce, signature,
        )

    def dispatch(self, message: AgentMessage, authorization: object) -> Mapping[str, Any]:
        self._check_message_lifecycle(message)
        request, token = authorization
        return self.boundary.execute(request, authorization=token).output_value()


class DatabaseBoundaryAdapter(_LifecycleBoundAdapter):
    boundary_type = "database"

    def __init__(
        self, boundary: object, *, boundary_id: str,
    ) -> None:
        from ..tools.database import Database
        if not isinstance(boundary, Database):
            raise TypeError("boundary must be a Database")
        self.boundary = boundary
        self.boundary_id = boundary_id
        self._lifecycle = None
        self._mission_registry = None
        self._mission_agent_id = None

    def authorize(self, message: AgentMessage) -> object:
        self._check_message_lifecycle(message)
        from ..tools.database import DatabaseRequest
        payload = dict(message.payload)
        request = DatabaseRequest(
            payload["operation"], payload["database"], payload["schema"], payload["table"],
            tuple(payload.get("columns", ())), tuple(payload.get("parameters", ())),
            payload.get("template", ""), tenant_id=payload.get("tenant_id"),
            agent_id=message.sender_agent_id, session_id=message.sender_session_id,
            tool_id=self.boundary_id,
        )
        self.boundary._prepare(request)
        token = self.boundary._issue_authorization(
            request, agent_id=message.sender_agent_id,
            session_id=message.sender_session_id, tool_id=self.boundary_id,
            expires_at=time.time() + 30.0,
            _capability=self.boundary._issuance_capability,
        )
        return request, token

    def dispatch(self, message: AgentMessage, authorization: object) -> Mapping[str, Any]:
        self._check_message_lifecycle(message)
        request, token = authorization
        return self.boundary.execute(
            request, authorization=token, agent_id=message.sender_agent_id,
            session_id=message.sender_session_id, tool_id=self.boundary_id,
        ).output_value()


class _AgentIssuanceCapability:
    __slots__ = ("_boundary",)

    def __init__(self, boundary: object) -> None:
        self._boundary = boundary


class AgentCommunicationBoundary:
    def __init__(
        self,
        policy: AgentMessagePolicy,
        downstream: Callable[[AgentMessage], Mapping[str, Any]],
        *,
        sender_contexts: Mapping[str, AgentSecurityContext] | None = None,
        policy_check: Callable[[AgentMessage], bool] | None = None,
        permission_check: Callable[[AgentMessage], bool] | None = None,
        delegation_check: Callable[[AgentMessage], bool] | None = None,
        dataflow_check: Callable[[AgentMessage], bool] | None = None,
        runtime_check: Callable[[AgentMessage], bool] | None = None,
        approval_check: Callable[[AgentMessage], bool] | None = None,
        output_check: Callable[[Mapping[str, Any]], bool] | None = None,
        audit_check: Callable[[AgentMessage], bool] | None = None,
        runtime_controller: RuntimeController | None = None,
        audit_logger: AuditLogger | None = None,
        delegation_registry: DelegationRegistry | None = None,
        application_id: str | None = None,
    ) -> None:
        if not isinstance(policy, AgentMessagePolicy) or not callable(downstream):
            raise TypeError("policy and downstream are required")
        self.policy = policy
        self._downstream = downstream
        self._contexts = dict(sender_contexts or {})
        self._secret = secrets.token_bytes(32)
        self._issuance_capability = _AgentIssuanceCapability(self)
        self._revoked: set[str] = set()
        self._used: set[str] = set()
        self._messages: set[str] = set()
        self._lock = threading.BoundedSemaphore(policy.max_concurrent)
        self._state = threading.RLock()
        self._checks = (policy_check, permission_check, delegation_check, dataflow_check, runtime_check, approval_check)
        self._output_check = output_check
        self._audit_check = audit_check
        self._runtime = runtime_controller
        self._audit_logger = audit_logger
        self._delegations = delegation_registry
        self._application_id = application_id
        self._release_issuer = None
        self._lifecycle: AgentLifecycleRegistry | None = None
        self._missions: object | None = None

    def _deny(self, category: str) -> None:
        raise AgentCommunicationDenied(category)

    def _bind_application_context(
        self,
        lifecycle: AgentLifecycleRegistry,
        audit_logger: AuditLogger,
        delegations: DelegationRegistry,
        missions: object | None = None,
    ) -> None:
        if not isinstance(lifecycle, AgentLifecycleRegistry):
            raise TypeError("application lifecycle context is required")
        if not isinstance(audit_logger, AuditLogger) or not isinstance(
            delegations, DelegationRegistry
        ):
            raise TypeError("application security context is invalid")
        if missions is not None:
            from ..identity.mission import MissionRegistry
            if not isinstance(missions, MissionRegistry):
                raise TypeError("application mission context is invalid")
        if self._lifecycle is not None and self._lifecycle is not lifecycle:
            raise ValueError("communication boundary is bound to another lifecycle context")
        if self._audit_logger is not None and self._audit_logger is not audit_logger:
            raise ValueError("communication boundary is bound to another audit context")
        if self._delegations is not None and self._delegations is not delegations:
            raise ValueError("communication boundary is bound to another delegation context")
        self._lifecycle = lifecycle
        self._audit_logger = audit_logger
        self._delegations = delegations
        self._missions = missions
        self._missions = missions

    def _security_integrations_present(self) -> bool:
        return (
            not any(check is None for check in self._checks)
            and self._output_check is not None
            and self._audit_check is not None
            and self._runtime is not None
        )

    def _check_agent_mission(
        self,
        agent_id: str,
        *,
        action: str,
        resource: str,
        tool: str,
        destination: str | None = None,
        defer_destination: bool = False,
        defer_data: bool = True,
        defer_context: bool = True,
    ) -> None:
        from ..identity.mission import MissionRegistry
        if not isinstance(self._missions, MissionRegistry):
            self._deny("mission_context_missing")
        decision = self._missions.check(
            agent_id,
            action=action,
            resource=resource,
            tool=tool,
            destination=destination,
            defer_destination=defer_destination,
            defer_data=defer_data,
            defer_context=defer_context,
            operation="agent_communication",
        )
        if decision.decision_type is not DecisionType.ALLOW:
            raise AgentCommunicationDenied("mission_denied")

    def _managed_mission_scope(
        self,
        managed_adapter: object,
        message: AgentMessage,
        authorization: object,
    ) -> dict[str, Any]:
        payload = dict(message.payload)
        request = authorization[0] if isinstance(authorization, tuple) else None
        boundary_type = getattr(managed_adapter, "boundary_type", None)
        if boundary_type == "mcp" and request is not None:
            return {
                "action": request.method,
                "resource": request.identity,
                "tool": request.identity if request.method == "tools/call" else request.method,
                "destination": request.server_id,
                "data": message.classification,
                "context": {
                    "protocol": "mcp",
                    "tenant": message.tenant_id or "",
                    "message_type": message.message_type,
                },
            }
        if boundary_type == "sandbox":
            return {
                "action": payload["action"],
                "resource": payload["resource"],
                "tool": managed_adapter.boundary_id,
                "destination": None,
                "data": "internal",
                "context": {
                    "protocol": "sandbox",
                    "tenant": message.tenant_id or "",
                    "message_type": message.message_type,
                },
            }
        if boundary_type == "network" and request is not None:
            return {
                "action": request.action,
                "resource": request.destination,
                "tool": managed_adapter.boundary_id,
                "destination": request.host,
                "data": request.classification.value,
                "context": {
                    "protocol": "network",
                    "tenant": message.tenant_id or "",
                    "message_type": message.message_type,
                },
            }
        if boundary_type == "filesystem" and request is not None:
            authorization_token = authorization[1]
            return {
                "action": request.operation,
                "resource": authorization_token.canonical_path,
                "tool": managed_adapter.boundary_id,
                "destination": None,
                "data": request.classification.value,
                "context": {
                    "protocol": "filesystem",
                    "tenant": message.tenant_id or "",
                    "message_type": message.message_type,
                },
            }
        if boundary_type == "database":
            if request is None:
                operation = str(payload["operation"]).lower()
                database = payload["database"]
                schema = payload["schema"]
                table = payload["table"]
                return {
                    "action": f"database.{operation}",
                    "resource": f"{database}:{schema}:{table}",
                    "tool": managed_adapter.boundary_id,
                    "destination": None,
                    "data": message.classification,
                    "context": {
                        "protocol": "database",
                        "database": database,
                        "schema": schema,
                        "table": table,
                        "columns": tuple(payload.get("columns", ())),
                        "message_type": message.message_type,
                        **(
                            {"tenant": payload["tenant_id"]}
                            if payload.get("tenant_id") is not None
                            else {}
                        ),
                    },
                }
            return {
                "action": request.action,
                "resource": f"{request.database}:{request.schema}:{request.table}",
                "tool": managed_adapter.boundary_id,
                "destination": None,
                "data": "internal",
                "context": {
                    "protocol": "database",
                    "database": request.database,
                    "schema": request.schema,
                    "table": request.table,
                    "columns": request.columns,
                    "message_type": message.message_type,
                    **(
                        {"tenant": request.tenant_id}
                        if request.tenant_id is not None
                        else {}
                    ),
                },
            }
        self._deny("managed_mission_scope_invalid")

    def validate(self, message: AgentMessage, context: AgentSecurityContext) -> None:
        if not isinstance(message, AgentMessage) or not isinstance(context, AgentSecurityContext):
            self._deny("malformed_message")
        if not context.authenticated or context.agent_id != message.sender_agent_id or context.session_id != message.sender_session_id:
            self._deny("authentication_denied")
        known = self._contexts.get(context.agent_id)
        if known is None or known != context:
            self._deny("unknown_sender_context")
        _require_active(self._lifecycle, message.sender_agent_id)
        _require_active(self._lifecycle, message.receiver_agent_id)
        self._check_agent_mission(
            message.sender_agent_id,
            action=message.message_type,
            resource=message.receiver_agent_id,
            tool="agent-communication",
            destination=None,
            defer_destination=True,
        )
        self._check_agent_mission(
            message.receiver_agent_id,
            action=message.message_type,
            resource=message.sender_agent_id,
            tool="agent-communication",
            destination=None,
            defer_destination=True,
        )
        if self.policy.allowed_senders and message.sender_agent_id not in self.policy.allowed_senders:
            self._deny("sender_denied")
        if self.policy.allowed_receivers and message.receiver_agent_id not in self.policy.allowed_receivers:
            self._deny("receiver_denied")
        if self.policy.allowed_tenants and message.tenant_id not in self.policy.allowed_tenants:
            self._deny("tenant_denied")
        if message.tenant_id != context.tenant_id:
            self._deny("tenant_scope_denied")
        if message.message_type not in self.policy.allowed_message_types:
            self._deny("message_type_denied")
        if message.message_type in {"capability_grant", "delegation"}:
            if self._delegations is None or message.parent_authority is None:
                self._deny("parent_authority_required")
        if message.parent_authority is not None:
            if (
                self._delegations is None
                or not self._delegations.is_active(message.parent_authority)
                or message.parent_authority.subject_identity != message.sender_agent_id
            ):
                self._deny("parent_authority_invalid")
        if message.delegation_authority is not None:
            if (
                self._delegations is None
                or not self._delegations.is_active(message.delegation_authority)
                or message.delegation_authority.issuer_identity != message.sender_agent_id
                or message.delegation_authority.subject_identity != message.receiver_agent_id
                or message.delegation_authority.parent_grant_id != (
                    message.parent_authority.grant_id
                    if message.parent_authority is not None else None
                )
            ):
                self._deny("delegation_authority_invalid")
        if message.chain_depth > self.policy.max_chain_depth:
            self._deny("chain_limit")
        if set(message.capability_scope) - set(self.policy.allowed_capabilities):
            self._deny("capability_denied")
        if set(message.capability_scope) - set(context.capabilities):
            self._deny("capability_scope_denied")
        if set(message.delegation_scope) - set(context.capabilities):
            self._deny("delegation_scope_denied")
        payload_size = len(_canonical(message.payload).encode())
        if payload_size > self.policy.max_payload_bytes or len(_canonical(message.canonical()).encode()) > self.policy.max_message_bytes:
            self._deny("message_limit")
        if message.expires_at <= time.time():
            self._deny("expired")

    def issue(
        self,
        message: AgentMessage,
        context: AgentSecurityContext,
        *,
        issuance_capability: object | None = None,
    ) -> AgentMessageAuthorization:
        if (
            not isinstance(issuance_capability, _AgentIssuanceCapability)
            or issuance_capability._boundary is not self
        ):
            self._deny("issuance_authority_required")
        self.validate(message, context)
        if not self._security_integrations_present():
            self._deny("security_integration_required")
        expires = min(message.expires_at, time.time() + self.policy.ttl_seconds)
        nonce = secrets.token_hex(16)
        from ..identity.mission import MissionRegistry
        if not isinstance(self._missions, MissionRegistry):
            self._deny("mission_context_missing")
        sender_binding = self._missions.binding_for(message.sender_agent_id)
        receiver_binding = self._missions.binding_for(message.receiver_agent_id)
        if sender_binding is None or receiver_binding is None:
            self._deny("mission_binding_missing")
        payload = _canonical({
            "message_digest": message.digest(),
            "sender_agent_id": message.sender_agent_id,
            "receiver_agent_id": message.receiver_agent_id,
            "sender_session_id": message.sender_session_id,
            "receiver_session_id": message.receiver_session_id,
            "message_type": message.message_type,
            "tenant_id": message.tenant_id,
            "sender_mission_id": sender_binding.mission_id,
            "sender_mission_version": sender_binding.mission_version,
            "receiver_mission_id": receiver_binding.mission_id,
            "receiver_mission_version": receiver_binding.mission_version,
            "expires_at": expires,
            "nonce": nonce,
        })
        return AgentMessageAuthorization(
            message.digest(), message.sender_agent_id, message.receiver_agent_id,
            message.sender_session_id, message.receiver_session_id, message.message_type,
            message.tenant_id,
            sender_binding.mission_id, sender_binding.mission_version,
            receiver_binding.mission_id, receiver_binding.mission_version,
            expires, nonce,
            hmac.new(self._secret, payload.encode(), hashlib.sha256).hexdigest(),
        )

    def attenuate(
        self,
        parent: DelegationGrant,
        *,
        delegating_agent: str,
        receiver_agent: str,
        capabilities: tuple[Capability, ...],
        expires_at: datetime,
        issuer_permissions: Any,
        session_id: str | None = None,
    ) -> DelegationGrant:
        if self._delegations is None:
            self._deny("delegation_integration_required")
        if parent.subject_identity != delegating_agent:
            self._deny("delegator_mismatch")
        try:
            return self._delegations.create(
                delegating_agent,
                receiver_agent,
                capabilities,
                expires_at=expires_at,
                parent=parent,
                session_id=session_id,
                issuer_permissions=issuer_permissions,
            )
        except Exception:
            self._deny("delegation_scope_exceeded")

    def revoke(self, authorization: AgentMessageAuthorization) -> None:
        self._revoked.add(authorization.nonce)

    def dispatch_managed(
        self,
        message: AgentMessage,
        authorization: AgentMessageAuthorization | None,
        context: AgentSecurityContext,
        adapter: ManagedBoundaryAdapter,
    ) -> AgentMessageResponse:
        return self.dispatch(
            message, authorization, context, managed_adapter=adapter
        )

    def dispatch(
        self,
        message: AgentMessage,
        authorization: AgentMessageAuthorization | None,
        context: AgentSecurityContext,
        *,
        managed_adapter: ManagedBoundaryAdapter | None = None,
        release_authorization: DataReleaseAuthorization | None = None,
        release_request: DataReleaseRequest | None = None,
        application_id: str | None = None,
    ) -> AgentMessageResponse:
        self.validate(message, context)
        if not isinstance(authorization, AgentMessageAuthorization):
            self._deny("authorization_required")
        if (authorization.message_digest != message.digest() or
            authorization.sender_agent_id != message.sender_agent_id or
            authorization.receiver_agent_id != message.receiver_agent_id or
            authorization.sender_session_id != message.sender_session_id or
            authorization.receiver_session_id != message.receiver_session_id or
            authorization.message_type != message.message_type or
            authorization.tenant_id != message.tenant_id):
            self._deny("scope_mismatch")
        effective_application_id = application_id or self._application_id
        if effective_application_id is not None and (
            release_authorization is not None or release_request is not None
        ):
            if release_authorization is None and release_request is not None and self._release_issuer is not None:
                try:
                    release_authorization = self._release_issuer(release_request)
                except Exception:
                    self._deny("release_authorization_issue_failed")
            if not verify_release_authorization(
                release_authorization,
                release_request,
                application_id=effective_application_id,
            ):
                self._deny("release_authorization_invalid")
            current_policy_version = getattr(
                self, "_release_policy_version_provider", lambda: None
            )()
            if (
                current_policy_version is not None
                and release_authorization.policy_version != current_policy_version
            ):
                self._deny("release_policy_version_changed")
            assert release_request is not None
            try:
                message_classification = FlowClassification(message.classification)
            except ValueError:
                self._deny("release_classification_invalid")
            if (
                release_request.agent_id != message.sender_agent_id
                or release_request.session_id != message.sender_session_id
                or release_request.destination != message.receiver_agent_id
                or release_request.operation != message.message_type
                or release_request.source_classification is not message_classification
                or release_request.destination_trust is DestinationTrust.UNKNOWN
            ):
                self._deny("release_authorization_scope")
        from ..identity.mission import MissionRegistry
        if not isinstance(self._missions, MissionRegistry):
            self._deny("mission_context_missing")
        sender_binding = self._missions.binding_for(message.sender_agent_id)
        receiver_binding = self._missions.binding_for(message.receiver_agent_id)
        if (
            sender_binding is None
            or receiver_binding is None
            or (
                sender_binding.mission_id,
                sender_binding.mission_version,
                receiver_binding.mission_id,
                receiver_binding.mission_version,
            )
            != (
                authorization.sender_mission_id,
                authorization.sender_mission_version,
                authorization.receiver_mission_id,
                authorization.receiver_mission_version,
            )
        ):
            self._deny("mission_version_changed")
        if authorization.expires_at <= time.time() or authorization.nonce in self._revoked:
            self._deny("authorization_expired")
        expected_payload = _canonical({
            "message_digest": authorization.message_digest,
            "sender_agent_id": authorization.sender_agent_id,
            "receiver_agent_id": authorization.receiver_agent_id,
            "sender_session_id": authorization.sender_session_id,
            "receiver_session_id": authorization.receiver_session_id,
            "message_type": authorization.message_type,
            "tenant_id": authorization.tenant_id,
            "sender_mission_id": authorization.sender_mission_id,
            "sender_mission_version": authorization.sender_mission_version,
            "receiver_mission_id": authorization.receiver_mission_id,
            "receiver_mission_version": authorization.receiver_mission_version,
            "expires_at": authorization.expires_at,
            "nonce": authorization.nonce,
        })
        expected = hmac.new(self._secret, expected_payload.encode(), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(expected, authorization.signature):
            self._deny("authorization_forged")
        if not self._lock.acquire(blocking=False):
            self._deny("concurrency_limit")
        try:
            with self._state:
                if authorization.nonce in self._used or (self.policy.one_time and message.message_id in self._messages):
                    self._deny("replay")
                self._used.add(authorization.nonce)
                self._messages.add(message.message_id)
            if not self._security_integrations_present():
                self._deny("security_integration_required")
            for check in self._checks:
                if not check(message):
                    self._deny("security_denied")
            runtime_decision = self._runtime.acquire(
                context.agent_id, "agent-communication", context.session_id
            )
            if runtime_decision is not None:
                self._deny("runtime_denied")
            runtime_acquired = True
            loop_decision = self._runtime.check_loop(
                ExecutionSignature(
                    context.agent_id,
                    context.session_id,
                    "agent-communication",
                    message.message_type,
                    message.receiver_agent_id,
                )
            )
            if loop_decision is not None:
                self._deny("loop_denied")
            audit_metadata = {
                "message_id": message.message_id,
                "sender_agent_id": message.sender_agent_id,
                "receiver_agent_id": message.receiver_agent_id,
                "message_type": message.message_type,
                "correlation_id": message.correlation_id,
                "decision": "pending",
            }
            if self._audit_logger is not None:
                audit_started = self._audit_logger.record(
                    AuditEventType.TOOL_EXECUTION_STARTED,
                    correlation_id=message.correlation_id,
                    agent_id=message.sender_agent_id,
                    session_id=message.sender_session_id,
                    tool_id="agent-communication",
                    action=message.message_type,
                    resource=message.receiver_agent_id,
                    metadata=audit_metadata,
                )
                if audit_started is None and self._audit_logger.policy.required:
                    self._deny("audit_security_denied")
            if managed_adapter is not None:
                if not isinstance(getattr(managed_adapter, "boundary_id", None), str):
                    self._deny("managed_boundary_invalid")
                bind_lifecycle = getattr(managed_adapter, "_bind_lifecycle", None)
                if not callable(bind_lifecycle) or self._lifecycle is None:
                    self._deny("lifecycle_denied")
                bind_lifecycle(self._lifecycle)
                bind_mission = getattr(managed_adapter, "_bind_mission", None)
                if not callable(bind_mission) or self._missions is None:
                    self._deny("mission_context_missing")
                bind_mission(self._missions, message.sender_agent_id)
                _require_active(self._lifecycle, message.sender_agent_id)
                _require_active(self._lifecycle, message.receiver_agent_id)
                self._check_agent_mission(
                    message.sender_agent_id,
                    action=message.message_type,
                    resource=message.receiver_agent_id,
                    tool="agent-communication",
                    destination=None,
                    defer_destination=True,
                )
                self._check_agent_mission(
                    message.receiver_agent_id,
                    action=message.message_type,
                    resource=message.sender_agent_id,
                    tool="agent-communication",
                    destination=None,
                    defer_destination=True,
                )
                tool_authorization = managed_adapter.authorize(message)
                from ..identity.mission import MissionRegistry
                if not isinstance(self._missions, MissionRegistry):
                    self._deny("mission_context_missing")
                scope = self._managed_mission_scope(
                    managed_adapter, message, tool_authorization
                )
                with self._missions.execution_guard(
                    message.sender_agent_id,
                    operation=f"{managed_adapter.boundary_type}_dispatch",
                    mission_id=authorization.sender_mission_id,
                    mission_version=authorization.sender_mission_version,
                    **scope,
                ) as sender_decision:
                    if sender_decision.decision_type is not DecisionType.ALLOW:
                        self._deny("mission_denied")
                    with self._missions.execution_guard(
                        message.receiver_agent_id,
                        action=message.message_type,
                        resource=message.sender_agent_id,
                        tool="agent-communication",
                        destination=None,
                        data=message.classification,
                        context={
                            "tenant": message.tenant_id or "",
                            "message_type": message.message_type,
                        },
                        operation="agent_communication",
                        mission_id=authorization.receiver_mission_id,
                        mission_version=authorization.receiver_mission_version,
                    ) as receiver_decision:
                        if receiver_decision.decision_type is not DecisionType.ALLOW:
                            self._deny("receiver_mission_denied")
                        result = managed_adapter.dispatch(message, tool_authorization)
            else:
                _require_active(self._lifecycle, message.sender_agent_id)
                _require_active(self._lifecycle, message.receiver_agent_id)
                self._check_agent_mission(
                    message.sender_agent_id,
                    action=message.message_type,
                    resource=message.receiver_agent_id,
                    tool="agent-communication",
                    destination=None,
                    defer_destination=True,
                )
                self._check_agent_mission(
                    message.receiver_agent_id,
                    action=message.message_type,
                    resource=message.sender_agent_id,
                    tool="agent-communication",
                    destination=None,
                    defer_destination=True,
                )
                from ..identity.mission import MissionRegistry
                if not isinstance(self._missions, MissionRegistry):
                    self._deny("mission_context_missing")
                with self._missions.execution_guard(
                    message.sender_agent_id,
                    action=message.message_type,
                    resource=message.receiver_agent_id,
                    tool="agent-communication",
                    destination=None,
                    operation="agent_communication",
                    mission_id=authorization.sender_mission_id,
                    mission_version=authorization.sender_mission_version,
                ) as sender_decision:
                    if sender_decision.decision_type is not DecisionType.ALLOW:
                        self._deny("mission_denied")
                    with self._missions.execution_guard(
                        message.receiver_agent_id,
                        action=message.message_type,
                        resource=message.sender_agent_id,
                        tool="agent-communication",
                        destination=None,
                        data=message.classification,
                        context={
                            "tenant": message.tenant_id or "",
                            "message_type": message.message_type,
                        },
                        operation="agent_communication",
                        mission_id=authorization.receiver_mission_id,
                        mission_version=authorization.receiver_mission_version,
                    ) as receiver_decision:
                        if receiver_decision.decision_type is not DecisionType.ALLOW:
                            self._deny("receiver_mission_denied")
                        result = self._downstream(message)
            if not isinstance(result, Mapping):
                self._deny("invalid_response")
            _json(dict(result), "response")
            if self._output_check is None or not self._output_check(result):
                self._deny("output_security_denied")
            audit_metadata = {
                "message_id": message.message_id,
                "sender_agent_id": message.sender_agent_id,
                "receiver_agent_id": message.receiver_agent_id,
                "message_type": message.message_type,
                "correlation_id": message.correlation_id,
                "decision": "allow",
            }
            if self._audit_logger is not None:
                self._audit_logger.record(
                    AuditEventType.TOOL_EXECUTION_SUCCEEDED,
                    correlation_id=message.correlation_id,
                    agent_id=message.sender_agent_id,
                    session_id=message.sender_session_id,
                    tool_id="agent-communication",
                    action=message.message_type,
                    resource=message.receiver_agent_id,
                    metadata=audit_metadata,
                )
            if self._audit_check is None or not self._audit_check(audit_metadata):
                self._deny("audit_security_denied")
            return AgentMessageResponse(message.message_id, True, dict(result))
        finally:
            if locals().get("runtime_acquired", False) and self._runtime is not None:
                self._runtime.release()
            self._lock.release()

    def forward(
        self,
        message: AgentMessage,
        context: AgentSecurityContext,
        receiver_agent_id: str,
        authorization: AgentMessageAuthorization | None = None,
        child_authority: DelegationGrant | None = None,
    ) -> AgentMessage:
        if not isinstance(authorization, AgentMessageAuthorization):
            self._deny("authorization_required")
        self.validate(message, context)
        self._verify_authorization(message, authorization)
        if child_authority is None:
            self._deny("delegation_required")
        if (
            child_authority.issuer_identity != context.agent_id
            or child_authority.subject_identity != receiver_agent_id
            or child_authority.parent_grant_id != (
                message.parent_authority.grant_id
                if message.parent_authority is not None else None
            )
        ):
            self._deny("delegation_chain_invalid")
        if (
            self._delegations is None
            or not self._delegations.is_active(child_authority)
            or child_authority.issuer_identity != context.agent_id
            or child_authority.subject_identity != receiver_agent_id
        ):
            self._deny("delegation_signature_invalid")
        if message.chain_depth >= self.policy.max_chain_depth:
            self._deny("chain_limit")
        return AgentMessage(
            message.message_id, message.message_type, context.agent_id,
            receiver_agent_id, context.session_id, message.receiver_session_id,
            message.conversation_id, message.parent_message_id, message.timestamp,
            message.expires_at, message.payload, message.capability_scope,
            message.delegation_scope, message.classification, message.correlation_id,
            message.protocol_version, message.tenant_id, message.chain_depth + 1,
            message.parent_authority,
            child_authority,
        )

    def _verify_authorization(
        self, message: AgentMessage, authorization: AgentMessageAuthorization
    ) -> None:
        if (
            authorization.message_digest != message.digest()
            or authorization.sender_agent_id != message.sender_agent_id
            or authorization.receiver_agent_id != message.receiver_agent_id
            or authorization.sender_session_id != message.sender_session_id
            or authorization.receiver_session_id != message.receiver_session_id
            or authorization.message_type != message.message_type
            or authorization.tenant_id != message.tenant_id
            or authorization.expires_at <= time.time()
            or authorization.nonce in self._revoked
        ):
            self._deny("scope_mismatch")
        expected_payload = _canonical({
            "message_digest": authorization.message_digest,
            "sender_agent_id": authorization.sender_agent_id,
            "receiver_agent_id": authorization.receiver_agent_id,
            "sender_session_id": authorization.sender_session_id,
            "receiver_session_id": authorization.receiver_session_id,
            "message_type": authorization.message_type,
            "tenant_id": authorization.tenant_id,
            "expires_at": authorization.expires_at,
            "nonce": authorization.nonce,
        })
        expected = hmac.new(
            self._secret, expected_payload.encode(), hashlib.sha256
        ).hexdigest()
        if not hmac.compare_digest(expected, authorization.signature):
            self._deny("authorization_forged")


__all__ = [
    "AgentMessagePolicy", "AgentMessage", "AgentMessageResponse",
    "AgentMessageAuthorization", "AgentSecurityContext",
    "AgentCommunicationBoundary", "AgentCommunicationDenied",
    "ManagedBoundaryAdapter", "ManagedBoundaryRegistration",
    "ManagedBoundaryRegistry",
    "MCPBoundaryAdapter", "SandboxBoundaryAdapter", "NetworkBoundaryAdapter",
    "FileSystemBoundaryAdapter", "DatabaseBoundaryAdapter",
]
