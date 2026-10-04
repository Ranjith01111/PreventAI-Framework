"""Deterministic, transport-neutral MCP-style protocol security boundary."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import hmac
import json
import re
import secrets
import threading
import time
from typing import Any, Callable, Mapping
from urllib.parse import unquote, urlsplit

from .security_request import ContractValidationError
from ..core.decision import DecisionType
from ..core.data_release import (
    DataReleaseAuthorization, DataReleaseRequest, DestinationTrust,
    verify_release_authorization,
)

_METHODS = frozenset({
    "tools/list", "tools/call", "resources/list", "resources/read",
    "prompts/list", "prompts/get", "initialize",
})
_NOTIFICATIONS = frozenset({"notifications/initialized", "notifications/cancelled"})
_IDENTIFIER = re.compile(r"^[A-Za-z0-9._:-]{1,256}$")
_SCHEMES = frozenset({"mcp", "urn"})


def _json(value: Any, field: str, depth: int = 0, *, max_depth: int = 16) -> None:
    if depth > max_depth:
        raise ContractValidationError(f"{field} exceeds nesting limit")
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            raise ContractValidationError(f"{field} contains a non-finite number")
        return
    if isinstance(value, list):
        for item in value:
            _json(item, field, depth + 1, max_depth=max_depth)
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str) or not _IDENTIFIER.fullmatch(key):
                raise ContractValidationError(f"{field} contains an invalid key")
            _json(item, field, depth + 1, max_depth=max_depth)
        return
    raise ContractValidationError(f"{field} contains an unsupported value")


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _text(value: Any, field: str, limit: int) -> str:
    if not isinstance(value, str) or not value or len(value) > limit or any(ord(c) < 32 for c in value):
        raise ContractValidationError(f"{field} is invalid")
    return value


def canonical_resource_uri(uri: str) -> str:
    _text(uri, "resource_uri", 2048)
    decoded = unquote(uri)
    if decoded != uri or ".." in decoded.split("/"):
        raise ContractValidationError("resource URI traversal is not allowed")
    parts = urlsplit(uri)
    if parts.scheme not in _SCHEMES or parts.netloc or "\\" in uri or not parts.path:
        raise ContractValidationError("resource URI is not allowed")
    return f"{parts.scheme}:{parts.path}" + (f"?{parts.query}" if parts.query else "")


@dataclass(frozen=True, slots=True)
class MCPPolicy:
    allowed_methods: tuple[str, ...] = tuple(sorted(_METHODS))
    allowed_capabilities: tuple[str, ...] = ()
    allowed_tools: tuple[str, ...] = ()
    allowed_resources: tuple[str, ...] = ()
    allowed_prompts: tuple[str, ...] = ()
    server_id: str = "preventai-server"
    allowed_clients: tuple[str, ...] = ()
    protocol_versions: tuple[str, ...] = ("2025-06-18",)
    max_message_bytes: int = 1_000_000
    max_arguments: int = 100
    max_string_length: int = 8192
    max_response_bytes: int = 1_000_000
    max_concurrent: int = 1
    authorization_ttl: float = 30.0

    def __post_init__(self) -> None:
        if not self.allowed_methods or any(m not in _METHODS for m in self.allowed_methods):
            raise ValueError("allowed_methods contains unsupported methods")
        for field in ("allowed_capabilities", "allowed_tools", "allowed_resources",
                      "allowed_prompts", "allowed_clients", "protocol_versions"):
            values = getattr(self, field)
            if not isinstance(values, tuple) or any(not isinstance(v, str) or not v for v in values):
                raise ValueError(f"{field} is invalid")
        for field in ("max_message_bytes", "max_arguments", "max_string_length",
                      "max_response_bytes", "max_concurrent"):
            value = getattr(self, field)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{field} must be positive")
        if not isinstance(self.authorization_ttl, (int, float)) or self.authorization_ttl <= 0:
            raise ValueError("authorization_ttl must be positive")


@dataclass(frozen=True, slots=True)
class MCPRequest:
    method: str
    request_id: str | int | None
    params: Mapping[str, Any]
    protocol_version: str
    server_id: str
    client_id: str
    session_id: str
    capabilities: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.method, str) or not self.method or len(self.method) > 256:
            raise ContractValidationError("invalid MCP method")
        if self.request_id is not None and (
            isinstance(self.request_id, bool)
            or not isinstance(self.request_id, (str, int))
            or (isinstance(self.request_id, str) and not self.request_id)
        ):
            raise ContractValidationError("invalid request id")
        if not isinstance(self.params, Mapping):
            raise ContractValidationError("params must be an object")
        _json(dict(self.params), "params")
        for field in ("protocol_version", "server_id", "client_id", "session_id"):
            _text(getattr(self, field), field, 256)
        if any(not isinstance(c, str) or not c for c in self.capabilities):
            raise ContractValidationError("invalid capability")

    @property
    def identity(self) -> str:
        if self.method == "tools/call":
            return str(self.params.get("name", ""))
        if self.method == "resources/read":
            return str(self.params.get("uri", ""))
        if self.method == "prompts/get":
            return str(self.params.get("name", ""))
        return self.method

    def canonical(self) -> dict[str, Any]:
        return {
            "method": self.method, "id": self.request_id, "params": dict(self.params),
            "protocol_version": self.protocol_version, "server_id": self.server_id,
            "client_id": self.client_id, "session_id": self.session_id,
            "capabilities": self.capabilities,
        }

    def digest(self) -> str:
        return _digest(self.canonical())

    @classmethod
    def from_json_rpc(cls, message: Mapping[str, Any], *, protocol_version: str,
                      server_id: str, client_id: str, session_id: str) -> MCPRequest:
        if not isinstance(message, Mapping) or set(message) - {"jsonrpc", "id", "method", "params"}:
            raise ContractValidationError("malformed JSON-RPC request")
        if message.get("jsonrpc") != "2.0" or not isinstance(message.get("method"), str):
            raise ContractValidationError("invalid JSON-RPC request")
        params = message.get("params", {})
        if not isinstance(params, Mapping):
            raise ContractValidationError("params must be an object")
        capabilities = params.get("capabilities", ())
        if not isinstance(capabilities, (list, tuple)):
            raise ContractValidationError("capabilities must be an array")
        return cls(message["method"], message.get("id"), params, protocol_version,
                   server_id, client_id, session_id, tuple(capabilities))


@dataclass(frozen=True, slots=True)
class MCPResponse:
    request_id: str | int
    result: Mapping[str, Any] | None = None
    error: str | None = None

    def __post_init__(self) -> None:
        if isinstance(self.request_id, bool) or not isinstance(self.request_id, (str, int)):
            raise ContractValidationError("invalid response id")
        if (self.result is None) == (self.error is None):
            raise ContractValidationError("response must contain exactly one result or error")
        if self.result is not None:
            _json(dict(self.result), "result")
        if self.error is not None:
            _text(self.error, "error", 256)

    def to_json_rpc(self) -> dict[str, Any]:
        return {"jsonrpc": "2.0", "id": self.request_id,
                "result": dict(self.result) if self.result is not None else None,
                "error": self.error}


@dataclass(frozen=True, slots=True)
class MCPAuthorization:
    request_digest: str
    server_id: str
    client_id: str
    session_id: str
    method: str
    identity: str
    expires_at: float
    nonce: str
    signature: str
    used: bool = False


class MCPExecutionDenied(PermissionError):
    def __init__(self, category: str):
        self.category = category
        super().__init__(category)


class MCPBoundary:
    """One authoritative dispatch boundary around an injected downstream handler."""

    def __init__(self, policy: MCPPolicy, downstream: Callable[[MCPRequest], Mapping[str, Any]],
                 *, policy_check: Callable[[MCPRequest], bool] | None = None,
                 permission_check: Callable[[MCPRequest], bool] | None = None,
                 delegation_check: Callable[[MCPRequest], bool] | None = None,
                 dataflow_check: Callable[[MCPRequest], bool] | None = None,
                 runtime_check: Callable[[MCPRequest], bool] | None = None,
                 approval_check: Callable[[MCPRequest], bool] | None = None,
                 output_check: Callable[[Mapping[str, Any]], bool] | None = None,
                 audit: Callable[[str, MCPRequest], None] | None = None) -> None:
        if not isinstance(policy, MCPPolicy) or not callable(downstream):
            raise TypeError("policy and downstream are required")
        self.policy = policy
        self._downstream = downstream
        self._secret = secrets.token_bytes(32)
        self._revoked: set[str] = set()
        self._used: set[str] = set()
        self._pending: set[str | int] = set()
        self._lock = threading.BoundedSemaphore(policy.max_concurrent)
        self._state_lock = threading.RLock()
        self._checks = (policy_check, permission_check, delegation_check, dataflow_check,
                        runtime_check, approval_check)
        self._output_check = output_check
        self._audit = audit
        self._mission_registry = None
        self._mission_agent_id: str | None = None
        self._application_id: str | None = None
        self._release_issuer = None

    def bind_mission_context(self, mission_registry: object, agent_id: str) -> None:
        from ..identity.mission import MissionRegistry
        if (
            not isinstance(mission_registry, MissionRegistry)
            or not mission_registry.application_owned
            or not isinstance(agent_id, str)
        ):
            raise TypeError("authoritative mission context is required")
        self._mission_registry = mission_registry
        self._mission_agent_id = agent_id

    def _deny(self, category: str) -> None:
        raise MCPExecutionDenied(category)

    def parse(self, message: Mapping[str, Any], *, protocol_version: str,
              client_id: str, session_id: str) -> MCPRequest:
        encoded = _canonical(message).encode("utf-8")
        if len(encoded) > self.policy.max_message_bytes:
            self._deny("message_limit")
        try:
            request = MCPRequest.from_json_rpc(
                message, protocol_version=protocol_version, server_id=self.policy.server_id,
                client_id=client_id, session_id=session_id,
            )
        except ContractValidationError:
            self._deny("malformed_request")
        if protocol_version not in self.policy.protocol_versions:
            self._deny("version_denied")
        if request.method not in self.policy.allowed_methods and request.method not in _NOTIFICATIONS:
            self._deny("method_denied")
        if self.policy.allowed_clients and client_id not in self.policy.allowed_clients:
            self._deny("client_denied")
        if len(request.params) > self.policy.max_arguments:
            self._deny("argument_limit")
        if request.method == "tools/call":
            name = request.params.get("name")
            if not isinstance(name, str) or name not in self.policy.allowed_tools:
                self._deny("tool_denied")
        elif request.method == "resources/read":
            try:
                canonical_resource_uri(str(request.params["uri"]))
            except (KeyError, ContractValidationError):
                self._deny("resource_denied")
            if not self.policy.allowed_resources or request.identity not in self.policy.allowed_resources:
                self._deny("resource_denied")
        elif request.method == "prompts/get":
            if not isinstance(request.params.get("name"), str) or (
                not self.policy.allowed_prompts or request.identity not in self.policy.allowed_prompts
            ):
                self._deny("prompt_denied")
        if any(len(s) > self.policy.max_string_length for s in json.dumps(dict(request.params), ensure_ascii=True).split()):
            self._deny("string_limit")
        if set(request.capabilities) - set(self.policy.allowed_capabilities):
            self._deny("capability_denied")
        return request

    def issue(self, request: MCPRequest) -> MCPAuthorization:
        if not isinstance(request, MCPRequest):
            self._deny("invalid_request")
        self._validate_request_against_policy(request)
        expires = time.time() + self.policy.authorization_ttl
        payload = f"{request.digest()}|{expires}"
        return MCPAuthorization(request.digest(), request.server_id, request.client_id,
                                request.session_id, request.method, request.identity,
                                expires, secrets.token_hex(16),
                                hmac.new(self._secret, payload.encode(), hashlib.sha256).hexdigest())

    def _validate_request_against_policy(self, request: MCPRequest) -> None:
        if request.protocol_version not in self.policy.protocol_versions:
            self._deny("version_denied")
        if request.server_id != self.policy.server_id:
            self._deny("server_denied")
        if self.policy.allowed_clients and request.client_id not in self.policy.allowed_clients:
            self._deny("client_denied")
        if request.method not in self.policy.allowed_methods:
            self._deny("method_denied")
        if request.method == "tools/call" and request.identity not in self.policy.allowed_tools:
            self._deny("tool_denied")
        if request.method == "resources/read":
            if not self.policy.allowed_resources or request.identity not in self.policy.allowed_resources:
                self._deny("resource_denied")
            canonical_resource_uri(request.identity)
        if request.method == "prompts/get" and (
            not self.policy.allowed_prompts or request.identity not in self.policy.allowed_prompts
        ):
            self._deny("prompt_denied")
        if set(request.capabilities) - set(self.policy.allowed_capabilities):
            self._deny("capability_denied")

    def revoke(self, authorization: MCPAuthorization) -> None:
        self._revoked.add(authorization.nonce)

    def dispatch(
        self,
        request: MCPRequest,
        authorization: MCPAuthorization | None,
        *,
        release_authorization: DataReleaseAuthorization | None = None,
        release_request: DataReleaseRequest | None = None,
        application_id: str | None = None,
    ) -> MCPResponse:
        if not isinstance(request, MCPRequest) or request.method in _NOTIFICATIONS:
            self._deny("notification_or_request_invalid")
        self._validate_request_against_policy(request)
        if self._mission_registry is None:
            self._deny("mission_context_missing")
        decision = self._mission_registry.check(
            self._mission_agent_id,
            action=request.method,
            resource=request.identity,
            tool=request.identity if request.method == "tools/call" else request.method,
            destination=request.server_id,
            data=None,
            context={"protocol": "mcp"},
            defer_data=True,
            defer_context=True,
            operation="mcp_dispatch",
        )
        if decision.decision_type is not DecisionType.ALLOW:
            self._deny("mission_denied")
        if application_id is not None and (
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
                application_id=application_id,
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
            if (
                release_request.destination != request.server_id
                or release_request.operation != request.method
                or release_request.tool_name != request.identity
                or release_request.session_id != request.session_id
                or release_request.agent_id != self._mission_agent_id
                or release_request.destination_trust is DestinationTrust.UNKNOWN
            ):
                self._deny("release_authorization_scope")
        if not isinstance(authorization, MCPAuthorization):
            self._deny("authorization_required")
        if authorization.request_digest != request.digest() or authorization.server_id != self.policy.server_id:
            self._deny("authorization_scope")
        if (authorization.client_id, authorization.session_id, authorization.method, authorization.identity) != (
            request.client_id, request.session_id, request.method, request.identity):
            self._deny("authorization_scope")
        if authorization.expires_at <= time.time() or authorization.nonce in self._revoked:
            self._deny("authorization_expired")
        expected = hmac.new(self._secret, f"{authorization.request_digest}|{authorization.expires_at}".encode(), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(expected, authorization.signature):
            self._deny("authorization_forged")
        if request.method in {"tools/call", "resources/read", "prompts/get"}:
            if any(check is None for check in self._checks):
                self._deny("security_integration_required")
        with self._state_lock:
            if authorization.used or authorization.nonce in self._used:
                self._deny("authorization_replay")
            self._used.add(authorization.nonce)
        for check in self._checks:
            if check is not None and not check(request):
                self._deny("framework_denied")
        if not self._lock.acquire(blocking=False):
            self._deny("concurrency_limit")
        try:
            if request.request_id in self._pending:
                self._deny("duplicate_request")
            self._pending.add(request.request_id)
            result = {} if request.method in {"tools/list", "resources/list", "prompts/list", "initialize"} else self._downstream(request)
            if not isinstance(result, Mapping) or not _json(result, "result") is None:
                self._deny("invalid_result")
            if len(_canonical(result).encode()) > self.policy.max_response_bytes:
                self._deny("response_limit")
            if self._output_check is not None and not self._output_check(result):
                self._deny("output_denied")
            if self._audit is not None:
                self._audit("accepted", request)
            return MCPResponse(request.request_id, result=dict(result))
        finally:
            self._lock.release()


__all__ = [
    "MCPPolicy", "MCPRequest", "MCPResponse", "MCPAuthorization",
    "MCPBoundary", "MCPExecutionDenied", "canonical_resource_uri",
]
