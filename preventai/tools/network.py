"""Policy-enforced outbound HTTP execution through one framework boundary."""

from __future__ import annotations

from dataclasses import dataclass
import base64
import hashlib
import hmac
import http.client
import ipaddress
import json
import math
import re
import secrets
import socket
import ssl
import threading
import time
from types import MappingProxyType
from typing import Any, Mapping
from urllib.parse import quote, urljoin, urlsplit, urlunsplit

from ..core.decision import Decision, DecisionReason, DecisionType, Severity
from ..core.dataflow import FlowClassification, FlowTrust
from ..audit.event import AuditEventType
from ..core.data_release import DataReleaseAuthorization, DataReleaseRequest, verify_release_authorization
from ..audit.logger import AuditLogger

_FRAMEWORK_ISSUER = object()
_HEADER_NAME = re.compile(r"^[!#$%&'*+\-.^_`|~0-9A-Za-z]+$")
_PERCENT_ESCAPE = re.compile(r"%(?![0-9A-Fa-f]{2})")
_REDIRECT_CODES = frozenset({301, 302, 303, 307, 308})
_MAX_BODY_CONSTRUCTION = 16 * 1024 * 1024
_MAX_HEADER_CONSTRUCTION = 64 * 1024


def _host(value: str, field: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 253 or "%" in value:
        raise ValueError(f"{field} is invalid")
    try:
        address = ipaddress.ip_address(value)
    except ValueError:
        try:
            normalized = value.rstrip(".").encode("idna").decode("ascii").lower()
        except UnicodeError as exc:
            raise ValueError(f"{field} is invalid") from exc
        if not normalized or any(
            not label
            or len(label) > 63
            or label.startswith("-")
            or label.endswith("-")
            or any(not (char.isascii() and (char.isalnum() or char == "-")) for char in label)
            for label in normalized.split(".")
        ):
            raise ValueError(f"{field} is invalid")
        return normalized
    if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped is not None:
        address = address.ipv4_mapped
    return address.compressed.lower()


def _canonical_url(value: str) -> tuple[str, str, str, int, str]:
    if not isinstance(value, str) or not value or len(value) > 8192:
        raise ValueError("url is invalid")
    if value != value.strip() or "\\" in value or any(ord(char) < 33 for char in value):
        raise ValueError("url contains invalid characters")
    try:
        parts = urlsplit(value)
        scheme = parts.scheme.lower()
        if scheme not in {"http", "https"} or not parts.netloc:
            raise ValueError("unsupported or missing URL scheme/host")
        if parts.username is not None or parts.password is not None or "@" in parts.netloc:
            raise ValueError("URL userinfo is not allowed")
        if parts.hostname is None:
            raise ValueError("URL host is missing")
        hostname = _host(parts.hostname, "host")
        parsed_port = parts.port
        port = (443 if scheme == "https" else 80) if parsed_port is None else parsed_port
        if not 1 <= port <= 65535:
            raise ValueError("port is out of range")
    except (ValueError, UnicodeError) as exc:
        raise ValueError("URL is malformed") from exc
    if _PERCENT_ESCAPE.search(parts.path) or _PERCENT_ESCAPE.search(parts.query):
        raise ValueError("URL contains malformed encoding")
    path = quote(parts.path or "/", safe="/%:@!$&'()*+,;=-._~")
    query = quote(parts.query, safe="/?%:@!$&'()*+,;=-._~")
    path = re.sub(r"%[0-9a-fA-F]{2}", lambda match: match.group(0).upper(), path)
    query = re.sub(r"%[0-9a-fA-F]{2}", lambda match: match.group(0).upper(), query)
    authority_host = f"[{hostname}]" if ":" in hostname else hostname
    default_port = (scheme == "http" and port == 80) or (scheme == "https" and port == 443)
    netloc = authority_host if default_port else f"{authority_host}:{port}"
    canonical = urlunsplit((scheme, netloc, path, query, ""))
    return canonical, scheme, hostname, port, f"{path}?{query}" if query else path


@dataclass(frozen=True, slots=True)
class NetworkPolicy:
    """An explicit outbound HTTP allowlist; an empty host list denies all."""

    allowed_schemes: tuple[str, ...] = ("https",)
    denied_schemes: tuple[str, ...] = ()
    allowed_hosts: tuple[str, ...] = ()
    denied_hosts: tuple[str, ...] = ()
    allowed_ports: tuple[int, ...] = ()
    denied_ports: tuple[int, ...] = ()
    allowed_ip_ranges: tuple[str, ...] = ()
    denied_ip_ranges: tuple[str, ...] = ()
    allowed_methods: tuple[str, ...] = ("GET",)
    max_request_bytes: int = 1_000_000
    max_response_bytes: int = 1_000_000
    max_redirects: int = 0
    timeout: float = 10.0
    require_https: bool = True
    max_concurrent_requests: int = 1

    def __post_init__(self) -> None:
        for field in ("allowed_schemes", "denied_schemes"):
            values = getattr(self, field)
            if not isinstance(values, tuple) or any(
                not isinstance(item, str) or item.lower() not in {"http", "https"}
                for item in values
            ):
                raise ValueError(f"{field} must contain supported schemes")
            object.__setattr__(self, field, tuple(item.lower() for item in values))
        for field in ("allowed_hosts", "denied_hosts"):
            values = getattr(self, field)
            if not isinstance(values, tuple) or any(not isinstance(item, str) or not item for item in values):
                raise ValueError(f"{field} must contain host patterns")
            normalized = []
            for item in values:
                wildcard = item.startswith("*.")
                normalized.append(("*." if wildcard else "") + _host(item[2:] if wildcard else item, field))
            object.__setattr__(self, field, tuple(normalized))
        for field in ("allowed_ports", "denied_ports"):
            values = getattr(self, field)
            if not isinstance(values, tuple) or any(
                not isinstance(port, int) or isinstance(port, bool) or not 1 <= port <= 65535
                for port in values
            ):
                raise ValueError(f"{field} must contain valid ports")
        for field in ("allowed_ip_ranges", "denied_ip_ranges"):
            values = getattr(self, field)
            if not isinstance(values, tuple) or any(not isinstance(value, str) for value in values):
                raise ValueError(f"{field} must contain CIDR strings")
            try:
                normalized = tuple(str(ipaddress.ip_network(value, strict=False)) for value in values)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{field} must contain valid IP networks") from exc
            object.__setattr__(self, field, normalized)
        if not isinstance(self.allowed_methods, tuple) or any(
            not isinstance(method, str) or not _HEADER_NAME.fullmatch(method)
            for method in self.allowed_methods
        ):
            raise ValueError("allowed_methods is invalid")
        object.__setattr__(self, "allowed_methods", tuple(method.upper() for method in self.allowed_methods))
        for field in ("max_request_bytes", "max_response_bytes", "max_concurrent_requests"):
            value = getattr(self, field)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{field} must be a positive integer")
        if not isinstance(self.max_redirects, int) or isinstance(self.max_redirects, bool) or self.max_redirects < 0:
            raise ValueError("max_redirects must be a non-negative integer")
        if (
            not isinstance(self.timeout, (int, float))
            or isinstance(self.timeout, bool)
            or not math.isfinite(self.timeout)
            or self.timeout <= 0
        ):
            raise ValueError("timeout must be positive")
        if not isinstance(self.require_https, bool):
            raise ValueError("require_https must be boolean")


@dataclass(frozen=True, slots=True)
class NetworkRequest:
    """A single explicit HTTP request; framework context is bound at execution."""

    url: str
    method: str = "GET"
    headers: Mapping[str, str] | None = None
    body: bytes = b""
    classification: FlowClassification = FlowClassification.INTERNAL
    source_trust: FlowTrust = FlowTrust.UNTRUSTED
    context: Mapping[str, Any] | None = None
    agent_id: str | None = None
    session_id: str | None = None
    tool_id: str | None = None

    def __post_init__(self) -> None:
        canonical, _, _, _, _ = _canonical_url(self.url)
        object.__setattr__(self, "url", canonical)
        if not isinstance(self.method, str) or not _HEADER_NAME.fullmatch(self.method):
            raise ValueError("method is invalid")
        object.__setattr__(self, "method", self.method.upper())
        if not isinstance(self.body, bytes) or len(self.body) > _MAX_BODY_CONSTRUCTION:
            raise ValueError("body must be bytes within the hard size limit")
        if not isinstance(self.classification, FlowClassification):
            raise ValueError("classification must be a FlowClassification")
        if not isinstance(self.source_trust, FlowTrust):
            raise ValueError("source_trust must be a FlowTrust")
        headers = {} if self.headers is None else dict(self.headers)
        if not isinstance(self.headers, (Mapping, type(None))):
            raise ValueError("headers must be a mapping")
        normalized_headers: dict[str, str] = {}
        for name, value in headers.items():
            if not isinstance(name, str) or not _HEADER_NAME.fullmatch(name):
                raise ValueError("header name is invalid")
            lowered = name.lower()
            if lowered in {
                "host", "forwarded", "x-forwarded-host", "x-forwarded-for",
                "content-length", "transfer-encoding", "proxy-authorization",
            } or lowered.startswith("x-forwarded-"):
                raise ValueError("destination and framing headers are controlled by the framework")
            if not isinstance(value, str) or "\r" in value or "\n" in value or "\x00" in value:
                raise ValueError("header value is invalid")
            if any(ord(char) < 32 or ord(char) == 127 for char in value):
                raise ValueError("header value contains a control character")
            if lowered in normalized_headers:
                raise ValueError("duplicate headers are not allowed")
            normalized_headers[lowered] = value
        header_size = sum(
            len(name.encode("ascii")) + len(value.encode("utf-8")) + 4
            for name, value in normalized_headers.items()
        )
        if header_size > _MAX_HEADER_CONSTRUCTION:
            raise ValueError("headers exceed the hard size limit")
        object.__setattr__(self, "headers", MappingProxyType(normalized_headers))
        context = {} if self.context is None else dict(self.context)
        if not isinstance(self.context, (Mapping, type(None))):
            raise ValueError("context must be a mapping")
        if any(
            not isinstance(key, str)
            or not key
            or not (value is None or isinstance(value, (str, bool, int, float)))
            for key, value in context.items()
        ):
            raise ValueError("context must contain scalar JSON values")
        try:
            encoded = json.dumps(context, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("context must contain canonical JSON values") from exc
        if len(encoded) > 8192:
            raise ValueError("context is too large")
        object.__setattr__(self, "context", MappingProxyType(context))
        for field in ("agent_id", "session_id", "tool_id"):
            value = getattr(self, field)
            if value is not None and (not isinstance(value, str) or not value.strip() or len(value) > 256):
                raise ValueError(f"{field} is invalid")

    @property
    def scheme(self) -> str:
        return _canonical_url(self.url)[1]

    @property
    def host(self) -> str:
        return _canonical_url(self.url)[2]

    @property
    def port(self) -> int:
        return _canonical_url(self.url)[3]

    @property
    def path(self) -> str:
        return _canonical_url(self.url)[4]

    @property
    def action(self) -> str:
        return f"http.{self.method.lower()}"

    @property
    def destination(self) -> str:
        return f"{self.scheme}://{self.host}:{self.port}"

    def canonical(self) -> dict[str, Any]:
        return {
            "url": self.url,
            "method": self.method,
            "headers": dict(sorted(self.headers.items())),
            "body": base64.b64encode(self.body).decode("ascii"),
            "classification": self.classification.value,
            "source_trust": self.source_trust.value,
            "context": dict(self.context),
            "agent_id": self.agent_id,
            "session_id": self.session_id,
            "tool_id": self.tool_id,
        }

    @property
    def digest(self) -> str:
        payload = json.dumps(self.canonical(), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class NetworkResponse:
    status: int
    headers: Mapping[str, str]
    body: bytes
    url: str
    redirects: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.status, int) or isinstance(self.status, bool) or not 100 <= self.status <= 599:
            raise ValueError("status is invalid")
        if not isinstance(self.body, bytes):
            raise ValueError("body must be bytes")
        if not isinstance(self.redirects, int) or isinstance(self.redirects, bool) or self.redirects < 0:
            raise ValueError("redirects must be a non-negative integer")
        canonical, _, _, _, _ = _canonical_url(self.url)
        object.__setattr__(self, "url", canonical)
        headers = dict(self.headers)
        if any(not isinstance(key, str) or not isinstance(value, str) for key, value in headers.items()):
            raise ValueError("response headers are invalid")
        object.__setattr__(self, "headers", MappingProxyType(headers))

    def output_value(self) -> dict[str, Any]:
        safe_headers = {
            key.lower(): value
            for key, value in self.headers.items()
            if key.lower() in {"content-type", "content-length", "etag", "last-modified"}
        }
        return {
            "status": self.status,
            "headers": safe_headers,
            "body": self.body.decode("utf-8", errors="strict"),
            "url": self.url.split("?", 1)[0],
            "redirects": self.redirects,
        }


@dataclass(frozen=True, slots=True)
class NetworkAuthorization:
    agent_id: str
    session_id: str
    tool_id: str
    action: str
    scheme: str
    host: str
    port: int
    request_digest: str
    resolved_addresses: tuple[str, ...]
    allow_redirects: bool
    expires_at: float
    _proof: str


@dataclass(frozen=True, slots=True)
class _NetworkPlan:
    request: NetworkRequest
    addresses: tuple[str, ...]
    internal: bool


class NetworkExecutionDenied(PermissionError):
    def __init__(self, decision: Decision) -> None:
        self.decision = decision
        super().__init__(f"network execution blocked: {decision.reason.value}")


class Network:
    """The sole framework-managed HTTP executor; absent policy means no access."""

    __slots__ = (
        "_policy", "audit_logger", "_secret", "_revoked", "_consumed", "_lock",
        "_concurrency", "__dict__",
    )

    def __init__(
        self,
        policy: NetworkPolicy,
        *,
        audit_logger: AuditLogger | None = None,
    ) -> None:
        if not isinstance(policy, NetworkPolicy):
            raise ValueError("policy must be a NetworkPolicy")
        self._policy = policy
        self.audit_logger = audit_logger
        self._secret = secrets.token_bytes(32)
        self._revoked: set[str] = set()
        self._consumed: set[str] = set()
        self._lock = threading.RLock()
        self._concurrency = threading.BoundedSemaphore(policy.max_concurrent_requests)
        self._mission_registry = None
        self._mission_agent_id: str | None = None
        self._mission_tool: str | None = None

    def bind_mission_context(
        self, mission_registry: object, agent_id: str, *, tool: str | None = None
    ) -> None:
        from ..identity.mission import MissionRegistry
        if (
            not isinstance(mission_registry, MissionRegistry)
            or not mission_registry.application_owned
            or not isinstance(agent_id, str)
        ):
            raise TypeError("authoritative mission context is required")
        self._mission_registry = mission_registry
        self._mission_agent_id = agent_id
        self._mission_tool = tool

    @property
    def policy(self) -> NetworkPolicy:
        return self._policy

    def prepare(self, request: NetworkRequest) -> _NetworkPlan:
        if not isinstance(request, NetworkRequest):
            raise NetworkExecutionDenied(self._blocked("explicit_request_required"))
        decision = self._policy_decision(request)
        if decision is not None:
            self._audit(AuditEventType.NETWORK_BLOCKED, request, decision)
            raise NetworkExecutionDenied(decision)
        try:
            addresses = self._resolve(request.host, request.port)
            internal = False
            for value in addresses:
                address = ipaddress.ip_address(value)
                if self._denied_ip(address):
                    decision = self._blocked("denied_ip_range", DecisionReason.NETWORK_SSRF_BLOCKED)
                    self._audit(AuditEventType.NETWORK_BLOCKED, request, decision)
                    raise NetworkExecutionDenied(decision)
                if self.policy.allowed_ip_ranges and not self._allowed_ip(address):
                    decision = self._blocked("ip_not_allowed", DecisionReason.NETWORK_POLICY_VIOLATION)
                    self._audit(AuditEventType.NETWORK_BLOCKED, request, decision)
                    raise NetworkExecutionDenied(decision)
                if not address.is_global:
                    if not self._allowed_ip(address):
                        decision = self._blocked("non_public_destination", DecisionReason.NETWORK_SSRF_BLOCKED)
                        self._audit(AuditEventType.NETWORK_BLOCKED, request, decision)
                        raise NetworkExecutionDenied(decision)
                    internal = True
        except NetworkExecutionDenied:
            raise
        except Exception:
            decision = self._blocked("dns_resolution_failed", DecisionReason.NETWORK_DNS_BLOCKED)
            self._audit(AuditEventType.NETWORK_BLOCKED, request, decision)
            raise NetworkExecutionDenied(decision)
        return _NetworkPlan(request, addresses, internal)

    def _issue_authorization(
        self,
        request: NetworkRequest,
        *,
        resolved_addresses: tuple[str, ...],
        ttl: float = 15.0,
        allow_redirects: bool = True,
        _issuer: object | None = None,
    ) -> NetworkAuthorization:
        if _issuer is not _FRAMEWORK_ISSUER:
            raise PermissionError("network authorization issuance is framework-only")
        if not isinstance(request, NetworkRequest) or not all(
            (request.agent_id, request.session_id, request.tool_id)
        ):
            raise PermissionError("network request context is incomplete")
        if not isinstance(resolved_addresses, tuple) or not resolved_addresses:
            raise PermissionError("validated DNS addresses are required")
        if not isinstance(allow_redirects, bool):
            raise ValueError("allow_redirects must be boolean")
        if (
            not isinstance(ttl, (int, float))
            or isinstance(ttl, bool)
            or not math.isfinite(ttl)
            or ttl <= 0
        ):
            raise ValueError("ttl must be a positive finite number")
        normalized_addresses = tuple(
            ipaddress.ip_address(address).compressed
            for address in resolved_addresses
        )
        if len(normalized_addresses) != len(set(normalized_addresses)):
            raise ValueError("validated DNS addresses must be unique")
        expiry = time.monotonic() + ttl
        values = (
            request.agent_id, request.session_id, request.tool_id, request.action,
            request.scheme, request.host, request.port, request.digest,
            normalized_addresses, allow_redirects, expiry,
        )
        proof = self._sign(values)
        return NetworkAuthorization(*values[:8], values[8], allow_redirects, expiry, proof)

    def revoke_authorization(self, authorization: NetworkAuthorization) -> None:
        if not isinstance(authorization, NetworkAuthorization):
            raise ValueError("authorization is invalid")
        with self._lock:
            self._revoked.add(authorization._proof)

    def _valid_authorization(
        self, request: NetworkRequest, authorization: NetworkAuthorization | None
    ) -> bool:
        try:
            if not isinstance(authorization, NetworkAuthorization):
                return False
            with self._lock:
                if (
                    authorization._proof in self._revoked
                    or authorization._proof in self._consumed
                ):
                    return False
                if not isinstance(authorization.expires_at, (int, float)) or isinstance(
                    authorization.expires_at, bool
                ):
                    return False
                if time.monotonic() >= authorization.expires_at:
                    return False
                values = (
                    authorization.agent_id, authorization.session_id, authorization.tool_id,
                    authorization.action, authorization.scheme, authorization.host,
                    authorization.port, authorization.request_digest,
                    authorization.resolved_addresses, authorization.allow_redirects,
                    authorization.expires_at,
                )
                valid = (
                    isinstance(authorization._proof, str)
                    and hmac.compare_digest(authorization._proof, self._sign(values))
                    and (authorization.agent_id, authorization.session_id, authorization.tool_id)
                    == (request.agent_id, request.session_id, request.tool_id)
                    and (authorization.action, authorization.scheme, authorization.host, authorization.port)
                    == (request.action, request.scheme, request.host, request.port)
                    and authorization.request_digest == request.digest
                    and isinstance(authorization.resolved_addresses, tuple)
                    and isinstance(authorization.allow_redirects, bool)
                    and bool(authorization.resolved_addresses)
                )
                if valid:
                    self._consumed.add(authorization._proof)
                return valid
        except Exception:
            return False

    def execute(
        self,
        request: NetworkRequest,
        *,
        authorization: NetworkAuthorization | None = None,
        release_authorization: DataReleaseAuthorization | None = None,
        release_request: DataReleaseRequest | None = None,
        application_id: str | None = None,
    ) -> NetworkResponse:
        if self._mission_registry is None:
            raise NetworkExecutionDenied(
                self._blocked("mission_context_missing", DecisionReason.MISSION_INVALID)
            )
        decision = self._mission_registry.check(
            self._mission_agent_id,
            action=request.action if isinstance(request, NetworkRequest) else "",
            resource=request.destination if isinstance(request, NetworkRequest) else "",
            tool=self._mission_tool or "",
            destination=request.host if isinstance(request, NetworkRequest) else None,
            data=request.classification.value if isinstance(request, NetworkRequest) else None,
            operation="network_execution",
        )
        if decision.decision_type is not DecisionType.ALLOW:
            raise NetworkExecutionDenied(decision)
        if application_id is not None and not verify_release_authorization(
            release_authorization, release_request, application_id=application_id
        ):
            raise NetworkExecutionDenied(
                self._blocked("release_authorization_invalid", DecisionReason.NETWORK_AUTHORIZATION_INVALID)
            )
        current_policy_version = getattr(
            self, "_release_policy_version_provider", lambda: None
        )()
        if (
            current_policy_version is not None
            and release_authorization is not None
            and release_authorization.policy_version != current_policy_version
        ):
            raise NetworkExecutionDenied(
                self._blocked("release_policy_version_changed", DecisionReason.NETWORK_AUTHORIZATION_INVALID)
            )
        if not isinstance(request, NetworkRequest) or not self._valid_authorization(request, authorization):
            decision = self._blocked("invalid_network_authorization", DecisionReason.NETWORK_AUTHORIZATION_INVALID)
            self._audit(AuditEventType.NETWORK_BLOCKED, request if isinstance(request, NetworkRequest) else None, decision)
            raise NetworkExecutionDenied(decision)
        decision = self._policy_decision(request)
        if decision is not None:
            self._audit(AuditEventType.NETWORK_BLOCKED, request, decision)
            raise NetworkExecutionDenied(decision)
        addresses = authorization.resolved_addresses
        try:
            if not self._addresses_allowed(addresses):
                decision = self._blocked("address_scope_changed", DecisionReason.NETWORK_SSRF_BLOCKED)
                self._audit(AuditEventType.NETWORK_BLOCKED, request, decision)
                raise NetworkExecutionDenied(decision)
        except NetworkExecutionDenied:
            raise
        except Exception:
            decision = self._blocked("address_validation_error", DecisionReason.NETWORK_DNS_BLOCKED)
            self._audit(AuditEventType.NETWORK_BLOCKED, request, decision)
            raise NetworkExecutionDenied(decision)
        if not self._concurrency.acquire(blocking=False):
            decision = self._blocked("concurrency_limit", DecisionReason.CONCURRENCY_LIMIT_EXCEEDED)
            self._audit(AuditEventType.NETWORK_BLOCKED, request, decision)
            raise NetworkExecutionDenied(decision)
        started = time.monotonic()
        try:
            self._audit(AuditEventType.NETWORK_REQUEST_STARTED, request)
            response = self._execute_redirects(
                request,
                addresses,
                started,
                allow_redirects=authorization.allow_redirects,
            )
            if len(response.body) > self.policy.max_response_bytes:
                decision = self._blocked("response_size_limit", DecisionReason.NETWORK_RESPONSE_BLOCKED)
                self._audit(AuditEventType.NETWORK_BLOCKED, request, decision)
                raise NetworkExecutionDenied(decision)
            self._audit(AuditEventType.NETWORK_REQUEST_COMPLETED, request, metadata={"status": response.status})
            return response
        except NetworkExecutionDenied:
            raise
        except (TimeoutError, socket.timeout):
            decision = self._blocked("timeout", DecisionReason.TIMEOUT)
            self._audit(AuditEventType.NETWORK_BLOCKED, request, decision)
            raise NetworkExecutionDenied(decision)
        except Exception:
            decision = self._blocked("network_execution_error", DecisionReason.NETWORK_EXECUTION_ERROR)
            self._audit(AuditEventType.NETWORK_BLOCKED, request, decision)
            raise NetworkExecutionDenied(decision)
        finally:
            self._concurrency.release()

    def _execute_redirects(
        self,
        request: NetworkRequest,
        addresses: tuple[str, ...],
        started: float,
        *,
        allow_redirects: bool = True,
    ) -> NetworkResponse:
        current = request
        count = 0
        while True:
            remaining = self.policy.timeout - (time.monotonic() - started)
            if remaining <= 0:
                raise NetworkExecutionDenied(self._blocked("timeout", DecisionReason.TIMEOUT))
            status, headers, body = self._request_once(current, addresses, remaining)
            header_size = sum(
                len(key.encode("utf-8")) + len(value.encode("utf-8")) + 4
                for key, value in headers.items()
            )
            if header_size + len(body) > self.policy.max_response_bytes:
                raise NetworkExecutionDenied(
                    self._blocked("response_size_limit", DecisionReason.NETWORK_RESPONSE_BLOCKED)
                )
            location = next((value for key, value in headers.items() if key.lower() == "location"), None)
            if status not in _REDIRECT_CODES or location is None:
                return NetworkResponse(status, headers, body, current.url, count)
            if not allow_redirects:
                raise NetworkExecutionDenied(
                    self._blocked(
                        "redirect_not_authorized",
                        DecisionReason.NETWORK_REDIRECT_BLOCKED,
                    )
                )
            if count >= self.policy.max_redirects:
                raise NetworkExecutionDenied(
                    self._blocked("redirect_limit", DecisionReason.NETWORK_REDIRECT_BLOCKED)
                )
            try:
                target = NetworkRequest(
                    url=urljoin(current.url, location),
                    method=("GET" if status == 303 else current.method),
                    headers=current.headers,
                    body=(b"" if status == 303 else current.body),
                    classification=current.classification,
                    source_trust=current.source_trust,
                    context=current.context,
                    agent_id=current.agent_id,
                    session_id=current.session_id,
                    tool_id=current.tool_id,
                )
            except Exception:
                raise NetworkExecutionDenied(
                    self._blocked("malformed_redirect", DecisionReason.NETWORK_REDIRECT_BLOCKED)
                )
            if (target.scheme, target.host, target.port) != (request.scheme, request.host, request.port):
                raise NetworkExecutionDenied(
                    self._blocked("cross_origin_redirect", DecisionReason.NETWORK_REDIRECT_BLOCKED)
                )
            decision = self._policy_decision(target)
            if decision is not None:
                raise NetworkExecutionDenied(
                    self._blocked("redirect_policy_denied", DecisionReason.NETWORK_REDIRECT_BLOCKED)
                )
            try:
                addresses = self._resolve(target.host, target.port)
            except Exception as exc:
                raise NetworkExecutionDenied(
                    self._blocked("redirect_dns_denied", DecisionReason.NETWORK_REDIRECT_BLOCKED)
                ) from exc
            if not self._addresses_allowed(addresses):
                raise NetworkExecutionDenied(
                    self._blocked("redirect_address_denied", DecisionReason.NETWORK_REDIRECT_BLOCKED)
                )
            current = target
            count += 1

    def _request_once(
        self, request: NetworkRequest, addresses: tuple[str, ...], timeout: float
    ) -> tuple[int, Mapping[str, str], bytes]:
        started = time.monotonic()

        def remaining() -> float:
            value = timeout - (time.monotonic() - started)
            if value <= 0:
                raise TimeoutError("network request timed out")
            return value

        connection_type = _PinnedHTTPSConnection if request.scheme == "https" else _PinnedHTTPConnection
        connection = connection_type(
            request.host,
            request.port,
            addresses=addresses,
            timeout=remaining(),
        )
        try:
            connection.putrequest(request.method, request.path, skip_host=True, skip_accept_encoding=True)
            connection.putheader("Host", _authority(request.host, request.port, request.scheme))
            for name, value in request.headers.items():
                connection.putheader(name, value)
            if request.body:
                connection.putheader("Content-Length", str(len(request.body)))
            if connection.sock is not None:
                connection.sock.settimeout(remaining())
            connection.endheaders(request.body if request.body else None)
            if connection.sock is not None:
                connection.sock.settimeout(remaining())
            response = connection.getresponse()
            raw_headers = response.getheaders()
            header_size = sum(
                len(key.encode("utf-8")) + len(value.encode("utf-8")) + 4
                for key, value in raw_headers
            )
            if header_size > self.policy.max_response_bytes:
                raise NetworkExecutionDenied(
                    self._blocked("response_size_limit", DecisionReason.NETWORK_RESPONSE_BLOCKED)
                )
            headers = dict(raw_headers)
            body = bytearray()
            limit = self.policy.max_response_bytes - header_size + 1
            while len(body) < limit:
                if connection.sock is not None:
                    connection.sock.settimeout(remaining())
                chunk = response.read1(min(64 * 1024, limit - len(body)))
                if not chunk:
                    break
                body.extend(chunk)
            return response.status, headers, bytes(body)
        finally:
            connection.close()

    def _policy_decision(self, request: NetworkRequest) -> Decision | None:
        if request.method not in self.policy.allowed_methods:
            return self._blocked("method_denied")
        if request.scheme not in self.policy.allowed_schemes or request.scheme in self.policy.denied_schemes:
            return self._blocked("scheme_denied", DecisionReason.NETWORK_SCHEME_BLOCKED)
        if self.policy.require_https and request.scheme != "https":
            return self._blocked("https_required", DecisionReason.NETWORK_SCHEME_BLOCKED)
        if not self._host_matches(request.host, self.policy.allowed_hosts):
            return self._blocked("host_not_allowed", DecisionReason.NETWORK_POLICY_VIOLATION)
        if self._host_matches(request.host, self.policy.denied_hosts):
            return self._blocked("host_denied", DecisionReason.NETWORK_POLICY_VIOLATION)
        if request.port in self.policy.denied_ports:
            return self._blocked("port_denied", DecisionReason.NETWORK_PORT_BLOCKED)
        if self.policy.allowed_ports and request.port not in self.policy.allowed_ports:
            return self._blocked("port_not_allowed", DecisionReason.NETWORK_PORT_BLOCKED)
        default_port = 443 if request.scheme == "https" else 80
        if not self.policy.allowed_ports and request.port != default_port:
            return self._blocked("custom_port_not_allowed", DecisionReason.NETWORK_PORT_BLOCKED)
        request_size = (
            len(request.method.encode("ascii"))
            + len(request.path.encode("utf-8"))
            + len(request.body)
            + len(_authority(request.host, request.port, request.scheme).encode("ascii"))
            + len("Host: \r\n")
            + sum(
                len(key.encode("ascii")) + len(value.encode("utf-8")) + 4
                for key, value in request.headers.items()
            )
            + (
                len(str(len(request.body))) + len("Content-Length: \r\n")
                if request.body
                else 0
            )
        )
        if request_size > self.policy.max_request_bytes:
            return self._blocked("request_size_limit", DecisionReason.NETWORK_POLICY_VIOLATION)
        return None

    def _resolve(self, host: str, port: int) -> tuple[str, ...]:
        try:
            address = ipaddress.ip_address(host)
        except ValueError:
            records = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
            resolved = []
            for record in records:
                text = record[4][0].split("%", 1)[0]
                address = ipaddress.ip_address(text)
                if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped is not None:
                    address = address.ipv4_mapped
                resolved.append(address.compressed)
        else:
            if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped is not None:
                address = address.ipv4_mapped
            resolved = [address.compressed]
        if not resolved:
            raise ValueError("hostname has no addresses")
        return tuple(sorted(set(resolved)))

    def _addresses_allowed(self, values: tuple[str, ...]) -> bool:
        if not values:
            return False
        for value in values:
            address = ipaddress.ip_address(value)
            if self._denied_ip(address):
                return False
            if self.policy.allowed_ip_ranges and not self._allowed_ip(address):
                return False
            if not address.is_global and not self._allowed_ip(address):
                return False
        return True

    def _allowed_ip(self, address: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
        return any(
            address.version == ipaddress.ip_network(value).version and address in ipaddress.ip_network(value)
            for value in self.policy.allowed_ip_ranges
        )

    def _denied_ip(self, address: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
        return (
            address.is_multicast
            or address.is_unspecified
            or address.is_reserved
            or any(
                address.version == ipaddress.ip_network(value).version
                and address in ipaddress.ip_network(value)
                for value in self.policy.denied_ip_ranges
            )
        )

    @staticmethod
    def _host_matches(host: str, patterns: tuple[str, ...]) -> bool:
        for pattern in patterns:
            if pattern.startswith("*."):
                suffix = pattern[2:]
                if host.endswith("." + suffix) and host != suffix:
                    return True
            elif host == pattern:
                return True
        return False

    def _sign(self, values: tuple[Any, ...]) -> str:
        canonical = json.dumps(values, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        return hmac.new(self._secret, canonical.encode("utf-8"), hashlib.sha256).hexdigest()

    @staticmethod
    def _blocked(
        category: str,
        reason: DecisionReason = DecisionReason.NETWORK_POLICY_VIOLATION,
    ) -> Decision:
        return Decision(
            DecisionType.BLOCK,
            reason,
            severity=Severity.HIGH,
            metadata={"stage": "network", "category": category},
        )

    def _audit(
        self,
        event_type: AuditEventType,
        request: NetworkRequest | None,
        decision: Decision | None = None,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if self.audit_logger is None:
            return
        try:
            self.audit_logger.record(
                event_type,
                agent_id=request.agent_id if request else None,
                session_id=request.session_id if request else None,
                tool_id=request.tool_id if request else None,
                action=request.action if request else None,
                resource=request.destination if request else None,
                decision=decision.decision_type.value if decision else None,
                reason=decision.reason.value if decision else None,
                metadata=metadata or (
                    {"category": decision.metadata.get("category", "blocked")} if decision else {}
                ),
            )
        except Exception:
            if self.audit_logger.policy.required:
                raise NetworkExecutionDenied(
                    Decision(
                        DecisionType.BLOCK,
                        DecisionReason.RUNTIME_SECURITY_ERROR,
                        severity=Severity.CRITICAL,
                        metadata={"stage": "network_audit"},
                    )
                )


def _authority(host: str, port: int, scheme: str) -> str:
    value = f"[{host}]" if ":" in host else host
    if (scheme == "https" and port != 443) or (scheme == "http" and port != 80):
        value += f":{port}"
    return value


class _PinnedHTTPConnection(http.client.HTTPConnection):
    def __init__(self, host: str, port: int, *, addresses: tuple[str, ...], timeout: float) -> None:
        super().__init__(host, port, timeout=timeout)
        self._addresses = addresses

    def connect(self) -> None:
        last_error: OSError | None = None
        deadline = time.monotonic() + float(self.timeout)
        for value in self._addresses:
            try:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("network connection timed out")
                self.sock = socket.create_connection((value, self.port), remaining)
                return
            except OSError as exc:
                last_error = exc
        if last_error is not None:
            raise last_error
        raise OSError("no validated destination address")


class _PinnedHTTPSConnection(_PinnedHTTPConnection):
    def __init__(self, host: str, port: int, *, addresses: tuple[str, ...], timeout: float) -> None:
        super().__init__(host, port, addresses=addresses, timeout=timeout)
        self._context = ssl.create_default_context()

    def connect(self) -> None:
        super().connect()
        if self.sock is None:
            raise OSError("TLS socket was not created")
        self.sock = self._context.wrap_socket(self.sock, server_hostname=self.host)
