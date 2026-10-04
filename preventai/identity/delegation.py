"""Bounded, signed delegation grants with capability attenuation."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import hmac
import json
import math
import threading
from types import MappingProxyType
from typing import Any, Iterable, Mapping
from uuid import uuid4

from ..core.decision import Decision, DecisionReason, DecisionType, Severity
from ..core.permissions import PermissionCategory, PermissionEngine
from .agent_identity import _identity_id


def _text(value: str, field: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 256:
        raise ValueError(f"{field} must be a non-empty string")
    if any(ord(char) < 32 for char in value):
        raise ValueError(f"{field} contains control characters")
    return value


def _timestamp(value: datetime, field: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise ValueError(f"{field} must be timezone-aware")
    return value.astimezone(timezone.utc)


def _immutable_context(value: Any) -> Any:
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) or not key for key in value):
            raise ValueError("context constraint keys must be non-empty strings")
        return MappingProxyType({
            key: _immutable_context(item) for key, item in value.items()
        })
    if isinstance(value, (list, tuple)):
        return tuple(_immutable_context(item) for item in value)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise ValueError("context constraints must contain structured scalar values")


def _plain_context(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _plain_context(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain_context(item) for item in value]
    return value


def _scope_contains(parent: tuple[str, ...], child: tuple[str, ...]) -> bool:
    if parent == ("*",):
        return True
    return child != ("*",) and set(child).issubset(parent)


@dataclass(frozen=True, slots=True)
class Capability:
    action: str
    resource: str = "*"
    tool: str = "*"
    category: PermissionCategory | None = None
    destinations: tuple[str, ...] = ("*",)
    data_classes: tuple[str, ...] = ("*",)
    context_constraints: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _text(self.action, "action")
        _text(self.resource, "resource")
        _text(self.tool, "tool")
        if self.category is not None and not isinstance(self.category, PermissionCategory):
            raise ValueError("category must be a PermissionCategory")
        for field_name in ("destinations", "data_classes"):
            values = getattr(self, field_name)
            if not isinstance(values, tuple) or not values or any(
                not isinstance(item, str) or not item.strip() for item in values
            ):
                raise ValueError(f"{field_name} must contain non-empty strings")
            if "*" in values and len(values) != 1:
                raise ValueError(f"{field_name} cannot combine wildcard and explicit scopes")
        if not isinstance(self.context_constraints, Mapping):
            raise ValueError("context_constraints must be a mapping")
        object.__setattr__(self, "context_constraints", _immutable_context(self.context_constraints))

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "resource": self.resource,
            "tool": self.tool,
            "category": self.category.value if self.category is not None else None,
            "destinations": self.destinations,
            "data_classes": self.data_classes,
            "context_constraints": _plain_context(self.context_constraints),
        }


@dataclass(frozen=True, slots=True)
class DelegationGrant:
    grant_id: str
    issuer_identity: str
    subject_identity: str
    capabilities: tuple[Capability, ...]
    issued_at: datetime
    expires_at: datetime
    depth: int = 1
    parent_grant_id: str | None = None
    session_id: str | None = None
    enabled: bool = True
    signature: str = ""

    def __post_init__(self) -> None:
        _text(self.grant_id, "grant_id")
        _identity_id(self.issuer_identity, "issuer_identity")
        _identity_id(self.subject_identity, "subject_identity")
        if not self.capabilities or any(not isinstance(item, Capability) for item in self.capabilities):
            raise ValueError("capabilities must contain Capability objects")
        issued = _timestamp(self.issued_at, "issued_at")
        expires = _timestamp(self.expires_at, "expires_at")
        if expires <= issued:
            raise ValueError("expires_at must be later than issued_at")
        if not isinstance(self.depth, int) or isinstance(self.depth, bool) or self.depth < 1:
            raise ValueError("depth must be positive")
        if self.parent_grant_id is not None:
            _text(self.parent_grant_id, "parent_grant_id")
        if self.session_id is not None:
            _text(self.session_id, "session_id")
        if not isinstance(self.enabled, bool):
            raise ValueError("enabled must be boolean")
        if not isinstance(self.signature, str):
            raise ValueError("signature must be a string")
        object.__setattr__(self, "issued_at", issued)
        object.__setattr__(self, "expires_at", expires)
        object.__setattr__(self, "capabilities", tuple(self.capabilities))

    def unsigned_dict(self) -> dict[str, Any]:
        return {
            "grant_id": self.grant_id,
            "issuer_identity": self.issuer_identity,
            "subject_identity": self.subject_identity,
            "capabilities": [item.to_dict() for item in self.capabilities],
            "issued_at": self.issued_at.isoformat().replace("+00:00", "Z"),
            "expires_at": self.expires_at.isoformat().replace("+00:00", "Z"),
            "depth": self.depth,
            "parent_grant_id": self.parent_grant_id,
            "session_id": self.session_id,
            "enabled": self.enabled,
        }

    def to_dict(self) -> dict[str, Any]:
        result = self.unsigned_dict()
        result["signature"] = self.signature
        return result

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=True)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "DelegationGrant":
        expected = {
            "grant_id", "issuer_identity", "subject_identity", "capabilities",
            "issued_at", "expires_at", "depth", "parent_grant_id", "session_id",
            "enabled", "signature",
        }
        if not isinstance(value, Mapping) or set(value) != expected:
            raise ValueError("grant fields do not exactly match the contract")
        try:
            capabilities = tuple(
                Capability(
                    item["action"],
                    item["resource"],
                    item["tool"],
                    PermissionCategory(item["category"]) if item["category"] is not None else None,
                    tuple(item.get("destinations", ("*",))),
                    tuple(item.get("data_classes", ("*",))),
                    item.get("context_constraints", {}),
                )
                for item in value["capabilities"]
            )
            return cls(
                grant_id=value["grant_id"],
                issuer_identity=value["issuer_identity"],
                subject_identity=value["subject_identity"],
                capabilities=capabilities,
                issued_at=datetime.fromisoformat(value["issued_at"].replace("Z", "+00:00")),
                expires_at=datetime.fromisoformat(value["expires_at"].replace("Z", "+00:00")),
                depth=value["depth"],
                parent_grant_id=value["parent_grant_id"],
                session_id=value["session_id"],
                enabled=value["enabled"],
                signature=value["signature"],
            )
        except (KeyError, TypeError, ValueError, AttributeError) as exc:
            raise ValueError("invalid delegation grant") from exc


def _glob_contains(parent: str, child: str) -> bool:
    if parent == "*":
        return True
    if parent == child:
        return True
    if "*" not in parent or "*" in child:
        return False
    prefix, suffix = parent.split("*", 1)
    return child.startswith(prefix) and child.endswith(suffix)


def _capability_contains(parent: Capability, child: Capability) -> bool:
    return (
        _glob_contains(parent.action, child.action)
        and _glob_contains(parent.resource, child.resource)
        and _glob_contains(parent.tool, child.tool)
        and (child.category is None or parent.category is child.category)
        and _scope_contains(parent.destinations, child.destinations)
        and _scope_contains(parent.data_classes, child.data_classes)
        and all(
            key in child.context_constraints
            and child.context_constraints[key] == value
            for key, value in parent.context_constraints.items()
        )
    )


class DelegationRegistry:
    """Thread-safe local signer, registry, and revocation store."""

    def __init__(self, signing_secret: str | None = None, *, max_depth: int = 3) -> None:
        if signing_secret is not None and (not isinstance(signing_secret, str) or not signing_secret):
            raise ValueError("signing_secret must be non-empty")
        if not isinstance(max_depth, int) or isinstance(max_depth, bool) or max_depth < 1:
            raise ValueError("max_depth must be positive")
        self._secret = (signing_secret or uuid4().hex).encode("utf-8")
        self.max_depth = max_depth
        self._revoked: set[str] = set()
        self._grants: dict[str, DelegationGrant] = {}
        self._lock = threading.RLock()

    def create(
        self,
        issuer_identity: str,
        subject_identity: str,
        capabilities: Iterable[Capability],
        *,
        expires_at: datetime,
        parent: DelegationGrant | None = None,
        session_id: str | None = None,
        issuer_permissions: PermissionEngine,
    ) -> DelegationGrant:
        _identity_id(issuer_identity, "issuer_identity")
        _identity_id(subject_identity, "subject_identity")
        capabilities = tuple(capabilities)
        if any(not isinstance(capability, Capability) for capability in capabilities):
            raise ValueError("capabilities must contain Capability objects")
        if parent is not None:
            if not isinstance(parent, DelegationGrant) or not self.verify(parent):
                raise ValueError("parent grant is not verified")
            with self._lock:
                if parent.grant_id in self._revoked or not parent.enabled:
                    raise ValueError("parent grant is revoked")
            if datetime.now(timezone.utc) >= parent.expires_at:
                raise ValueError("parent grant is expired")
            if parent.subject_identity != issuer_identity:
                raise ValueError("parent grant subject does not match issuer")
            if parent.depth >= self.max_depth:
                raise ValueError("delegation depth exceeded")
            expires_at = min(
                _timestamp(expires_at, "expires_at"),
                parent.expires_at,
            )
            if expires_at > parent.expires_at:
                expires_at = parent.expires_at
            if parent.session_id is not None and session_id != parent.session_id:
                raise ValueError("child grant session exceeds parent")
            if any(
                not any(_capability_contains(parent_capability, child) for parent_capability in parent.capabilities)
                for child in capabilities
            ):
                raise ValueError("delegated capability exceeds parent scope")
            depth = parent.depth + 1
            parent_grant_id = parent.grant_id
        else:
            depth = 1
            parent_grant_id = None
            expires_at = _timestamp(expires_at, "expires_at")
        if not isinstance(issuer_permissions, PermissionEngine):
            raise ValueError("issuer_permissions must be a PermissionEngine")
        for capability in capabilities:
            if not any(
                _capability_contains(
                    Capability(rule.action, rule.resource, rule.tool, rule.category),
                    capability,
                )
                for rule in issuer_permissions.rules
            ):
                raise ValueError("delegated capability exceeds issuer authority")
        now = datetime.now(timezone.utc)
        grant = DelegationGrant(
            grant_id=str(uuid4()),
            issuer_identity=issuer_identity,
            subject_identity=subject_identity,
            capabilities=capabilities,
            issued_at=now,
            expires_at=expires_at,
            depth=depth,
            parent_grant_id=parent_grant_id,
            session_id=session_id,
            signature="",
        )
        signed = self._sign(grant.unsigned_dict())
        grant = DelegationGrant(
            grant_id=grant.grant_id,
            issuer_identity=grant.issuer_identity,
            subject_identity=grant.subject_identity,
            capabilities=grant.capabilities,
            issued_at=grant.issued_at,
            expires_at=grant.expires_at,
            depth=grant.depth,
            parent_grant_id=grant.parent_grant_id,
            session_id=grant.session_id,
            enabled=grant.enabled,
            signature=signed,
        )
        with self._lock:
            if grant.grant_id in self._grants:
                raise ValueError("grant_id collision")
            self._grants[grant.grant_id] = grant
        return grant

    def verify(self, grant: DelegationGrant) -> bool:
        return isinstance(grant, DelegationGrant) and hmac.compare_digest(
            grant.signature, self._sign(grant.unsigned_dict())
        )

    def is_active(self, grant: DelegationGrant) -> bool:
        """Return whether a signed grant is currently usable as a parent."""
        if not self.verify(grant):
            return False
        with self._lock:
            if grant.grant_id in self._revoked or not grant.enabled:
                return False
        return datetime.now(timezone.utc) < grant.expires_at

    def revoke(self, grant_id: str) -> None:
        _text(grant_id, "grant_id")
        with self._lock:
            self._revoked.add(grant_id)

    def evaluate(
        self,
        grant: DelegationGrant,
        *,
        authenticated_identity: str,
        action: str,
        resource: str,
        tool: str,
        session_id: str | None = None,
        category: PermissionCategory | None = None,
        destination: str | None = None,
        data_class: str | None = None,
        context: Mapping[str, Any] | None = None,
    ) -> Decision:
        try:
            with self._lock:
                if grant.grant_id in self._revoked or not grant.enabled:
                    return self._blocked(DecisionReason.DELEGATION_REVOKED)
                if not self.verify(grant):
                    return self._blocked(DecisionReason.DELEGATION_SIGNATURE_INVALID)
                if grant.subject_identity != authenticated_identity:
                    return self._blocked(DecisionReason.DELEGATION_SUBJECT_MISMATCH)
                if grant.session_id is not None and grant.session_id != session_id:
                    return self._blocked(DecisionReason.DELEGATION_SUBJECT_MISMATCH)
                if datetime.now(timezone.utc) >= grant.expires_at:
                    return self._blocked(DecisionReason.DELEGATION_EXPIRED)
                requested = Capability(
                    action,
                    resource,
                    tool,
                    category,
                    (destination,) if destination is not None else ("*",),
                    (data_class,) if data_class is not None else ("*",),
                    context or {},
                )
                if not any(_capability_contains(capability, requested) for capability in grant.capabilities):
                    return self._blocked(DecisionReason.DELEGATION_SCOPE_EXCEEDED)
                return Decision(
                    DecisionType.ALLOW,
                    DecisionReason.REQUEST_ALLOWED,
                    metadata={"grant_id": grant.grant_id, "delegation_depth": grant.depth},
                )
        except Exception:
            return self._blocked(DecisionReason.DELEGATION_SECURITY_ERROR)

    def _sign(self, value: Mapping[str, Any]) -> str:
        canonical = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
        return hmac.new(self._secret, canonical, hashlib.sha256).hexdigest()

    @staticmethod
    def _blocked(reason: DecisionReason) -> Decision:
        return Decision(
            DecisionType.BLOCK,
            reason,
            severity=Severity.HIGH,
            metadata={"stage": "delegation"},
        )
