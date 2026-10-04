"""Application-owned integrity authority for security configuration."""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from datetime import datetime, timezone
from enum import Enum
import hashlib
import json
import threading
from types import MappingProxyType
from typing import Any, Mapping

from ..audit import AuditEventType, AuditLogger
from .permissions import PermissionEngine
from .policy import PolicyEvaluator

_APPLICATION_CONTEXT_TOKEN = object()


def _canonical(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _canonical(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_canonical(item) for item in value]
    if is_dataclass(value):
        return {field.name: _canonical(getattr(value, field.name)) for field in fields(value)}
    if hasattr(value, "to_json") and callable(value.to_json):
        return json.loads(value.to_json())
    if hasattr(value, "__dict__"):
        return {
            key: _canonical(item)
            for key, item in sorted(vars(value).items())
            if not key.startswith("_")
        }
    raise TypeError("security configuration is not deterministically serializable")


def configuration_digest(policy: PolicyEvaluator, permissions: PermissionEngine, runtime: Any) -> str:
    payload = _canonical({
        "policy_type": f"{type(policy).__module__}.{type(policy).__qualname__}",
        "policy": policy,
        "permissions": permissions,
        "runtime": runtime,
    })
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class SecurityConfigurationSnapshot:
    configuration_id: str
    version: int
    digest: str
    policy: PolicyEvaluator
    permissions: PermissionEngine
    runtime: Any
    state: str = "active"
    expires_at: datetime | None = None


class PolicyIntegrityRegistry:
    """Single application-owned authority for active security configuration."""

    def __init__(self, *, audit_logger: AuditLogger, _owner_token: object | None = None) -> None:
        if _owner_token is not _APPLICATION_CONTEXT_TOKEN:
            raise ValueError("policy integrity registry must be created by PreventApp")
        self._audit = audit_logger
        self._lock = threading.RLock()
        self._snapshots: dict[tuple[str, int], SecurityConfigurationSnapshot] = {}
        self._active: tuple[str, int] | None = None

    def register(
        self,
        *,
        configuration_id: str,
        version: int,
        policy: PolicyEvaluator,
        permissions: PermissionEngine,
        runtime: Any,
        activate: bool = True,
        expires_at: datetime | None = None,
    ) -> SecurityConfigurationSnapshot:
        if not isinstance(configuration_id, str) or not configuration_id.strip():
            raise ValueError("configuration_id must be non-empty")
        if not isinstance(version, int) or isinstance(version, bool) or version < 1:
            raise ValueError("version must be a positive integer")
        if not isinstance(permissions, PermissionEngine):
            raise TypeError("permissions must be a PermissionEngine")
        digest = configuration_digest(policy, permissions, runtime)
        snapshot = SecurityConfigurationSnapshot(
            configuration_id.strip(), version, digest, policy, permissions, runtime,
            "active", expires_at,
        )
        if expires_at is not None and (
            not isinstance(expires_at, datetime) or expires_at.tzinfo is None
        ):
            raise ValueError("expires_at must be timezone-aware")
        with self._lock:
            key = (snapshot.configuration_id, snapshot.version)
            if key in self._snapshots:
                raise ValueError("security configuration version already exists")
            if self._active is not None and version <= self._active[1]:
                raise ValueError("security configuration version is stale")
            previous_active = self._active
            self._snapshots[key] = snapshot
            try:
                self._record(AuditEventType.POLICY_REGISTERED, snapshot)
                if activate:
                    self._active = key
                    self._record(AuditEventType.POLICY_ACTIVATED, snapshot)
            except Exception:
                self._snapshots.pop(key, None)
                self._active = previous_active
                raise
            return snapshot

    def active(self) -> SecurityConfigurationSnapshot:
        with self._lock:
            if self._active is None:
                raise RuntimeError("no active security configuration")
            snapshot = self._snapshots.get(self._active)
            if snapshot is None or snapshot.state != "active":
                raise RuntimeError("active security configuration is invalid")
            if snapshot.expires_at is not None and snapshot.expires_at <= datetime.now(timezone.utc):
                raise RuntimeError("active security configuration is expired")
            if configuration_digest(snapshot.policy, snapshot.permissions, snapshot.runtime) != snapshot.digest:
                self._record(AuditEventType.POLICY_INTEGRITY_MISMATCH, snapshot)
                raise RuntimeError("active security configuration integrity mismatch")
            self._record(AuditEventType.POLICY_VERIFIED, snapshot)
            return snapshot

    def get_active_version(self, configuration_id: str) -> str:
        with self._lock:
            snapshot = self.active()
            if snapshot.configuration_id != configuration_id:
                raise RuntimeError("active security configuration identity mismatch")
            return str(snapshot.version)

    def activate(self, configuration_id: str, version: int) -> SecurityConfigurationSnapshot:
        with self._lock:
            snapshot = self._snapshots.get((configuration_id, version))
            if snapshot is None or snapshot.state != "active":
                raise ValueError("unknown or invalid security configuration")
            if configuration_id != "application-security" and self._active is not None:
                raise ValueError("security configuration identity mismatch")
            if self._active is not None and version < self._active[1]:
                raise ValueError("security configuration version is stale")
            if configuration_digest(snapshot.policy, snapshot.permissions, snapshot.runtime) != snapshot.digest:
                self._record(AuditEventType.POLICY_INTEGRITY_MISMATCH, snapshot)
                raise ValueError("security configuration integrity mismatch")
            self._active = (configuration_id, version)
            self._record(AuditEventType.POLICY_ACTIVATED, snapshot)
            return snapshot

    def revoke(self, configuration_id: str, version: int) -> None:
        with self._lock:
            snapshot = self._snapshots.get((configuration_id, version))
            if snapshot is None:
                raise ValueError("unknown security configuration")
            updated = SecurityConfigurationSnapshot(
                snapshot.configuration_id, snapshot.version, snapshot.digest,
                snapshot.policy, snapshot.permissions, snapshot.runtime, "revoked",
                snapshot.expires_at,
            )
            self._snapshots[(configuration_id, version)] = updated
            previous_active = self._active
            if self._active == (configuration_id, version):
                self._active = None
            try:
                self._record(AuditEventType.POLICY_REVOKED, updated)
            except Exception:
                self._snapshots[(configuration_id, version)] = snapshot
                self._active = previous_active
                raise

    def _record(self, event_type: AuditEventType, snapshot: SecurityConfigurationSnapshot) -> None:
        event = self._audit.record(
            event_type,
            action="security_configuration",
            resource=snapshot.configuration_id,
            metadata={
                "configuration_id": snapshot.configuration_id,
                "version": snapshot.version,
                "digest": f"sha256:{snapshot.digest}",
                "state": snapshot.state,
            },
        )
        if event is None and self._audit.policy.required:
            raise RuntimeError("mandatory policy integrity audit event was not recorded")
