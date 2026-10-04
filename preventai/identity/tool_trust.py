"""Application-owned tool supply-chain identity, integrity, and trust."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
import hashlib
import inspect
import threading
import re
from types import MappingProxyType
from typing import Any, Callable, Mapping

from ..audit.event import AuditEventType
from ..audit.logger import AuditLogger

_APPLICATION_CONTEXT_TOKEN = object()


def _text(value: str, field: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 256:
        raise ValueError(f"{field} must be a non-empty string")
    value = value.strip()
    if any(ord(char) < 32 for char in value):
        raise ValueError(f"{field} contains control characters")
    return value


def _version(value: str) -> str:
    value = _text(value, "version")
    if not re.fullmatch(r"[0-9]+(?:\.[0-9]+){1,2}(?:[-+][0-9A-Za-z.-]+)?", value):
        raise ValueError("version must be a canonical numeric version")
    return value


def validate_tool_version(value: str) -> str:
    """Validate and return a canonical tool version."""
    return _version(value)


def implementation_digest(function: Callable[..., Any]) -> str:
    if not callable(function):
        raise TypeError("tool implementation must be callable")
    try:
        source = inspect.getsource(function)
        payload = source.encode("utf-8")
    except (OSError, TypeError):
        code = getattr(function, "__code__", None)
        if code is None:
            raise ValueError("tool implementation has no stable source representation")
        payload = repr(code).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class ToolTrustState(str, Enum):
    REGISTERED = "registered"
    VERIFIED = "verified"
    TRUSTED = "trusted"
    SUSPENDED = "suspended"
    REVOKED = "revoked"
    EXPIRED = "expired"


@dataclass(frozen=True, slots=True)
class ToolProvenance:
    provider: str
    source: str
    package: str | None = None
    publisher: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "provider", _text(self.provider, "provider"))
        object.__setattr__(self, "source", _text(self.source, "source"))
        if self.package is not None:
            object.__setattr__(self, "package", _text(self.package, "package"))
        if self.publisher is not None:
            object.__setattr__(self, "publisher", _text(self.publisher, "publisher"))


@dataclass(frozen=True, slots=True)
class ToolTrustRecord:
    tool_id: str
    name: str
    version: str
    provenance: ToolProvenance
    integrity_digest: str
    state: ToolTrustState
    registered_at: datetime
    expires_at: datetime | None = None
    issuer: str = "application"

    def __post_init__(self) -> None:
        _text(self.tool_id, "tool_id")
        _text(self.name, "name")
        object.__setattr__(self, "version", _version(self.version))
        if not isinstance(self.provenance, ToolProvenance):
            raise TypeError("provenance must be ToolProvenance")
        if not isinstance(self.integrity_digest, str) or len(self.integrity_digest) != 64:
            raise ValueError("integrity_digest must be a SHA-256 digest")
        if not isinstance(self.state, ToolTrustState):
            raise ValueError("state must be a ToolTrustState")
        if self.expires_at is not None and self.expires_at.tzinfo is None:
            raise ValueError("expires_at must be timezone-aware")


class ToolTrustRegistry:
    """Single app-owned authority for tool provenance and integrity."""

    def __init__(
        self,
        *,
        audit_logger: AuditLogger | None = None,
        _owner_token: object | None = None,
    ) -> None:
        if _owner_token is not _APPLICATION_CONTEXT_TOKEN:
            raise ValueError("tool trust registry must be created by PreventApp")
        self._audit = audit_logger or AuditLogger()
        self._records: dict[tuple[str, str], ToolTrustRecord] = {}
        self._lock = threading.RLock()

    def register(
        self,
        *,
        tool_id: str,
        name: str,
        version: str,
        provenance: ToolProvenance,
        integrity_digest: str,
        issuer: str = "application",
        expires_at: datetime | None = None,
        state: ToolTrustState = ToolTrustState.REGISTERED,
    ) -> ToolTrustRecord:
        record = ToolTrustRecord(
            tool_id, name, version, provenance, integrity_digest, state,
            datetime.now(timezone.utc), expires_at, issuer,
        )
        with self._lock:
            key = (record.tool_id, record.version)
            if key in self._records:
                raise ValueError("tool trust record already exists")
            self._records[key] = record
            self._record(AuditEventType.TOOL_REGISTERED, record)
            return record

    def trust(self, tool_id: str, version: str) -> ToolTrustRecord:
        return self._transition(tool_id, version, ToolTrustState.TRUSTED, AuditEventType.TOOL_TRUSTED)

    def revoke(self, tool_id: str, version: str) -> ToolTrustRecord:
        return self._transition(tool_id, version, ToolTrustState.REVOKED, AuditEventType.TOOL_REVOKED)

    def suspend(self, tool_id: str, version: str) -> ToolTrustRecord:
        return self._transition(tool_id, version, ToolTrustState.SUSPENDED, AuditEventType.TOOL_UNTRUSTED)

    def get(self, tool_id: str, version: str) -> ToolTrustRecord | None:
        with self._lock:
            return self._records.get((tool_id, version))

    def check(
        self,
        *,
        tool_id: str,
        name: str,
        version: str,
        provenance: ToolProvenance,
        integrity_digest: str,
    ) -> tuple[bool, str]:
        with self._lock:
            record = self._records.get((tool_id, version))
            if record is None:
                if any(existing_id == tool_id for existing_id, _ in self._records):
                    self._record(
                        AuditEventType.TOOL_VERSION_MISMATCH,
                        ToolTrustRecord(
                            tool_id, name, version, provenance, integrity_digest,
                            ToolTrustState.REGISTERED, datetime.now(timezone.utc),
                        ),
                    )
                    return False, "version_mismatch"
                return False, "unknown_tool"
            if record.name != name or record.provenance != provenance:
                self._record(AuditEventType.TOOL_PROVENANCE_MISMATCH, record)
                return False, "provenance_mismatch"
            if record.integrity_digest != integrity_digest:
                self._record(AuditEventType.TOOL_INTEGRITY_MISMATCH, record)
                return False, "integrity_mismatch"
            if record.expires_at is not None and record.expires_at <= datetime.now(timezone.utc):
                if record.state is not ToolTrustState.EXPIRED:
                    record = ToolTrustRecord(
                        record.tool_id, record.name, record.version, record.provenance,
                        record.integrity_digest, ToolTrustState.EXPIRED,
                        record.registered_at, record.expires_at, record.issuer,
                    )
                    self._records[(record.tool_id, record.version)] = record
                self._record(AuditEventType.TOOL_EXPIRED, record)
                return False, "expired"
            if record.state is not ToolTrustState.TRUSTED:
                return False, record.state.value
            self._record(AuditEventType.TOOL_VERIFIED, record)
            return True, "trusted"

    def _transition(
        self, tool_id: str, version: str, state: ToolTrustState, event_type: AuditEventType
    ) -> ToolTrustRecord:
        with self._lock:
            current = self._records.get((tool_id, version))
            if current is None:
                raise ValueError("unknown tool trust record")
            if current.state in {ToolTrustState.REVOKED, ToolTrustState.EXPIRED}:
                raise ValueError("trust state cannot be reactivated")
            updated = ToolTrustRecord(
                current.tool_id, current.name, current.version, current.provenance,
                current.integrity_digest, state, current.registered_at,
                current.expires_at, current.issuer,
            )
            self._records[(tool_id, version)] = updated
            self._record(event_type, updated)
            return updated

    def _record(self, event_type: AuditEventType, record: ToolTrustRecord) -> None:
        event = self._audit.record(
            event_type,
            tool_id=record.tool_id,
            action=record.name,
            resource=f"{record.tool_id}@{record.version}",
            metadata={
                "version": record.version,
                "provider": record.provenance.provider,
                "source": record.provenance.source,
                "integrity_digest": f"sha256:{record.integrity_digest}",
                "state": record.state.value,
            },
        )
        if event is None and self._audit.policy.required:
            raise RuntimeError("mandatory tool trust audit event was not recorded")
