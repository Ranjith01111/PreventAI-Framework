"""Application-owned deterministic admission registry for managed agents."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import hashlib
import json
import threading
from types import MappingProxyType
from typing import Any, Mapping

from ..audit.event import AuditEventType
from ..audit.logger import AuditLogger
from ..core.decision import Decision, DecisionReason, DecisionType, Severity

_APPLICATION_CONTEXT_TOKEN = object()


def _text(value: str, field: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 256:
        raise ValueError(f"{field} must be a non-empty string")
    value = value.strip()
    if any(ord(char) < 32 for char in value):
        raise ValueError(f"{field} contains control characters")
    return value


def _stable_hash(value: str | None) -> str:
    payload = (value or "").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _configuration_digest(
    *,
    application_id: str,
    agent_id: str,
    version: str,
    provenance: str,
    session_id: str | None,
    parent_agent_id: str | None,
    mission_id: str | None,
    mission_version: int | None,
    capabilities: tuple[str, ...],
    tools: tuple[str, ...],
    policy_version: str | None,
) -> str:
    return hashlib.sha256(
        _canonical_json(
            {
                "application_id": application_id,
                "agent_id": agent_id,
                "version": version,
                "provenance": provenance,
                "session_id": session_id,
                "parent_agent_id": parent_agent_id,
                "mission_id": mission_id,
                "mission_version": mission_version,
                "capabilities": list(capabilities),
                "tools": list(tools),
                "policy_version": policy_version,
            }
        ).encode("utf-8")
    ).hexdigest()


class AgentAdmissionState(str, Enum):
    UNKNOWN = "unknown"
    ADMITTED = "admitted"
    ACTIVE = "active"
    SUSPENDED = "suspended"
    REVOKED = "revoked"
    EXPIRED = "expired"


@dataclass(frozen=True, slots=True)
class AgentAttestation:
    agent_id: str
    application_id: str
    version: str
    provenance: str
    mission_id: str | None = None
    mission_version: int | None = None
    capability_envelope: tuple[str, ...] = ()
    tool_set: tuple[str, ...] = ()
    policy_version: str | None = None
    parent_agent_id: str | None = None
    session_id: str | None = None
    configuration_digest: str = ""
    issued_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self) -> None:
        object.__setattr__(self, "agent_id", _text(self.agent_id, "agent_id"))
        object.__setattr__(self, "application_id", _text(self.application_id, "application_id"))
        object.__setattr__(self, "version", _text(self.version, "version"))
        object.__setattr__(self, "provenance", _text(self.provenance, "provenance"))
        if self.mission_id is not None:
            object.__setattr__(self, "mission_id", _text(self.mission_id, "mission_id"))
        if self.mission_version is not None:
            if not isinstance(self.mission_version, int) or isinstance(self.mission_version, bool) or self.mission_version < 1:
                raise ValueError("mission_version must be a positive integer")
        if not isinstance(self.capability_envelope, tuple):
            raise ValueError("capability_envelope must be a tuple")
        if not isinstance(self.tool_set, tuple):
            raise ValueError("tool_set must be a tuple")
        object.__setattr__(self, "capability_envelope", tuple(_text(item, "capability") for item in self.capability_envelope))
        object.__setattr__(self, "tool_set", tuple(_text(item, "tool") for item in self.tool_set))
        if self.policy_version is not None:
            object.__setattr__(self, "policy_version", _text(self.policy_version, "policy_version"))
        if self.parent_agent_id is not None:
            object.__setattr__(self, "parent_agent_id", _text(self.parent_agent_id, "parent_agent_id"))
        if self.session_id is not None:
            object.__setattr__(self, "session_id", _text(self.session_id, "session_id"))
        if not isinstance(self.configuration_digest, str) or len(self.configuration_digest) != 64:
            raise ValueError("configuration_digest must be a SHA-256 hex digest")
        if not isinstance(self.issued_at, datetime) or self.issued_at.tzinfo is None:
            raise ValueError("issued_at must be timezone-aware")
        if not isinstance(self.expires_at, datetime) or self.expires_at.tzinfo is None:
            raise ValueError("expires_at must be timezone-aware")
        if self.expires_at <= self.issued_at:
            raise ValueError("expires_at must be later than issued_at")

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "application_id": self.application_id,
            "version": self.version,
            "provenance": self.provenance,
            "mission_id": self.mission_id,
            "mission_version": self.mission_version,
            "capability_envelope": list(self.capability_envelope),
            "tool_set": list(self.tool_set),
            "policy_version": self.policy_version,
            "parent_agent_id": self.parent_agent_id,
            "session_id": self.session_id,
            "configuration_digest": self.configuration_digest,
            "issued_at": self.issued_at.isoformat().replace("+00:00", "Z"),
            "expires_at": self.expires_at.isoformat().replace("+00:00", "Z"),
        }

    def digest(self) -> str:
        return hashlib.sha256(_canonical_json(self.to_dict()).encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class AgentAdmissionRequest:
    agent_id: str
    application_id: str
    version: str = "1.0.0"
    provenance: str = "framework"
    session_id: str | None = None
    parent_agent_id: str | None = None
    mission_id: str | None = None
    mission_version: int | None = None
    capabilities: tuple[str, ...] = ()
    tools: tuple[str, ...] = ()
    policy_version: str | None = None
    configuration_digest: str = ""
    integrity_digest: str = ""
    attestation_digest: str | None = None
    requested_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc) + __import__("datetime").timedelta(minutes=30))
    context: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "agent_id", _text(self.agent_id, "agent_id"))
        object.__setattr__(self, "application_id", _text(self.application_id, "application_id"))
        object.__setattr__(self, "version", _text(self.version, "version"))
        object.__setattr__(self, "provenance", _text(self.provenance, "provenance"))
        if self.session_id is not None:
            object.__setattr__(self, "session_id", _text(self.session_id, "session_id"))
        if self.parent_agent_id is not None:
            object.__setattr__(self, "parent_agent_id", _text(self.parent_agent_id, "parent_agent_id"))
        if self.mission_id is not None:
            object.__setattr__(self, "mission_id", _text(self.mission_id, "mission_id"))
        if self.mission_version is not None:
            if not isinstance(self.mission_version, int) or isinstance(self.mission_version, bool) or self.mission_version < 1:
                raise ValueError("mission_version must be a positive integer")
        object.__setattr__(self, "capabilities", tuple(_text(item, "capabilities") for item in self.capabilities))
        object.__setattr__(self, "tools", tuple(_text(item, "tools") for item in self.tools))
        if self.policy_version is not None:
            object.__setattr__(self, "policy_version", _text(self.policy_version, "policy_version"))
        if not isinstance(self.configuration_digest, str) or len(self.configuration_digest) != 64:
            raise ValueError("configuration_digest must be a SHA-256 hex digest")
        if not isinstance(self.integrity_digest, str) or len(self.integrity_digest) != 64:
            raise ValueError("integrity_digest must be a SHA-256 hex digest")
        if self.attestation_digest is not None:
            if not isinstance(self.attestation_digest, str) or len(self.attestation_digest) != 64:
                raise ValueError("attestation_digest must be a SHA-256 hex digest")
        if not isinstance(self.requested_at, datetime) or self.requested_at.tzinfo is None:
            raise ValueError("requested_at must be timezone-aware")
        if not isinstance(self.expires_at, datetime) or self.expires_at.tzinfo is None:
            raise ValueError("expires_at must be timezone-aware")
        if self.expires_at <= self.requested_at:
            raise ValueError("expires_at must be later than requested_at")
        if not isinstance(self.context, Mapping):
            raise ValueError("context must be a mapping")
        object.__setattr__(self, "context", MappingProxyType(dict(self.context)))

    @property
    def attestation(self) -> AgentAttestation:
        return AgentAttestation(
            agent_id=self.agent_id,
            application_id=self.application_id,
            version=self.version,
            provenance=self.provenance,
            mission_id=self.mission_id,
            mission_version=self.mission_version,
            capability_envelope=self.capabilities,
            tool_set=self.tools,
            policy_version=self.policy_version,
            parent_agent_id=self.parent_agent_id,
            session_id=self.session_id,
            configuration_digest=self.configuration_digest,
            issued_at=self.requested_at,
            expires_at=self.expires_at,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "application_id": self.application_id,
            "version": self.version,
            "provenance": self.provenance,
            "session_id": self.session_id,
            "parent_agent_id": self.parent_agent_id,
            "mission_id": self.mission_id,
            "mission_version": self.mission_version,
            "capabilities": list(self.capabilities),
            "tools": list(self.tools),
            "policy_version": self.policy_version,
            "configuration_digest": self.configuration_digest,
            "integrity_digest": self.integrity_digest,
            "attestation_digest": self.attestation_digest,
            "requested_at": self.requested_at.isoformat().replace("+00:00", "Z"),
            "expires_at": self.expires_at.isoformat().replace("+00:00", "Z"),
            "context": dict(self.context),
        }

    def digest(self) -> str:
        return hashlib.sha256(_canonical_json(self.to_dict()).encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class AgentAdmissionRecord:
    agent_id: str
    application_id: str
    state: AgentAdmissionState
    admission_id: str
    session_id: str | None
    parent_agent_id: str | None
    mission_id: str | None
    mission_version: int | None
    version: str
    provenance: str
    capability_envelope: tuple[str, ...]
    tool_set: tuple[str, ...]
    policy_version: str | None
    configuration_digest: str
    integrity_digest: str
    attestation_digest: str
    issued_at: datetime
    expires_at: datetime
    updated_at: datetime
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "agent_id", _text(self.agent_id, "agent_id"))
        object.__setattr__(self, "application_id", _text(self.application_id, "application_id"))
        if not isinstance(self.state, AgentAdmissionState):
            raise ValueError("state must be an AgentAdmissionState")
        object.__setattr__(self, "admission_id", _text(self.admission_id, "admission_id"))
        if self.session_id is not None:
            object.__setattr__(self, "session_id", _text(self.session_id, "session_id"))
        if self.parent_agent_id is not None:
            object.__setattr__(self, "parent_agent_id", _text(self.parent_agent_id, "parent_agent_id"))
        if self.mission_id is not None:
            object.__setattr__(self, "mission_id", _text(self.mission_id, "mission_id"))
        if self.mission_version is not None:
            if not isinstance(self.mission_version, int) or isinstance(self.mission_version, bool) or self.mission_version < 1:
                raise ValueError("mission_version must be a positive integer")
        object.__setattr__(self, "version", _text(self.version, "version"))
        object.__setattr__(self, "provenance", _text(self.provenance, "provenance"))
        object.__setattr__(self, "capability_envelope", tuple(_text(item, "capability") for item in self.capability_envelope))
        object.__setattr__(self, "tool_set", tuple(_text(item, "tool") for item in self.tool_set))
        if self.policy_version is not None:
            object.__setattr__(self, "policy_version", _text(self.policy_version, "policy_version"))
        if not isinstance(self.configuration_digest, str) or len(self.configuration_digest) != 64:
            raise ValueError("configuration_digest must be a SHA-256 hex digest")
        if not isinstance(self.integrity_digest, str) or len(self.integrity_digest) != 64:
            raise ValueError("integrity_digest must be a SHA-256 hex digest")
        if not isinstance(self.attestation_digest, str) or len(self.attestation_digest) != 64:
            raise ValueError("attestation_digest must be a SHA-256 hex digest")
        if not isinstance(self.issued_at, datetime) or self.issued_at.tzinfo is None:
            raise ValueError("issued_at must be timezone-aware")
        if not isinstance(self.expires_at, datetime) or self.expires_at.tzinfo is None:
            raise ValueError("expires_at must be timezone-aware")
        if self.expires_at <= self.issued_at:
            raise ValueError("expires_at must be later than issued_at")
        if not isinstance(self.updated_at, datetime) or self.updated_at.tzinfo is None:
            raise ValueError("updated_at must be timezone-aware")
        if not isinstance(self.metadata, Mapping):
            raise ValueError("metadata must be a mapping")
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


class AgentAdmissionRegistry:
    """Application-owned admission registry for framework-managed agents."""

    def __init__(
        self,
        *,
        application_id: str,
        audit_logger: AuditLogger | None = None,
        _owner_token: object | None = None,
    ) -> None:
        if _owner_token is not _APPLICATION_CONTEXT_TOKEN:
            raise ValueError("agent admission registry must be created by PreventApp")
        self.application_id = _text(application_id, "application_id")
        self._audit = audit_logger or AuditLogger()
        self._records: dict[str, AgentAdmissionRecord] = {}
        self._lock = threading.RLock()

    def _record(self, event_type: AuditEventType, record: AgentAdmissionRecord, **metadata: Any) -> None:
        self._audit.record(
            event_type,
            agent_id=record.agent_id,
            session_id=record.session_id,
            identity=record.application_id,
            metadata={
                "application_id": record.application_id,
                "state": record.state.value,
                "mission_id": record.mission_id,
                "policy_version": record.policy_version,
                "version": record.version,
                **metadata,
            },
        )

    def _make_record(
        self,
        *,
        agent_id: str,
        application_id: str,
        session_id: str | None,
        parent_agent_id: str | None,
        mission_id: str | None,
        mission_version: int | None,
        capabilities: tuple[str, ...],
        tools: tuple[str, ...],
        version: str,
        provenance: str,
        policy_version: str | None,
        configuration_digest: str,
        integrity_digest: str,
        expires_at: datetime | None,
        metadata: Mapping[str, Any] | None = None,
    ) -> AgentAdmissionRecord:
        requested_at = datetime.now(timezone.utc)
        if expires_at is None:
            expires_at = requested_at + __import__("datetime").timedelta(minutes=60)
        now = requested_at
        attestation = AgentAttestation(
            agent_id=agent_id,
            application_id=application_id,
            version=version,
            provenance=provenance,
            mission_id=mission_id,
            mission_version=mission_version,
            capability_envelope=capabilities,
            tool_set=tools,
            policy_version=policy_version,
            parent_agent_id=parent_agent_id,
            session_id=session_id,
            configuration_digest=configuration_digest,
            issued_at=requested_at,
            expires_at=expires_at,
        )
        digest = attestation.digest()
        return AgentAdmissionRecord(
            agent_id=agent_id,
            application_id=application_id,
            state=AgentAdmissionState.ADMITTED,
            admission_id=f"admission:{agent_id}:{digest[:12]}",
            session_id=session_id,
            parent_agent_id=parent_agent_id,
            mission_id=mission_id,
            mission_version=mission_version,
            version=version,
            provenance=provenance,
            capability_envelope=capabilities,
            tool_set=tools,
            policy_version=policy_version,
            configuration_digest=configuration_digest,
            integrity_digest=integrity_digest,
            attestation_digest=digest,
            issued_at=now,
            expires_at=expires_at,
            updated_at=now,
            metadata=metadata or {},
        )

    def admit(
        self,
        agent_id: str,
        *,
        application_id: str | None = None,
        session_id: str | None = None,
        parent_agent_id: str | None = None,
        mission_id: str | None = None,
        mission_version: int | None = None,
        capabilities: tuple[str, ...] = (),
        tools: tuple[str, ...] = (),
        version: str = "1.0.0",
        provenance: str = "framework",
        policy_version: str | None = None,
        configuration_digest: str | None = None,
        integrity_digest: str | None = None,
        expires_at: datetime | None = None,
    ) -> AgentAdmissionRecord:
        effective_app = application_id or self.application_id
        if effective_app != self.application_id:
            raise ValueError("admission must belong to this application")
        if not isinstance(agent_id, str) or not agent_id.strip():
            raise ValueError("agent_id must be a non-empty string")
        if configuration_digest is None:
            configuration_digest = _configuration_digest(
                application_id=effective_app,
                agent_id=agent_id,
                version=version,
                provenance=provenance,
                session_id=session_id,
                parent_agent_id=parent_agent_id,
                mission_id=mission_id,
                mission_version=mission_version,
                capabilities=tuple(capabilities),
                tools=tuple(tools),
                policy_version=policy_version,
            )
        elif configuration_digest != _configuration_digest(
            application_id=effective_app,
            agent_id=agent_id,
            version=version,
            provenance=provenance,
            session_id=session_id,
            parent_agent_id=parent_agent_id,
            mission_id=mission_id,
            mission_version=mission_version,
            capabilities=tuple(capabilities),
            tools=tuple(tools),
            policy_version=policy_version,
        ):
            raise ValueError("configuration_digest does not match admission fields")
        if integrity_digest is None:
            integrity_digest = hashlib.sha256(b"framework-managed-agent-envelope").hexdigest()
        with self._lock:
            current = self._records.get(agent_id)
            if current is not None and current.application_id != effective_app:
                raise ValueError("cross-application substitution is denied")
            record = self._make_record(
                agent_id=agent_id,
                application_id=effective_app,
                session_id=session_id,
                parent_agent_id=parent_agent_id,
                mission_id=mission_id,
                mission_version=mission_version,
                capabilities=tuple(capabilities),
                tools=tuple(tools),
                version=version,
                provenance=provenance,
                policy_version=policy_version,
                configuration_digest=configuration_digest,
                integrity_digest=integrity_digest,
                expires_at=expires_at,
            )
            self._record(AuditEventType.AGENT_ADMISSION_REQUESTED, record)
            self._records[agent_id] = record
            try:
                self._record(AuditEventType.AGENT_ADMISSION_ALLOWED, record)
            except Exception:
                self._records.pop(agent_id, None)
                raise
            return record

    def get(self, agent_id: str) -> AgentAdmissionRecord | None:
        with self._lock:
            record = self._records.get(agent_id)
            if record is None:
                return None
            return record

    def get_state(self, agent_id: str) -> AgentAdmissionState | None:
        record = self.get(agent_id)
        return record.state if record is not None else None

    def suspend(self, agent_id: str) -> AgentAdmissionRecord:
        return self._transition(agent_id, AgentAdmissionState.SUSPENDED)

    def revoke(self, agent_id: str) -> AgentAdmissionRecord:
        return self._transition(agent_id, AgentAdmissionState.REVOKED)

    def expire(self, agent_id: str) -> AgentAdmissionRecord:
        return self._transition(agent_id, AgentAdmissionState.EXPIRED)

    def _transition(self, agent_id: str, target: AgentAdmissionState) -> AgentAdmissionRecord:
        with self._lock:
            current = self._records.get(agent_id)
            if current is None:
                raise ValueError("unknown agent admission")
            if target is AgentAdmissionState.ADMITTED:
                raise ValueError("admitted state is reserved for initial registration")
            now = datetime.now(timezone.utc)
            updated = AgentAdmissionRecord(
                agent_id=current.agent_id,
                application_id=current.application_id,
                state=target,
                admission_id=current.admission_id,
                session_id=current.session_id,
                parent_agent_id=current.parent_agent_id,
                mission_id=current.mission_id,
                mission_version=current.mission_version,
                version=current.version,
                provenance=current.provenance,
                capability_envelope=current.capability_envelope,
                tool_set=current.tool_set,
                policy_version=current.policy_version,
                configuration_digest=current.configuration_digest,
                integrity_digest=current.integrity_digest,
                attestation_digest=current.attestation_digest,
                issued_at=current.issued_at,
                expires_at=current.expires_at,
                updated_at=now,
                metadata={**current.metadata, "transition": target.value},
            )
            self._records[agent_id] = updated
            event_type = {
                AgentAdmissionState.SUSPENDED: AuditEventType.AGENT_ADMISSION_SUSPENDED,
                AgentAdmissionState.REVOKED: AuditEventType.AGENT_ADMISSION_REVOKED,
                AgentAdmissionState.EXPIRED: AuditEventType.AGENT_ADMISSION_EXPIRED,
            }[target]
            try:
                self._record(event_type, updated)
            except Exception:
                self._records[agent_id] = current
                raise
            return updated

    def verify(
        self,
        *,
        agent_id: str,
        application_id: str,
        session_id: str | None = None,
        parent_agent_id: str | None = None,
        mission_id: str | None = None,
        mission_version: int | None = None,
        capabilities: tuple[str, ...] = (),
        tools: tuple[str, ...] = (),
        version: str | None = None,
        provenance: str | None = None,
        configuration_digest: str | None = None,
        policy_version: str | None = None,
    ) -> Decision:
        with self._lock:
            record = self._records.get(agent_id)
            if record is None:
                return Decision(
                    DecisionType.BLOCK,
                    DecisionReason.INVALID_REQUEST,
                    severity=Severity.CRITICAL,
                    metadata={"stage": "agent_admission", "error": "unknown_agent"},
                )
            if record.application_id != application_id:
                return Decision(
                    DecisionType.BLOCK,
                    DecisionReason.INVALID_REQUEST,
                    severity=Severity.CRITICAL,
                    metadata={"stage": "agent_admission", "error": "application_mismatch"},
                )
            expected_configuration_digest = _configuration_digest(
                application_id=record.application_id,
                agent_id=record.agent_id,
                version=record.version,
                provenance=record.provenance,
                session_id=record.session_id,
                parent_agent_id=record.parent_agent_id,
                mission_id=record.mission_id,
                mission_version=record.mission_version,
                capabilities=record.capability_envelope,
                tools=record.tool_set,
                policy_version=record.policy_version,
            )
            if record.configuration_digest != expected_configuration_digest:
                return Decision(
                    DecisionType.BLOCK,
                    DecisionReason.INVALID_REQUEST,
                    severity=Severity.CRITICAL,
                    metadata={"stage": "agent_admission", "error": "integrity_mismatch"},
                )
            expected_attestation = AgentAttestation(
                agent_id=record.agent_id,
                application_id=record.application_id,
                version=record.version,
                provenance=record.provenance,
                mission_id=record.mission_id,
                mission_version=record.mission_version,
                capability_envelope=record.capability_envelope,
                tool_set=record.tool_set,
                policy_version=record.policy_version,
                parent_agent_id=record.parent_agent_id,
                session_id=record.session_id,
                configuration_digest=record.configuration_digest,
                issued_at=record.issued_at,
                expires_at=record.expires_at,
            )
            if record.attestation_digest != expected_attestation.digest():
                return Decision(
                    DecisionType.BLOCK,
                    DecisionReason.INVALID_REQUEST,
                    severity=Severity.CRITICAL,
                    metadata={"stage": "agent_admission", "error": "attestation_mismatch"},
                )
            if record.state in {AgentAdmissionState.SUSPENDED, AgentAdmissionState.REVOKED, AgentAdmissionState.EXPIRED}:
                return Decision(
                    DecisionType.BLOCK,
                    DecisionReason.INVALID_REQUEST,
                    severity=Severity.CRITICAL,
                    metadata={"stage": "agent_admission", "error": record.state.value},
                )
            if record.state is AgentAdmissionState.UNKNOWN:
                return Decision(DecisionType.BLOCK, DecisionReason.INVALID_REQUEST, severity=Severity.CRITICAL, metadata={"stage": "agent_admission", "error": "unknown_state"})
            if record.session_id is not None and session_id is not None and record.session_id != session_id:
                return Decision(DecisionType.BLOCK, DecisionReason.INVALID_REQUEST, severity=Severity.CRITICAL, metadata={"stage": "agent_admission", "error": "session_mismatch"})
            if record.session_id is not None and session_id is None:
                return Decision(DecisionType.BLOCK, DecisionReason.INVALID_REQUEST, severity=Severity.CRITICAL, metadata={"stage": "agent_admission", "error": "missing_session"})
            if record.parent_agent_id is not None and record.parent_agent_id != parent_agent_id:
                return Decision(DecisionType.BLOCK, DecisionReason.INVALID_REQUEST, severity=Severity.CRITICAL, metadata={"stage": "agent_admission", "error": "parent_mismatch"})
            if record.mission_id is not None and record.mission_id != mission_id:
                return Decision(DecisionType.BLOCK, DecisionReason.INVALID_REQUEST, severity=Severity.CRITICAL, metadata={"stage": "agent_admission", "error": "mission_mismatch"})
            if record.mission_version is not None and record.mission_version != mission_version:
                return Decision(DecisionType.BLOCK, DecisionReason.INVALID_REQUEST, severity=Severity.CRITICAL, metadata={"stage": "agent_admission", "error": "mission_version_mismatch"})
            if version is not None and record.version != version:
                return Decision(DecisionType.BLOCK, DecisionReason.INVALID_REQUEST, severity=Severity.CRITICAL, metadata={"stage": "agent_admission", "error": "version_mismatch"})
            if provenance is not None and record.provenance != provenance:
                return Decision(DecisionType.BLOCK, DecisionReason.INVALID_REQUEST, severity=Severity.CRITICAL, metadata={"stage": "agent_admission", "error": "provenance_mismatch"})
            if configuration_digest is not None and record.configuration_digest != configuration_digest:
                return Decision(DecisionType.BLOCK, DecisionReason.INVALID_REQUEST, severity=Severity.CRITICAL, metadata={"stage": "agent_admission", "error": "integrity_mismatch"})
            if record.policy_version is not None and record.policy_version != policy_version:
                return Decision(DecisionType.BLOCK, DecisionReason.INVALID_REQUEST, severity=Severity.CRITICAL, metadata={"stage": "agent_admission", "error": "policy_mismatch"})
            if not set(capabilities).issubset(record.capability_envelope):
                return Decision(DecisionType.BLOCK, DecisionReason.PRIVILEGE_ESCALATION, severity=Severity.CRITICAL, metadata={"stage": "agent_admission", "error": "capability_escalation"})
            if not set(tools).issubset(record.tool_set):
                return Decision(DecisionType.BLOCK, DecisionReason.PRIVILEGE_ESCALATION, severity=Severity.CRITICAL, metadata={"stage": "agent_admission", "error": "tool_escalation"})
            if record.expires_at <= datetime.now(timezone.utc):
                expired = AgentAdmissionRecord(
                    agent_id=record.agent_id,
                    application_id=record.application_id,
                    state=AgentAdmissionState.EXPIRED,
                    admission_id=record.admission_id,
                    session_id=record.session_id,
                    parent_agent_id=record.parent_agent_id,
                    mission_id=record.mission_id,
                    mission_version=record.mission_version,
                    version=record.version,
                    provenance=record.provenance,
                    capability_envelope=record.capability_envelope,
                    tool_set=record.tool_set,
                    policy_version=record.policy_version,
                    configuration_digest=record.configuration_digest,
                    integrity_digest=record.integrity_digest,
                    attestation_digest=record.attestation_digest,
                    issued_at=record.issued_at,
                    expires_at=record.expires_at,
                    updated_at=datetime.now(timezone.utc),
                    metadata={**record.metadata, "expired": True},
                )
                self._records[agent_id] = expired
                try:
                    self._record(AuditEventType.AGENT_ADMISSION_EXPIRED, expired)
                except Exception:
                    self._records[agent_id] = record
                return Decision(DecisionType.BLOCK, DecisionReason.INVALID_REQUEST, severity=Severity.CRITICAL, metadata={"stage": "agent_admission", "error": "expired"})
            return Decision(DecisionType.ALLOW, DecisionReason.REQUEST_ALLOWED, metadata={"stage": "agent_admission", "agent_id": agent_id})
