"""Immutable, safe audit event structures."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
import json
from types import MappingProxyType
from typing import Any, Mapping


class AuditEventType(str, Enum):
    AUTHENTICATION = "authentication"
    DELEGATION = "delegation"
    POLICY_EVALUATION = "policy_evaluation"
    PERMISSION_EVALUATION = "permission_evaluation"
    SCHEMA_VALIDATION = "schema_validation"
    SECRET_DETECTION = "secret_detection"
    PII_DETECTION = "pii_detection"
    INJECTION_DETECTION = "injection_detection"
    CONTENT_POLICY = "content_policy"
    DATA_FLOW = "data_flow"
    RATE_LIMIT = "rate_limit"
    BUDGET = "budget"
    TIMEOUT = "timeout"
    TOOL_CALL_LIMIT = "tool_call_limit"
    CONCURRENCY = "concurrency"
    LOOP_DETECTION = "loop_detection"
    CIRCUIT_BREAKER = "circuit_breaker"
    APPROVAL_REQUEST = "approval_request"
    APPROVAL_DECISION = "approval_decision"
    TOOL_EXECUTION_STARTED = "tool_execution_started"
    TOOL_EXECUTION_SUCCEEDED = "tool_execution_succeeded"
    TOOL_EXECUTION_FAILED = "tool_execution_failed"
    OUTPUT_VALIDATION = "output_validation"
    MEMORY_READ = "memory_read"
    MEMORY_WRITE = "memory_write"
    MEMORY_UPDATE = "memory_update"
    MEMORY_DELETE = "memory_delete"
    MEMORY_BLOCKED = "memory_blocked"
    SANDBOX_VALIDATION = "sandbox_validation"
    SANDBOX_BLOCKED = "sandbox_blocked"
    SANDBOX_EXECUTION_STARTED = "sandbox_execution_started"
    SANDBOX_EXECUTION_SUCCEEDED = "sandbox_execution_succeeded"
    SANDBOX_EXECUTION_FAILED = "sandbox_execution_failed"
    NETWORK_BLOCKED = "network_blocked"
    NETWORK_REQUEST_STARTED = "network_request_started"
    NETWORK_REQUEST_COMPLETED = "network_request_completed"
    FILESYSTEM_BLOCKED = "filesystem_blocked"
    FILESYSTEM_OPERATION_STARTED = "filesystem_operation_started"
    FILESYSTEM_OPERATION_COMPLETED = "filesystem_operation_completed"
    LIFECYCLE_TRANSITION = "lifecycle_transition"
    LIFECYCLE_BLOCKED = "lifecycle_blocked"
    MISSION_VALIDATION = "mission_validation"
    MISSION_BLOCKED = "mission_blocked"
    MISSION_ALLOWED = "mission_allowed"
    MISSION_SCOPE_DENIED = "mission_scope_denied"
    MISSION_REVOKED = "mission_revoked"
    MISSION_EXPIRED = "mission_expired"
    MISSION_INVALID = "mission_invalid"
    MISSION_VERSION_INVALID = "mission_version_invalid"
    MISSION_SCOPE_ESCALATION = "mission_scope_escalation"
    TOOL_REGISTERED = "tool_registered"
    TOOL_VERIFIED = "tool_verified"
    TOOL_TRUSTED = "tool_trusted"
    TOOL_UNTRUSTED = "tool_untrusted"
    TOOL_REVOKED = "tool_revoked"
    TOOL_EXPIRED = "tool_expired"
    TOOL_INTEGRITY_MISMATCH = "tool_integrity_mismatch"
    TOOL_PROVENANCE_MISMATCH = "tool_provenance_mismatch"
    TOOL_VERSION_MISMATCH = "tool_version_mismatch"
    TOOL_SCOPE_DENIED = "tool_scope_denied"
    AGENT_ADMISSION_REQUESTED = "agent_admission_requested"
    AGENT_ADMISSION_ALLOWED = "agent_admission_allowed"
    AGENT_ADMISSION_BLOCKED = "agent_admission_blocked"
    AGENT_ADMISSION_APPROVAL_REQUIRED = "agent_admission_approval_required"
    AGENT_ADMISSION_ACTIVATED = "agent_admission_activated"
    AGENT_ADMISSION_SUSPENDED = "agent_admission_suspended"
    AGENT_ADMISSION_REVOKED = "agent_admission_revoked"
    AGENT_ADMISSION_EXPIRED = "agent_admission_expired"
    AGENT_IDENTITY_MISMATCH = "agent_identity_mismatch"
    AGENT_APPLICATION_MISMATCH = "agent_application_mismatch"
    AGENT_PROVENANCE_MISMATCH = "agent_provenance_mismatch"
    AGENT_VERSION_MISMATCH = "agent_version_mismatch"
    AGENT_INTEGRITY_MISMATCH = "agent_integrity_mismatch"
    AGENT_PARENT_DELEGATOR_MISMATCH = "agent_parent_delegator_mismatch"
    AGENT_CAPABILITY_ESCALATION_DENIED = "agent_capability_escalation_denied"
    AGENT_CROSS_APP_SUBSTITUTION = "agent_cross_app_substitution"
    AGENT_CROSS_AGENT_SUBSTITUTION = "agent_cross_agent_substitution"
    AGENT_STALE_ADMISSION = "agent_stale_admission"
    AGENT_TOCTOU_REJECTION = "agent_toctou_rejection"
    POLICY_REGISTERED = "policy_registered"
    POLICY_VERIFIED = "policy_verified"
    POLICY_ACTIVATED = "policy_activated"
    POLICY_REVOKED = "policy_revoked"
    POLICY_INTEGRITY_MISMATCH = "policy_integrity_mismatch"
    POLICY_VERSION_MISMATCH = "policy_version_mismatch"
    POLICY_UPDATE_DENIED = "policy_update_denied"
    DATA_RELEASE_EVALUATED = "data_release_evaluated"
    DATA_RELEASE_ALLOWED = "data_release_allowed"
    DATA_RELEASE_BLOCKED = "data_release_blocked"
    DATA_RELEASE_APPROVAL_REQUESTED = "data_release_approval_requested"
    DATA_DECLASSIFICATION_REQUESTED = "data_declassification_requested"
    DATA_DECLASSIFICATION_APPROVED = "data_declassification_approved"
    DATA_DECLASSIFICATION_DENIED = "data_declassification_denied"
    DATA_DESTINATION_DENIED = "data_destination_denied"
    DATA_CLASSIFICATION_MISMATCH = "data_classification_mismatch"
    DATA_LINEAGE_FAILURE = "data_lineage_failure"
    DATA_FORGED_CONTEXT = "data_forged_context"
    DATA_RELEASE_TOCTOU_FAILURE = "data_release_toctou_failure"
    DATA_RELEASE_DISPATCH_PREVENTED = "data_release_dispatch_prevented"


@dataclass(frozen=True, slots=True)
class AuditVerification:
    valid: bool
    failure_type: str | None = None
    sequence: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "failure_type": self.failure_type,
            "sequence": self.sequence,
        }


def _text(value: str | None, field: str) -> None:
    if value is not None and (
        not isinstance(value, str) or not value.strip() or len(value) > 256
    ):
        raise ValueError(f"{field} must be non-empty when provided")


def _safe_value(value: Any, field: str = "metadata") -> None:
    forbidden = (
        "password", "secret", "token", "credential", "private_key",
        "api_key", "payload", "prompt", "argument", "result", "email",
        "phone", "address", "credit_card",
    )
    if value is None or isinstance(value, (str, bool, int)):
        if isinstance(value, str) and any(word in field.lower() for word in forbidden):
            raise ValueError(f"{field} is not safe for audit")
        return
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            raise ValueError(f"{field} contains an invalid number")
        return
    if isinstance(value, list):
        for item in value:
            _safe_value(item, field)
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str) or not key.strip():
                raise ValueError(f"{field} keys must be non-empty strings")
            _safe_value(item, key)
        return
    raise ValueError(f"{field} contains an unsupported value")


@dataclass(frozen=True, slots=True)
class AuditEvent:
    event_id: str
    sequence: int
    timestamp: datetime
    event_type: AuditEventType
    agent_id: str | None = None
    session_id: str | None = None
    tool_id: str | None = None
    action: str | None = None
    resource: str | None = None
    identity: str | None = None
    decision: str | None = None
    reason: str | None = None
    severity: str | None = None
    rule_id: str | None = None
    correlation_id: str | None = None
    previous_event_hash: str | None = None
    event_hash: str | None = None
    integrity_tag: str | None = None
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.event_id, str) or not self.event_id.strip():
            raise ValueError("event_id must be non-empty")
        if not isinstance(self.sequence, int) or isinstance(self.sequence, bool) or self.sequence <= 0:
            raise ValueError("sequence must be positive")
        if not isinstance(self.timestamp, datetime) or self.timestamp.tzinfo is None:
            raise ValueError("timestamp must be timezone-aware")
        if not isinstance(self.event_type, AuditEventType):
            raise ValueError("event_type must be an AuditEventType")
        for field in (
            "agent_id", "session_id", "tool_id", "action", "resource", "identity",
            "decision", "reason", "severity", "rule_id", "correlation_id",
            "previous_event_hash", "event_hash", "integrity_tag",
        ):
            _text(getattr(self, field), field)
        metadata = {} if self.metadata is None else dict(self.metadata)
        if not isinstance(self.metadata, (Mapping, type(None))):
            raise ValueError("metadata must be an object")
        _safe_value(metadata)
        object.__setattr__(self, "timestamp", self.timestamp.astimezone(timezone.utc))
        object.__setattr__(self, "metadata", MappingProxyType(metadata))

    def unsigned_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "sequence": self.sequence,
            "timestamp": self.timestamp.isoformat().replace("+00:00", "Z"),
            "event_type": self.event_type.value,
            "agent_id": self.agent_id,
            "session_id": self.session_id,
            "tool_id": self.tool_id,
            "action": self.action,
            "resource": self.resource,
            "identity": self.identity,
            "decision": self.decision,
            "reason": self.reason,
            "severity": self.severity,
            "rule_id": self.rule_id,
            "correlation_id": self.correlation_id,
            "previous_event_hash": self.previous_event_hash,
            "metadata": dict(self.metadata),
        }

    def to_dict(self) -> dict[str, Any]:
        value = self.unsigned_dict()
        value["event_hash"] = self.event_hash
        value["integrity_tag"] = self.integrity_tag
        return value

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=True, sort_keys=True, separators=(",", ":"))
