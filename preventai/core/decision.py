"""Immutable, serializable primitives for deterministic security decisions."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
import math
from types import MappingProxyType
from typing import Any, Mapping

from ..protocol.security_request import ContractValidationError


class DecisionType(str, Enum):
    """The possible outcomes of a future deterministic decision engine."""

    ALLOW = "allow"
    BLOCK = "block"
    WARN = "warn"
    APPROVAL_REQUIRED = "approval_required"


class DecisionReason(str, Enum):
    """Machine-readable reasons that explain a decision outcome."""

    REQUEST_ALLOWED = "request_allowed"
    REQUEST_BLOCKED = "request_blocked"
    SECURITY_WARNING = "security_warning"
    APPROVAL_REQUIRED = "approval_required"
    INVALID_REQUEST = "invalid_request"
    RULE_MATCH = "rule_match"
    POLICY_MATCH = "policy_match"
    NO_POLICY = "no_policy"
    CONFLICTING_POLICIES = "conflicting_policies"
    PERMISSION_DENIED = "permission_denied"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    MISSING_PERMISSION = "missing_permission"
    SECURITY_EVALUATION_ERROR = "security_evaluation_error"
    SCHEMA_VALIDATION_FAILED = "schema_validation_failed"
    SECRETS_DETECTED = "secrets_detected"
    PII_DETECTED = "pii_detected"
    INJECTION_DETECTED = "injection_detected"
    CONTENT_POLICY_VIOLATION = "content_policy_violation"
    OUTPUT_SECURITY_ERROR = "output_security_error"
    OUTPUT_SCHEMA_VALIDATION_FAILED = "output_schema_validation_failed"
    OUTPUT_POLICY_VIOLATION = "output_policy_violation"
    DATA_FLOW_VIOLATION = "data_flow_violation"
    DATA_FLOW_NO_POLICY = "data_flow_no_policy"
    DATA_FLOW_EVALUATION_ERROR = "data_flow_evaluation_error"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    TIMEOUT = "timeout"
    BUDGET_EXCEEDED = "budget_exceeded"
    TOOL_CALL_LIMIT_EXCEEDED = "tool_call_limit_exceeded"
    CONCURRENCY_LIMIT_EXCEEDED = "concurrency_limit_exceeded"
    RUNTIME_SECURITY_ERROR = "runtime_security_error"
    LOOP_DETECTED = "loop_detected"
    CIRCUIT_OPEN = "circuit_open"
    AUTHENTICATION_FAILED = "authentication_failed"
    AUTHENTICATION_ERROR = "authentication_error"
    LIFECYCLE_INVALID = "lifecycle_invalid"
    MISSION_INVALID = "mission_invalid"
    MISSION_EXPIRED = "mission_expired"
    MISSION_REVOKED = "mission_revoked"
    MISSION_SCOPE_DENIED = "mission_scope_denied"
    MISSION_VERSION_INVALID = "mission_version_invalid"
    AGENT_ADMISSION_REQUIRED = "agent_admission_required"
    AGENT_UNKNOWN = "agent_unknown"
    AGENT_IDENTITY_MISMATCH = "agent_identity_mismatch"
    AGENT_APPLICATION_MISMATCH = "agent_application_mismatch"
    AGENT_SESSION_MISMATCH = "agent_session_mismatch"
    AGENT_PARENT_MISMATCH = "agent_parent_mismatch"
    AGENT_MISSION_MISMATCH = "agent_mission_mismatch"
    AGENT_POLICY_MISMATCH = "agent_policy_mismatch"
    AGENT_VERSION_MISMATCH = "agent_version_mismatch"
    AGENT_PROVENANCE_MISMATCH = "agent_provenance_mismatch"
    AGENT_INTEGRITY_MISMATCH = "agent_integrity_mismatch"
    AGENT_CAPABILITY_ESCALATION = "agent_capability_escalation"
    AGENT_REVOKED = "agent_revoked"
    AGENT_SUSPENDED = "agent_suspended"
    AGENT_EXPIRED = "agent_expired"
    AGENT_STALE_ADMISSION = "agent_stale_admission"
    AGENT_TOCTOU_REJECTION = "agent_toctou_rejection"
    DELEGATION_INVALID = "delegation_invalid"
    DELEGATION_EXPIRED = "delegation_expired"
    DELEGATION_REVOKED = "delegation_revoked"
    DELEGATION_SUBJECT_MISMATCH = "delegation_subject_mismatch"
    DELEGATION_DEPTH_EXCEEDED = "delegation_depth_exceeded"
    DELEGATION_SCOPE_EXCEEDED = "delegation_scope_exceeded"
    DELEGATION_AUTHORITY_EXCEEDED = "delegation_authority_exceeded"
    DELEGATION_SIGNATURE_INVALID = "delegation_signature_invalid"
    DELEGATION_SECURITY_ERROR = "delegation_security_error"
    APPROVAL_INVALID = "approval_invalid"
    APPROVAL_EXPIRED = "approval_expired"
    APPROVAL_REVOKED = "approval_revoked"
    APPROVAL_SUBJECT_MISMATCH = "approval_subject_mismatch"
    APPROVAL_REQUEST_MISMATCH = "approval_request_mismatch"
    APPROVAL_ALREADY_CONSUMED = "approval_already_consumed"
    APPROVAL_SECURITY_ERROR = "approval_security_error"
    SANDBOX_POLICY_VIOLATION = "sandbox_policy_violation"
    SANDBOX_SECURITY_ERROR = "sandbox_security_error"
    SANDBOX_TIMEOUT = "sandbox_timeout"
    NETWORK_POLICY_VIOLATION = "network_policy_violation"
    NETWORK_AUTHORIZATION_INVALID = "network_authorization_invalid"
    NETWORK_SSRF_BLOCKED = "network_ssrf_blocked"
    NETWORK_DNS_BLOCKED = "network_dns_blocked"
    NETWORK_REDIRECT_BLOCKED = "network_redirect_blocked"
    NETWORK_SCHEME_BLOCKED = "network_scheme_blocked"
    NETWORK_PORT_BLOCKED = "network_port_blocked"
    NETWORK_RESPONSE_BLOCKED = "network_response_blocked"
    NETWORK_EXECUTION_ERROR = "network_execution_error"
    FILESYSTEM_POLICY_VIOLATION = "filesystem_policy_violation"
    FILESYSTEM_AUTHORIZATION_INVALID = "filesystem_authorization_invalid"
    FILESYSTEM_EXECUTION_ERROR = "filesystem_execution_error"


class Severity(str, Enum):
    """Severity associated with a decision when one is applicable."""

    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


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


@dataclass(frozen=True, slots=True)
class Decision:
    """A decision value with no policy evaluation or side effects."""

    decision_type: DecisionType
    reason: DecisionReason
    rule_id: str | None = None
    severity: Severity | None = None
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.decision_type, DecisionType):
            raise ContractValidationError("decision_type must be a DecisionType")
        if not isinstance(self.reason, DecisionReason):
            raise ContractValidationError("reason must be a DecisionReason")
        if self.rule_id is not None and (
            not isinstance(self.rule_id, str)
            or not self.rule_id.strip()
            or len(self.rule_id) > 256
        ):
            raise ContractValidationError("rule_id must be a non-empty string when provided")
        if self.severity is not None and not isinstance(self.severity, Severity):
            raise ContractValidationError("severity must be a Severity when provided")
        metadata = {} if self.metadata is None else self.metadata
        if not isinstance(metadata, Mapping):
            raise ContractValidationError("metadata must be an object")
        metadata = dict(metadata)
        _json_value(metadata, "metadata")
        object.__setattr__(self, "metadata", MappingProxyType(metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision_type": self.decision_type.value,
            "reason": self.reason.value,
            "rule_id": self.rule_id,
            "severity": self.severity.value if self.severity is not None else None,
            "metadata": dict(self.metadata),
        }

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(), ensure_ascii=True, sort_keys=True, separators=(",", ":")
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Decision:
        if not isinstance(value, Mapping):
            raise ContractValidationError("decision must be an object")
        expected = {"decision_type", "reason", "rule_id", "severity", "metadata"}
        if set(value) != expected:
            raise ContractValidationError("decision fields do not exactly match the contract")
        try:
            severity = value["severity"]
            return cls(
                decision_type=DecisionType(value["decision_type"]),
                reason=DecisionReason(value["reason"]),
                rule_id=value["rule_id"],
                severity=Severity(severity) if severity is not None else None,
                metadata=value["metadata"],
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ContractValidationError("invalid decision") from exc

    @classmethod
    def from_json(cls, value: str) -> Decision:
        if not isinstance(value, str):
            raise ContractValidationError("JSON input must be a string")
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ContractValidationError("invalid JSON") from exc
        return cls.from_dict(decoded)
