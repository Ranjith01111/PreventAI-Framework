"""Signed, exact-request, in-memory human approval tokens."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import hmac
import json
import threading
from typing import Any
from uuid import uuid4

from ..core.decision import Decision, DecisionReason, DecisionType, Severity
from ..core.data_release import DataReleaseRequest
from ..protocol.security_request import SecurityRequest


def _aware(value: datetime, field: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise ValueError(f"{field} must be timezone-aware")
    return value.astimezone(timezone.utc)


@dataclass(frozen=True, slots=True)
class ApprovalRequest:
    approval_request_id: str
    request_id: str
    request_digest: str
    authenticated_identity: str
    session_id: str
    agent_id: str
    tool_id: str
    action: str
    resource: str
    created_at: datetime
    expires_at: datetime
    reason: str
    policy_rule_id: str | None = None

    def __post_init__(self) -> None:
        for field in (
            "approval_request_id", "request_id", "request_digest",
            "authenticated_identity", "session_id", "agent_id", "tool_id",
            "action", "resource", "reason",
        ):
            value = getattr(self, field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field} must be non-empty")
        created = _aware(self.created_at, "created_at")
        expires = _aware(self.expires_at, "expires_at")
        if expires <= created:
            raise ValueError("expires_at must be later than created_at")
        object.__setattr__(self, "created_at", created)
        object.__setattr__(self, "expires_at", expires)

    def to_dict(self) -> dict[str, Any]:
        return {
            "approval_request_id": self.approval_request_id,
            "request_id": self.request_id,
            "request_digest": self.request_digest,
            "authenticated_identity": self.authenticated_identity,
            "session_id": self.session_id,
            "agent_id": self.agent_id,
            "tool_id": self.tool_id,
            "action": self.action,
            "resource": self.resource,
            "created_at": self.created_at.isoformat().replace("+00:00", "Z"),
            "expires_at": self.expires_at.isoformat().replace("+00:00", "Z"),
            "reason": self.reason,
            "policy_rule_id": self.policy_rule_id,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=True)


@dataclass(frozen=True, slots=True)
class Approval:
    approval_id: str
    approval_request_id: str
    request_digest: str
    subject_identity: str
    session_id: str
    action: str
    tool_id: str
    resource: str
    approver_identity: str
    issued_at: datetime
    expires_at: datetime
    one_time: bool
    signature: str

    def __post_init__(self) -> None:
        for field in (
            "approval_id", "approval_request_id", "request_digest",
            "subject_identity", "session_id", "action", "tool_id",
            "resource", "approver_identity", "signature",
        ):
            if not isinstance(getattr(self, field), str) or not getattr(self, field).strip():
                raise ValueError(f"{field} must be non-empty")
        issued = _aware(self.issued_at, "issued_at")
        expires = _aware(self.expires_at, "expires_at")
        if expires <= issued:
            raise ValueError("expires_at must be later than issued_at")
        object.__setattr__(self, "issued_at", issued)
        object.__setattr__(self, "expires_at", expires)

    def _unsigned(self) -> dict[str, Any]:
        return {
            "approval_id": self.approval_id,
            "approval_request_id": self.approval_request_id,
            "request_digest": self.request_digest,
            "subject_identity": self.subject_identity,
            "session_id": self.session_id,
            "action": self.action,
            "tool_id": self.tool_id,
            "resource": self.resource,
            "approver_identity": self.approver_identity,
            "issued_at": self.issued_at.isoformat().replace("+00:00", "Z"),
            "expires_at": self.expires_at.isoformat().replace("+00:00", "Z"),
            "one_time": self.one_time,
        }

    def to_dict(self) -> dict[str, Any]:
        result = self._unsigned()
        result["signature"] = self.signature
        return result

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=True)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Approval":
        expected = {
            "approval_id", "approval_request_id", "request_digest",
            "subject_identity", "session_id", "action", "tool_id", "resource",
            "approver_identity", "issued_at", "expires_at", "one_time", "signature",
        }
        if not isinstance(value, dict) or set(value) != expected:
            raise ValueError("approval fields do not exactly match the contract")
        try:
            return cls(
                approval_id=value["approval_id"],
                approval_request_id=value["approval_request_id"],
                request_digest=value["request_digest"],
                subject_identity=value["subject_identity"],
                session_id=value["session_id"],
                action=value["action"],
                tool_id=value["tool_id"],
                resource=value["resource"],
                approver_identity=value["approver_identity"],
                issued_at=datetime.fromisoformat(value["issued_at"].replace("Z", "+00:00")),
                expires_at=datetime.fromisoformat(value["expires_at"].replace("Z", "+00:00")),
                one_time=value["one_time"],
                signature=value["signature"],
            )
        except (KeyError, TypeError, ValueError, AttributeError) as exc:
            raise ValueError("invalid approval") from exc


class ApprovalManager:
    def __init__(self, signing_secret: str | None = None) -> None:
        if signing_secret is not None and (not isinstance(signing_secret, str) or not signing_secret):
            raise ValueError("signing_secret must be non-empty")
        self._secret = (signing_secret or uuid4().hex).encode()
        self._requests: dict[str, ApprovalRequest] = {}
        self._revoked: set[str] = set()
        self._consumed: set[str] = set()
        self._lock = threading.RLock()

    @staticmethod
    def digest_request(request: SecurityRequest) -> str:
        if not isinstance(request, SecurityRequest):
            raise ValueError("request must be a SecurityRequest")
        payload = request.to_dict()
        payload.pop("timestamp", None)
        payload.pop("request_id", None)
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    @staticmethod
    def digest_release_request(request: DataReleaseRequest) -> str:
        if not isinstance(request, DataReleaseRequest):
            raise ValueError("request must be a DataReleaseRequest")
        payload = {
            "agent_id": request.agent_id,
            "session_id": request.session_id,
            "source": request.source,
            "source_classification": request.source_classification.value,
            "lineage": request.lineage,
            "destination": request.destination,
            "destination_trust": request.destination_trust.value,
            "operation": request.operation,
            "tool_id": request.tool_id,
            "tool_name": request.tool_name,
            "mission_id": request.mission_id,
            "mission_version": request.mission_version,
            "transformation": request.transformation.value,
            "target_classification": (
                request.target_classification.value
                if request.target_classification is not None
                else None
            ),
            "field_paths": request.field_paths,
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def request_release(
        self,
        request: DataReleaseRequest,
        *,
        identity: str,
        reason: str,
        expires_at: datetime,
        policy_rule_id: str | None = None,
    ) -> ApprovalRequest:
        if not isinstance(identity, str) or not identity.strip():
            raise ValueError("identity must be non-empty")
        now = datetime.now(timezone.utc)
        result = ApprovalRequest(
            approval_request_id=str(uuid4()),
            request_id=self.digest_release_request(request),
            request_digest=self.digest_release_request(request),
            authenticated_identity=identity,
            session_id=request.session_id,
            agent_id=request.agent_id,
            tool_id=request.tool_id,
            action=request.operation,
            resource=request.destination,
            created_at=now,
            expires_at=expires_at,
            reason=reason,
            policy_rule_id=policy_rule_id,
        )
        with self._lock:
            self._requests[result.approval_request_id] = result
        return result

    def verify_release(
        self,
        approval: Approval,
        request: DataReleaseRequest,
        *,
        authenticated_identity: str,
        consume: bool = True,
    ) -> Decision:
        try:
            digest = self.digest_release_request(request)
            with self._lock:
                if approval.approval_id in self._revoked:
                    return self._blocked(DecisionReason.APPROVAL_REVOKED)
                if approval.one_time and approval.approval_id in self._consumed:
                    return self._blocked(DecisionReason.APPROVAL_ALREADY_CONSUMED)
                if not hmac.compare_digest(approval.signature, self._sign(approval._unsigned())):
                    return self._blocked(DecisionReason.APPROVAL_INVALID)
                if (
                    approval.subject_identity != authenticated_identity
                    or approval.request_digest != digest
                    or approval.action != request.operation
                    or approval.tool_id != request.tool_id
                    or approval.resource != request.destination
                    or approval.session_id != request.session_id
                ):
                    return self._blocked(DecisionReason.APPROVAL_REQUEST_MISMATCH)
                if datetime.now(timezone.utc) >= approval.expires_at:
                    return self._blocked(DecisionReason.APPROVAL_EXPIRED)
                if consume and approval.one_time:
                    self._consumed.add(approval.approval_id)
            return Decision(DecisionType.ALLOW, DecisionReason.REQUEST_ALLOWED)
        except Exception:
            return self._blocked(DecisionReason.APPROVAL_SECURITY_ERROR)

    def request(
        self,
        request: SecurityRequest,
        *,
        identity: str,
        reason: str,
        expires_at: datetime,
        policy_rule_id: str | None = None,
    ) -> ApprovalRequest:
        now = datetime.now(timezone.utc)
        result = ApprovalRequest(
            approval_request_id=str(uuid4()),
            request_id=request.request_id,
            request_digest=self.digest_request(request),
            authenticated_identity=identity,
            session_id=request.session.session_id,
            agent_id=request.agent.agent_id,
            tool_id=request.resource.resource_id,
            action=request.action.name,
            resource=request.resource.resource_id,
            created_at=now,
            expires_at=expires_at,
            reason=reason,
            policy_rule_id=policy_rule_id,
        )
        with self._lock:
            self._requests[result.approval_request_id] = result
        return result

    def approve(
        self,
        pending: ApprovalRequest,
        *,
        approved_by: str,
        one_time: bool = True,
        expires_at: datetime | None = None,
    ) -> Approval:
        with self._lock:
            if self._requests.get(pending.approval_request_id) != pending:
                raise ValueError("unknown approval request")
        issued = datetime.now(timezone.utc)
        expiry = expires_at or pending.expires_at
        unsigned = {
            "approval_id": str(uuid4()),
            "approval_request_id": pending.approval_request_id,
            "request_digest": pending.request_digest,
            "subject_identity": pending.authenticated_identity,
            "session_id": pending.session_id,
            "action": pending.action,
            "tool_id": pending.tool_id,
            "resource": pending.resource,
            "approver_identity": approved_by,
            "issued_at": issued.isoformat().replace("+00:00", "Z"),
            "expires_at": _aware(expiry, "expires_at").isoformat().replace("+00:00", "Z"),
            "one_time": one_time,
        }
        signature = self._sign(unsigned)
        approval = Approval(
            approval_id=unsigned["approval_id"],
            approval_request_id=pending.approval_request_id,
            request_digest=pending.request_digest,
            subject_identity=pending.authenticated_identity,
            session_id=pending.session_id,
            action=pending.action,
            tool_id=pending.tool_id,
            resource=pending.resource,
            approver_identity=approved_by,
            issued_at=issued,
            expires_at=expiry,
            one_time=one_time,
            signature=signature,
        )
        return approval

    def revoke(self, approval_id: str) -> None:
        with self._lock:
            self._revoked.add(approval_id)

    def verify(
        self,
        approval: Approval,
        request: SecurityRequest,
        *,
        authenticated_identity: str,
        consume: bool = True,
    ) -> Decision:
        try:
            digest = self.digest_request(request)
            with self._lock:
                if approval.approval_id in self._revoked:
                    return self._blocked(DecisionReason.APPROVAL_REVOKED)
                if approval.one_time and approval.approval_id in self._consumed:
                    return self._blocked(DecisionReason.APPROVAL_ALREADY_CONSUMED)
                if not hmac.compare_digest(approval.signature, self._sign(approval._unsigned())):
                    return self._blocked(DecisionReason.APPROVAL_INVALID)
                if approval.subject_identity != authenticated_identity:
                    return self._blocked(DecisionReason.APPROVAL_SUBJECT_MISMATCH)
                if (
                    approval.request_digest != digest
                    or approval.action != request.action.name
                    or approval.tool_id != request.resource.resource_id
                    or approval.resource != request.resource.resource_id
                    or approval.session_id != request.session.session_id
                ):
                    return self._blocked(DecisionReason.APPROVAL_REQUEST_MISMATCH)
                if datetime.now(timezone.utc) >= approval.expires_at:
                    return self._blocked(DecisionReason.APPROVAL_EXPIRED)
                if consume and approval.one_time:
                    self._consumed.add(approval.approval_id)
            return Decision(DecisionType.ALLOW, DecisionReason.REQUEST_ALLOWED)
        except Exception:
            return self._blocked(DecisionReason.APPROVAL_SECURITY_ERROR)

    def _sign(self, value: dict[str, Any]) -> str:
        canonical = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
        return hmac.new(self._secret, canonical, hashlib.sha256).hexdigest()

    @staticmethod
    def _blocked(reason: DecisionReason) -> Decision:
        return Decision(
            DecisionType.BLOCK,
            reason,
            severity=Severity.HIGH,
            metadata={"stage": "approval"},
        )
