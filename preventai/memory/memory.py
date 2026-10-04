"""Deterministic, bounded, in-process protected memory."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
from enum import Enum
from copy import deepcopy
from types import MappingProxyType
import threading
from typing import Any, Callable, Mapping
from uuid import uuid4

from ..audit.event import AuditEventType
from ..audit.logger import AuditLogger
from ..core.dataflow import (
    DataFlow, DataFlowDestination, DataFlowEngine, DataSource, FlowClassification, FlowTrust
)
from ..core.decision import Decision, DecisionReason, DecisionType, Severity
from ..core.data_release import (
    DataReleaseAuthorization, DataReleaseRequest, DestinationTrust,
    verify_release_authorization,
)
from .memory_isolation import MemoryIsolation
from .memory_policy import MemoryAccessContext, MemoryOperation, MemoryPolicy
from .memory_validation import encoded_size, json_value, text


class MemoryStatus(str, Enum):
    ACTIVE = "active"
    DELETED = "deleted"


@dataclass(frozen=True, slots=True)
class MemoryEntry:
    memory_id: str
    owner_agent_id: str
    session_id: str
    namespace: str
    classification: FlowClassification
    trust: FlowTrust
    created_at: datetime
    updated_at: datetime
    expires_at: datetime | None
    version: int
    status: MemoryStatus
    metadata: Mapping[str, Any]
    payload: Any

    def __post_init__(self) -> None:
        for field in ("memory_id", "owner_agent_id", "session_id", "namespace"):
            text(getattr(self, field), field)
        if not isinstance(self.classification, FlowClassification):
            raise ValueError("classification must be a FlowClassification")
        if not isinstance(self.trust, FlowTrust):
            raise ValueError("trust must be a FlowTrust")
        for field in ("created_at", "updated_at"):
            value = getattr(self, field)
            if not isinstance(value, datetime) or value.tzinfo is None:
                raise ValueError(f"{field} must be timezone-aware")
        if self.expires_at is not None and self.expires_at.tzinfo is None:
            raise ValueError("expires_at must be timezone-aware")
        if not isinstance(self.version, int) or self.version < 1:
            raise ValueError("version must be positive")
        if not isinstance(self.status, MemoryStatus):
            raise ValueError("status must be a MemoryStatus")
        json_value(self.payload, "payload")
        if not isinstance(self.metadata, Mapping):
            raise ValueError("metadata must be an object")
        metadata = dict(self.metadata)
        forbidden = ("secret", "token", "password", "credential", "api_key", "email", "phone", "address")
        if any(any(word in key.lower() for word in forbidden) for key in metadata):
            raise ValueError("sensitive metadata is not permitted")
        json_value(metadata, "metadata")
        object.__setattr__(self, "created_at", self.created_at.astimezone(timezone.utc))
        object.__setattr__(self, "updated_at", self.updated_at.astimezone(timezone.utc))
        object.__setattr__(self, "metadata", MappingProxyType(metadata))

    def expired(self, now: datetime | None = None) -> bool:
        return self.expires_at is not None and self.expires_at <= (now or datetime.now(timezone.utc))


class MemoryAccessDenied(PermissionError):
    def __init__(self, decision: Decision) -> None:
        self.decision = decision
        super().__init__(decision.reason.value)


class MemoryStore:
    def __init__(
        self,
        *,
        policy: MemoryPolicy,
        isolation: MemoryIsolation | None = None,
        max_entry_bytes: int = 1_000_000,
        max_entries_per_namespace: int = 1_000,
        max_total_bytes: int = 10_000_000,
        audit_logger: AuditLogger | None = None,
        data_flow_engine: DataFlowEngine | None = None,
        permission_checker: Callable[[MemoryOperation, MemoryAccessContext, MemoryEntry | None], Decision] | None = None,
        authentication_checker: Callable[[MemoryAccessContext], bool] | None = None,
        application_id: str | None = None,
        release_issuer: Callable[[DataReleaseRequest], DataReleaseAuthorization] | None = None,
    ) -> None:
        if not isinstance(policy, MemoryPolicy):
            raise ValueError("policy must be a MemoryPolicy")
        for name, value in (("max_entry_bytes", max_entry_bytes), ("max_entries_per_namespace", max_entries_per_namespace), ("max_total_bytes", max_total_bytes)):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        self.policy, self.isolation = policy, isolation or MemoryIsolation()
        self.max_entry_bytes, self.max_entries_per_namespace, self.max_total_bytes = max_entry_bytes, max_entries_per_namespace, max_total_bytes
        self.audit_logger, self.data_flow_engine = audit_logger, data_flow_engine
        self.permission_checker, self.authentication_checker = permission_checker, authentication_checker
        self.application_id = application_id
        self.release_issuer = release_issuer
        self._entries: dict[str, MemoryEntry] = {}
        self._sizes: dict[str, int] = {}
        self._lock = threading.RLock()

    def write(self, context: MemoryAccessContext, payload: Any, *, namespace: str | None = None,
              classification: FlowClassification = FlowClassification.PRIVATE, trust: FlowTrust = FlowTrust.TRUSTED,
              memory_id: str | None = None, expires_at: datetime | None = None,
              metadata: Mapping[str, Any] | None = None,
              release_authorization: DataReleaseAuthorization | None = None,
              release_request: DataReleaseRequest | None = None) -> MemoryEntry:
        now = datetime.now(timezone.utc)
        namespace = namespace or context.namespace
        memory_id = memory_id or str(uuid4())
        entry = MemoryEntry(memory_id, context.agent_id, context.session_id, namespace, classification, trust,
                            now, now, expires_at, 1, MemoryStatus.ACTIVE, metadata or {}, deepcopy(payload))
        with self._lock:
            if namespace != context.namespace:
                if release_authorization is None and release_request is not None and self.release_issuer is not None:
                    try:
                        release_authorization = self.release_issuer(release_request)
                    except Exception:
                        raise self._deny(
                            context,
                            entry,
                            self._block("memory_release_authorization_issue_failed"),
                        )
                if self.application_id is None or not verify_release_authorization(
                    release_authorization,
                    release_request,
                    application_id=self.application_id,
                ):
                    raise self._deny(
                        context,
                        entry,
                        self._block("memory_release_authorization_required"),
                    )
                assert release_request is not None
                if (
                    release_request.agent_id != context.agent_id
                    or release_request.session_id != context.session_id
                    or release_request.destination != namespace
                    or release_request.source_classification is not classification
                    or release_request.destination_trust is DestinationTrust.UNKNOWN
                ):
                    raise self._deny(
                        context,
                        entry,
                        self._block("memory_release_scope_mismatch"),
                    )
            authorization_context = (
                replace(context, namespace=namespace)
                if namespace != context.namespace and release_authorization is not None
                else context
            )
            self._authorize(MemoryOperation.WRITE, authorization_context, entry)
            if memory_id in self._entries and self._entries[memory_id].status is MemoryStatus.ACTIVE:
                raise self._deny(context, entry, self._block("duplicate_memory_id"))
            self._flow(context, entry, "memory.write")
            size = encoded_size(entry.payload)
            if size > self.max_entry_bytes or sum(self._sizes.values()) + size > self.max_total_bytes:
                raise self._deny(context, entry, self._block("memory_limit"))
            if sum(1 for item in self._entries.values() if item.namespace == namespace and item.status is MemoryStatus.ACTIVE) >= self.max_entries_per_namespace:
                raise self._deny(context, entry, self._block("namespace_limit"))
            self._audit(AuditEventType.MEMORY_WRITE, context, entry, Decision(DecisionType.ALLOW, DecisionReason.REQUEST_ALLOWED))
            self._entries[memory_id], self._sizes[memory_id] = entry, size
            return self._copy_entry(entry)

    def read(self, context: MemoryAccessContext, memory_id: str) -> MemoryEntry:
        with self._lock:
            entry = self._get(memory_id)
            self._authorize(MemoryOperation.READ, context, entry)
            self._flow(context, entry, "memory.read")
            self._audit(AuditEventType.MEMORY_READ, context, entry, Decision(DecisionType.ALLOW, DecisionReason.REQUEST_ALLOWED))
            return self._copy_entry(entry)

    def update(self, context: MemoryAccessContext, memory_id: str, payload: Any, *, expected_version: int,
               classification: FlowClassification | None = None, trust: FlowTrust | None = None,
               expires_at: datetime | None = None, metadata: Mapping[str, Any] | None = None) -> MemoryEntry:
        with self._lock:
            old = self._get(memory_id)
            self._authorize(MemoryOperation.UPDATE, context, old)
            if expected_version != old.version:
                raise self._deny(context, old, self._block("version_conflict"))
            new_class = classification or old.classification
            new_trust = FlowTrust.UNTRUSTED if old.trust is FlowTrust.UNTRUSTED or trust is FlowTrust.UNTRUSTED else old.trust
            if new_class is not old.classification and _rank(new_class) < _rank(old.classification):
                raise self._deny(context, old, self._block("classification_downgrade"))
            now = datetime.now(timezone.utc)
            entry = MemoryEntry(old.memory_id, old.owner_agent_id, old.session_id, old.namespace, new_class, new_trust,
                                old.created_at, now, expires_at if expires_at is not None else old.expires_at,
                                old.version + 1, old.status, metadata or old.metadata, deepcopy(payload))
            size = encoded_size(entry.payload)
            if size > self.max_entry_bytes or sum(self._sizes.values()) - self._sizes[memory_id] + size > self.max_total_bytes:
                raise self._deny(context, old, self._block("memory_limit"))
            self._audit(AuditEventType.MEMORY_UPDATE, context, entry, Decision(DecisionType.ALLOW, DecisionReason.REQUEST_ALLOWED))
            self._entries[memory_id], self._sizes[memory_id] = entry, size
            return self._copy_entry(entry)

    def delete(self, context: MemoryAccessContext, memory_id: str) -> None:
        with self._lock:
            entry = self._get(memory_id)
            self._authorize(MemoryOperation.DELETE, context, entry)
            deleted = MemoryEntry(entry.memory_id, entry.owner_agent_id, entry.session_id, entry.namespace,
                entry.classification, entry.trust, entry.created_at, datetime.now(timezone.utc), entry.expires_at,
                entry.version + 1, MemoryStatus.DELETED, entry.metadata, entry.payload)
            self._audit(AuditEventType.MEMORY_DELETE, context, entry, Decision(DecisionType.ALLOW, DecisionReason.REQUEST_ALLOWED))
            self._entries[memory_id] = deleted
            self._sizes.pop(memory_id, None)

    def share(self, context: MemoryAccessContext, memory_id: str, grantee_agent_id: str, *,
              grantee_session_id: str | None = None, expires_at: datetime | None = None):
        with self._lock:
            entry = self._get(memory_id)
            self._authorize(MemoryOperation.SHARE, context, entry)
            return self.isolation.share(context, grantee_agent_id, entry.namespace,
                                        grantee_session_id=grantee_session_id, expires_at=expires_at)

    def revoke_share(self, share_id: str) -> None:
        self.isolation.revoke(share_id)

    def _get(self, memory_id: str) -> MemoryEntry:
        text(memory_id, "memory_id")
        entry = self._entries.get(memory_id)
        if entry is None or entry.status is not MemoryStatus.ACTIVE or entry.expired():
            raise MemoryAccessDenied(self._block("memory_unavailable"))
        return entry

    @staticmethod
    def _copy_entry(entry: MemoryEntry) -> MemoryEntry:
        return MemoryEntry(
            entry.memory_id, entry.owner_agent_id, entry.session_id, entry.namespace,
            entry.classification, entry.trust, entry.created_at, entry.updated_at,
            entry.expires_at, entry.version, entry.status, dict(entry.metadata),
            deepcopy(entry.payload),
        )

    def _authorize(self, operation, context, entry):
        try:
            if not isinstance(context, MemoryAccessContext) or (self.authentication_checker and not self.authentication_checker(context)):
                raise MemoryAccessDenied(self._block("authentication"))
            isolation = self.isolation.check(context, entry.owner_agent_id, entry.session_id, entry.namespace)
            if isolation.decision_type is not DecisionType.ALLOW:
                raise self._deny(context, entry, isolation)
            if context.delegation is not None:
                if context.delegation_registry is None:
                    raise self._deny(context, entry, self._block("delegation_context"))
                delegation = context.delegation_registry.evaluate(
                    context.delegation,
                    authenticated_identity=context.authenticated_identity,
                    action=f"memory.{operation.value}",
                    resource=entry.namespace,
                    tool="memory",
                    session_id=context.session_id,
                )
                if delegation.decision_type is not DecisionType.ALLOW:
                    raise self._deny(context, entry, delegation)
            decision = self.policy.evaluate(operation, context, entry.classification, entry.trust)
            if decision.decision_type is not DecisionType.ALLOW:
                raise self._deny(context, entry, decision)
            if self.permission_checker is not None:
                permission = self.permission_checker(operation, context, entry)
                if not isinstance(permission, Decision) or permission.decision_type is not DecisionType.ALLOW:
                    decision = permission if isinstance(permission, Decision) else self._block("permission")
                    raise self._deny(context, entry, decision)
        except MemoryAccessDenied:
            raise
        except Exception:
            raise MemoryAccessDenied(self._block("security_error"))

    def _flow(self, context, entry, action):
        if self.data_flow_engine is None:
            return
        flow = DataFlow(DataSource("memory." + entry.namespace, entry.classification, entry.trust),
                        DataFlowDestination("memory." + context.namespace), action,
                        agent=context.agent_id, resource=entry.memory_id)
        try:
            decision = self.data_flow_engine.check(flow)
        except Exception:
            raise self._deny(context, entry, self._block("data_flow_error"))
        if decision.decision_type is not DecisionType.ALLOW:
            raise self._deny(context, entry, decision)

    def _deny(self, context, entry, decision):
        self._audit(AuditEventType.MEMORY_BLOCKED, context, entry, decision)
        return MemoryAccessDenied(decision)

    def _audit(self, event_type, context, entry, decision):
        if self.audit_logger is None:
            return
        try:
            metadata = {
                "memory_id": entry.memory_id,
                "namespace": entry.namespace,
                "classification": entry.classification.value,
                "operation": event_type.value,
            }
            if context.delegation is not None:
                metadata["grant_id"] = context.delegation.grant_id
            self.audit_logger.record(event_type, correlation_id=entry.memory_id, agent_id=context.agent_id,
                session_id=context.session_id, resource=entry.memory_id, action=event_type.value,
                decision=decision.decision_type.value, reason=decision.reason.value,
                metadata=metadata)
        except Exception:
            if self.audit_logger.policy.required:
                raise MemoryAccessDenied(self._block("audit"))

    @staticmethod
    def _block(category):
        return Decision(DecisionType.BLOCK, DecisionReason.RUNTIME_SECURITY_ERROR,
                        severity=Severity.CRITICAL, metadata={"category": category})


def _rank(value: FlowClassification) -> int:
    return {FlowClassification.PUBLIC: 0, FlowClassification.INTERNAL: 1, FlowClassification.PRIVATE: 2,
            FlowClassification.SENSITIVE: 3, FlowClassification.SECRET: 4}[value]
