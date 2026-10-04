"""Deterministic agent lifecycle governance."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
import threading
from types import MappingProxyType
from typing import Any, Mapping

from ..audit.event import AuditEventType
from ..audit.logger import AuditLogger
from ..core.decision import Decision, DecisionReason, DecisionType, Severity
from .agent_identity import AgentIdentity, _identity_id


class LifecycleState(str, Enum):
    CREATED = "created"
    REGISTERED = "registered"
    ACTIVE = "active"
    SUSPENDED = "suspended"
    QUARANTINED = "quarantined"
    REVOKED = "revoked"
    TERMINATED = "terminated"


class LifecycleError(ValueError):
    """Raised for invalid lifecycle administration operations."""


_TRANSITIONS = {
    LifecycleState.CREATED: frozenset({LifecycleState.REGISTERED}),
    LifecycleState.REGISTERED: frozenset({LifecycleState.ACTIVE}),
    LifecycleState.ACTIVE: frozenset({
        LifecycleState.SUSPENDED, LifecycleState.QUARANTINED,
        LifecycleState.REVOKED, LifecycleState.TERMINATED,
    }),
    LifecycleState.SUSPENDED: frozenset({
        LifecycleState.ACTIVE, LifecycleState.REVOKED, LifecycleState.TERMINATED,
    }),
    LifecycleState.QUARANTINED: frozenset({
        LifecycleState.ACTIVE, LifecycleState.REVOKED, LifecycleState.TERMINATED,
    }),
    LifecycleState.REVOKED: frozenset({LifecycleState.TERMINATED}),
    LifecycleState.TERMINATED: frozenset(),
}


@dataclass(frozen=True, slots=True)
class LifecycleRecord:
    agent_id: str
    state: LifecycleState
    registered_at: datetime
    updated_at: datetime


class AgentLifecycleRegistry:
    """Thread-safe, instance-local lifecycle registry."""

    __slots__ = ("_record_snapshot", "_lock", "_audit", "_audit_failed")

    def __init__(self, *, audit_logger: AuditLogger | None = None) -> None:
        if audit_logger is not None and not isinstance(audit_logger, AuditLogger):
            raise TypeError("audit_logger must be an AuditLogger")
        self._record_snapshot = MappingProxyType({})
        self._lock = threading.RLock()
        self._audit = audit_logger or AuditLogger()
        self._audit_failed = False

    @property
    def _records(self) -> Mapping[str, LifecycleRecord]:
        return self._record_snapshot

    @staticmethod
    def _valid_record(agent_id: str, record: object) -> bool:
        try:
            return (
                isinstance(record, LifecycleRecord)
                and record.agent_id == agent_id
                and isinstance(record.state, LifecycleState)
                and isinstance(record.registered_at, datetime)
                and record.registered_at.tzinfo is not None
                and isinstance(record.updated_at, datetime)
                and record.updated_at.tzinfo is not None
                and record.updated_at >= record.registered_at
            )
        except Exception:
            return False

    def register(self, agent: AgentIdentity | str) -> LifecycleRecord:
        agent_id = agent.agent_id if isinstance(agent, AgentIdentity) else agent
        _identity_id(agent_id, "agent_id")
        if isinstance(agent, AgentIdentity) and not agent.enabled:
            raise LifecycleError("disabled identity cannot be registered")
        now = datetime.now(timezone.utc)
        with self._lock:
            if agent_id in self._record_snapshot:
                self._record_block(agent_id, LifecycleState.REGISTERED, LifecycleState.REGISTERED, "already_registered")
                raise LifecycleError("agent is already registered")
            record = LifecycleRecord(agent_id, LifecycleState.REGISTERED, now, now)
            self._record_transition(agent_id, LifecycleState.CREATED, LifecycleState.REGISTERED, now)
            self._record_snapshot = MappingProxyType({**self._record_snapshot, agent_id: record})
            return record

    def get_state(self, agent_id: str) -> LifecycleState | None:
        try:
            _identity_id(agent_id, "agent_id")
        except ValueError:
            return None
        with self._lock:
            record = self._record_snapshot.get(agent_id)
            return record.state if self._valid_record(agent_id, record) else None

    def get_record(self, agent_id: str) -> LifecycleRecord | None:
        try:
            _identity_id(agent_id, "agent_id")
        except ValueError:
            return None
        with self._lock:
            record = self._record_snapshot.get(agent_id)
            if not self._valid_record(agent_id, record):
                return None
            return record

    def transition(self, agent_id: str, target_state: LifecycleState) -> LifecycleRecord:
        _identity_id(agent_id, "agent_id")
        if not isinstance(target_state, LifecycleState):
            raise LifecycleError("unknown lifecycle state")
        with self._lock:
            current = self._record_snapshot.get(agent_id)
            if current is None:
                raise LifecycleError("unknown agent")
            if not self._valid_record(agent_id, current):
                self._record_block(
                    agent_id, getattr(current, "state", None), target_state, "unknown_state"
                )
                raise LifecycleError("unknown lifecycle state")
            if target_state not in _TRANSITIONS.get(current.state, frozenset()):
                self._record_block(agent_id, current.state, target_state, "invalid_transition")
                raise LifecycleError("invalid lifecycle transition")
            now = datetime.now(timezone.utc)
            self._record_transition(agent_id, current.state, target_state, now)
            updated = LifecycleRecord(agent_id, target_state, current.registered_at, now)
            self._record_snapshot = MappingProxyType({**self._record_snapshot, agent_id: updated})
            return updated

    def activate(self, agent_id: str) -> LifecycleRecord:
        return self.transition(agent_id, LifecycleState.ACTIVE)

    def suspend(self, agent_id: str) -> LifecycleRecord:
        return self.transition(agent_id, LifecycleState.SUSPENDED)

    def quarantine(self, agent_id: str) -> LifecycleRecord:
        return self.transition(agent_id, LifecycleState.QUARANTINED)

    def revoke(self, agent_id: str) -> LifecycleRecord:
        return self.transition(agent_id, LifecycleState.REVOKED)

    def terminate(self, agent_id: str) -> LifecycleRecord:
        return self.transition(agent_id, LifecycleState.TERMINATED)

    def is_active(self, agent_id: str) -> bool:
        return self.get_state(agent_id) is LifecycleState.ACTIVE

    def is_executable(self, agent_id: str) -> bool:
        return self.is_active(agent_id)

    def is_communicable(self, agent_id: str) -> bool:
        return self.is_active(agent_id)

    def check(self, agent_id: str, *, operation: str = "privileged_operation") -> Decision:
        with self._lock:
            if self._audit_failed:
                return Decision(
                    DecisionType.BLOCK,
                    DecisionReason.LIFECYCLE_INVALID,
                    severity=Severity.CRITICAL,
                    metadata={"operation": operation, "stage": "lifecycle_audit"},
                )
            if self.is_active(agent_id):
                return Decision(DecisionType.ALLOW, DecisionReason.REQUEST_ALLOWED)
            safe_agent_id = agent_id if isinstance(agent_id, str) else "<invalid>"
            try:
                state = self.get_state(agent_id)
                self._record_block(
                    safe_agent_id,
                    state,
                    state,
                    "lifecycle_invalid",
                    operation=operation,
                )
            except Exception:
                return Decision(
                    DecisionType.BLOCK,
                    DecisionReason.LIFECYCLE_INVALID,
                    severity=Severity.CRITICAL,
                    metadata={"operation": operation, "stage": "lifecycle_audit"},
                )
            return Decision(
                DecisionType.BLOCK,
                DecisionReason.LIFECYCLE_INVALID,
                severity=Severity.HIGH,
                metadata={"operation": operation, "agent_id": safe_agent_id},
            )

    def _record_transition(
        self, agent_id: str, previous: LifecycleState,
        target: LifecycleState, timestamp: datetime,
    ) -> None:
        if self._audit is not None:
            try:
                event = self._audit.record(
                    AuditEventType.LIFECYCLE_TRANSITION,
                    timestamp=timestamp,
                    agent_id=agent_id,
                    decision=DecisionType.ALLOW.value,
                    reason="lifecycle_transition",
                    metadata={"previous_state": previous.value, "new_state": target.value},
                )
                if event is None:
                    raise RuntimeError("mandatory lifecycle audit event was not recorded")
            except Exception:
                self._audit_failed = True
                raise

    def _record_block(
        self, agent_id: str, previous: LifecycleState | Any,
        target: LifecycleState | Any, reason: str, *, operation: str = "transition",
    ) -> None:
        if self._audit is not None:
            try:
                event = self._audit.record(
                    AuditEventType.LIFECYCLE_BLOCKED,
                    agent_id=agent_id,
                    decision=DecisionType.BLOCK.value,
                    reason=reason,
                    metadata={
                        "previous_state": previous.value if isinstance(previous, LifecycleState) else "unknown",
                        "requested_state": target.value if isinstance(target, LifecycleState) else "unknown",
                        "operation": operation,
                    },
                )
                if event is None:
                    raise RuntimeError("mandatory lifecycle audit event was not recorded")
            except Exception:
                self._audit_failed = True
                raise
