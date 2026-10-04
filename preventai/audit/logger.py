"""Thread-safe in-memory append-only audit logger."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import hmac
import json
import threading
from typing import Any, Iterable
from uuid import uuid4

from .event import AuditEvent, AuditEventType, AuditVerification


@dataclass(frozen=True, slots=True)
class AuditPolicy:
    enabled: bool = True
    required: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool) or not isinstance(self.required, bool):
            raise ValueError("enabled and required must be boolean")


class AuditLogger:
    def __init__(self, *, secret: str | None = None, policy: AuditPolicy | None = None) -> None:
        if secret is not None and (not isinstance(secret, str) or not secret):
            raise ValueError("secret must be non-empty")
        self.policy = policy or AuditPolicy()
        self._secret = (secret or uuid4().hex).encode("utf-8")
        self._events: list[AuditEvent] = []
        self._lock = threading.RLock()

    def record(
        self,
        event_type: AuditEventType,
        *,
        correlation_id: str | None = None,
        timestamp: datetime | None = None,
        metadata: dict[str, Any] | None = None,
        **fields: str | None,
    ) -> AuditEvent | None:
        if not self.policy.enabled:
            return None
        with self._lock:
            sequence = len(self._events) + 1
            previous = self._events[-1].event_hash if self._events else None
            event = AuditEvent(
                event_id=str(uuid4()),
                sequence=sequence,
                timestamp=timestamp or datetime.now(timezone.utc),
                event_type=event_type,
                correlation_id=correlation_id,
                previous_event_hash=previous,
                metadata=metadata,
                **fields,
            )
            event_hash = self._hash(event.unsigned_dict())
            tag = self._tag(event_hash)
            event = AuditEvent(
                event_id=event.event_id,
                sequence=event.sequence,
                timestamp=event.timestamp,
                event_type=event.event_type,
                agent_id=event.agent_id,
                session_id=event.session_id,
                tool_id=event.tool_id,
                action=event.action,
                resource=event.resource,
                identity=event.identity,
                decision=event.decision,
                reason=event.reason,
                severity=event.severity,
                rule_id=event.rule_id,
                correlation_id=event.correlation_id,
                previous_event_hash=event.previous_event_hash,
                event_hash=event_hash,
                integrity_tag=tag,
                metadata=event.metadata,
            )
            self._events.append(event)
            return event

    def events(self) -> tuple[AuditEvent, ...]:
        with self._lock:
            return tuple(self._events)

    def trace(self, correlation_id: str) -> tuple[AuditEvent, ...]:
        with self._lock:
            return tuple(event for event in self._events if event.correlation_id == correlation_id)

    def verify(self) -> AuditVerification:
        with self._lock:
            previous = None
            for expected_sequence, event in enumerate(self._events, start=1):
                if event.sequence != expected_sequence:
                    return AuditVerification(False, "SEQUENCE_MISMATCH", event.sequence)
                if event.previous_event_hash != previous:
                    return AuditVerification(False, "PREVIOUS_HASH_MISMATCH", event.sequence)
                event_hash = self._hash(event.unsigned_dict())
                if not hmac.compare_digest(event.event_hash or "", event_hash):
                    return AuditVerification(False, "HASH_CHAIN_MISMATCH", event.sequence)
                if not hmac.compare_digest(event.integrity_tag or "", self._tag(event_hash)):
                    return AuditVerification(False, "HMAC_MISMATCH", event.sequence)
                previous = event_hash
            return AuditVerification(True)

    @staticmethod
    def _canonical(value: dict[str, Any]) -> bytes:
        return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode()

    def _hash(self, value: dict[str, Any]) -> str:
        return hashlib.sha256(self._canonical(value)).hexdigest()

    def _tag(self, event_hash: str) -> str:
        return hmac.new(self._secret, event_hash.encode(), hashlib.sha256).hexdigest()
