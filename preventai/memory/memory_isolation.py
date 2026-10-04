"""Ownership and explicit sharing checks for memory."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
import threading
from uuid import uuid4

from ..core.dataflow import FlowClassification, FlowTrust
from ..core.decision import Decision, DecisionReason, DecisionType, Severity
from .memory_policy import MemoryAccessContext
from .memory_validation import text


@dataclass(frozen=True, slots=True)
class MemoryShare:
    share_id: str
    owner_agent_id: str
    owner_session_id: str
    namespace: str
    grantee_agent_id: str
    grantee_session_id: str | None = None
    expires_at: datetime | None = None
    revoked: bool = False

    def __post_init__(self) -> None:
        for field in ("share_id", "owner_agent_id", "owner_session_id", "namespace", "grantee_agent_id"):
            text(getattr(self, field), field)
        if self.grantee_session_id is not None:
            text(self.grantee_session_id, "grantee_session_id")
        if self.expires_at is not None and (
            self.expires_at.tzinfo is None
        ):
            raise ValueError("expires_at must be timezone-aware")


class MemoryIsolation:
    def __init__(self) -> None:
        self._shares: dict[str, MemoryShare] = {}
        self._lock = threading.RLock()

    def share(
        self,
        owner: MemoryAccessContext,
        grantee_agent_id: str,
        namespace: str,
        *,
        grantee_session_id: str | None = None,
        expires_at: datetime | None = None,
    ) -> MemoryShare:
        text(grantee_agent_id, "grantee_agent_id")
        text(namespace, "namespace")
        if expires_at is not None and expires_at.tzinfo is None:
            raise ValueError("expires_at must be timezone-aware")
        share = MemoryShare(
            str(uuid4()), owner.agent_id, owner.session_id, namespace,
            grantee_agent_id, grantee_session_id, expires_at,
        )
        with self._lock:
            self._shares[share.share_id] = share
        return share

    def revoke(self, share_id: str) -> None:
        with self._lock:
            share = self._shares.get(share_id)
            if share is None:
                raise KeyError("share not found")
            self._shares[share_id] = replace(share, revoked=True)

    def check(
        self,
        context: MemoryAccessContext,
        owner_agent_id: str,
        owner_session_id: str,
        namespace: str,
        *,
        now: datetime | None = None,
    ) -> Decision:
        if (
            context.agent_id == owner_agent_id
            and context.session_id == owner_session_id
            and context.namespace == namespace
        ):
            return Decision(DecisionType.ALLOW, DecisionReason.REQUEST_ALLOWED)
        current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
        with self._lock:
            valid = any(
                not share.revoked
                and share.owner_agent_id == owner_agent_id
                and share.owner_session_id == owner_session_id
                and share.namespace == namespace
                and context.namespace == namespace
                and share.grantee_agent_id == context.agent_id
                and (share.grantee_session_id is None or share.grantee_session_id == context.session_id)
                and (share.expires_at is None or share.expires_at > current)
                for share in self._shares.values()
            )
        if valid:
            return Decision(DecisionType.ALLOW, DecisionReason.REQUEST_ALLOWED)
        return Decision(
            DecisionType.BLOCK,
            DecisionReason.PERMISSION_DENIED,
            severity=Severity.HIGH,
            metadata={"category": "memory_isolation"},
        )
