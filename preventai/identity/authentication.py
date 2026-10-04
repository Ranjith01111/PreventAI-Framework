"""In-process HMAC-based identity authentication."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import hmac
import secrets
import threading
from typing import Any

from ..core.decision import Decision, DecisionReason, DecisionType, Severity
from .agent_identity import AgentIdentity, _identity_id


class AuthenticationStatus(str, Enum):
    AUTHENTICATED = "authenticated"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class Credential:
    identity_id: str
    secret: str

    def __post_init__(self) -> None:
        _identity_id(self.identity_id)
        if not isinstance(self.secret, str) or not self.secret:
            raise ValueError("secret must be a non-empty string")

    def __repr__(self) -> str:
        return f"Credential(identity_id={self.identity_id!r}, secret=<redacted>)"


@dataclass(frozen=True, slots=True)
class AuthenticationResult:
    status: AuthenticationStatus
    identity_id: str | None = None
    session_id: str | None = None

    @property
    def authenticated(self) -> bool:
        return self.status is AuthenticationStatus.AUTHENTICATED

    def to_decision(self) -> Decision:
        if self.authenticated:
            return Decision(
                decision_type=DecisionType.ALLOW,
                reason=DecisionReason.REQUEST_ALLOWED,
                metadata={"authentication": "authenticated"},
            )
        return Decision(
            decision_type=DecisionType.BLOCK,
            reason=DecisionReason.AUTHENTICATION_FAILED,
            severity=Severity.HIGH,
            metadata={"authentication": "failed"},
        )


@dataclass
class _Record:
    identity: AgentIdentity
    digest: bytes
    revoked: bool = False


class Authenticator:
    """Thread-safe local credential registry for development/runtime use."""

    def __init__(self) -> None:
        self._records: dict[str, _Record] = {}
        self._sessions: dict[str, str] = {}
        self._lock = threading.RLock()

    def register(self, identity: AgentIdentity, credential: Credential) -> None:
        if not isinstance(identity, AgentIdentity) or not isinstance(credential, Credential):
            raise ValueError("identity and credential types are invalid")
        if identity.agent_id != credential.identity_id:
            raise ValueError("credential identity does not match agent identity")
        with self._lock:
            if identity.agent_id in self._records:
                raise ValueError("identity is already registered")
            self._records[identity.agent_id] = _Record(
                identity=identity,
                digest=self._digest(credential.secret),
            )

    def has_identity(self, identity_id: str) -> bool:
        _identity_id(identity_id)
        with self._lock:
            return identity_id in self._records

    def authenticate(
        self,
        identity_id: str,
        presented_credential: Credential | str,
        *,
        session_id: str | None = None,
    ) -> AuthenticationResult:
        try:
            _identity_id(identity_id)
            secret = (
                presented_credential.secret
                if isinstance(presented_credential, Credential)
                else presented_credential
            )
            if not isinstance(secret, str):
                return AuthenticationResult(AuthenticationStatus.FAILED)
            with self._lock:
                record = self._records.get(identity_id)
                if (
                    record is None
                    or not record.identity.enabled
                    or record.revoked
                    or (
                        isinstance(presented_credential, Credential)
                        and presented_credential.identity_id != identity_id
                    )
                ):
                    return AuthenticationResult(AuthenticationStatus.FAILED)
                valid = hmac.compare_digest(record.digest, self._digest(secret))
                if valid and session_id is not None:
                    bound = self._sessions.get(identity_id)
                    if bound is not None and bound != session_id:
                        valid = False
                    else:
                        self._sessions[identity_id] = session_id
                return AuthenticationResult(
                    AuthenticationStatus.AUTHENTICATED if valid else AuthenticationStatus.FAILED,
                    identity_id if valid else None,
                    session_id if valid else None,
                )
        except Exception:
            return AuthenticationResult(AuthenticationStatus.FAILED)

    def rotate(self, identity_id: str, new_credential: Credential) -> None:
        if not isinstance(new_credential, Credential) or new_credential.identity_id != identity_id:
            raise ValueError("credential identity does not match identity")
        with self._lock:
            record = self._records.get(identity_id)
            if record is None:
                raise ValueError("identity is not registered")
            record.digest = self._digest(new_credential.secret)
            record.revoked = False
            self._sessions.pop(identity_id, None)

    def revoke(self, identity_id: str) -> None:
        with self._lock:
            record = self._records.get(identity_id)
            if record is None:
                raise ValueError("identity is not registered")
            record.revoked = True
            self._sessions.pop(identity_id, None)

    def disable(self, identity_id: str) -> None:
        self._set_enabled(identity_id, False)

    def enable(self, identity_id: str) -> None:
        self._set_enabled(identity_id, True)

    def _set_enabled(self, identity_id: str, enabled: bool) -> None:
        with self._lock:
            record = self._records.get(identity_id)
            if record is None:
                raise ValueError("identity is not registered")
            record.identity = AgentIdentity(
                record.identity.agent_id,
                record.identity.name,
                enabled,
                record.identity.metadata,
            )

    @staticmethod
    def _digest(secret: str) -> bytes:
        return hmac.new(b"preventai-local-credential", secret.encode("utf-8"), hashlib.sha256).digest()
