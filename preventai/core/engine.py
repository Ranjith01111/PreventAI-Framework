"""Deterministic decision engine boundary."""

from __future__ import annotations

from typing import Any

from ..protocol.security_request import ContractValidationError
from .context import EvaluationContext
from .decision import Decision, DecisionReason, DecisionType, Severity
from .permissions import PermissionEngine
from .policy import DefaultDenyPolicy, PolicyEvaluator


class SecurityEngine:
    """Validate a request and delegate decision creation to a policy seam."""

    __slots__ = ("_policy", "_permissions", "_integrity_registry")

    def __init__(
        self,
        policy: PolicyEvaluator | None = None,
        permissions: PermissionEngine | None = None,
        integrity_registry: Any | None = None,
    ) -> None:
        self._policy = policy if policy is not None else DefaultDenyPolicy()
        self._permissions = permissions if permissions is not None else PermissionEngine()
        self._integrity_registry = integrity_registry

    def evaluate(self, request: Any) -> Decision:
        """Return a decision without executing tools or causing side effects."""

        if self._integrity_registry is not None:
            try:
                snapshot = self._integrity_registry.active()
                if self._policy is not snapshot.policy or self._permissions is not snapshot.permissions:
                    return Decision(
                        decision_type=DecisionType.BLOCK,
                        reason=DecisionReason.SECURITY_EVALUATION_ERROR,
                        severity=Severity.CRITICAL,
                        metadata={"stage": "policy_integrity", "error": "configuration_substitution"},
                    )
            except Exception:
                return Decision(
                    decision_type=DecisionType.BLOCK,
                    reason=DecisionReason.SECURITY_EVALUATION_ERROR,
                    severity=Severity.CRITICAL,
                    metadata={"stage": "policy_integrity"},
                )
        try:
            context = EvaluationContext.from_request(request)
        except Exception:
            return Decision(
                decision_type=DecisionType.BLOCK,
                reason=DecisionReason.INVALID_REQUEST,
                severity=Severity.CRITICAL,
                metadata={"stage": "context"},
            )
        try:
            decision = self._policy.evaluate(context)
        except Exception:
            return Decision(
                decision_type=DecisionType.BLOCK,
                reason=DecisionReason.SECURITY_EVALUATION_ERROR,
                severity=Severity.CRITICAL,
                metadata={"stage": "policy"},
            )
        if not isinstance(decision, Decision):
            return Decision(
                decision_type=DecisionType.BLOCK,
                reason=DecisionReason.SECURITY_EVALUATION_ERROR,
                severity=Severity.CRITICAL,
                metadata={"stage": "policy"},
            )
        if decision.decision_type is DecisionType.ALLOW:
            try:
                permission = self._permissions.evaluate(context)
            except Exception:
                return Decision(
                    decision_type=DecisionType.BLOCK,
                    reason=DecisionReason.SECURITY_EVALUATION_ERROR,
                    severity=Severity.CRITICAL,
                    metadata={"stage": "permission"},
                )
            if permission.decision_type is not DecisionType.ALLOW:
                return permission
        return decision

    def integrity_check(self) -> Decision | None:
        """Revalidate the app-owned security snapshot without evaluating a request."""
        if self._integrity_registry is None:
            return None
        try:
            snapshot = self._integrity_registry.active()
            if self._policy is not snapshot.policy or self._permissions is not snapshot.permissions:
                return Decision(
                    decision_type=DecisionType.BLOCK,
                    reason=DecisionReason.SECURITY_EVALUATION_ERROR,
                    severity=Severity.CRITICAL,
                    metadata={"stage": "policy_integrity", "error": "configuration_substitution"},
                )
            return None
        except Exception:
            return Decision(
                decision_type=DecisionType.BLOCK,
                reason=DecisionReason.SECURITY_EVALUATION_ERROR,
                severity=Severity.CRITICAL,
                metadata={"stage": "policy_integrity"},
            )

    @property
    def permissions(self) -> PermissionEngine:
        return self._permissions
