"""Deterministic explicit data-flow labels, policies, and evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import fnmatch
from types import MappingProxyType
from typing import Any, Iterable, Mapping

from ..protocol.security_request import DataClassification, SourceTrust
from .decision import Decision, DecisionReason, DecisionType, Severity


class FlowClassification(str, Enum):
    PUBLIC = DataClassification.PUBLIC.value
    INTERNAL = DataClassification.INTERNAL.value
    PRIVATE = DataClassification.PRIVATE.value
    SENSITIVE = DataClassification.SENSITIVE.value
    SECRET = DataClassification.SECRET.value


class FlowTrust(str, Enum):
    TRUSTED = SourceTrust.TRUSTED.value
    UNTRUSTED = SourceTrust.UNTRUSTED.value


@dataclass(frozen=True, slots=True)
class DataSource:
    name: str
    classification: FlowClassification
    trust: FlowTrust

    def __post_init__(self) -> None:
        _text(self.name, "source.name")
        if not isinstance(self.classification, FlowClassification):
            raise ValueError("source.classification must be a FlowClassification")
        if not isinstance(self.trust, FlowTrust):
            raise ValueError("source.trust must be a FlowTrust")


@dataclass(frozen=True, slots=True)
class DataFlowDestination:
    name: str

    def __post_init__(self) -> None:
        _text(self.name, "destination.name")


@dataclass(frozen=True, slots=True)
class DataFlow:
    source: DataSource
    destination: DataFlowDestination
    action: str
    classification: FlowClassification | None = None
    trust: FlowTrust | None = None
    agent: str | None = None
    tool: str | None = None
    resource: str | None = None
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.source, DataSource):
            raise ValueError("source must be a DataSource")
        if not isinstance(self.destination, DataFlowDestination):
            raise ValueError("destination must be a DataFlowDestination")
        _text(self.action, "action")
        if self.classification is not None and not isinstance(self.classification, FlowClassification):
            raise ValueError("classification must be a FlowClassification")
        if self.trust is not None and not isinstance(self.trust, FlowTrust):
            raise ValueError("trust must be a FlowTrust")
        for field in ("agent", "tool", "resource"):
            value = getattr(self, field)
            if value is not None:
                _text(value, field)
        metadata = {} if self.metadata is None else dict(self.metadata)
        if not isinstance(self.metadata, (Mapping, type(None))):
            raise ValueError("metadata must be an object")
        object.__setattr__(self, "metadata", MappingProxyType(metadata))
        object.__setattr__(
            self,
            "classification",
            self.classification or self.source.classification,
        )
        object.__setattr__(self, "trust", self.trust or self.source.trust)


@dataclass(frozen=True, slots=True)
class DataFlowPolicy:
    rule_id: str
    decision_type: DecisionType
    source: str | None = None
    destination: str | None = None
    action: str | None = None
    classification: FlowClassification | None = None
    trust: FlowTrust | None = None
    agent: str | None = None
    tool: str | None = None
    resource: str | None = None
    priority: int = 0
    severity: Severity | None = None
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        _text(self.rule_id, "rule_id")
        if not isinstance(self.decision_type, DecisionType):
            raise ValueError("decision_type must be a DecisionType")
        if not isinstance(self.priority, int) or isinstance(self.priority, bool) or self.priority < 0:
            raise ValueError("priority must be a non-negative integer")
        for field in ("source", "destination", "action", "agent", "tool", "resource"):
            value = getattr(self, field)
            if value is not None:
                _text(value, field)
        if self.classification is not None and not isinstance(self.classification, FlowClassification):
            raise ValueError("classification must be a FlowClassification")
        if self.trust is not None and not isinstance(self.trust, FlowTrust):
            raise ValueError("trust must be a FlowTrust")
        if self.severity is not None and not isinstance(self.severity, Severity):
            raise ValueError("severity must be a Severity")
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata or {})))

    def matches(self, flow: DataFlow) -> bool:
        values = {
            "source": flow.source.name,
            "destination": flow.destination.name,
            "action": flow.action,
            "classification": flow.classification.value,
            "trust": flow.trust.value,
            "agent": flow.agent or "",
            "tool": flow.tool or "",
            "resource": flow.resource or "",
        }
        for field in values:
            expected = getattr(self, field)
            if expected is not None:
                expected_value = expected.value if isinstance(expected, Enum) else expected
                if not fnmatch.fnmatchcase(values[field], expected_value):
                    return False
        return True


class DataFlowEngine:
    """Evaluate explicit flow policies with priority and conflict detection."""

    def __init__(
        self,
        policies: Iterable[DataFlowPolicy] = (),
        *,
        default_decision: DecisionType = DecisionType.BLOCK,
    ) -> None:
        if not isinstance(default_decision, DecisionType):
            raise ValueError("default_decision must be a DecisionType")
        self._policies = tuple(policies)
        if any(not isinstance(policy, DataFlowPolicy) for policy in self._policies):
            raise ValueError("policies must contain DataFlowPolicy objects")
        self._default = default_decision

    def check(self, flow: DataFlow) -> Decision:
        try:
            if not isinstance(flow, DataFlow):
                return self._blocked("malformed_flow")
            matches = [policy for policy in self._policies if policy.matches(flow)]
            if not matches:
                return self._decision(self._default, "missing_policy")
            highest = max(policy.priority for policy in matches)
            selected = [policy for policy in matches if policy.priority == highest]
            decisions = {policy.decision_type for policy in selected}
            if len(decisions) > 1:
                return self._blocked("conflicting_policies")
            policy = selected[0]
            return self._decision(
                policy.decision_type,
                "flow_policy_match",
                rule_id=policy.rule_id,
                severity=policy.severity,
                metadata=dict(policy.metadata),
            )
        except Exception:
            return self._blocked("evaluation_error")

    @staticmethod
    def _decision(
        decision_type: DecisionType,
        reason: str,
        *,
        rule_id: str | None = None,
        severity: Severity | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> Decision:
        reasons = {
            "missing_policy": DecisionReason.DATA_FLOW_NO_POLICY,
            "flow_policy_match": DecisionReason.REQUEST_ALLOWED,
        }
        if decision_type is DecisionType.BLOCK and reason == "flow_policy_match":
            mapped_reason = DecisionReason.DATA_FLOW_VIOLATION
        else:
            mapped_reason = reasons.get(reason, DecisionReason.DATA_FLOW_VIOLATION)
        return Decision(
            decision_type=decision_type,
            reason=mapped_reason,
            rule_id=rule_id,
            severity=severity or (Severity.HIGH if decision_type is DecisionType.BLOCK else None),
            metadata=dict(metadata or {}),
        )

    @staticmethod
    def _blocked(reason: str) -> Decision:
        return Decision(
            decision_type=DecisionType.BLOCK,
            reason=(
                DecisionReason.DATA_FLOW_EVALUATION_ERROR
                if reason == "evaluation_error"
                else DecisionReason.DATA_FLOW_VIOLATION
            ),
            severity=Severity.CRITICAL,
            metadata={"category": reason},
        )


def propagate_classification(*labels: FlowClassification) -> FlowClassification:
    if not labels or any(not isinstance(label, FlowClassification) for label in labels):
        raise ValueError("at least one valid classification is required")
    return max(labels, key=lambda label: _CLASSIFICATION_ORDER[label])


def propagate_trust(*trusts: FlowTrust) -> FlowTrust:
    if not trusts or any(not isinstance(trust, FlowTrust) for trust in trusts):
        raise ValueError("at least one valid trust value is required")
    return FlowTrust.TRUSTED if all(trust is FlowTrust.TRUSTED for trust in trusts) else FlowTrust.UNTRUSTED


def _text(value: Any, field: str) -> None:
    if not isinstance(value, str) or not value.strip() or len(value) > 256:
        raise ValueError(f"{field} must be a non-empty string")


_CLASSIFICATION_ORDER = {
    FlowClassification.PUBLIC: 0,
    FlowClassification.INTERNAL: 1,
    FlowClassification.PRIVATE: 2,
    FlowClassification.SENSITIVE: 3,
    FlowClassification.SECRET: 4,
}
