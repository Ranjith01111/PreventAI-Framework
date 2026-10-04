"""Deterministic, rule-based policy evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import fnmatch
import json
import math
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Protocol

from ..protocol.security_request import ContractValidationError
from .context import EvaluationContext
from .decision import Decision, DecisionReason, DecisionType, Severity


def _json_value(value: Any, field: str) -> None:
    if value is None or isinstance(value, (str, bool, int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            raise ContractValidationError(f"{field} contains an invalid number")
        return
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


def _pattern(value: Any, field: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip() or len(value) > 256:
        raise ContractValidationError(f"{field} must be a non-empty string or null")
    return value


class PolicyEvaluator(Protocol):
    """Interface implemented by deterministic policy evaluators."""

    def evaluate(self, context: EvaluationContext) -> Decision:
        """Return a decision without executing side effects."""


@dataclass(frozen=True, slots=True)
class PolicyRule:
    """One explicit policy rule; omitted matchers mean any value."""

    rule_id: str
    decision_type: DecisionType
    priority: int = 0
    action: str | None = None
    resource: str | None = None
    agent: str | None = None
    tool: str | None = None
    conditions: Mapping[str, Any] | None = None
    severity: Severity | None = None
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.rule_id, str) or not self.rule_id.strip() or len(self.rule_id) > 256:
            raise ContractValidationError("rule_id must be a non-empty string")
        if not isinstance(self.decision_type, DecisionType):
            raise ContractValidationError("decision_type must be a DecisionType")
        if not isinstance(self.priority, int) or isinstance(self.priority, bool) or self.priority < 0:
            raise ContractValidationError("priority must be a non-negative integer")
        for field in ("action", "resource", "agent", "tool"):
            _pattern(getattr(self, field), field)
        if self.severity is not None and not isinstance(self.severity, Severity):
            raise ContractValidationError("severity must be a Severity")
        conditions = {} if self.conditions is None else dict(self.conditions)
        metadata = {} if self.metadata is None else dict(self.metadata)
        if not isinstance(self.conditions, (Mapping, type(None))):
            raise ContractValidationError("conditions must be an object")
        if not isinstance(self.metadata, (Mapping, type(None))):
            raise ContractValidationError("metadata must be an object")
        _json_value(conditions, "conditions")
        _json_value(metadata, "metadata")
        object.__setattr__(self, "conditions", MappingProxyType(conditions))
        object.__setattr__(self, "metadata", MappingProxyType(metadata))

    def matches(self, context: EvaluationContext) -> bool:
        request = context.request
        values = {
            "action": request.action.name,
            "resource": request.resource.resource_id,
            "agent": request.agent.agent_id,
            "tool": request.tool.name,
        }
        for field in ("action", "resource", "agent", "tool"):
            expected = getattr(self, field)
            if expected is not None and not fnmatch.fnmatchcase(values[field], expected):
                return False
        for path, expected in self.conditions.items():
            actual = _condition_value(request, path)
            if isinstance(expected, str):
                if not isinstance(actual, str) or not fnmatch.fnmatchcase(actual, expected):
                    return False
            elif actual != expected:
                return False
        return True

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "decision_type": self.decision_type.value,
            "priority": self.priority,
            "action": self.action,
            "resource": self.resource,
            "agent": self.agent,
            "tool": self.tool,
            "conditions": dict(self.conditions),
            "severity": self.severity.value if self.severity else None,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> PolicyRule:
        if not isinstance(value, Mapping):
            raise ContractValidationError("policy rule must be an object")
        expected = {
            "rule_id", "decision_type", "priority", "action", "resource",
            "agent", "tool", "conditions", "severity", "metadata",
        }
        if set(value) != expected:
            raise ContractValidationError("policy rule fields do not exactly match the contract")
        try:
            return cls(
                rule_id=value["rule_id"],
                decision_type=DecisionType(value["decision_type"]),
                priority=value["priority"],
                action=value["action"],
                resource=value["resource"],
                agent=value["agent"],
                tool=value["tool"],
                conditions=value["conditions"],
                severity=Severity(value["severity"]) if value["severity"] is not None else None,
                metadata=value["metadata"],
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ContractValidationError("invalid policy rule") from exc


def _condition_value(request: Any, path: str) -> Any:
    roots = {
        "requested_by": request.requested_by.value,
        "data.classification": request.data.classification.value,
        "data.source_trust": request.data.source_trust.value,
        "resource.type": request.resource.resource_type,
    }
    if path in roots:
        return roots[path]
    if path.startswith("context."):
        return request.context.get(path[8:])
    if path.startswith("metadata."):
        return request.metadata.get(path[9:])
    if path.startswith("action.parameters."):
        return request.action.parameters.get(path[18:])
    raise ContractValidationError(f"unsupported policy condition: {path}")


def _decision_for_rule(rule: PolicyRule) -> Decision:
    reasons = {
        DecisionType.ALLOW: DecisionReason.REQUEST_ALLOWED,
        DecisionType.BLOCK: DecisionReason.REQUEST_BLOCKED,
        DecisionType.WARN: DecisionReason.SECURITY_WARNING,
        DecisionType.APPROVAL_REQUIRED: DecisionReason.APPROVAL_REQUIRED,
    }
    return Decision(
        decision_type=rule.decision_type,
        reason=reasons[rule.decision_type],
        rule_id=rule.rule_id,
        severity=rule.severity,
        metadata=dict(rule.metadata),
    )


class PolicyEngine:
    """Evaluates explicit rules by priority and fails closed on conflicts."""

    def __init__(
        self,
        rules: Iterable[PolicyRule] = (),
        *,
        default_decision: DecisionType = DecisionType.BLOCK,
    ) -> None:
        if not isinstance(default_decision, DecisionType):
            raise ContractValidationError("default_decision must be a DecisionType")
        rules = tuple(rules)
        if any(not isinstance(rule, PolicyRule) for rule in rules):
            raise ContractValidationError("rules must contain only PolicyRule objects")
        ids = [rule.rule_id for rule in rules]
        if len(ids) != len(set(ids)):
            raise ContractValidationError("rule_id values must be unique")
        self._rules = tuple(sorted(rules, key=lambda rule: (-rule.priority, rule.rule_id)))
        self._default_decision = default_decision

    @property
    def rules(self) -> tuple[PolicyRule, ...]:
        return self._rules

    def evaluate(self, context: EvaluationContext) -> Decision:
        matched = [rule for rule in self._rules if rule.matches(context)]
        if not matched:
            return Decision(
                decision_type=self._default_decision,
                reason=DecisionReason.NO_POLICY,
                severity=Severity.HIGH if self._default_decision is DecisionType.BLOCK else None,
                metadata={"matched_rules": []},
            )
        highest = matched[0].priority
        selected = [rule for rule in matched if rule.priority == highest]
        decisions = {rule.decision_type for rule in selected}
        if len(decisions) > 1:
            return Decision(
                decision_type=DecisionType.BLOCK,
                reason=DecisionReason.CONFLICTING_POLICIES,
                severity=Severity.CRITICAL,
                metadata={
                    "conflict": True,
                    "rule_ids": [rule.rule_id for rule in selected],
                },
            )
        return _decision_for_rule(selected[0])

    def to_json(self) -> str:
        return json.dumps(
            {
                "default_decision": self._default_decision.value,
                "rules": [rule.to_dict() for rule in self._rules],
            },
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> PolicyEngine:
        if not isinstance(value, Mapping) or set(value) != {"default_decision", "rules"}:
            raise ContractValidationError("policy fields do not exactly match the contract")
        try:
            rules = value["rules"]
            if not isinstance(rules, list):
                raise ContractValidationError("policy rules must be an array")
            return cls(
                (PolicyRule.from_dict(item) for item in rules),
                default_decision=DecisionType(value["default_decision"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ContractValidationError("invalid policy") from exc

    @classmethod
    def from_json(cls, value: str) -> PolicyEngine:
        if not isinstance(value, str):
            raise ContractValidationError("JSON input must be a string")
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ContractValidationError("invalid JSON") from exc
        return cls.from_dict(decoded)


class DefaultDenyPolicy:
    """Compatibility policy used when no policy evaluator is configured."""

    def evaluate(self, context: EvaluationContext) -> Decision:
        return Decision(
            decision_type=DecisionType.BLOCK,
            reason=DecisionReason.NO_POLICY,
            severity=Severity.HIGH,
            metadata={"request_id": context.request.request_id},
        )
