"""Explicit deterministic memory access policies."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import fnmatch
from typing import Any, Iterable

from ..core.decision import Decision, DecisionReason, DecisionType, Severity
from ..core.dataflow import FlowClassification, FlowTrust
from .memory_validation import text


class MemoryOperation(str, Enum):
    READ = "read"
    WRITE = "write"
    UPDATE = "update"
    DELETE = "delete"
    SHARE = "share"


@dataclass(frozen=True, slots=True)
class MemoryAccessContext:
    authenticated_identity: str
    agent_id: str
    session_id: str
    namespace: str
    delegation: Any = None
    delegation_registry: Any = None

    def __post_init__(self) -> None:
        for field in ("authenticated_identity", "agent_id", "session_id", "namespace"):
            text(getattr(self, field), field)
        if self.authenticated_identity != self.agent_id:
            raise ValueError("authenticated identity must match agent_id")


@dataclass(frozen=True, slots=True)
class MemoryPolicyRule:
    rule_id: str
    operation: MemoryOperation
    decision_type: DecisionType
    identity: str = "*"
    agent: str = "*"
    session: str = "*"
    namespace: str = "*"
    classification: FlowClassification | None = None
    trust: FlowTrust | None = None
    priority: int = 0

    def __post_init__(self) -> None:
        text(self.rule_id, "rule_id")
        if not isinstance(self.operation, MemoryOperation):
            raise ValueError("operation must be a MemoryOperation")
        if not isinstance(self.decision_type, DecisionType):
            raise ValueError("decision_type must be a DecisionType")
        if self.priority < 0 or isinstance(self.priority, bool):
            raise ValueError("priority must be a non-negative integer")
        for field in ("identity", "agent", "session", "namespace"):
            text(getattr(self, field), field)
        if self.classification is not None and not isinstance(self.classification, FlowClassification):
            raise ValueError("classification must be a FlowClassification")
        if self.trust is not None and not isinstance(self.trust, FlowTrust):
            raise ValueError("trust must be a FlowTrust")

    def matches(self, context: MemoryAccessContext, classification: FlowClassification, trust: FlowTrust) -> bool:
        return (
            fnmatch.fnmatchcase(context.authenticated_identity, self.identity)
            and fnmatch.fnmatchcase(context.agent_id, self.agent)
            and fnmatch.fnmatchcase(context.session_id, self.session)
            and fnmatch.fnmatchcase(context.namespace, self.namespace)
            and (self.classification is None or self.classification is classification)
            and (self.trust is None or self.trust is trust)
        )


class MemoryPolicy:
    def __init__(self, rules: Iterable[MemoryPolicyRule] = ()) -> None:
        self._rules = tuple(rules)
        if any(not isinstance(rule, MemoryPolicyRule) for rule in self._rules):
            raise ValueError("rules must contain MemoryPolicyRule objects")
        ids = [rule.rule_id for rule in self._rules]
        if len(ids) != len(set(ids)):
            raise ValueError("rule_id values must be unique")

    @property
    def rules(self) -> tuple[MemoryPolicyRule, ...]:
        return self._rules

    def evaluate(
        self,
        operation: MemoryOperation,
        context: MemoryAccessContext,
        classification: FlowClassification,
        trust: FlowTrust,
    ) -> Decision:
        matches = [
            rule for rule in self._rules
            if rule.operation is operation and rule.matches(context, classification, trust)
        ]
        if not matches:
            return Decision(
                DecisionType.BLOCK,
                DecisionReason.NO_POLICY,
                severity=Severity.HIGH,
                metadata={"operation": operation.value},
            )
        highest = max(rule.priority for rule in matches)
        selected = [rule for rule in matches if rule.priority == highest]
        if len({rule.decision_type for rule in selected}) != 1:
            return Decision(
                DecisionType.BLOCK,
                DecisionReason.CONFLICTING_POLICIES,
                severity=Severity.CRITICAL,
                metadata={"operation": operation.value},
            )
        rule = selected[0]
        return Decision(
            rule.decision_type,
            DecisionReason.POLICY_MATCH,
            rule_id=rule.rule_id,
            severity=Severity.HIGH if rule.decision_type is DecisionType.BLOCK else None,
            metadata={"operation": operation.value},
        )
