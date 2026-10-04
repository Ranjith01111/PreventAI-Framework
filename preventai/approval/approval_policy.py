"""Explicit deterministic rules for requiring human approval."""

from __future__ import annotations

from dataclasses import dataclass
import fnmatch
from typing import Iterable

from ..core.context import EvaluationContext


@dataclass(frozen=True, slots=True)
class ApprovalRule:
    rule_id: str
    action: str | None = None
    resource: str | None = None
    tool: str | None = None
    agent: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.rule_id, str) or not self.rule_id.strip():
            raise ValueError("rule_id must be non-empty")
        for field in ("action", "resource", "tool", "agent"):
            value = getattr(self, field)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"{field} must be non-empty when provided")

    def matches(self, context: EvaluationContext) -> bool:
        values = {
            "action": context.request.action.name,
            "resource": context.request.resource.resource_id,
            "tool": context.request.tool.name,
            "agent": context.request.agent.agent_id,
        }
        return all(
            expected is None or fnmatch.fnmatchcase(values[field], expected)
            for field, expected in (
                ("action", self.action),
                ("resource", self.resource),
                ("tool", self.tool),
                ("agent", self.agent),
            )
        )


class ApprovalPolicy:
    def __init__(self, rules: Iterable[ApprovalRule] = ()) -> None:
        self._rules = tuple(rules)
        if any(not isinstance(rule, ApprovalRule) for rule in self._rules):
            raise ValueError("rules must contain ApprovalRule objects")

    def match(self, context: EvaluationContext) -> ApprovalRule | None:
        for rule in self._rules:
            if rule.matches(context):
                return rule
        return None
