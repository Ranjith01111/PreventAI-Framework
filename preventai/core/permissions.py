"""Explicit least-privilege permission evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import fnmatch
from types import MappingProxyType
from typing import Any, Iterable, Mapping

from ..protocol.security_request import ContractValidationError
from .context import EvaluationContext
from .decision import Decision, DecisionReason, DecisionType, Severity


class PermissionCategory(str, Enum):
    READ = "read"
    WRITE = "write"
    UPDATE = "update"
    DELETE = "delete"
    EXECUTE = "execute"
    NETWORK = "network"
    FILE = "file"
    DATABASE = "database"


_VERB_CATEGORIES = {
    "read": PermissionCategory.READ,
    "get": PermissionCategory.READ,
    "list": PermissionCategory.READ,
    "write": PermissionCategory.WRITE,
    "create": PermissionCategory.WRITE,
    "update": PermissionCategory.UPDATE,
    "delete": PermissionCategory.DELETE,
    "execute": PermissionCategory.EXECUTE,
    "run": PermissionCategory.EXECUTE,
}


def _categories(context: EvaluationContext) -> frozenset[PermissionCategory]:
    request = context.request
    first = request.action.name.split(".", 1)[0].lower()
    verb = request.action.name.rsplit(".", 1)[-1].lower()
    result = set()
    if first == "database" or request.resource.resource_type.lower() == "database":
        result.add(PermissionCategory.DATABASE)
    if first in {"file", "filesystem"} or request.resource.resource_type.lower() in {"file", "filesystem"}:
        result.add(PermissionCategory.FILE)
    if first in {"http", "network", "email", "web"}:
        result.add(PermissionCategory.NETWORK)
    if verb in _VERB_CATEGORIES:
        result.add(_VERB_CATEGORIES[verb])
    else:
        result.add(PermissionCategory.READ)
    return frozenset(result)


@dataclass(frozen=True, slots=True)
class PermissionRule:
    """A single explicit grant scoped to an agent, tool, and resource."""

    permission_id: str
    category: PermissionCategory
    action: str = "*"
    agent: str = "*"
    tool: str = "*"
    resource: str = "*"
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.permission_id, str) or not self.permission_id.strip():
            raise ContractValidationError("permission_id must be a non-empty string")
        if not isinstance(self.category, PermissionCategory):
            raise ContractValidationError("category must be a PermissionCategory")
        for field in ("action", "agent", "tool", "resource"):
            value = getattr(self, field)
            if not isinstance(value, str) or not value.strip() or len(value) > 256:
                raise ContractValidationError(f"{field} must be a non-empty string")
        metadata = {} if self.metadata is None else dict(self.metadata)
        if not isinstance(self.metadata, (Mapping, type(None))):
            raise ContractValidationError("metadata must be an object")
        object.__setattr__(self, "metadata", MappingProxyType(metadata))

    def matches(self, context: EvaluationContext) -> bool:
        request = context.request
        return (
            self.category in _categories(context)
            and fnmatch.fnmatchcase(request.action.name, self.action)
            and fnmatch.fnmatchcase(request.agent.agent_id, self.agent)
            and fnmatch.fnmatchcase(request.tool.name, self.tool)
            and fnmatch.fnmatchcase(request.resource.resource_id, self.resource)
        )


class PermissionEngine:
    """Evaluates only explicit grants; no grant means deny."""

    def __init__(self, rules: Iterable[PermissionRule] = ()) -> None:
        self._rules = tuple(rules)
        if any(not isinstance(rule, PermissionRule) for rule in self._rules):
            raise ContractValidationError("rules must contain only PermissionRule objects")
        ids = [rule.permission_id for rule in self._rules]
        if len(ids) != len(set(ids)):
            raise ContractValidationError("permission_id values must be unique")

    @property
    def rules(self) -> tuple[PermissionRule, ...]:
        return self._rules

    def evaluate(
        self,
        context: EvaluationContext,
        *,
        delegating_permissions: PermissionEngine | None = None,
    ) -> Decision:
        matched = [rule for rule in self._rules if rule.matches(context)]
        if not matched:
            return self._denied("no matching permission", DecisionReason.MISSING_PERMISSION)
        if delegating_permissions is not None:
            authority = delegating_permissions.evaluate(context)
            if authority.decision_type is not DecisionType.ALLOW:
                return self._denied(
                    "requested permission exceeds delegating authority",
                    DecisionReason.PRIVILEGE_ESCALATION,
                )
        return Decision(
            decision_type=DecisionType.ALLOW,
            reason=DecisionReason.REQUEST_ALLOWED,
            rule_id=matched[0].permission_id,
            metadata={"permission_ids": [rule.permission_id for rule in matched]},
        )

    @staticmethod
    def _denied(message: str, reason: DecisionReason) -> Decision:
        return Decision(
            decision_type=DecisionType.BLOCK,
            reason=reason,
            severity=Severity.HIGH,
            metadata={"error": message},
        )
