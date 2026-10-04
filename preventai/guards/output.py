"""Deterministic protection for values returned by tools."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from typing import Any, Iterable

from ..core.decision import Decision, DecisionReason, DecisionType, Severity
from .content import ContentGuard
from .pii import PIIGuard
from .schema import SchemaGuard
from .secrets import SecretsGuard


class OutputGuard:
    """Validate and protect a tool result before it crosses the output boundary."""

    _DEFAULT_TYPES = frozenset(
        {"str", "int", "float", "bool", "list", "dict", "NoneType"}
    )

    def __init__(
        self,
        *,
        output_schema: Mapping[str, Any] | None = None,
        block_secrets: bool = True,
        pii_policy: str = "warn",
        blocked_phrases: Iterable[str] | None = None,
        custom_patterns: Mapping[str, str] | None = None,
        enabled_categories: Iterable[str] | None = None,
        max_output_bytes: int | None = None,
        allowed_output_types: Iterable[str] | None = None,
        max_depth: int = 16,
    ) -> None:
        if output_schema is not None and not isinstance(output_schema, Mapping):
            raise ValueError("output_schema must be an object")
        if not isinstance(block_secrets, bool):
            raise ValueError("block_secrets must be boolean")
        if pii_policy not in {"allow", "warn", "block"}:
            raise ValueError("pii_policy must be allow, warn, or block")
        if max_output_bytes is not None and (
            not isinstance(max_output_bytes, int)
            or isinstance(max_output_bytes, bool)
            or max_output_bytes < 0
        ):
            raise ValueError("max_output_bytes must be a non-negative integer")
        types = self._DEFAULT_TYPES if allowed_output_types is None else frozenset(
            allowed_output_types
        )
        if any(not isinstance(item, str) or item not in self._DEFAULT_TYPES for item in types):
            raise ValueError("allowed_output_types contains an unsupported type")
        self.output_schema = output_schema
        self.block_secrets = block_secrets
        self.pii_policy = pii_policy
        self.max_output_bytes = max_output_bytes
        self.allowed_output_types = types
        self._schema = output_schema
        self._secrets = SecretsGuard(max_depth=max_depth)
        self._pii = PIIGuard(max_depth=max_depth)
        self._content = ContentGuard(
            blocked_phrases=blocked_phrases,
            custom_patterns=custom_patterns,
            enabled_categories=enabled_categories,
            max_depth=max_depth,
        )

    def validate(self, result: Any, context: Any = None) -> Decision:
        del context
        try:
            type_name = type(result).__name__
            if type_name not in self.allowed_output_types:
                return self._blocked("unsupported_output_type", "$", Severity.HIGH)
            serialized_size = self._serialized_size(result)
            if self.max_output_bytes is not None and serialized_size > self.max_output_bytes:
                return self._blocked("output_size_exceeded", "$", Severity.HIGH)
            if self._schema is not None:
                schema_decision = SchemaGuard.validate(result, self._schema)
                if schema_decision.decision_type is not DecisionType.ALLOW:
                    return Decision(
                        decision_type=DecisionType.BLOCK,
                        reason=DecisionReason.OUTPUT_SCHEMA_VALIDATION_FAILED,
                        severity=Severity.HIGH,
                        metadata={"category": "output_schema", "location": "$"},
                    )
            if self.block_secrets:
                secret_decision = self._secrets.scan(result)
                if secret_decision.decision_type is not DecisionType.ALLOW:
                    return self._blocked(
                        "secret",
                        str(secret_decision.metadata.get("location", "$")),
                        Severity.CRITICAL,
                    )
            pii_decision = self._pii.scan(result)
            if pii_decision.decision_type is not DecisionType.ALLOW:
                if self.pii_policy == "block":
                    return self._blocked(
                        "pii",
                        str(pii_decision.metadata.get("location", "$")),
                        Severity.HIGH,
                    )
                if self.pii_policy == "warn":
                    return Decision(
                        decision_type=DecisionType.WARN,
                        reason=DecisionReason.SECURITY_WARNING,
                        severity=Severity.MEDIUM,
                        metadata={"category": "pii", "location": "$"},
                    )
            content_decision = self._content.scan(result)
            if content_decision.decision_type is not DecisionType.ALLOW:
                return self._blocked(
                    "output_policy",
                    str(content_decision.metadata.get("location", "$")),
                    Severity.HIGH,
                )
            return Decision(
                decision_type=DecisionType.ALLOW,
                reason=DecisionReason.REQUEST_ALLOWED,
                metadata={"output": "validated"},
            )
        except Exception:
            return Decision(
                decision_type=DecisionType.BLOCK,
                reason=DecisionReason.OUTPUT_SECURITY_ERROR,
                severity=Severity.CRITICAL,
                metadata={"stage": "output_guard"},
            )

    @staticmethod
    def _serialized_size(value: Any) -> int:
        encoded = json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return len(encoded)

    @staticmethod
    def _blocked(category: str, location: str, severity: Severity) -> Decision:
        reason = (
            DecisionReason.OUTPUT_SCHEMA_VALIDATION_FAILED
            if category == "output_schema"
            else DecisionReason.OUTPUT_POLICY_VIOLATION
        )
        return Decision(
            decision_type=DecisionType.BLOCK,
            reason=reason,
            severity=severity,
            metadata={"category": category, "location": location},
        )
