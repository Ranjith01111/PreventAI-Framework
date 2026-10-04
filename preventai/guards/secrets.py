"""Deterministic detection of common secret material."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import re
from typing import Any, Iterable

from ..core.decision import Decision, DecisionReason, DecisionType, Severity


class SecretsGuard:
    """Scan JSON-like tool input and block detected secret material.

    Diagnostics contain only a category and structural location. Values are
    never copied into decisions or exception messages.
    """

    DEFAULT_FIELDS = frozenset(
        {
            "password",
            "passwd",
            "pwd",
            "secret",
            "api_key",
            "apikey",
            "access_token",
            "auth_token",
            "refresh_token",
            "client_secret",
            "private_key",
            "credential",
            "credentials",
        }
    )
    DEFAULT_PATTERNS = (
        ("private_key", re.compile(r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----")),
        ("bearer_token", re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]{8,}")),
        ("jwt", re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{4,}\.[A-Za-z0-9_-]{4,}\b")),
        ("api_key", re.compile(r"\b(?:sk|pk|ghp|github_pat|xox[baprs])[-_][A-Za-z0-9_-]{8,}\b")),
        ("cloud_key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
        (
            "connection_string",
            re.compile(r"(?i)\b(?:mongodb(?:\+srv)?|postgres(?:ql)?|mysql|redis)://[^/\s:@]+:[^@\s]+@"),
        ),
        (
            "environment_secret",
            re.compile(
                r"(?i)\b(?:api[_-]?key|access[_-]?token|client[_-]?secret|password|secret)\s*=\s*[^\s]+"
            ),
        ),
    )

    def __init__(
        self,
        *,
        enabled_categories: Iterable[str] | None = None,
        sensitive_fields: Iterable[str] | None = None,
        custom_patterns: Mapping[str, str] | None = None,
        max_depth: int = 16,
    ) -> None:
        if not isinstance(max_depth, int) or isinstance(max_depth, bool) or max_depth < 0:
            raise ValueError("max_depth must be a non-negative integer")
        self._enabled = (
            frozenset(enabled_categories) if enabled_categories is not None else None
        )
        self._fields = frozenset(
            field.lower() for field in (sensitive_fields or self.DEFAULT_FIELDS)
        )
        if any(not isinstance(field, str) or not field.strip() for field in self._fields):
            raise ValueError("sensitive_fields must contain non-empty strings")
        patterns = list(self.DEFAULT_PATTERNS)
        if custom_patterns is not None:
            for category, pattern in custom_patterns.items():
                if not isinstance(category, str) or not category.strip():
                    raise ValueError("custom pattern categories must be non-empty strings")
                patterns.append((category, re.compile(pattern)))
        self._patterns = tuple(patterns)
        self._max_depth = max_depth

    def scan(self, value: Any) -> Decision:
        try:
            finding = self._scan(value, "$", 0, set())
        except Exception:
            return self._blocked("scanner_error")
        if finding is None:
            return Decision(
                decision_type=DecisionType.ALLOW,
                reason=DecisionReason.REQUEST_ALLOWED,
                metadata={"secrets": "none_detected"},
            )
        category, location = finding
        return self._blocked(category, location)

    def _scan(
        self, value: Any, location: str, depth: int, active: set[int]
    ) -> tuple[str, str] | None:
        if depth > self._max_depth:
            return ("scanner_error", location)
        if isinstance(value, str):
            return self._scan_string(value, location)
        if value is None or isinstance(value, (bool, int, float)):
            return None
        if isinstance(value, Mapping):
            identity = id(value)
            if identity in active:
                return ("scanner_error", location)
            active.add(identity)
            try:
                for key, child in value.items():
                    if not isinstance(key, str):
                        return ("scanner_error", location)
                    child_location = f"{location}.{key}" if location != "$" else key
                    if key.lower() in self._fields and self._is_non_empty_secret_value(child):
                        category = self._field_category(key)
                        if self._enabled is None or category in self._enabled:
                            return category, child_location
                    finding = self._scan(child, child_location, depth + 1, active)
                    if finding is not None:
                        return finding
            finally:
                active.remove(identity)
            return None
        if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
            identity = id(value)
            if identity in active:
                return ("scanner_error", location)
            active.add(identity)
            try:
                for index, child in enumerate(value):
                    finding = self._scan(child, f"{location}[{index}]", depth + 1, active)
                    if finding is not None:
                        return finding
            finally:
                active.remove(identity)
            return None
        return ("scanner_error", location)

    def _scan_string(self, value: str, location: str) -> tuple[str, str] | None:
        for category, pattern in self._patterns:
            if (self._enabled is None or category in self._enabled) and pattern.search(value):
                return category, location
        return None

    @staticmethod
    def _is_non_empty_secret_value(value: Any) -> bool:
        if isinstance(value, str):
            return bool(value.strip())
        return value is not None

    @staticmethod
    def _field_category(field: str) -> str:
        normalized = field.lower()
        if "password" in normalized or normalized in {"passwd", "pwd"}:
            return "password"
        if "token" in normalized:
            return "access_token"
        if "key" in normalized:
            return "api_key"
        if "private" in normalized:
            return "private_key"
        if "credential" in normalized:
            return "credential"
        return "secret"

    @staticmethod
    def _blocked(category: str, location: str | None = None) -> Decision:
        metadata = {"category": category}
        if location is not None:
            metadata["location"] = location
        return Decision(
            decision_type=DecisionType.BLOCK,
            reason=DecisionReason.SECRETS_DETECTED,
            severity=Severity.CRITICAL,
            metadata=metadata,
        )
