"""Deterministic detection of high-confidence instruction injection signals."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import re
import unicodedata
from typing import Any, Iterable

from ..core.decision import Decision, DecisionReason, DecisionType, Severity


class InjectionGuard:
    """Scan untrusted JSON-like values for instruction-control manipulation.

    This guard uses bounded, deterministic structural patterns. It does not
    decode arbitrary encodings, stringify arbitrary objects, or attempt to
    detect every possible natural-language injection.
    """

    _RULES = (
        (
            "instruction_override",
            "INJ-OVERRIDE-001",
            re.compile(
                r"\b(?:ignore|disregard|forget|override)\s+"
                r"(?:all\s+)?(?:the\s+)?(?:previous|prior|current|system|developer)\s+"
                r"(?:instructions?|directions?|rules?)\b"
            ),
        ),
        (
            "role_hijack",
            "INJ-ROLE-001",
            re.compile(
                r"\b(?:you\s+are\s+now|act\s+as|pretend\s+(?:to\s+be|you\s+are)|"
                r"assume\s+the\s+role\s+of)\s+(?:the\s+)?"
                r"(?:system(?:\s+administrator)?|developer|administrator|admin|root)\b"
            ),
        ),
        (
            "policy_bypass",
            "INJ-POLICY-001",
            re.compile(
                r"\b(?:bypass|circumvent|ignore|disregard|skip|evade)\s+"
                r"(?:the\s+)?(?:security|safety|authorization|permission|policy|"
                r"policies|restrictions?|rules?)\b"
            ),
        ),
        (
            "prompt_extraction",
            "INJ-EXTRACT-001",
            re.compile(
                r"\b(?:show|reveal|print|display|expose|output|give\s+me)\s+"
                r"(?:the\s+)?(?:system|developer|hidden|internal)\s+"
                r"(?:prompt|instructions?|policy|configuration)\b"
            ),
        ),
        (
            "tool_control_manipulation",
            "INJ-TOOL-001",
            re.compile(
                r"\b(?:call|invoke|run|execute)\s+(?:the\s+)?(?:tool|function)\s+"
                r"(?:directly|without\s+(?:permission|authorization|approval))\b|"
                r"\b(?:ignore|bypass|override)\s+(?:the\s+)?tool\s+"
                r"(?:restrictions?|authorization|permissions?)\b"
            ),
        ),
        (
            "authority_escalation",
            "INJ-AUTHORITY-001",
            re.compile(
                r"\b(?:i\s+am|i'm)\s+(?:the\s+)?(?:admin|administrator|developer|"
                r"system|root)\b|\b(?:system|administrator|admin|root)\s+override\b"
            ),
        ),
        (
            "security_control_disablement",
            "INJ-CONTROL-001",
            re.compile(
                r"\b(?:disable|turn\s+off|deactivate|remove)\s+"
                r"(?:security|safety|validation|permission|authorization|"
                r"access\s+control|guard|checks?)\b"
            ),
        ),
        (
            "instruction_chain_break",
            "INJ-CHAIN-001",
            re.compile(
                r"\b(?:stop|terminate|end|break)\s+(?:following|obeying|using)\s+"
                r"(?:the\s+)?(?:current|existing|previous)\s+"
                r"(?:instructions?|rules?|policy)\b|"
                r"\b(?:terminate|end|break)\s+(?:the\s+)?instruction\s+"
                r"(?:chain|hierarchy)\b"
            ),
        ),
    )
    _DEFAULT_CATEGORIES = frozenset(rule[0] for rule in _RULES)
    _ZERO_WIDTH = re.compile(r"[\u200b-\u200d\ufeff]")
    _WHITESPACE = re.compile(r"\s+")
    _SAFE_KEY = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

    def __init__(
        self,
        *,
        enabled_categories: Iterable[str] | None = None,
        custom_patterns: Mapping[str, str] | None = None,
        max_depth: int = 16,
    ) -> None:
        if not isinstance(max_depth, int) or isinstance(max_depth, bool) or max_depth < 0:
            raise ValueError("max_depth must be a non-negative integer")
        if enabled_categories is None:
            self._enabled = None
        else:
            categories = frozenset(enabled_categories)
            if any(not isinstance(category, str) or not category.strip() for category in categories):
                raise ValueError("enabled_categories must contain non-empty strings")
            self._enabled = categories

        patterns: list[tuple[str, str, re.Pattern[str]]] = list(self._RULES)
        if custom_patterns is not None:
            for category, pattern in custom_patterns.items():
                if not isinstance(category, str) or not category.strip():
                    raise ValueError("custom pattern categories must be non-empty strings")
                if not isinstance(pattern, str):
                    raise ValueError("custom patterns must be strings")
                try:
                    compiled = re.compile(pattern, re.IGNORECASE)
                except re.error as exc:
                    raise ValueError("custom injection pattern is invalid") from exc
                patterns.append((category, f"INJ-CUSTOM-{len(patterns) + 1:03d}", compiled))
        self._patterns = tuple(patterns)
        self._max_depth = max_depth

    def scan(self, value: Any) -> Decision:
        try:
            finding = self._scan(value, "$", 0, set())
        except Exception:
            return self._blocked("scanner_error", "$", "INJ-SCANNER-001")
        if finding is None:
            return Decision(
                decision_type=DecisionType.ALLOW,
                reason=DecisionReason.REQUEST_ALLOWED,
                metadata={"injection": "none_detected"},
            )
        category, location, rule_id = finding
        return self._blocked(category, location, rule_id)

    def _scan(
        self, value: Any, location: str, depth: int, active: set[int]
    ) -> tuple[str, str, str] | None:
        if depth > self._max_depth:
            return "scanner_error", location, "INJ-SCANNER-001"
        if isinstance(value, str):
            return self._scan_string(value, location)
        if value is None or isinstance(value, (bool, int, float)):
            return None
        if isinstance(value, Mapping):
            identity = id(value)
            if identity in active:
                return "scanner_error", location, "INJ-SCANNER-001"
            active.add(identity)
            try:
                for key, child in value.items():
                    if not isinstance(key, str):
                        return "scanner_error", location, "INJ-SCANNER-001"
                    child_location = self._location_for_key(location, key)
                    finding = self._scan(child, child_location, depth + 1, active)
                    if finding is not None:
                        return finding
            finally:
                active.remove(identity)
            return None
        if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
            identity = id(value)
            if identity in active:
                return "scanner_error", location, "INJ-SCANNER-001"
            active.add(identity)
            try:
                for index, child in enumerate(value):
                    finding = self._scan(child, f"{location}[{index}]", depth + 1, active)
                    if finding is not None:
                        return finding
            finally:
                active.remove(identity)
            return None
        return "scanner_error", location, "INJ-SCANNER-001"

    def _scan_string(self, value: str, location: str) -> tuple[str, str, str] | None:
        normalized = self._normalize(value)
        for category, rule_id, pattern in self._patterns:
            if self._enabled_allows(category) and pattern.search(normalized):
                return category, location, rule_id
        return None

    def _enabled_allows(self, category: str) -> bool:
        return self._enabled is None or category in self._enabled

    @classmethod
    def _normalize(cls, value: str) -> str:
        normalized = unicodedata.normalize("NFKC", value).casefold()
        normalized = cls._ZERO_WIDTH.sub("", normalized)
        return cls._WHITESPACE.sub(" ", normalized).strip()

    @classmethod
    def _location_for_key(cls, location: str, key: str) -> str:
        if cls._SAFE_KEY.fullmatch(key):
            return f"{location}.{key}" if location != "$" else key
        return f"{location}[key]"

    @staticmethod
    def _blocked(category: str, location: str, rule_id: str) -> Decision:
        return Decision(
            decision_type=DecisionType.BLOCK,
            reason=DecisionReason.INJECTION_DETECTED,
            rule_id=rule_id,
            severity=Severity.HIGH if category != "scanner_error" else Severity.CRITICAL,
            metadata={"category": category, "location": location},
        )
