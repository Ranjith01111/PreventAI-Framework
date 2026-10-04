"""Deterministic enforcement of explicitly configured content policies."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import re
import unicodedata
from typing import Any, Iterable

from ..core.decision import Decision, DecisionReason, DecisionType, Severity


class ContentGuard:
    """Scan JSON-like values against explicit application content rules.

    The default guard has no blocked content rules. It provides deterministic
    policy enforcement, not semantic harmful-content moderation.
    """

    _WHITESPACE = re.compile(r"\s+")
    _ZERO_WIDTH = re.compile(r"[\u200b-\u200d\ufeff]")
    _SAFE_KEY = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

    def __init__(
        self,
        *,
        blocked_phrases: Iterable[str] | None = None,
        custom_patterns: Mapping[str, str] | None = None,
        field_rules: Mapping[str, Iterable[str]] | None = None,
        enabled_categories: Iterable[str] | None = None,
        max_depth: int = 16,
        case_sensitive: bool = False,
    ) -> None:
        if not isinstance(max_depth, int) or isinstance(max_depth, bool) or max_depth < 0:
            raise ValueError("max_depth must be a non-negative integer")
        if not isinstance(case_sensitive, bool):
            raise ValueError("case_sensitive must be boolean")
        self._enabled = self._validate_categories(enabled_categories)
        self._case_sensitive = case_sensitive
        self._max_depth = max_depth
        self._phrase_rules = tuple(
            self._compile_phrase(phrase, f"CONTENT-PHRASE-{index:03d}")
            for index, phrase in enumerate(
                sorted(self._validate_phrases(blocked_phrases)), 1
            )
        )
        patterns: list[tuple[str, str, re.Pattern[str]]] = []
        raw_patterns = custom_patterns or {}
        if not isinstance(raw_patterns, Mapping):
            raise ValueError("custom_patterns must be a mapping")
        if any(not isinstance(rule_id, str) or not rule_id.strip() for rule_id in raw_patterns):
            raise ValueError("custom pattern identifiers must be non-empty strings")
        for rule_id, pattern in sorted(raw_patterns.items()):
            if not isinstance(pattern, str):
                raise ValueError("custom patterns must be strings")
            try:
                flags = 0 if case_sensitive else re.IGNORECASE
                patterns.append((rule_id, "prohibited_pattern", re.compile(pattern, flags)))
            except re.error as exc:
                raise ValueError("custom content pattern is invalid") from exc
        self._patterns = tuple(patterns)
        self._field_rules = self._compile_field_rules(field_rules or {})

    def scan(self, value: Any) -> Decision:
        try:
            finding = self._scan(value, "$", 0, set(), None)
        except Exception:
            return self._blocked("scanner_error", "$", "CONTENT-SCANNER-001")
        if finding is None:
            return Decision(
                decision_type=DecisionType.ALLOW,
                reason=DecisionReason.REQUEST_ALLOWED,
                metadata={"content": "no_policy_violation"},
            )
        category, location, rule_id = finding
        return self._blocked(category, location, rule_id)

    def _scan(
        self,
        value: Any,
        location: str,
        depth: int,
        active: set[int],
        field: str | None,
    ) -> tuple[str, str, str] | None:
        if depth > self._max_depth:
            return "scanner_error", location, "CONTENT-SCANNER-001"
        if isinstance(value, str):
            return self._scan_string(value, location, field)
        if value is None or isinstance(value, (bool, int, float)):
            return None
        if isinstance(value, Mapping):
            identity = id(value)
            if identity in active:
                return "scanner_error", location, "CONTENT-SCANNER-001"
            active.add(identity)
            try:
                for key, child in value.items():
                    if not isinstance(key, str):
                        return "scanner_error", location, "CONTENT-SCANNER-001"
                    child_location = self._location_for_key(location, key)
                    finding = self._scan(child, child_location, depth + 1, active, key)
                    if finding is not None:
                        return finding
            finally:
                active.remove(identity)
            return None
        if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
            identity = id(value)
            if identity in active:
                return "scanner_error", location, "CONTENT-SCANNER-001"
            active.add(identity)
            try:
                for index, child in enumerate(value):
                    finding = self._scan(child, f"{location}[{index}]", depth + 1, active, field)
                    if finding is not None:
                        return finding
            finally:
                active.remove(identity)
            return None
        return "scanner_error", location, "CONTENT-SCANNER-001"

    def _scan_string(
        self, value: str, location: str, field: str | None
    ) -> tuple[str, str, str] | None:
        normalized = self._normalize(value)
        if self._enabled_allows("prohibited_phrase"):
            for rule_id, pattern in self._phrase_rules:
                if pattern.search(normalized):
                    return "prohibited_phrase", location, rule_id
        if self._enabled_allows("prohibited_pattern"):
            for rule_id, category, pattern in self._patterns:
                if pattern.search(normalized):
                    return category, location, rule_id
        if field is not None and self._enabled_allows("restricted_field"):
            for rule_id, pattern in self._field_rules.get(field.casefold(), ()):
                if pattern.search(normalized):
                    return "restricted_field", location, rule_id
        return None

    def _compile_field_rules(
        self, field_rules: Mapping[str, Iterable[str]]
    ) -> dict[str, tuple[tuple[str, re.Pattern[str]], ...]]:
        compiled: dict[str, tuple[tuple[str, re.Pattern[str]], ...]] = {}
        for field, phrases in field_rules.items():
            if not isinstance(field, str) or not field.strip():
                raise ValueError("field rule names must be non-empty strings")
            values = self._validate_phrases(phrases)
            compiled[field.casefold()] = tuple(
                self._compile_phrase(phrase, f"CONTENT-FIELD-{index:03d}")
                for index, phrase in enumerate(sorted(values), 1)
            )
        return compiled

    @staticmethod
    def _validate_phrases(phrases: Iterable[str] | None) -> set[str]:
        if phrases is None:
            return set()
        if isinstance(phrases, str):
            phrases = [phrases]
        try:
            values = set(phrases)
        except (TypeError, ValueError) as exc:
            raise ValueError("content phrases must be iterable strings") from exc
        if any(not isinstance(phrase, str) or not phrase.strip() for phrase in values):
            raise ValueError("content phrases must contain non-empty strings")
        return values

    def _compile_phrase(self, phrase: str, rule_id: str) -> tuple[str, re.Pattern[str]]:
        normalized = self._normalize(phrase)
        escaped = re.escape(normalized).replace(r"\ ", r"\s+")
        flags = 0 if self._case_sensitive else re.IGNORECASE
        return rule_id, re.compile(rf"(?<!\w){escaped}(?!\w)", flags)

    @classmethod
    def _normalize(cls, value: str) -> str:
        normalized = unicodedata.normalize("NFKC", value)
        normalized = cls._ZERO_WIDTH.sub("", normalized)
        return cls._WHITESPACE.sub(" ", normalized).strip()

    def _validate_categories(
        self, categories: Iterable[str] | None
    ) -> frozenset[str] | None:
        if categories is None:
            return None
        values = frozenset(categories)
        allowed = {"prohibited_phrase", "prohibited_pattern", "restricted_field", "custom_policy"}
        if any(not isinstance(category, str) or category not in allowed for category in values):
            raise ValueError("enabled_categories contains an unsupported category")
        return values

    def _enabled_allows(self, category: str) -> bool:
        return self._enabled is None or category in self._enabled

    @classmethod
    def _location_for_key(cls, location: str, key: str) -> str:
        if cls._SAFE_KEY.fullmatch(key):
            return f"{location}.{key}" if location != "$" else key
        return f"{location}[key]"

    @staticmethod
    def _blocked(category: str, location: str, rule_id: str) -> Decision:
        return Decision(
            decision_type=DecisionType.BLOCK,
            reason=DecisionReason.CONTENT_POLICY_VIOLATION,
            rule_id=rule_id,
            severity=Severity.HIGH if category != "scanner_error" else Severity.CRITICAL,
            metadata={"category": category, "location": location},
        )
