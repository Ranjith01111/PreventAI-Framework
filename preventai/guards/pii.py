"""Deterministic detection of personal and sensitive data."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import ipaddress
import re
from typing import Any, Iterable

from ..core.decision import Decision, DecisionReason, DecisionType, Severity


class PIIGuard:
    """Scan JSON-like values for conservative PII indicators.

    Findings contain only a category and structural location. Values are
    never copied into decisions or exception messages.
    """

    DEFAULT_FIELDS = frozenset(
        {
            "email",
            "phone",
            "mobile",
            "address",
            "physical_address",
            "date_of_birth",
            "dob",
            "national_id",
            "passport",
            "tax_id",
            "user_id",
            "customer_id",
        }
    )
    EMAIL = re.compile(r"(?i)(?<![\w.+-])[\w.!#$%&'*+/=?^`{|}~-]+@[\w-]+(?:\.[\w-]+)+(?![\w.-])")
    PHONE = re.compile(r"(?<!\d)(?:\+\d{1,3}[\s.-]?)?(?:\(?\d{2,4}\)?[\s.-]?){2,5}\d{2,4}(?!\d)")
    CARD = re.compile(r"(?<!\d)(?:\d[ -]?){13,19}(?!\d)")

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
        patterns = []
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
                metadata={"pii": "none_detected"},
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
                    category = self._field_category(key)
                    if (
                        category is not None
                        and self._enabled_allows(category)
                        and self._field_value_is_pii(category, child)
                    ):
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
        if self._enabled_allows("email") and self.EMAIL.search(value):
            return "email", location
        if self._enabled_allows("ip_address") and self._is_ip(value):
            return "ip_address", location
        if self._enabled_allows("credit_card_like") and self._is_card(value):
            return "credit_card_like", location
        if self._enabled_allows("phone") and self._is_phone(value):
            return "phone", location
        for category, pattern in self._patterns:
            if self._enabled_allows(category) and pattern.search(value):
                return category, location
        return None

    def _field_category(self, field: str) -> str | None:
        normalized = field.lower()
        if normalized not in self._fields:
            return None
        if normalized in {"email"}:
            return "email"
        if normalized in {"phone", "mobile"}:
            return "phone"
        if normalized in {"address", "physical_address"}:
            return "physical_address"
        if normalized in {"date_of_birth", "dob"}:
            return "date_of_birth"
        if normalized in {"user_id", "customer_id"}:
            return "user_identifier"
        return "government_id_like"

    def _enabled_allows(self, category: str) -> bool:
        return self._enabled is None or category in self._enabled

    @staticmethod
    def _non_empty(value: Any) -> bool:
        return value is not None and (not isinstance(value, str) or bool(value.strip()))

    @classmethod
    def _field_value_is_pii(cls, category: str, value: Any) -> bool:
        if not cls._non_empty(value):
            return False
        if category == "user_identifier":
            # Preserve short opaque application IDs such as C001; longer
            # explicit user identifiers remain protected by field context.
            return not isinstance(value, str) or len(value.strip()) >= 6
        return True

    @staticmethod
    def _is_ip(value: str) -> bool:
        try:
            ipaddress.ip_address(value.strip())
        except ValueError:
            return False
        return True

    @classmethod
    def _is_phone(cls, value: str) -> bool:
        match = cls.PHONE.search(value)
        if match is None:
            return False
        text = match.group(0)
        digits = sum(character.isdigit() for character in text)
        return 10 <= digits <= 15 and (
            text.startswith("+") or any(character in text for character in " -.()")
        )

    @classmethod
    def _is_card(cls, value: str) -> bool:
        match = cls.CARD.search(value)
        if match is None:
            return False
        digits = re.sub(r"\D", "", match.group(0))
        if not 13 <= len(digits) <= 19:
            return False
        total = 0
        for index, digit in enumerate(reversed(digits)):
            number = int(digit)
            if index % 2:
                number *= 2
                if number > 9:
                    number -= 9
            total += number
        return total % 10 == 0

    @staticmethod
    def _blocked(category: str, location: str | None = None) -> Decision:
        metadata = {"category": category}
        if location is not None:
            metadata["location"] = location
        return Decision(
            decision_type=DecisionType.BLOCK,
            reason=DecisionReason.PII_DETECTED,
            severity=Severity.HIGH,
            metadata=metadata,
        )
