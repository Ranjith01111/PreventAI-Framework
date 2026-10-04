"""Small deterministic schema subset for validating tool arguments."""

from __future__ import annotations

from collections.abc import Mapping
import math
import re
from typing import Any

from ..core.decision import Decision, DecisionReason, DecisionType, Severity


_TYPES = {"object", "string", "integer", "number", "boolean", "array", "null"}
_COMMON_KEYS = {"type", "enum", "minimum", "maximum", "min_length", "max_length", "pattern"}


class SchemaGuard:
    """Validate JSON-like values against the supported PreventAI schema subset."""

    @staticmethod
    def validate(arguments: Any, schema: Any) -> Decision:
        try:
            errors = []
            SchemaGuard._validate_schema(schema, "$")
            SchemaGuard._validate_value(arguments, schema, "$", errors)
            if errors:
                return SchemaGuard._blocked(errors[0])
            return Decision(
                decision_type=DecisionType.ALLOW,
                reason=DecisionReason.REQUEST_ALLOWED,
                metadata={"schema": "validated"},
            )
        except Exception:
            return SchemaGuard._blocked("invalid_schema")

    @staticmethod
    def _blocked(code: str) -> Decision:
        return Decision(
            decision_type=DecisionType.BLOCK,
            reason=DecisionReason.SCHEMA_VALIDATION_FAILED,
            severity=Severity.HIGH,
            metadata={"error": code},
        )

    @staticmethod
    def _validate_schema(schema: Any, path: str) -> None:
        if not isinstance(schema, Mapping):
            raise ValueError("schema must be an object")
        schema_type = schema.get("type")
        if not isinstance(schema_type, str) or schema_type not in _TYPES:
            raise ValueError("schema type is invalid")
        allowed_keys = set(_COMMON_KEYS)
        if schema_type == "object":
            allowed_keys.update({"properties", "required", "additional_properties"})
        if schema_type == "array":
            allowed_keys.add("items")
        if set(schema) - allowed_keys:
            raise ValueError("schema contains unsupported keywords")
        properties = schema.get("properties")
        if schema_type == "object":
            if properties is not None:
                if not isinstance(properties, Mapping):
                    raise ValueError("properties must be an object")
                for name, child in properties.items():
                    if not isinstance(name, str) or not name:
                        raise ValueError("property names must be non-empty strings")
                    SchemaGuard._validate_schema(child, f"{path}.{name}")
            required = schema.get("required", [])
            if not isinstance(required, list) or any(
                not isinstance(name, str) or not name for name in required
            ):
                raise ValueError("required must be a list of names")
            if len(required) != len(set(required)):
                raise ValueError("required names must be unique")
            additional = schema.get("additional_properties", True)
            if not isinstance(additional, bool):
                raise ValueError("additional_properties must be boolean")
        elif properties is not None or "required" in schema or "additional_properties" in schema:
            raise ValueError("object constraints require type object")
        if "enum" in schema:
            enum = schema["enum"]
            if not isinstance(enum, list) or not enum:
                raise ValueError("enum must be a non-empty list")
            for item in enum:
                SchemaGuard._json_value(item)
        for name in ("minimum", "maximum"):
            if name in schema and (
                not isinstance(schema[name], (int, float))
                or isinstance(schema[name], bool)
                or not math.isfinite(schema[name])
            ):
                raise ValueError(f"{name} must be a finite number")
        if "minimum" in schema and "maximum" in schema and schema["minimum"] > schema["maximum"]:
            raise ValueError("minimum cannot exceed maximum")
        for name in ("min_length", "max_length"):
            if name in schema and (
                not isinstance(schema[name], int)
                or isinstance(schema[name], bool)
                or schema[name] < 0
            ):
                raise ValueError(f"{name} must be a non-negative integer")
        if "min_length" in schema and "max_length" in schema and schema["min_length"] > schema["max_length"]:
            raise ValueError("min_length cannot exceed max_length")
        if "pattern" in schema:
            if not isinstance(schema["pattern"], str):
                raise ValueError("pattern must be a string")
            re.compile(schema["pattern"])
        if schema_type == "array" and "items" in schema:
            SchemaGuard._validate_schema(schema["items"], f"{path}[]")

    @staticmethod
    def _validate_value(value: Any, schema: Mapping[str, Any], path: str, errors: list[str]) -> None:
        schema_type = schema["type"]
        valid_type = {
            "object": isinstance(value, Mapping),
            "string": isinstance(value, str),
            "integer": isinstance(value, int) and not isinstance(value, bool),
            "number": isinstance(value, (int, float)) and not isinstance(value, bool),
            "boolean": isinstance(value, bool),
            "array": isinstance(value, list),
            "null": value is None,
        }[schema_type]
        if schema_type == "number" and valid_type:
            valid_type = math.isfinite(value)
        if not valid_type:
            errors.append(f"{path}.type")
            return
        if "enum" in schema and value not in schema["enum"]:
            errors.append(f"{path}.enum")
        if schema_type in {"integer", "number"}:
            if "minimum" in schema and value < schema["minimum"]:
                errors.append(f"{path}.minimum")
            if "maximum" in schema and value > schema["maximum"]:
                errors.append(f"{path}.maximum")
        if schema_type == "string":
            if "min_length" in schema and len(value) < schema["min_length"]:
                errors.append(f"{path}.min_length")
            if "max_length" in schema and len(value) > schema["max_length"]:
                errors.append(f"{path}.max_length")
            if "pattern" in schema and re.search(schema["pattern"], value) is None:
                errors.append(f"{path}.pattern")
        if schema_type == "object":
            properties = schema.get("properties", {})
            required = schema.get("required", [])
            for name in required:
                if name not in value:
                    errors.append(f"{path}.{name}.required")
            if schema.get("additional_properties", True) is False:
                for name in value:
                    if name not in properties:
                        errors.append(f"{path}.{name}.additional")
            for name, child in properties.items():
                if name in value:
                    SchemaGuard._validate_value(value[name], child, f"{path}.{name}", errors)
        if schema_type == "array" and "items" in schema:
            for index, item in enumerate(value):
                SchemaGuard._validate_value(item, schema["items"], f"{path}[{index}]", errors)

    @staticmethod
    def _json_value(value: Any) -> None:
        if value is None or isinstance(value, (str, bool, int)):
            return
        if isinstance(value, float) and math.isfinite(value):
            return
        if isinstance(value, list):
            for item in value:
                SchemaGuard._json_value(item)
            return
        if isinstance(value, Mapping):
            for key, item in value.items():
                if not isinstance(key, str):
                    raise ValueError("object keys must be strings")
                SchemaGuard._json_value(item)
            return
        raise ValueError("value is not JSON-compatible")
