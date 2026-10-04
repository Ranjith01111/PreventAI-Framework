"""Validated, side-effect-free input context for policy evaluators."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

from ..protocol.security_request import ContractValidationError, SecurityRequest


def validate_request(request: Any) -> SecurityRequest:
    """Validate the engine boundary and return a trusted request instance."""

    if not isinstance(request, SecurityRequest):
        raise ContractValidationError("engine input must be a SecurityRequest")
    # Re-serialize nested data to validate the complete immutable contract
    # boundary without permitting policy code to observe mutable caller state.
    SecurityRequest.from_dict(request.to_dict())
    return request


@dataclass(frozen=True, slots=True)
class EvaluationContext:
    """A stable request snapshot passed to policy evaluators."""

    request: SecurityRequest
    request_payload: str

    @classmethod
    def from_request(cls, request: Any) -> EvaluationContext:
        validated = validate_request(request)
        return cls(validated, validated.to_json())

    @property
    def action_name(self) -> str:
        return self.request.action.name

    @property
    def resource_id(self) -> str:
        return self.request.resource.resource_id

    @property
    def context(self) -> Mapping[str, Any]:
        return MappingProxyType(dict(self.request.context))
