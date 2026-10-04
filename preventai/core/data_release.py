"""Deterministic data-release and declassification authorization."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping
from time import monotonic

from .dataflow import FlowClassification, FlowTrust
from .decision import Decision, DecisionReason, DecisionType, Severity


class DestinationTrust(str, Enum):
    INTERNAL = "internal"
    TRUSTED_EXTERNAL = "trusted_external"
    RESTRICTED_EXTERNAL = "restricted_external"
    PUBLIC = "public"
    UNKNOWN = "unknown"


class Transformation(str, Enum):
    NONE = "none"
    REDACTION = "redaction"
    MASKING = "masking"
    AGGREGATION = "aggregation"


_RELEASE_AUTHORITY = object()


@dataclass(frozen=True, slots=True)
class DataReleaseAuthorization:
    application_id: str
    agent_id: str
    session_id: str
    tool_id: str
    operation: str
    tool_name: str
    source: str
    source_classification: FlowClassification
    lineage: tuple[str, ...]
    destination: str
    destination_trust: DestinationTrust
    field_paths: tuple[str, ...]
    transformation: Transformation
    target_classification: FlowClassification | None
    mission_id: str | None
    mission_version: int | None
    policy_version: int
    expires_at: float
    _authority: object

    @classmethod
    def issue(
        cls,
        request: "DataReleaseRequest",
        *,
        application_id: str,
        policy_version: int,
        _authority: object,
        ttl: float = 30.0,
    ) -> "DataReleaseAuthorization":
        if _authority is not _RELEASE_AUTHORITY:
            raise PermissionError("release authorization is framework-issued")
        return cls(
            application_id=application_id,
            agent_id=request.agent_id,
            session_id=request.session_id,
            tool_id=request.tool_id,
            operation=request.operation,
            tool_name=request.tool_name,
            source=request.source,
            source_classification=request.source_classification,
            lineage=request.lineage,
            destination=request.destination,
            destination_trust=request.destination_trust,
            field_paths=request.field_paths,
            transformation=request.transformation,
            target_classification=request.target_classification,
            mission_id=request.mission_id,
            mission_version=request.mission_version,
            policy_version=request.policy_version if request.policy_version is not None else policy_version,
            expires_at=monotonic() + ttl,
            _authority=_authority,
        )

    def matches(
        self,
        request: "DataReleaseRequest",
        *,
        application_id: str,
        policy_version: int | None = None,
    ) -> bool:
        return (
            self._authority is _RELEASE_AUTHORITY
            and self.application_id == application_id
            and monotonic() < self.expires_at
            and self.agent_id == request.agent_id
            and self.session_id == request.session_id
            and self.tool_id == request.tool_id
            and self.operation == request.operation
            and self.tool_name == request.tool_name
            and self.source == request.source
            and self.source_classification is request.source_classification
            and self.lineage == request.lineage
            and self.destination == request.destination
            and self.destination_trust is request.destination_trust
            and self.field_paths == request.field_paths
            and self.transformation is request.transformation
            and self.target_classification is request.target_classification
            and self.mission_id == request.mission_id
            and self.mission_version == request.mission_version
            and (
                (request.policy_version is None and (policy_version is None or self.policy_version == policy_version))
                or (request.policy_version is not None and self.policy_version == request.policy_version)
            )
        )


def verify_release_authorization(
    authorization: DataReleaseAuthorization | None,
    request: DataReleaseRequest | None,
    *,
    application_id: str,
    policy_version: int | None = None,
) -> bool:
    return (
        isinstance(authorization, DataReleaseAuthorization)
        and isinstance(request, DataReleaseRequest)
        and authorization.matches(
            request,
            application_id=application_id,
            policy_version=policy_version,
        )
    )


@dataclass(frozen=True, slots=True)
class DataReleaseRequest:
    agent_id: str
    session_id: str
    source: str
    source_classification: FlowClassification
    lineage: tuple[str, ...]
    destination: str
    destination_trust: DestinationTrust
    operation: str
    tool_id: str
    tool_name: str
    mission_id: str | None = None
    mission_version: int | None = None
    transformation: Transformation = Transformation.NONE
    target_classification: FlowClassification | None = None
    field_paths: tuple[str, ...] = ()
    policy_version: int | None = None

    def __post_init__(self) -> None:
        for field in ("agent_id", "session_id", "source", "destination", "operation", "tool_id", "tool_name"):
            value = getattr(self, field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field} must be non-empty")
        if not isinstance(self.source_classification, FlowClassification):
            raise ValueError("source_classification must be a FlowClassification")
        if not isinstance(self.destination_trust, DestinationTrust):
            raise ValueError("destination_trust must be a DestinationTrust")
        if not isinstance(self.transformation, Transformation):
            raise ValueError("transformation must be a Transformation")
        if not isinstance(self.lineage, tuple) or not self.lineage or any(
            not isinstance(item, str) or not item.strip() for item in self.lineage
        ):
            raise ValueError("lineage must be a non-empty tuple")
        if any(not isinstance(item, str) or not item.strip() for item in self.field_paths):
            raise ValueError("field_paths must contain non-empty strings")
        if self.mission_id is not None and (not isinstance(self.mission_id, str) or not self.mission_id.strip()):
            raise ValueError("mission_id must be a non-empty string when set")
        if self.mission_version is not None and (
            not isinstance(self.mission_version, int)
            or isinstance(self.mission_version, bool)
            or self.mission_version < 1
        ):
            raise ValueError("mission_version must be a positive integer when set")
        if self.policy_version is not None and (
            not isinstance(self.policy_version, int)
            or isinstance(self.policy_version, bool)
            or self.policy_version <= 0
        ):
            raise ValueError("policy_version must be a positive integer when set")
        if self.target_classification is not None and not isinstance(
            self.target_classification, FlowClassification
        ):
            raise ValueError("target_classification must be a FlowClassification")
        if self.transformation is Transformation.NONE and self.target_classification is not None:
            raise ValueError("target classification requires an explicit transformation")


@dataclass(frozen=True, slots=True)
class DataReleasePolicy:
    version: int = 1
    allowed_destinations: tuple[str, ...] = ()
    allowed_destination_trust: tuple[DestinationTrust, ...] = (DestinationTrust.INTERNAL,)
    allowed_classifications: tuple[FlowClassification, ...] = (
        FlowClassification.PUBLIC,
        FlowClassification.INTERNAL,
    )
    allowed_transformations: tuple[Transformation, ...] = ()
    require_approval_for: tuple[FlowClassification, ...] = ()
    allowed_field_paths: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.version, int) or isinstance(self.version, bool) or self.version <= 0:
            raise ValueError("version must be positive")
        if any(not isinstance(value, str) or not value.strip() for value in self.allowed_destinations):
            raise ValueError("allowed_destinations must contain non-empty strings")
        if any(not isinstance(value, DestinationTrust) for value in self.allowed_destination_trust):
            raise ValueError("allowed_destination_trust is invalid")
        if any(not isinstance(value, FlowClassification) for value in self.allowed_classifications):
            raise ValueError("allowed_classifications is invalid")
        if any(not isinstance(value, Transformation) for value in self.allowed_transformations):
            raise ValueError("allowed_transformations is invalid")
        if any(not isinstance(value, FlowClassification) for value in self.require_approval_for):
            raise ValueError("require_approval_for is invalid")
        if any(not isinstance(value, str) or not value.strip() for value in self.allowed_field_paths):
            raise ValueError("allowed_field_paths is invalid")


def evaluate_data_release(
    request: DataReleaseRequest,
    policy: DataReleasePolicy,
) -> Decision:
    if not request.lineage:
        return _blocked("missing_lineage")
    if policy.allowed_destinations and request.destination not in policy.allowed_destinations:
        return _blocked("destination_denied")
    if request.destination_trust not in policy.allowed_destination_trust:
        return _blocked("destination_trust_denied")
    if request.source_classification not in policy.allowed_classifications:
        if (
            request.transformation not in policy.allowed_transformations
            or request.target_classification is None
            or request.target_classification not in policy.allowed_classifications
        ):
            return _blocked("declassification_denied")
    if request.field_paths and (
        not policy.allowed_field_paths
        or any(path not in policy.allowed_field_paths for path in request.field_paths)
    ):
        return _blocked("field_restriction_denied")
    if request.transformation not in policy.allowed_transformations and request.transformation is not Transformation.NONE:
        return _blocked("transformation_denied")
    if request.source_classification in policy.require_approval_for:
        return Decision(
            DecisionType.APPROVAL_REQUIRED,
            DecisionReason.APPROVAL_REQUIRED,
            severity=Severity.HIGH,
            metadata={"stage": "data_release", "classification": request.source_classification.value},
        )
    return Decision(
        DecisionType.ALLOW,
        DecisionReason.REQUEST_ALLOWED,
        metadata={"stage": "data_release", "destination": request.destination},
    )


def _blocked(error: str) -> Decision:
    return Decision(
        DecisionType.BLOCK,
        DecisionReason.DATA_FLOW_VIOLATION,
        severity=Severity.HIGH,
        metadata={"stage": "data_release", "error": error},
    )
