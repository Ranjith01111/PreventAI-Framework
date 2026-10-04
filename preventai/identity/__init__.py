"""Deterministic local identity and authentication primitives."""

from .agent_identity import AgentIdentity, ToolIdentity
from .tool_trust import (
    ToolProvenance, ToolTrustRecord, ToolTrustRegistry, ToolTrustState,
    implementation_digest, validate_tool_version,
)
from .authentication import (
    AuthenticationResult,
    AuthenticationStatus,
    Authenticator,
    Credential,
)
from .delegation import Capability, DelegationGrant, DelegationRegistry
from .lifecycle import AgentLifecycleRegistry, LifecycleError, LifecycleRecord, LifecycleState
from .mission import Mission, MissionBinding, MissionError, MissionRegistry, MissionState
from .admission import (
    AgentAdmissionRecord,
    AgentAdmissionRegistry,
    AgentAdmissionRequest,
    AgentAdmissionState,
    AgentAttestation,
)

__all__ = [
    "AgentIdentity",
    "ToolIdentity",
    "ToolProvenance", "ToolTrustRecord", "ToolTrustRegistry", "ToolTrustState",
    "implementation_digest", "validate_tool_version",
    "Credential",
    "Authenticator",
    "AuthenticationResult",
    "AuthenticationStatus",
    "Capability",
    "DelegationGrant",
    "DelegationRegistry",
    "AgentLifecycleRegistry",
    "LifecycleError",
    "LifecycleRecord",
    "LifecycleState",
    "Mission",
    "MissionBinding",
    "MissionError",
    "MissionRegistry",
    "MissionState",
    "AgentAdmissionState",
    "AgentAttestation",
    "AgentAdmissionRequest",
    "AgentAdmissionRecord",
    "AgentAdmissionRegistry",
]
