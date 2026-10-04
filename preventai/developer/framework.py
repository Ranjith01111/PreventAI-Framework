"""Simple synchronous framework APIs backed by the security engine."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from contextlib import contextmanager
from functools import wraps
from enum import Enum
import hashlib
import inspect
import ipaddress
import json
from typing import Any, Callable, Iterator, Mapping
from uuid import uuid5, NAMESPACE_URL

from ..core.context import EvaluationContext
from ..core.decision import Decision, DecisionReason, DecisionType, Severity
from ..core.engine import SecurityEngine
from ..core.permissions import PermissionCategory, PermissionEngine
from ..core.policy import DefaultDenyPolicy, PolicyEvaluator
from ..core.integrity import (
    PolicyIntegrityRegistry,
    _APPLICATION_CONTEXT_TOKEN as _POLICY_INTEGRITY_CONTEXT_TOKEN,
)
from ..core.data_release import (
    DataReleaseAuthorization, DataReleasePolicy, DataReleaseRequest, DestinationTrust,
    _RELEASE_AUTHORITY, evaluate_data_release,
)
from ..guards.schema import SchemaGuard
from ..guards.secrets import SecretsGuard
from ..guards.pii import PIIGuard
from ..guards.injection import InjectionGuard
from ..guards.content import ContentGuard
from ..guards.output import OutputGuard
from ..core.dataflow import (
    DataFlow,
    DataFlowDestination,
    DataFlowEngine,
    DataSource,
)
from ..approval import Approval, ApprovalManager, ApprovalPolicy
from ..audit import AuditEventType, AuditLogger
from ..runtime import RuntimeController, RuntimePolicy
from ..runtime import ExecutionSignature
from ..tools import Sandbox, SandboxCommand, SandboxPolicy
from ..tools.sandbox import _FRAMEWORK_ISSUER
from ..tools import Network, NetworkExecutionDenied, NetworkPolicy, NetworkRequest
from ..tools.network import _FRAMEWORK_ISSUER as _NETWORK_FRAMEWORK_ISSUER
from ..tools import FileSystem, FileSystemExecutionDenied, FileSystemPolicy, FileSystemRequest
from ..tools import Database, DatabaseExecutionDenied, DatabasePolicy, DatabaseRequest
from ..core.dataflow import FlowClassification, FlowTrust, propagate_classification, propagate_trust
from ..identity import (
    AgentIdentity as RegisteredAgentIdentity,
    Authenticator,
    AgentAdmissionRegistry,
    AgentAdmissionState,
    AgentLifecycleRegistry,
    Capability,
    Credential,
    DelegationGrant,
    DelegationRegistry,
    LifecycleState,
    Mission,
    MissionBinding,
    MissionRegistry,
    ToolProvenance,
    ToolTrustRegistry,
    implementation_digest,
    validate_tool_version,
)
from ..identity.tool_trust import _APPLICATION_CONTEXT_TOKEN as _TOOL_TRUST_CONTEXT_TOKEN
from ..identity.admission import _APPLICATION_CONTEXT_TOKEN as _ADMISSION_CONTEXT_TOKEN
import time
from ..protocol.security_request import (
    Action,
    AgentIdentity as ProtocolAgentIdentity,
    DataClassification,
    DataMetadata,
    RequestedBy,
    Resource,
    SecurityRequest,
    SessionIdentity,
    SourceTrust,
    Tool as ProtocolTool,
)


class ToolExecutionDenied(PermissionError):
    """Raised when the security engine blocks a tool invocation."""

    def __init__(self, decision: Decision) -> None:
        self.decision = decision
        super().__init__(f"tool execution blocked: {decision.reason.value}")


class ToolApprovalRequired(PermissionError):
    """Raised when a tool requires approval before it can execute."""

    def __init__(self, decision: Decision) -> None:
        self.decision = decision
        super().__init__("tool execution requires approval")


def _blocked_release(error: str) -> Decision:
    return Decision(
        DecisionType.BLOCK,
        DecisionReason.DATA_FLOW_VIOLATION,
        severity=Severity.HIGH,
        metadata={"stage": "data_release", "error": error},
    )


class AppState(str, Enum):
    NEW = "new"
    READY = "ready"
    RUNNING = "running"


def _stable_id(namespace: str, name: str) -> str:
    return str(uuid5(NAMESPACE_URL, f"preventai:{namespace}:{name}"))


def _agent_admission_configuration_digest(
    *,
    application_id: str,
    agent_id: str,
    version: str = "1.0.0",
    provenance: str = "framework",
    session_id: str | None = None,
    parent_agent_id: str | None = None,
    mission_id: str | None = None,
    mission_version: int | None = None,
    capabilities: tuple[str, ...] = (),
    tools: tuple[str, ...] = (),
    policy_version: str | None = None,
) -> str:
    return hashlib.sha256(
        json.dumps(
            {
                "application_id": application_id,
                "agent_id": agent_id,
                "version": version,
                "provenance": provenance,
                "session_id": session_id,
                "parent_agent_id": parent_agent_id,
                "mission_id": mission_id,
                "mission_version": mission_version,
                "capabilities": list(capabilities),
                "tools": list(tools),
                "policy_version": policy_version,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _action_parameters(args: tuple[Any, ...], kwargs: Mapping[str, Any], *, filesystem: bool) -> dict[str, Any]:
    if not filesystem:
        return {"args": list(args), "kwargs": dict(kwargs)}

    def summarize(value: Any) -> Any:
        if isinstance(value, bytes):
            return {
                "sha256": hashlib.sha256(value).hexdigest(),
                "length": len(value),
            }
        if isinstance(value, list):
            return [summarize(item) for item in value]
        if isinstance(value, tuple):
            return [summarize(item) for item in value]
        if isinstance(value, dict):
            return {key: summarize(item) for key, item in value.items()}
        return value

    return summarize({"args": list(args), "kwargs": dict(kwargs)})


def _guard_arguments(arguments: Mapping[str, Any], *, filesystem: bool) -> Mapping[str, Any]:
    if not filesystem:
        return arguments

    def decode(value: Any) -> Any:
        if isinstance(value, bytes):
            return value.decode("utf-8")
        if isinstance(value, list):
            return [decode(item) for item in value]
        if isinstance(value, tuple):
            return [decode(item) for item in value]
        if isinstance(value, dict):
            return {key: decode(item) for key, item in value.items()}
        return value

    return decode(dict(arguments))


class Tool:
    """A registered synchronous callable whose invocation is security-gated."""

    def __init__(
        self,
        name: str,
        function: Callable[..., Any],
        tool_id: str | None = None,
        input_schema: Mapping[str, Any] | None = None,
        content_guard: ContentGuard | None = None,
        output_schema: Mapping[str, Any] | None = None,
        output_guard: OutputGuard | None = None,
        data_source: DataSource | None = None,
        data_destination: DataFlowDestination | None = None,
        runtime_policy: RuntimePolicy | None = None,
        sandbox_policy: SandboxPolicy | None = None,
        network_policy: NetworkPolicy | None = None,
        filesystem_policy: FileSystemPolicy | None = None,
        database_policy: DatabasePolicy | None = None,
        version: str = "1.0.0",
        provenance: ToolProvenance | None = None,
        runtime_controller: RuntimeController | None = None,
    ) -> None:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("tool name must be a non-empty string")
        if not callable(function):
            raise TypeError("tool function must be callable")
        self.name = name
        self.tool_id = tool_id or _stable_id("tool", name)
        self.version = validate_tool_version(version)
        self._tool_trust_declared = version != "1.0.0" or provenance is not None
        self.provenance = provenance or ToolProvenance(
            provider="local", source=f"{function.__module__}:{function.__qualname__}"
        )
        self.integrity_digest = implementation_digest(function)
        self._tool_trust_registry: ToolTrustRegistry | None = None
        self._require_tool_trust = True
        self.input_schema = input_schema
        if content_guard is not None and not isinstance(content_guard, ContentGuard):
            raise TypeError("content_guard must be a ContentGuard")
        self._content_guard = content_guard or ContentGuard()
        if output_guard is not None and not isinstance(output_guard, OutputGuard):
            raise TypeError("output_guard must be an OutputGuard")
        self._output_guard = (
            output_guard
            if output_guard is not None
            else OutputGuard(output_schema=output_schema)
        )
        if data_source is not None and not isinstance(data_source, DataSource):
            raise TypeError("data_source must be a DataSource")
        if data_destination is not None and not isinstance(data_destination, DataFlowDestination):
            raise TypeError("data_destination must be a DataFlowDestination")
        self._data_source = data_source
        self._data_destination = data_destination
        self._data_flow_engine: DataFlowEngine | None = None
        if runtime_policy is not None and not isinstance(runtime_policy, RuntimePolicy):
            raise TypeError("runtime_policy must be a RuntimePolicy")
        if runtime_controller is not None and not isinstance(runtime_controller, RuntimeController):
            raise TypeError("runtime_controller must be a RuntimeController")
        self._runtime_policy = runtime_policy
        self._runtime_controller = runtime_controller
        if sandbox_policy is not None and not isinstance(sandbox_policy, SandboxPolicy):
            raise TypeError("sandbox_policy must be a SandboxPolicy")
        self._sandbox_policy = sandbox_policy
        if network_policy is not None and not isinstance(network_policy, NetworkPolicy):
            raise TypeError("network_policy must be a NetworkPolicy")
        if filesystem_policy is not None and not isinstance(filesystem_policy, FileSystemPolicy):
            raise TypeError("filesystem_policy must be a FileSystemPolicy")
        if database_policy is not None and not isinstance(database_policy, DatabasePolicy):
            raise TypeError("database_policy must be a DatabasePolicy")
        if sum(policy is not None for policy in (sandbox_policy, network_policy, filesystem_policy, database_policy)) > 1:
            raise ValueError("a tool can use only one restricted execution boundary")
        self._network_policy = network_policy
        self._filesystem_policy = filesystem_policy
        self._database_policy = database_policy
        self._network: Network | None = None
        self._filesystem: FileSystem | None = None
        self._database: Database | None = None
        self._filesystem_authorizer: Callable[..., Any] | None = None
        self._sandbox: Sandbox | None = None
        self._authenticator: Authenticator | None = None
        self._lifecycle_registry: AgentLifecycleRegistry | None = None
        self._mission_registry: MissionRegistry | None = None
        self._delegation_registry: DelegationRegistry | None = None
        self._approval_manager: ApprovalManager | None = None
        self._approval_policy: ApprovalPolicy | None = None
        self._approval: Approval | None = None
        self._pending_approval_request = None
        self._release_approval: Approval | None = None
        self._pending_release_approval_request = None
        self._release_approval_policy_version: int | None = None
        self._audit_logger: AuditLogger | None = None
        self._function = function
        self._engine: SecurityEngine | None = None
        self._data_flow_engine: DataFlowEngine | None = None
        self._runtime_controller: RuntimeController | None = None
        self._data_release_policy: DataReleasePolicy | None = None
        self._release_authorization: DataReleaseAuthorization | None = None
        self._release_request: DataReleaseRequest | None = None
        self._agent: Agent | None = None
        self._admission_registry: AgentAdmissionRegistry | None = None
        wraps(function)(self)

    def _bind_tool_trust(self, registry: ToolTrustRegistry, *, required: bool) -> None:
        if not isinstance(registry, ToolTrustRegistry):
            raise TypeError("tool trust registry is required")
        self._tool_trust_registry = registry
        self._require_tool_trust = True

    def _require_active_admission(
        self,
        *,
        mission_binding: MissionBinding | None = None,
        session_id: str | None = None,
        application_id: str | None = None,
        policy_version: str | None = None,
    ) -> None:
        if self._agent is None:
            raise ToolExecutionDenied(
                Decision(
                    DecisionType.BLOCK,
                    DecisionReason.INVALID_REQUEST,
                    severity=Severity.CRITICAL,
                    metadata={"stage": "agent_admission", "error": "agent_unbound"},
                )
            )
        self._admission_registry = getattr(self._agent, "_admission_registry", self._admission_registry)
        if self._admission_registry is None:
            raise ToolExecutionDenied(
                Decision(
                    DecisionType.BLOCK,
                    DecisionReason.INVALID_REQUEST,
                    severity=Severity.CRITICAL,
                    metadata={"stage": "agent_admission", "error": "registry_missing"},
                )
            )
        current_application_id = application_id or getattr(self._agent, "_application_id", "preventai")
        if policy_version is None:
            try:
                policy_version = self._agent._policy_integrity.get_active_version(
                    "application-security"
                )
            except Exception as exc:
                raise ToolExecutionDenied(
                    Decision(
                        DecisionType.BLOCK,
                        DecisionReason.INVALID_REQUEST,
                        severity=Severity.CRITICAL,
                        metadata={"stage": "agent_admission", "error": "policy_integrity"},
                    )
                ) from exc
        decision = self._admission_registry.verify(
            agent_id=self._agent.agent_id,
            application_id=current_application_id,
            session_id=session_id or self._agent.session_id,
            mission_id=mission_binding.mission_id if mission_binding is not None else None,
            mission_version=mission_binding.mission_version if mission_binding is not None else None,
            capabilities=(),
            tools=(self.name,),
            version="1.0.0",
            provenance="framework",
            configuration_digest=_agent_admission_configuration_digest(
                application_id=current_application_id,
                agent_id=self._agent.agent_id,
                version="1.0.0",
                provenance="framework",
                session_id=self._agent.session_id,
                parent_agent_id=self._admission_registry.get(self._agent.agent_id).parent_agent_id,
                mission_id=self._admission_registry.get(self._agent.agent_id).mission_id,
                mission_version=self._admission_registry.get(self._agent.agent_id).mission_version,
                capabilities=self._admission_registry.get(self._agent.agent_id).capability_envelope,
                tools=self._admission_registry.get(self._agent.agent_id).tool_set,
                policy_version=policy_version,
            ),
            policy_version=policy_version,
        )
        if decision.decision_type is not DecisionType.ALLOW:
            raise ToolExecutionDenied(decision)
        record = self._admission_registry.get(self._agent.agent_id)
        if record is None or record.state not in {
            AgentAdmissionState.ADMITTED,
            AgentAdmissionState.ACTIVE,
        }:
            raise ToolExecutionDenied(
                Decision(
                    DecisionType.BLOCK,
                    DecisionReason.INVALID_REQUEST,
                    severity=Severity.CRITICAL,
                    metadata={"stage": "agent_admission", "error": "admission_state_invalid"},
                )
            )

    def _check_tool_trust(self) -> None:
        if not self._require_tool_trust:
            return
        if self._tool_trust_registry is None:
            raise ToolExecutionDenied(Decision(
                DecisionType.BLOCK, DecisionReason.INVALID_REQUEST,
                severity=Severity.CRITICAL,
                metadata={"stage": "tool_trust", "error": "registry_missing"},
            ))
        try:
            current_integrity_digest = implementation_digest(self._function)
            allowed, reason = self._tool_trust_registry.check(
                tool_id=self.tool_id, name=self.name, version=self.version,
                provenance=self.provenance, integrity_digest=current_integrity_digest,
            )
        except Exception as exc:
            raise ToolExecutionDenied(Decision(
                DecisionType.BLOCK, DecisionReason.INVALID_REQUEST,
                severity=Severity.CRITICAL,
                metadata={"stage": "tool_trust", "error": "resolution_failed"},
            )) from exc
        if not allowed:
            raise ToolExecutionDenied(Decision(
                DecisionType.BLOCK, DecisionReason.INVALID_REQUEST,
                severity=Severity.HIGH,
                metadata={"stage": "tool_trust", "error": reason},
            ))

    def _check_data_release(
        self,
        mission_binding: MissionBinding | None,
        *,
        destination: str | None = None,
        destination_trust: DestinationTrust = DestinationTrust.INTERNAL,
        source_classification: FlowClassification | None = None,
        lineage: tuple[str, ...] | None = None,
        field_paths: tuple[str, ...] = (),
        bind_authorization: bool = False,
    ) -> None:
        if self._data_release_policy is None or self._data_source is None or self._data_destination is None:
            return
        request = DataReleaseRequest(
            agent_id=self._agent.agent_id,
            session_id=self._agent.session_id,
            source=self._data_source.name,
            source_classification=source_classification or self._data_source.classification,
            lineage=lineage or (self._data_source.name,),
            destination=destination or self._data_destination.name,
            destination_trust=destination_trust,
            operation=self.name,
            tool_id=self.tool_id,
            tool_name=self.name,
            mission_id=mission_binding.mission_id if mission_binding else None,
            mission_version=mission_binding.mission_version if mission_binding else None,
            field_paths=field_paths,
            policy_version=self._data_release_policy.version,
        )
        decision = evaluate_data_release(request, self._data_release_policy)
        if bind_authorization and (
            self._release_authorization is not None
            and not self._release_authorization.matches(
                request,
                application_id=(
                    self._agent._application_id
                    if self._data_release_policy is not None
                    else None
                ),
            )
        ):
            self._audit(
                AuditEventType.DATA_FORGED_CONTEXT,
                agent_id=self._agent.agent_id,
                session_id=self._agent.session_id,
                tool_id=self.tool_id,
                action=self.name,
                resource=request.destination,
                decision=DecisionType.BLOCK.value,
                reason="release_context_mismatch",
            )
            raise ToolExecutionDenied(_blocked_release("release_context_mismatch"))
        audit_fields = {
            "agent_id": self._agent.agent_id,
            "session_id": self._agent.session_id,
            "tool_id": self.tool_id,
            "action": self.name,
            "resource": destination or self._data_destination.name,
            "decision": decision.decision_type.value,
            "reason": decision.reason.value,
            "severity": decision.severity.value if decision.severity else None,
            "metadata": {"destination": destination or self._data_destination.name},
        }
        self._audit(AuditEventType.DATA_RELEASE_EVALUATED, **audit_fields)
        if decision.decision_type is not DecisionType.ALLOW:
            error = str(decision.metadata.get("error", ""))
            event_type = AuditEventType.DATA_RELEASE_BLOCKED
            if decision.decision_type is DecisionType.APPROVAL_REQUIRED:
                event_type = AuditEventType.DATA_RELEASE_APPROVAL_REQUESTED
            elif error == "destination_denied" or error == "destination_trust_denied":
                event_type = AuditEventType.DATA_DESTINATION_DENIED
            elif error == "missing_lineage":
                event_type = AuditEventType.DATA_LINEAGE_FAILURE
            elif error == "declassification_denied":
                event_type = AuditEventType.DATA_DECLASSIFICATION_DENIED
            elif error in {"transformation_denied", "field_restriction_denied"}:
                event_type = AuditEventType.DATA_CLASSIFICATION_MISMATCH
            self._audit(event_type, **audit_fields)
            if decision.decision_type is DecisionType.APPROVAL_REQUIRED:
                if self._approval_manager is not None and self._release_approval is not None:
                    if (
                        self._release_approval_policy_version is None
                        or self._release_approval_policy_version != self._data_release_policy.version
                    ):
                        self._audit(
                            AuditEventType.DATA_RELEASE_TOCTOU_FAILURE,
                            agent_id=self._agent.agent_id,
                            session_id=self._agent.session_id,
                            tool_id=self.tool_id,
                            action=self.name,
                            resource=request.destination,
                            decision=DecisionType.BLOCK.value,
                            reason="release_policy_version_changed",
                        )
                        raise ToolExecutionDenied(_blocked_release("release_policy_changed"))
                    approval_decision = self._approval_manager.verify_release(
                        self._release_approval,
                        request,
                        authenticated_identity=self._agent.agent_id,
                        consume=bind_authorization,
                    )
                    if approval_decision.decision_type is not DecisionType.ALLOW:
                        raise ToolExecutionDenied(approval_decision)
                    decision = approval_decision
                    audit_fields["decision"] = DecisionType.ALLOW.value
                    audit_fields["reason"] = DecisionReason.REQUEST_ALLOWED.value
                else:
                    if self._approval_manager is not None:
                        self._pending_release_approval_request = (
                            self._approval_manager.request_release(
                                request,
                                identity=self._agent.agent_id,
                                reason=decision.reason.value,
                                expires_at=datetime.now(timezone.utc) + timedelta(minutes=5),
                            )
                        )
                        self._release_approval_policy_version = self._data_release_policy.version
                    raise ToolApprovalRequired(decision)
            if decision.decision_type is not DecisionType.ALLOW:
                raise ToolExecutionDenied(decision)
        if bind_authorization:
            authorization = DataReleaseAuthorization.issue(
                request,
                application_id=(
                    self._agent._application_id
                    if self._data_release_policy is not None
                    else None
                ),
                policy_version=self._data_release_policy.version,
                _authority=_RELEASE_AUTHORITY,
            )
        else:
            authorization = None
        if bind_authorization and (
            self._release_authorization is not None
            and self._release_authorization.policy_version != self._data_release_policy.version
        ):
            self._audit(
                AuditEventType.DATA_RELEASE_TOCTOU_FAILURE,
                agent_id=self._agent.agent_id,
                session_id=self._agent.session_id,
                tool_id=self.tool_id,
                action=self.name,
                resource=request.destination,
                decision=DecisionType.BLOCK.value,
                reason="release_policy_version_changed",
            )
            raise ToolExecutionDenied(_blocked_release("release_policy_changed"))
        if authorization is not None:
            self._release_authorization = authorization
            self._release_request = request
        self._audit(AuditEventType.DATA_RELEASE_ALLOWED, **audit_fields)
    def _bind(
        self,
        agent: Agent,
        engine: SecurityEngine,
        data_flow_engine: DataFlowEngine | None = None,
        audit_logger: AuditLogger | None = None,
    ) -> None:
        if self._agent is not None and self._agent is not agent:
            raise ValueError("tool is already bound to another agent")
        self._agent = agent
        self._engine = engine
        self._data_flow_engine = data_flow_engine
        self._audit_logger = audit_logger
        self._admission_registry = getattr(agent, "_admission_registry", None)
        if self._sandbox_policy is not None:
            self._sandbox = Sandbox(
                self._sandbox_policy,
                audit_logger=audit_logger,
            )
        if self._network_policy is not None:
            self._network = Network(self._network_policy, audit_logger=audit_logger)
            self._network._release_policy_version_provider = (
                lambda: self._data_release_policy.version
                if self._data_release_policy is not None
                else None
            )
        if self._filesystem_policy is not None:
            self._filesystem = FileSystem(self._filesystem_policy, audit_logger=audit_logger)
            self._filesystem_authorizer = self._filesystem._bind_framework_authorizer(self)
            self._filesystem._release_policy_version_provider = (
                lambda: self._data_release_policy.version
                if self._data_release_policy is not None
                else None
            )
        if self._database_policy is not None:
            self._database = Database(self._database_policy)
            self._database._release_policy_version_provider = (
                lambda: self._data_release_policy.version
                if self._data_release_policy is not None
                else None
            )

    def use_approval(self, approval: Approval) -> None:
        if not isinstance(approval, Approval):
            raise TypeError("approval must be an Approval")
        if self._approval_manager is None:
            raise RuntimeError("tool is not registered with an ApprovalManager")
        self._approval = approval

    def use_release_approval(self, approval: Approval) -> None:
        if not isinstance(approval, Approval):
            raise TypeError("approval must be an Approval")
        if self._approval_manager is None:
            raise RuntimeError("tool is not registered with an ApprovalManager")
        self._release_approval = approval

    @property
    def pending_approval_request(self):
        return self._pending_approval_request

    @property
    def pending_release_approval_request(self):
        return self._pending_release_approval_request

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if self._engine is None or self._agent is None:
            raise ToolExecutionDenied(
                Decision(
                    decision_type=DecisionType.BLOCK,
                    reason=DecisionReason.INVALID_REQUEST,
                    metadata={"error": "tool is not registered with a PreventApp"},
                )
            )
        return self.execute(*args, **kwargs)

    def execute(self, *args: Any, **kwargs: Any) -> Any:
        if (
            self._engine is None
            or self._agent is None
            or self._agent._app_state is not AppState.RUNNING
        ):
            raise ToolExecutionDenied(
                Decision(
                    decision_type=DecisionType.BLOCK,
                    reason=DecisionReason.INVALID_REQUEST,
                    metadata={"error": "tool is not registered with a PreventApp"},
                )
            )
        self._check_tool_trust()
        self._require_active_admission(
            session_id=self._agent.session_id,
            application_id=self._agent._application_id,
        )
        if self._authenticator is not None:
                try:
                    auth = self._authenticator.authenticate(
                        self._agent.agent_id,
                        self._agent._credential,
                        session_id=self._agent.session_id,
                    )
                    if not auth.authenticated:
                        raise ToolExecutionDenied(auth.to_decision())
                except ToolExecutionDenied:
                    raise
                except Exception:
                    raise ToolExecutionDenied(
                        Decision(
                            decision_type=DecisionType.BLOCK,
                            reason=DecisionReason.AUTHENTICATION_ERROR,
                            severity=Severity.CRITICAL,
                            metadata={"stage": "authentication"},
                        )
                    )
        self._require_active_lifecycle("tool_execution")
        mission_binding = (
            self._mission_registry.binding_for(self._agent.agent_id)
            if self._mission_registry is not None
            else None
        )
        try:
            request = SecurityRequest(
                protocol_version="1.0",
                request_id=_stable_id("request", f"{self._agent.agent_id}:{self.name}"),
                timestamp=datetime.now(timezone.utc),
                agent=ProtocolAgentIdentity(self._agent.agent_id, "developer_agent"),
                session=SessionIdentity(self._agent.session_id),
                requested_by=RequestedBy.AGENT,
                action=Action(
                    self.name,
                    _action_parameters(
                        args, kwargs, filesystem=self._filesystem_policy is not None
                    ),
                ),
                resource=Resource(self.tool_id, "tool"),
                tool=ProtocolTool(self.name, "1.0"),
                data=DataMetadata(DataClassification.INTERNAL, SourceTrust.TRUSTED),
                context={"framework": "preventai"},
                metadata={
                    "agent_id": self._agent.agent_id,
                    "mission_id": mission_binding.mission_id if mission_binding else None,
                    "mission_version": (
                        mission_binding.mission_version if mission_binding else None
                    ),
                },
            )
        except Exception:
            raise ToolExecutionDenied(
                Decision(
                    decision_type=DecisionType.BLOCK,
                    reason=DecisionReason.INVALID_REQUEST,
                    metadata={"error": "request_construction"},
                )
            )
        correlation_id = request.request_id
        self._audit(
            AuditEventType.TOOL_EXECUTION_STARTED,
            correlation_id=correlation_id,
            agent_id=self._agent.agent_id,
            session_id=self._agent.session_id,
            tool_id=self.tool_id,
            action=self.name,
            resource=self.tool_id,
        )
        self._require_valid_mission(
            operation="tool_execution",
            action=request.action.name,
            resource=request.resource.resource_id,
            tool=request.tool.name,
            destination=None,
            data=request.data.classification.value,
            context=None,
            defer_destination=self._network is not None,
            defer_data=any((self._network, self._filesystem, self._database)),
            defer_context=any((self._network, self._filesystem, self._database)),
            mission_binding=mission_binding,
        )
        if self._agent._delegation is not None:
            delegation_decision = self._agent._delegation_registry.evaluate(
                self._agent._delegation,
                authenticated_identity=self._agent.agent_id,
                action=request.action.name,
                resource=request.resource.resource_id,
                tool=request.tool.name,
                session_id=self._agent.session_id,
                context=request.context,
            )
            if delegation_decision.decision_type is not DecisionType.ALLOW:
                raise ToolExecutionDenied(delegation_decision)
        approval_rule = (
            self._approval_policy.match(EvaluationContext.from_request(request))
            if self._approval_policy is not None
            else None
        )
        decision = self._engine.evaluate(request)
        self._audit_decision(AuditEventType.POLICY_EVALUATION, decision, request)
        if approval_rule is not None and decision.decision_type is DecisionType.ALLOW:
            decision = Decision(
                DecisionType.APPROVAL_REQUIRED,
                DecisionReason.APPROVAL_REQUIRED,
                rule_id=approval_rule.rule_id,
            )
        filesystem_approval_required = (
            self._filesystem_policy is not None
            and decision.decision_type is DecisionType.APPROVAL_REQUIRED
        )
        database_approval_required = (
            self._database_policy is not None
            and decision.decision_type is DecisionType.APPROVAL_REQUIRED
        )
        if decision.decision_type is DecisionType.APPROVAL_REQUIRED and not filesystem_approval_required and not database_approval_required:
            approval_decision = self._verify_approval(request, consume=False)
            if approval_decision is not None:
                if approval_decision.decision_type is not DecisionType.ALLOW:
                    raise ToolApprovalRequired(decision) if self._approval is None else ToolExecutionDenied(approval_decision)
            else:
                if self._approval_manager is not None:
                    self._pending_approval_request = self._approval_manager.request(
                        request,
                        identity=self._agent.agent_id,
                        reason=decision.reason.value,
                        expires_at=datetime.now(timezone.utc) + timedelta(minutes=5),
                        policy_rule_id=decision.rule_id,
                    )
                raise ToolApprovalRequired(decision)
        if decision.decision_type is not DecisionType.ALLOW and not filesystem_approval_required:
            raise ToolExecutionDenied(decision)
        try:
            arguments = dict(inspect.signature(self._function).bind(*args, **kwargs).arguments)
        except Exception:
            if self.input_schema is not None:
                raise ToolExecutionDenied(
                    Decision(
                        decision_type=DecisionType.BLOCK,
                        reason=DecisionReason.SCHEMA_VALIDATION_FAILED,
                        metadata={"error": "invalid_arguments"},
                    )
                )
            arguments = {"args": list(args), "kwargs": dict(kwargs)}
        if self.input_schema is not None:
            try:
                schema_decision = SchemaGuard.validate(arguments, self.input_schema)
            except Exception:
                schema_decision = Decision(
                    decision_type=DecisionType.BLOCK,
                    reason=DecisionReason.SCHEMA_VALIDATION_FAILED,
                    metadata={"error": "validator_error"},
                )
            if schema_decision.decision_type is not DecisionType.ALLOW:
                self._audit_decision(AuditEventType.SCHEMA_VALIDATION, schema_decision, request)
                raise ToolExecutionDenied(schema_decision)
        guard_arguments = _guard_arguments(
            arguments, filesystem=self._filesystem_policy is not None
        )
        try:
            secrets_decision = SecretsGuard().scan(guard_arguments)
        except Exception:
            secrets_decision = Decision(
                decision_type=DecisionType.BLOCK,
                reason=DecisionReason.SECRETS_DETECTED,
                metadata={"category": "scanner_error"},
            )
        if secrets_decision.decision_type is not DecisionType.ALLOW:
            self._audit_decision(AuditEventType.SECRET_DETECTION, secrets_decision, request)
            raise ToolExecutionDenied(secrets_decision)
        try:
            pii_decision = PIIGuard().scan(guard_arguments)
        except Exception:
            pii_decision = Decision(
                decision_type=DecisionType.BLOCK,
                reason=DecisionReason.PII_DETECTED,
                metadata={"category": "scanner_error"},
            )
        if pii_decision.decision_type is not DecisionType.ALLOW:
            self._audit_decision(AuditEventType.PII_DETECTION, pii_decision, request)
            raise ToolExecutionDenied(pii_decision)
        try:
            injection_decision = InjectionGuard().scan(guard_arguments)
        except Exception:
            injection_decision = Decision(
                decision_type=DecisionType.BLOCK,
                reason=DecisionReason.INJECTION_DETECTED,
                metadata={"category": "scanner_error"},
            )
        if injection_decision.decision_type is not DecisionType.ALLOW:
            self._audit_decision(AuditEventType.INJECTION_DETECTION, injection_decision, request)
            raise ToolExecutionDenied(injection_decision)
        try:
            content_decision = self._content_guard.scan(guard_arguments)
        except Exception:
            content_decision = Decision(
                decision_type=DecisionType.BLOCK,
                reason=DecisionReason.CONTENT_POLICY_VIOLATION,
                metadata={"category": "scanner_error"},
            )
        if content_decision.decision_type is not DecisionType.ALLOW:
            self._audit_decision(AuditEventType.CONTENT_POLICY, content_decision, request)
            raise ToolExecutionDenied(content_decision)
        if self._data_flow_engine is not None:
            if self._data_source is None or self._data_destination is None:
                raise ToolExecutionDenied(
                    Decision(
                        decision_type=DecisionType.BLOCK,
                        reason=DecisionReason.DATA_FLOW_VIOLATION,
                        severity=Severity.CRITICAL,
                        metadata={"category": "missing_flow_endpoint"},
                    )
                )
            flow_decision = self._data_flow_engine.check(
                DataFlow(
                    source=self._data_source,
                    destination=self._data_destination,
                    action=self.name,
                    agent=self._agent.agent_id,
                    tool=self.name,
                    resource=self.tool_id,
                )
            )
            self._audit_decision(AuditEventType.DATA_FLOW, flow_decision, request)
            filesystem_flow_approval_required = (
                self._filesystem_policy is not None
                and flow_decision.decision_type is DecisionType.APPROVAL_REQUIRED
            )
            if flow_decision.decision_type is DecisionType.APPROVAL_REQUIRED and not filesystem_flow_approval_required:
                approval_decision = self._verify_approval(request, consume=False)
                if approval_decision is None:
                    raise ToolApprovalRequired(flow_decision)
                if approval_decision.decision_type is not DecisionType.ALLOW:
                    raise ToolExecutionDenied(approval_decision)
            if flow_decision.decision_type is not DecisionType.ALLOW and not filesystem_flow_approval_required:
                raise ToolExecutionDenied(flow_decision)
        else:
            filesystem_flow_approval_required = False
        controller = self._runtime_controller
        acquired = False
        if controller is not None:
            try:
                runtime_decision = controller.acquire(
                    self._agent.agent_id, self.tool_id, self._agent.session_id
                )
            except Exception:
                runtime_decision = Decision(
                    decision_type=DecisionType.BLOCK,
                    reason=DecisionReason.RUNTIME_SECURITY_ERROR,
                    severity=Severity.CRITICAL,
                    metadata={"stage": "runtime"},
                )
            if runtime_decision is not None:
                self._audit_decision(AuditEventType.RATE_LIMIT, runtime_decision, request)
                raise ToolExecutionDenied(runtime_decision)
            acquired = True
            runtime_decision = controller.check_loop(
                ExecutionSignature(
                    agent_id=self._agent.agent_id,
                    session_id=self._agent.session_id,
                    tool_id=self.tool_id,
                    action=self.name,
                    resource=self.tool_id,
                )
            )
            if runtime_decision is not None:
                self._audit_decision(AuditEventType.LOOP_DETECTION, runtime_decision, request)
                controller.release()
                raise ToolExecutionDenied(runtime_decision)
            runtime_decision = controller.check_circuit(
                self._agent.agent_id, self.tool_id
            )
            if runtime_decision is not None:
                self._audit_decision(AuditEventType.CIRCUIT_BREAKER, runtime_decision, request)
                controller.release()
                raise ToolExecutionDenied(runtime_decision)
        network_approval_is_exact_request = (
            self._network_policy is not None
            and self._approval is not None
            and self._approval_manager is not None
            and self._approval.request_digest
            != self._approval_manager.digest_request(request)
        )
        filesystem_approval_is_exact_request = (
            self._filesystem_policy is not None
            and self._approval is not None
            and self._approval_manager is not None
            and self._approval.request_digest
            != self._approval_manager.digest_request(request)
        )
        final_approval = (
            None
            if network_approval_is_exact_request or filesystem_approval_is_exact_request
            else self._verify_approval(request, consume=True)
        )
        if final_approval is not None and final_approval.decision_type is not DecisionType.ALLOW:
            raise ToolExecutionDenied(final_approval)
        started_at = time.monotonic()
        try:
            self._require_active_lifecycle("tool_execution")
            with self._mission_execution_guard(
                operation="tool_execution_final",
                action=request.action.name,
                resource=request.resource.resource_id,
                tool=request.tool.name,
                destination=None,
                data=request.data.classification.value,
                context=None,
                defer_destination=self._network is not None,
                defer_data=any((self._network, self._filesystem, self._database)),
                defer_context=any((self._network, self._filesystem, self._database)),
                mission_binding=mission_binding,
            ):
                integrity_decision = self._engine.integrity_check()
                if integrity_decision is not None:
                    raise ToolExecutionDenied(integrity_decision)
                self._check_tool_trust()
                self._check_data_release(mission_binding)
                result = self._function(*args, **kwargs)
            if self._sandbox is not None:
                integrity_decision = self._engine.integrity_check()
                if integrity_decision is not None:
                    raise ToolExecutionDenied(integrity_decision)
                self._check_tool_trust()
                self._check_data_release(mission_binding)
                if not isinstance(result, SandboxCommand):
                    raise ToolExecutionDenied(
                        Decision(
                            DecisionType.BLOCK,
                            DecisionReason.SANDBOX_POLICY_VIOLATION,
                            severity=Severity.HIGH,
                            metadata={"category": "sandbox_command_required"},
                        )
                    )
                self._require_active_lifecycle("sandbox_dispatch")
                with self._mission_execution_guard(
                    operation="sandbox_dispatch",
                    action=self.name,
                    resource=self.tool_id,
                    tool=self.name,
                    destination=None,
                    data=DataClassification.INTERNAL.value,
                    context={"framework": "preventai", "sandbox": True},
                    mission_binding=mission_binding,
                ):
                    sandbox_result = self._sandbox.execute(
                        result,
                        authorization=self._sandbox._issue_authorization(
                            agent_id=self._agent.agent_id,
                            session_id=self._agent.session_id,
                            tool_id=self.tool_id,
                            action=self.name,
                            resource=self.tool_id,
                            command=result,
                            _issuer=_FRAMEWORK_ISSUER,
                        ),
                        agent_id=self._agent.agent_id,
                        session_id=self._agent.session_id,
                        tool_id=self.tool_id,
                        action=self.name,
                        resource=self.tool_id,
                    )
                if sandbox_result.decision.decision_type is not DecisionType.ALLOW:
                    raise ToolExecutionDenied(sandbox_result.decision)
                result = {
                    "returncode": sandbox_result.returncode,
                    "stdout": sandbox_result.stdout,
                    "stderr": sandbox_result.stderr,
                    "timed_out": sandbox_result.timed_out,
                    "output_truncated": sandbox_result.output_truncated,
                }
            if self._network_policy is not None:
                integrity_decision = self._engine.integrity_check()
                if integrity_decision is not None:
                    raise ToolExecutionDenied(integrity_decision)
                self._check_tool_trust()
                if not isinstance(result, NetworkRequest):
                    raise ToolExecutionDenied(Network._blocked("explicit_network_request_required"))
                self._check_data_release(
                    mission_binding,
                    destination=result.destination,
                    destination_trust=(
                        DestinationTrust.INTERNAL
                        if result.host in {"localhost", "127.0.0.1", "::1"}
                        else DestinationTrust.UNKNOWN
                    ),
                    source_classification=result.classification,
                    bind_authorization=True,
                )
                result = self._execute_network(result, mission_binding=mission_binding)
            if self._filesystem_policy is not None:
                integrity_decision = self._engine.integrity_check()
                if integrity_decision is not None:
                    raise ToolExecutionDenied(integrity_decision)
                self._check_tool_trust()
                if not isinstance(result, FileSystemRequest):
                    raise ToolExecutionDenied(FileSystem._blocked("explicit_filesystem_request_required"))
                self._check_data_release(
                    mission_binding,
                    destination=result.destination_path or result.path,
                    source_classification=result.classification,
                    bind_authorization=True,
                )
                result = self._execute_filesystem(
                    result,
                    base_approval_required=filesystem_approval_required,
                    base_flow_approval_required=filesystem_flow_approval_required,
                    mission_binding=mission_binding,
                )
            if self._database_policy is not None:
                integrity_decision = self._engine.integrity_check()
                if integrity_decision is not None:
                    raise ToolExecutionDenied(integrity_decision)
                self._check_tool_trust()
                if not isinstance(result, DatabaseRequest):
                    raise ToolExecutionDenied(Database._blocked("explicit_database_request_required"))
                self._check_data_release(
                    mission_binding,
                    destination=f"{result.database}:{result.schema}:{result.table}",
                    source_classification=(
                        result.classification
                        or self._data_source.classification
                        if self._data_source is not None
                        else result.classification
                    ),
                    field_paths=tuple(result.columns),
                    bind_authorization=True,
                )
                result = self._execute_database(
                    result,
                    base_approval_required=database_approval_required,
                    mission_binding=mission_binding,
                )
        except Exception:
            self._audit(
                AuditEventType.TOOL_EXECUTION_FAILED,
                correlation_id=correlation_id,
                agent_id=self._agent.agent_id,
                session_id=self._agent.session_id,
                tool_id=self.tool_id,
                action=self.name,
                resource=self.tool_id,
                metadata={"failure_type": "callable_exception"},
            )
            if acquired:
                failure_decision = controller.record_failure(
                    self._agent.agent_id, self.tool_id
                )
                controller.release()
                if failure_decision is not None:
                    raise ToolExecutionDenied(failure_decision)
            raise
        elapsed = time.monotonic() - started_at
        if controller is not None:
            runtime_decision = controller.record(elapsed)
            success_decision = controller.record_success(
                self._agent.agent_id, self.tool_id
            )
            if success_decision is not None:
                runtime_decision = success_decision
            controller.release()
            if runtime_decision is not None:
                event_type = (
                    AuditEventType.TIMEOUT
                    if runtime_decision.reason is DecisionReason.TIMEOUT
                    else AuditEventType.BUDGET
                    if runtime_decision.reason is DecisionReason.BUDGET_EXCEEDED
                    else AuditEventType.RATE_LIMIT
                )
                self._audit_decision(event_type, runtime_decision, request)
                raise ToolExecutionDenied(runtime_decision)
        try:
            output_decision = self._output_guard.validate(result, request)
        except Exception:
            output_decision = Decision(
                decision_type=DecisionType.BLOCK,
                reason=DecisionReason.OUTPUT_SECURITY_ERROR,
                severity=Severity.CRITICAL,
                metadata={"stage": "output_guard"},
            )
        if output_decision.decision_type is DecisionType.BLOCK:
            self._audit_decision(AuditEventType.OUTPUT_VALIDATION, output_decision, request)
            raise ToolExecutionDenied(output_decision)
        if output_decision.decision_type is DecisionType.APPROVAL_REQUIRED:
            raise ToolApprovalRequired(output_decision)
        if (
            self._data_release_policy is not None
            and self._data_source is not None
            and self._data_destination is not None
            and not any((
                self._network_policy,
                self._filesystem_policy,
                self._database_policy,
            ))
        ):
            self._check_data_release(
                mission_binding,
                bind_authorization=True,
            )
        self._audit(
            AuditEventType.TOOL_EXECUTION_SUCCEEDED,
            correlation_id=correlation_id,
            agent_id=self._agent.agent_id,
            session_id=self._agent.session_id,
            tool_id=self.tool_id,
            action=self.name,
            resource=self.tool_id,
            metadata={"elapsed_seconds": elapsed},
        )
        return result

    def _require_active_lifecycle(self, operation: str) -> None:
        try:
            registry = self._lifecycle_registry
            agent = self._agent
            decision = (
                registry.check(agent.agent_id, operation=operation)
                if registry is not None and agent is not None
                else None
            )
        except Exception:
            decision = None
        if decision is None or decision.decision_type is not DecisionType.ALLOW:
            raise ToolExecutionDenied(
                Decision(
                    DecisionType.BLOCK,
                    DecisionReason.LIFECYCLE_INVALID,
                    severity=Severity.CRITICAL,
                    metadata={"stage": "lifecycle", "operation": operation},
                )
            )

    def _mission_request(
        self,
        *,
        operation: str,
        action: str,
        resource: str,
        tool: str,
        destination: str | None,
        data: str | None,
        context: Mapping[str, Any] | None,
        defer_destination: bool = False,
        defer_data: bool = False,
        defer_context: bool = False,
        mission_binding: MissionBinding | None = None,
    ) -> dict[str, Any]:
        return {
            "operation": operation,
            "action": action,
            "resource": resource,
            "tool": tool,
            "destination": destination,
            "data": data,
            "context": context,
            "defer_destination": defer_destination,
            "defer_data": defer_data,
            "defer_context": defer_context,
            "mission_id": mission_binding.mission_id if mission_binding else None,
            "mission_version": mission_binding.mission_version if mission_binding else None,
        }

    def _require_valid_mission(
        self,
        *,
        operation: str,
        action: str,
        resource: str,
        tool: str,
        destination: str | None = None,
        data: str | None = None,
        context: Mapping[str, Any] | None = None,
        defer_destination: bool = False,
        defer_data: bool = False,
        defer_context: bool = False,
        mission_binding: MissionBinding | None = None,
    ) -> None:
        if self._mission_registry is None or self._agent is None or mission_binding is None:
            raise ToolExecutionDenied(
                Decision(
                    DecisionType.BLOCK,
                    DecisionReason.MISSION_INVALID,
                    severity=Severity.CRITICAL,
                    metadata={"stage": "mission_binding"},
                )
            )
        decision = self._mission_registry.check(
            self._agent.agent_id,
            **self._mission_request(
                operation=operation,
                action=action,
                resource=resource,
                tool=tool,
                destination=destination,
                data=data,
                context=context,
                defer_destination=defer_destination,
                defer_data=defer_data,
                defer_context=defer_context,
                mission_binding=mission_binding,
            ),
        )
        if decision.decision_type is not DecisionType.ALLOW:
            raise ToolExecutionDenied(decision)

    @contextmanager
    def _mission_execution_guard(
        self,
        *,
        operation: str,
        action: str,
        resource: str,
        tool: str,
        destination: str | None,
        data: str | None,
        context: Mapping[str, Any] | None,
        defer_destination: bool = False,
        defer_data: bool = False,
        defer_context: bool = False,
        mission_binding: MissionBinding | None = None,
    ) -> Iterator[None]:
        if self._mission_registry is None or self._agent is None or mission_binding is None:
            raise ToolExecutionDenied(
                Decision(
                    DecisionType.BLOCK,
                    DecisionReason.MISSION_INVALID,
                    severity=Severity.CRITICAL,
                    metadata={"stage": "mission_binding"},
                )
            )
        request = self._mission_request(
            operation=operation,
            action=action,
            resource=resource,
            tool=tool,
            destination=destination,
            data=data,
            context=context,
            defer_destination=defer_destination,
            defer_data=defer_data,
            defer_context=defer_context,
            mission_binding=mission_binding,
        )
        with self._mission_registry.execution_guard(self._agent.agent_id, **request) as decision:
            if decision.decision_type is not DecisionType.ALLOW:
                raise ToolExecutionDenied(decision)
            yield

    def _audit(self, event_type: AuditEventType, **kwargs: Any) -> None:
        if self._audit_logger is None:
            return
        try:
            self._audit_logger.record(event_type, **kwargs)
        except Exception:
            if self._audit_logger.policy.required:
                raise ToolExecutionDenied(
                    Decision(
                        DecisionType.BLOCK,
                        DecisionReason.RUNTIME_SECURITY_ERROR,
                        severity=Severity.CRITICAL,
                        metadata={"stage": "audit"},
                    )
                )

    def _audit_decision(
        self,
        event_type: AuditEventType,
        decision: Decision,
        request: SecurityRequest,
    ) -> None:
        resource = (
            "filesystem"
            if request.context.get("filesystem") is True
            else request.resource.resource_id
        )
        tool_id = (
            self.tool_id
            if request.context.get("filesystem") is True
            else request.resource.resource_id
        )
        self._audit(
            event_type,
            correlation_id=request.request_id,
            agent_id=request.agent.agent_id,
            session_id=request.session.session_id,
            tool_id=tool_id,
            action=request.action.name,
            resource=resource,
            decision=decision.decision_type.value,
            reason=decision.reason.value,
            severity=decision.severity.value if decision.severity else None,
            rule_id=decision.rule_id,
        )
    def _verify_approval(self, request: SecurityRequest, *, consume: bool) -> Decision | None:
        if self._approval_manager is None or self._approval is None:
            return None
        return self._approval_manager.verify(
            self._approval,
            request,
            authenticated_identity=self._agent.agent_id,
            consume=consume,
        )

    def _execute_filesystem(
        self,
        result: Any,
        *,
        base_approval_required: bool,
        base_flow_approval_required: bool,
        mission_binding: MissionBinding,
    ) -> Any:
        self._require_active_lifecycle("filesystem_execution")
        if not isinstance(result, FileSystemRequest) or self._filesystem is None:
            raise ToolExecutionDenied(
                FileSystem._blocked("explicit_filesystem_request_required")
            )
        assert self._agent is not None and self._engine is not None
        for field, expected in (
            ("agent_id", self._agent.agent_id),
            ("session_id", self._agent.session_id),
            ("tool_id", self.tool_id),
        ):
            supplied = getattr(result, field)
            if supplied is not None and supplied != expected:
                raise ToolExecutionDenied(FileSystem._blocked("filesystem_context_mismatch"))
        request = replace(
            result,
            agent_id=self._agent.agent_id,
            session_id=self._agent.session_id,
            tool_id=self.tool_id,
        )
        security_request = self._filesystem_security_request(request)

        if self._agent._delegation is not None:
            decision = self._delegation_registry.evaluate(
                self._agent._delegation,
                authenticated_identity=self._agent.agent_id,
                action=request.action,
                resource=request.path,
                tool=self.name,
                session_id=self._agent.session_id,
                category=PermissionCategory.FILE,
                data_class=request.classification.value,
                context={
                    "operation": request.operation,
                    "destination_path": request.destination_path,
                },
            )
            if decision.decision_type is not DecisionType.ALLOW:
                self._audit_decision(AuditEventType.DELEGATION, decision, security_request)
                raise ToolExecutionDenied(decision)

        decision = self._engine.evaluate(security_request)
        self._audit_decision(AuditEventType.POLICY_EVALUATION, decision, security_request)
        policy_approval_required = (
            base_approval_required
            or decision.decision_type is DecisionType.APPROVAL_REQUIRED
        )
        if decision.decision_type not in {
            DecisionType.ALLOW, DecisionType.APPROVAL_REQUIRED
        }:
            raise ToolExecutionDenied(decision)

        file_permissions = PermissionEngine(
            rule for rule in self._engine.permissions.rules
            if rule.category is PermissionCategory.FILE
        )
        permission = file_permissions.evaluate(EvaluationContext.from_request(security_request))
        self._audit_decision(AuditEventType.PERMISSION_EVALUATION, permission, security_request)
        if permission.decision_type is not DecisionType.ALLOW:
            raise ToolExecutionDenied(permission)

        try:
            plan = self._filesystem._prepare(request)
        except FileSystemExecutionDenied as exc:
            self._audit_decision(AuditEventType.FILESYSTEM_BLOCKED, exc.decision, security_request)
            raise ToolExecutionDenied(exc.decision) from exc

        approval_rule = (
            self._approval_policy.match(EvaluationContext.from_request(security_request))
            if self._approval_policy is not None else None
        )
        flow_approval_required = base_flow_approval_required
        if request.operation in {"read", "list", "stat"}:
            if self._data_flow_engine is None or self._data_destination is None:
                raise ToolExecutionDenied(FileSystem._blocked("filesystem_data_flow_not_configured"))
            source = DataSource(
                f"filesystem:{plan.path}",
                propagate_classification(
                    request.classification,
                    self._data_source.classification if self._data_source is not None else request.classification,
                ),
                FlowTrust.UNTRUSTED,
            )
            flow = DataFlow(
                source=source,
                destination=self._data_destination,
                action=request.action,
                agent=self._agent.agent_id,
                tool=self.name,
                resource=request.path,
            )
            flow_decision = self._data_flow_engine.check(flow)
        elif request.operation in {"write", "create", "append"}:
            if self._data_flow_engine is None or self._data_source is None:
                raise ToolExecutionDenied(FileSystem._blocked("filesystem_data_flow_not_configured"))
            flow = DataFlow(
                source=self._data_source,
                destination=DataFlowDestination(f"filesystem:{plan.path}"),
                action=request.action,
                classification=propagate_classification(
                    self._data_source.classification, request.classification
                ),
                trust=propagate_trust(self._data_source.trust, request.source_trust),
                agent=self._agent.agent_id,
                tool=self.name,
                resource=request.path,
            )
            flow_decision = self._data_flow_engine.check(flow)
        elif request.operation == "rename":
            if self._data_flow_engine is None:
                raise ToolExecutionDenied(FileSystem._blocked("filesystem_data_flow_not_configured"))
            assert plan.destination is not None
            flow = DataFlow(
                source=DataSource(
                    f"filesystem:{plan.path}",
                    request.classification,
                    FlowTrust.UNTRUSTED,
                ),
                destination=DataFlowDestination(f"filesystem:{plan.destination}"),
                action=request.action,
                agent=self._agent.agent_id,
                tool=self.name,
                resource=request.path,
            )
            flow_decision = self._data_flow_engine.check(flow)
        else:
            flow_decision = Decision(DecisionType.ALLOW, DecisionReason.REQUEST_ALLOWED)
        self._audit_decision(AuditEventType.DATA_FLOW, flow_decision, security_request)
        flow_approval_required = (
            flow_approval_required
            or flow_decision.decision_type is DecisionType.APPROVAL_REQUIRED
        )
        if flow_decision.decision_type not in {
            DecisionType.ALLOW, DecisionType.APPROVAL_REQUIRED
        }:
            raise ToolExecutionDenied(flow_decision)

        approval_required = policy_approval_required or approval_rule is not None or flow_approval_required
        if approval_required:
            approval_decision = self._verify_approval(security_request, consume=False)
            if approval_decision is not None and approval_decision.decision_type is not DecisionType.ALLOW:
                self._audit_decision(AuditEventType.APPROVAL_DECISION, approval_decision, security_request)
                raise ToolExecutionDenied(approval_decision)
            if approval_decision is None:
                required = Decision(
                    DecisionType.APPROVAL_REQUIRED,
                    DecisionReason.APPROVAL_REQUIRED,
                    rule_id=approval_rule.rule_id if approval_rule else decision.rule_id,
                )
                self._audit_decision(AuditEventType.APPROVAL_DECISION, required, security_request)
                if self._approval_manager is not None:
                    try:
                        self._pending_approval_request = self._approval_manager.request(
                            security_request,
                            identity=self._agent.agent_id,
                            reason="filesystem_approval_required",
                            expires_at=datetime.now(timezone.utc) + timedelta(minutes=5),
                            policy_rule_id=required.rule_id,
                        )
                    except Exception as exc:
                        raise ToolExecutionDenied(FileSystem._blocked("filesystem_approval_error")) from exc
                raise ToolApprovalRequired(required)
            final_approval = self._verify_approval(security_request, consume=True)
            if final_approval is None or final_approval.decision_type is not DecisionType.ALLOW:
                error = final_approval or FileSystem._blocked("filesystem_approval_missing")
                self._audit_decision(AuditEventType.APPROVAL_DECISION, error, security_request)
                raise ToolExecutionDenied(error)

        if self._filesystem_authorizer is None:
            raise ToolExecutionDenied(FileSystem._blocked("filesystem_authorization_unavailable"))
        authorization = self._filesystem_authorizer(request, plan)
        try:
            self._require_active_lifecycle("filesystem_dispatch")
            with self._mission_execution_guard(
                operation="filesystem_dispatch",
                action=request.operation,
                resource=str(plan.path),
                tool=self.name,
                destination=None,
                data=request.classification.value,
                context={
                    "framework": "preventai",
                    "filesystem": True,
                    "operation": request.operation,
                },
                mission_binding=mission_binding,
            ):
                response = self._filesystem.execute(
                    request,
                    authorization=authorization,
                    release_authorization=self._release_authorization,
                    release_request=self._release_request,
                    application_id=(
                        self._agent._application_id
                        if self._data_release_policy is not None
                        else None
                    ),
                )
            return response.output_value()
        except FileSystemExecutionDenied as exc:
            raise ToolExecutionDenied(exc.decision) from exc

    def _execute_database(
        self,
        result: Any,
        *,
        base_approval_required: bool,
        mission_binding: MissionBinding,
    ) -> Any:
            self._require_active_lifecycle("database_execution")
            if not isinstance(result, DatabaseRequest) or self._database is None:
                raise ToolExecutionDenied(Database._blocked("explicit_database_request_required"))
            assert self._agent is not None and self._engine is not None
            for field, expected in (("agent_id", self._agent.agent_id), ("session_id", self._agent.session_id), ("tool_id", self.tool_id)):
                supplied = getattr(result, field)
                if supplied is not None and supplied != expected:
                    raise ToolExecutionDenied(Database._blocked("database_context_mismatch"))
            request = replace(result, agent_id=self._agent.agent_id, session_id=self._agent.session_id, tool_id=self.tool_id)
            security_request = self._database_security_request(request)
            if self._agent._delegation is not None:
                decision = self._delegation_registry.evaluate(
                    self._agent._delegation, authenticated_identity=self._agent.agent_id,
                    action=request.action, resource=f"{request.database}:{request.schema}:{request.table}",
                    tool=self.name, session_id=self._agent.session_id,
                    category=PermissionCategory.DATABASE,
                    data_class=(
                        request.classification.value
                        if request.classification is not None
                        else None
                    ),
                    context={
                        "database": request.database,
                        "schema": request.schema,
                        "table": request.table,
                        "columns": request.columns,
                        "tenant": request.tenant_id,
                        "operation": request.operation,
                    },
                )
                if decision.decision_type is not DecisionType.ALLOW:
                    raise ToolExecutionDenied(decision)
            permission = PermissionEngine(
                rule for rule in self._engine.permissions.rules if rule.category is PermissionCategory.DATABASE
            ).evaluate(EvaluationContext.from_request(security_request))
            if permission.decision_type is not DecisionType.ALLOW:
                raise ToolExecutionDenied(permission)
            policy_decision = self._engine.evaluate(security_request)
            if policy_decision.decision_type not in {DecisionType.ALLOW, DecisionType.APPROVAL_REQUIRED}:
                raise ToolExecutionDenied(policy_decision)
            if self._data_flow_engine is None:
                raise ToolExecutionDenied(Database._blocked("database_data_flow_not_configured"))
            flow = DataFlow(
                source=self._data_source or DataSource("database-request", FlowClassification.INTERNAL, FlowTrust.TRUSTED),
                destination=self._data_destination or DataFlowDestination(f"database:{request.database}:{request.schema}:{request.table}"),
                action=request.action, agent=self._agent.agent_id, tool=self.name,
                resource=f"{request.database}:{request.schema}:{request.table}",
            )
            flow_decision = self._data_flow_engine.check(flow)
            if flow_decision.decision_type not in {DecisionType.ALLOW, DecisionType.APPROVAL_REQUIRED}:
                raise ToolExecutionDenied(flow_decision)
            if base_approval_required or policy_decision.decision_type is DecisionType.APPROVAL_REQUIRED or flow_decision.decision_type is DecisionType.APPROVAL_REQUIRED:
                approval = self._verify_approval(security_request, consume=False)
                if approval is None or approval.decision_type is not DecisionType.ALLOW:
                    raise ToolApprovalRequired(policy_decision)
            authorization = self._database._issue_authorization(
                request, agent_id=self._agent.agent_id, session_id=self._agent.session_id,
                tool_id=self.tool_id, expires_at=time.time() + 30,
                _capability=self._database._issuance_capability
            )
            try:
                self._require_active_lifecycle("database_dispatch")
                with self._mission_execution_guard(
                    operation="database_dispatch",
                    action=request.action,
                    resource=f"{request.database}:{request.schema}:{request.table}",
                    tool=self.name,
                    destination=None,
                    data=(
                        request.classification.value
                        if request.classification is not None
                        else None
                    ),
                    context={
                        "framework": "preventai",
                        "database": request.database,
                        "schema": request.schema,
                        "table": request.table,
                        "tenant": request.tenant_id,
                        "operation": request.operation,
                        "columns": request.columns,
                    },
                    mission_binding=mission_binding,
                ):
                    response = self._database.execute(
                        request, authorization=authorization, agent_id=self._agent.agent_id,
                        session_id=self._agent.session_id, tool_id=self.tool_id,
                        release_authorization=self._release_authorization,
                        release_request=self._release_request,
                        application_id=(
                            self._agent._application_id
                            if self._data_release_policy is not None
                            else None
                        ),
                    )
            except DatabaseExecutionDenied as exc:
                raise ToolExecutionDenied(exc.decision) from exc
            return response.output_value()

    def _database_security_request(self, request: DatabaseRequest) -> SecurityRequest:
        return SecurityRequest(
            protocol_version="1.0", request_id=_stable_id("database", request.digest()),
            timestamp=datetime.now(timezone.utc),
            agent=ProtocolAgentIdentity(self._agent.agent_id, "developer_agent"),
            session=SessionIdentity(self._agent.session_id), requested_by=RequestedBy.AGENT,
            action=Action(request.action, {"database": request.database, "schema": request.schema, "table": request.table}),
            resource=Resource(f"{request.database}:{request.schema}:{request.table}", "database"),
            tool=ProtocolTool(self.name, "1.0"),
            data=DataMetadata(
                (
                    DataClassification(request.classification.value)
                    if request.classification is not None
                    else DataClassification.INTERNAL
                ),
                SourceTrust.TRUSTED,
            ),
            context={"framework": "preventai", "database": True},
            metadata={"agent_id": self._agent.agent_id},
        )
    def _filesystem_security_request(self, request: FileSystemRequest) -> SecurityRequest:
        assert self._agent is not None
        return SecurityRequest(
            protocol_version="1.0",
            request_id=_stable_id(
                "filesystem-request",
                f"{self.tool_id}:{request.digest()}",
            ),
            timestamp=datetime.now(timezone.utc),
            agent=ProtocolAgentIdentity(self._agent.agent_id, "developer_agent"),
            session=SessionIdentity(self._agent.session_id),
            requested_by=RequestedBy.AGENT,
            action=Action(
                request.action,
                {
                    "path": request.path,
                    "destination_path": request.destination_path,
                    "request_digest": request.digest(),
                    "overwrite": request.overwrite,
                },
            ),
            resource=Resource(request.path, "file"),
            tool=ProtocolTool(self.name, "1.0"),
            data=DataMetadata(
                DataClassification(request.classification.value),
                SourceTrust(request.source_trust.value),
            ),
            context={"framework": "preventai", "filesystem": True},
            metadata={"tool_id": self.tool_id},
        )

    def _execute_network(
        self, result: Any, *, mission_binding: MissionBinding
    ) -> Any:
        self._require_active_lifecycle("network_execution")
        if not isinstance(result, NetworkRequest) or self._network is None:
            raise ToolExecutionDenied(
                Network._blocked("explicit_network_request_required")
            )
        assert self._agent is not None and self._engine is not None
        request = result
        for name, expected in (
            ("agent_id", self._agent.agent_id),
            ("session_id", self._agent.session_id),
            ("tool_id", self.tool_id),
        ):
            supplied = getattr(request, name)
            if supplied is not None and supplied != expected:
                raise ToolExecutionDenied(
                    Network._blocked("network_context_mismatch")
                )
        request = replace(
            request,
            agent_id=self._agent.agent_id,
            session_id=self._agent.session_id,
            tool_id=self.tool_id,
        )
        security_request = self._network_security_request(request)

        if self._agent._delegation is not None:
            decision = self._delegation_registry.evaluate(
                self._agent._delegation,
                authenticated_identity=self._agent.agent_id,
                action=request.action,
                resource=request.destination,
                tool=self.name,
                session_id=self._agent.session_id,
                category=PermissionCategory.NETWORK,
                destination=request.host,
                data_class=request.classification.value,
                context={"operation": request.action, "host": request.host},
            )
            if decision.decision_type is not DecisionType.ALLOW:
                self._audit_decision(AuditEventType.DELEGATION, decision, security_request)
                raise ToolExecutionDenied(decision)

        decision = self._engine.evaluate(security_request)
        self._audit_decision(AuditEventType.POLICY_EVALUATION, decision, security_request)
        if decision.decision_type not in {DecisionType.ALLOW, DecisionType.APPROVAL_REQUIRED}:
            raise ToolExecutionDenied(decision)
        network_permissions = PermissionEngine(
            rule for rule in self._engine.permissions.rules
            if rule.category is PermissionCategory.NETWORK
        )
        permission = network_permissions.evaluate(
            EvaluationContext.from_request(security_request)
        )
        self._audit_decision(AuditEventType.PERMISSION_EVALUATION, permission, security_request)
        if permission.decision_type is not DecisionType.ALLOW:
            raise ToolExecutionDenied(permission)
        policy_approval_required = decision.decision_type is DecisionType.APPROVAL_REQUIRED
        approval_rule = (
            self._approval_policy.match(EvaluationContext.from_request(security_request))
            if self._approval_policy is not None
            else None
        )
        self._network_approval(
            security_request,
            required=policy_approval_required or approval_rule is not None,
            rule_id=approval_rule.rule_id if approval_rule else decision.rule_id,
            consume=False,
        )

        if self._data_flow_engine is None or self._data_source is None:
            raise ToolExecutionDenied(
                Network._blocked("network_data_flow_not_configured")
            )
        policy_decision = self._network._policy_decision(request)
        if policy_decision is not None:
            self._audit_decision(
                AuditEventType.NETWORK_BLOCKED, policy_decision, security_request
            )
            raise ToolExecutionDenied(policy_decision)
        try:
            literal = ipaddress.ip_address(request.host)
            preflight_internal = not literal.is_global
        except ValueError:
            preflight_internal = False
        preflight_destination = DataFlowDestination(
            ("internal." if preflight_internal else "external.") + request.host
        )
        flow_decision = self._network_data_flow_decision(
            request, preflight_destination
        )
        self._audit_decision(AuditEventType.DATA_FLOW, flow_decision, security_request)
        flow_approval_required = (
            flow_decision.decision_type is DecisionType.APPROVAL_REQUIRED
        )
        if flow_decision.decision_type not in {
            DecisionType.ALLOW,
            DecisionType.APPROVAL_REQUIRED,
        }:
            raise ToolExecutionDenied(flow_decision)
        if flow_approval_required:
            self._network_approval(
                security_request,
                required=True,
                rule_id=flow_decision.rule_id,
                consume=False,
            )
        try:
            plan = self._network.prepare(request)
        except NetworkExecutionDenied as exc:
            self._audit_decision(AuditEventType.NETWORK_BLOCKED, exc.decision, security_request)
            raise ToolExecutionDenied(exc.decision) from exc

        destination = DataFlowDestination(
            ("internal." if plan.internal else "external.") + request.host
        )
        if destination.name != preflight_destination.name:
            flow_decision = self._network_data_flow_decision(request, destination)
            self._audit_decision(
                AuditEventType.DATA_FLOW, flow_decision, security_request
            )
        flow_approval_required = (
            flow_approval_required
            or flow_decision.decision_type is DecisionType.APPROVAL_REQUIRED
        )
        if flow_decision.decision_type not in {
            DecisionType.ALLOW,
            DecisionType.APPROVAL_REQUIRED,
        }:
            raise ToolExecutionDenied(flow_decision)
        if flow_approval_required:
            self._network_approval(
                security_request, required=True, rule_id=flow_decision.rule_id, consume=False
            )

        self._network_approval(
            security_request,
            required=(
                policy_approval_required
                or approval_rule is not None
                or flow_approval_required
            ),
            rule_id=(
                approval_rule.rule_id
                if approval_rule
                else flow_decision.rule_id
                if flow_approval_required
                else decision.rule_id
            ),
            consume=True,
        )
        authorization = self._network._issue_authorization(
            request,
            resolved_addresses=plan.addresses,
            allow_redirects=not (
                policy_approval_required
                or approval_rule is not None
                or flow_approval_required
            ),
            _issuer=_NETWORK_FRAMEWORK_ISSUER,
        )
        try:
            self._require_active_lifecycle("network_dispatch")
            with self._mission_execution_guard(
                operation="network_dispatch",
                action=request.action,
                resource=request.destination,
                tool=self.name,
                destination=request.host,
                data=request.classification.value,
                context={
                    "framework": "preventai",
                    "network": True,
                    "operation": request.action,
                },
                mission_binding=mission_binding,
            ):
                response = self._network.execute(
                    request,
                    authorization=authorization,
                    release_authorization=self._release_authorization,
                    release_request=self._release_request,
                    application_id=(
                        self._agent._application_id
                        if self._data_release_policy is not None
                        else None
                    ),
                )
            value = response.output_value()
        except NetworkExecutionDenied as exc:
            self._audit_decision(AuditEventType.NETWORK_BLOCKED, exc.decision, security_request)
            raise ToolExecutionDenied(exc.decision) from exc
        except Exception as exc:
            decision = Network._blocked("network_response_validation_failed")
            self._audit_decision(AuditEventType.NETWORK_BLOCKED, decision, security_request)
            raise ToolExecutionDenied(decision) from exc
        return value

    def _network_data_flow_decision(
        self,
        request: NetworkRequest,
        destination: DataFlowDestination,
    ) -> Decision:
        assert self._agent is not None
        assert self._data_flow_engine is not None
        assert self._data_source is not None
        return self._data_flow_engine.check(
            DataFlow(
                source=self._data_source,
                destination=destination,
                action=request.action,
                classification=propagate_classification(
                    self._data_source.classification, request.classification
                ),
                trust=propagate_trust(self._data_source.trust, request.source_trust),
                agent=self._agent.agent_id,
                tool=self.name,
                resource=request.destination,
            )
        )

    def _network_security_request(self, request: NetworkRequest) -> SecurityRequest:
        assert self._agent is not None
        classification = DataClassification(request.classification.value)
        trust = (
            SourceTrust.TRUSTED
            if request.source_trust is FlowTrust.TRUSTED
            else SourceTrust.UNTRUSTED
        )
        return SecurityRequest(
            protocol_version="1.0",
            request_id=_stable_id("network-request", f"{self.tool_id}:{request.digest}"),
            timestamp=datetime.now(timezone.utc),
            agent=ProtocolAgentIdentity(self._agent.agent_id, "developer_agent"),
            session=SessionIdentity(self._agent.session_id),
            requested_by=RequestedBy.AGENT,
            action=Action(
                request.action,
                {
                    "scheme": request.scheme,
                    "host": request.host,
                    "port": request.port,
                    "digest": request.digest,
                },
            ),
            resource=Resource(request.destination, "network"),
            tool=ProtocolTool(self.name, "1.0"),
            data=DataMetadata(classification, trust),
            context={"framework": "preventai", "network": True},
            metadata={"tool_id": self.tool_id},
        )

    def _network_approval(
        self,
        request: SecurityRequest,
        *,
        required: bool,
        rule_id: str | None,
        consume: bool,
    ) -> None:
        if not required:
            return
        decision = self._verify_approval(request, consume=consume)
        if decision is not None and decision.decision_type is not DecisionType.ALLOW:
            self._audit_decision(AuditEventType.APPROVAL_DECISION, decision, request)
            raise ToolExecutionDenied(decision)
        if decision is None:
            required_decision = Decision(
                DecisionType.APPROVAL_REQUIRED,
                DecisionReason.APPROVAL_REQUIRED,
                rule_id=rule_id,
            )
            self._audit_decision(
                AuditEventType.APPROVAL_DECISION, required_decision, request
            )
            if self._approval_manager is not None:
                try:
                    self._pending_approval_request = self._approval_manager.request(
                        request,
                        identity=self._agent.agent_id,
                        reason="network_approval_required",
                        expires_at=datetime.now(timezone.utc) + timedelta(minutes=5),
                        policy_rule_id=rule_id,
                    )
                except Exception as exc:
                    raise ToolExecutionDenied(
                        Decision(
                            DecisionType.BLOCK,
                            DecisionReason.APPROVAL_SECURITY_ERROR,
                            severity=Severity.CRITICAL,
                            metadata={"stage": "network_approval"},
                        )
                    ) from exc
            raise ToolApprovalRequired(
                required_decision
            )


class Agent:
    """A stable developer agent identity and its registered tools."""

    def __init__(
        self,
        name: str,
        agent_id: str | None = None,
        *,
        credential: Credential | None = None,
    ) -> None:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("agent name must be a non-empty string")
        self.name = name
        self.agent_id = agent_id or _stable_id("agent", name)
        self.identity = RegisteredAgentIdentity(self.agent_id, name)
        if credential is not None and not isinstance(credential, Credential):
            raise TypeError("credential must be a Credential")
        if credential is not None and credential.identity_id != self.agent_id:
            raise ValueError("credential identity does not match agent")
        self._credential = credential
        self.session_id = _stable_id("session", self.agent_id)
        self._tools: dict[str, Tool] = {}
        self._tool_trust_registry: ToolTrustRegistry | None = None
        self._require_tool_trust = True
        self._engine: SecurityEngine | None = None
        self._app_state: AppState | None = None
        self._authenticator: Authenticator | None = None
        self._lifecycle_registry: AgentLifecycleRegistry | None = None
        self._mission_registry: MissionRegistry | None = None
        self._data_flow_engine: DataFlowEngine | None = None
        self._runtime_controller: RuntimeController | None = None
        self._application_id = "preventai"
        self._admission_registry: AgentAdmissionRegistry | None = None
        self._policy_integrity: PolicyIntegrityRegistry | None = None
        self._delegation_registry: DelegationRegistry | None = None
        self._delegation: DelegationGrant | None = None
        self._app_agents: Mapping[str, Agent] = {}
        self._audit_logger: AuditLogger | None = None
        self._approval_manager: ApprovalManager | None = None

    @property
    def tools(self) -> Mapping[str, Tool]:
        return dict(self._tools)

    @property
    def application_id(self) -> str:
        return self._application_id

    def verify_admission(
        self,
        *,
        session_id: str | None = None,
        mission_id: str | None = None,
        mission_version: int | None = None,
        capabilities: tuple[str, ...] = (),
        tools: tuple[str, ...] = (),
        policy_version: str | None = None,
    ) -> Decision:
        if self._admission_registry is None:
            return Decision(
                DecisionType.BLOCK,
                DecisionReason.INVALID_REQUEST,
                severity=Severity.CRITICAL,
                metadata={"stage": "agent_admission", "error": "registry_missing"},
            )
        return self._admission_registry.verify(
            agent_id=self.agent_id,
            application_id=self._application_id,
            session_id=session_id or self.session_id,
            mission_id=mission_id,
            mission_version=mission_version,
            capabilities=tuple(capabilities),
            tools=tuple(tools),
            policy_version=policy_version,
        )

    def use_delegation(self, grant: DelegationGrant) -> None:
        if not isinstance(grant, DelegationGrant):
            raise TypeError("grant must be a DelegationGrant")
        if self._delegation_registry is None:
            raise RuntimeError("agent is not registered with a PreventApp")
        self._delegation = grant

    def delegate(
        self,
        target_agent: str | Agent,
        capabilities: Any,
        *,
        expires_at: datetime,
        session_id: str | None = None,
    ) -> DelegationGrant:
        if self._delegation_registry is None or self._authenticator is None:
            raise PermissionError("delegation requires authenticated agents")
        target_id = target_agent.agent_id if isinstance(target_agent, Agent) else target_agent
        capabilities = tuple(capabilities)
        if not isinstance(target_id, str) or target_id not in self._app_agents:
            raise PermissionError("delegation subject is not registered")
        if not self._authenticator.has_identity(target_id):
            raise PermissionError("delegation subject is not authenticated")
        auth = self._authenticator.authenticate(
            self.agent_id, self._credential, session_id=self.session_id
        )
        if not auth.authenticated:
            raise PermissionError("delegation issuer authentication failed")
        if (
            self._lifecycle_registry is None
            or self._lifecycle_registry.check(
                self.agent_id, operation="delegation_issuance"
            ).decision_type is not DecisionType.ALLOW
        ):
            raise PermissionError("delegation issuer lifecycle is not active")
        if (
            self._mission_registry is not None
            and self._mission_registry.binding_for(self.agent_id) is not None
            and any(
                not self._mission_registry.allows_capability(self.agent_id, capability)
                for capability in capabilities
            )
        ):
            raise PermissionError("delegated capability exceeds issuer mission")
        return self._delegation_registry.create(
            self.agent_id,
            target_id,
            capabilities,
            expires_at=expires_at,
            parent=self._delegation,
            session_id=session_id,
            issuer_permissions=self._engine.permissions,
        )

    def tool(
        self,
        name: str | None = None,
        *,
        tool_id: str | None = None,
        input_schema: Mapping[str, Any] | None = None,
        content_guard: ContentGuard | None = None,
        output_schema: Mapping[str, Any] | None = None,
        output_guard: OutputGuard | None = None,
        data_source: DataSource | None = None,
        data_destination: DataFlowDestination | None = None,
        runtime_policy: RuntimePolicy | None = None,
        sandbox_policy: SandboxPolicy | None = None,
        network_policy: NetworkPolicy | None = None,
        filesystem_policy: FileSystemPolicy | None = None,
        database_policy: DatabasePolicy | None = None,
        version: str = "1.0.0",
        provenance: ToolProvenance | None = None,
    ):
        def register(function: Callable[..., Any]) -> Tool:
            tool_name = name or function.__name__
            stable_tool_id = tool_id or _stable_id(
                f"tool:{self.agent_id}", tool_name
            )
            tool = Tool(
                tool_name,
                function,
                stable_tool_id,
                input_schema,
                content_guard,
                output_schema,
                output_guard,
                data_source,
                data_destination,
                runtime_policy,
                sandbox_policy=sandbox_policy,
                network_policy=network_policy,
                filesystem_policy=filesystem_policy,
                database_policy=database_policy,
                version=version,
                provenance=provenance,
            )
            if tool.name in self._tools:
                raise ValueError(f"tool already registered: {tool.name}")
            self._tools[tool.name] = tool
            if self._engine is not None:
                tool._bind(
                    self,
                    self._engine,
                    self._data_flow_engine,
                    self._audit_logger,
                )
                tool._authenticator = self._authenticator
                tool._lifecycle_registry = self._lifecycle_registry
                tool._mission_registry = self._mission_registry
                tool._data_release_policy = getattr(self, "_data_release_policy", None)
                if self._tool_trust_registry is not None:
                    tool._bind_tool_trust(
                        self._tool_trust_registry, required=self._require_tool_trust
                    )
                tool._delegation_registry = self._delegation_registry
                tool._approval_manager = self._approval_manager
                tool._approval_policy = self._approval_policy
                tool._audit_logger = self._audit_logger
                tool._runtime_controller = (
                    RuntimeController(
                        tool._runtime_policy,
                        application_id=self._application_id,
                    )
                    if tool._runtime_policy is not None
                    else self._runtime_controller
                )
            return tool

        return register

    def _bind(
        self,
        engine: SecurityEngine,
        data_flow_engine: DataFlowEngine | None = None,
        runtime_controller: RuntimeController | None = None,
        application_id: str = "preventai",
        authenticator: Authenticator | None = None,
        approval_manager: ApprovalManager | None = None,
        delegation_registry: DelegationRegistry | None = None,
        app_agents: Mapping[str, Agent] | None = None,
        approval_policy: ApprovalPolicy | None = None,
        data_release_policy: DataReleasePolicy | None = None,
        audit_logger: AuditLogger | None = None,
        lifecycle_registry: AgentLifecycleRegistry | None = None,
    ) -> None:
        if self._application_id != "preventai" and self._application_id != application_id:
            raise ValueError("agent is already bound to another PreventApp")
        self._engine = engine
        self._data_flow_engine = data_flow_engine
        self._runtime_controller = runtime_controller
        self._application_id = application_id
        self._authenticator = authenticator
        self._lifecycle_registry = lifecycle_registry
        self._mission_registry = getattr(self, "_mission_registry", None)
        if approval_manager is not None and not isinstance(approval_manager, ApprovalManager):
            raise TypeError("approval_manager must be an ApprovalManager")
        self._approval_manager = approval_manager
        if approval_policy is not None and not isinstance(approval_policy, ApprovalPolicy):
            raise TypeError("approval_policy must be an ApprovalPolicy")
        self._approval_policy = approval_policy
        if data_release_policy is not None and not isinstance(data_release_policy, DataReleasePolicy):
            raise TypeError("data_release_policy must be a DataReleasePolicy")
        self._data_release_policy = data_release_policy
        self._audit_logger = audit_logger
        self._delegation_registry = delegation_registry
        self._mission_registry = getattr(self, "_mission_registry", None)
        self._app_agents = app_agents or {}
        for tool in self._tools.values():
            tool._bind(self, engine, data_flow_engine, audit_logger)
            tool._authenticator = authenticator
            tool._lifecycle_registry = lifecycle_registry
            tool._delegation_registry = delegation_registry
            tool._mission_registry = self._mission_registry
            tool._approval_manager = self._approval_manager
            tool._approval_policy = self._approval_policy
            tool._audit_logger = audit_logger


class PreventApp:
    """Application container for multiple secure agents and tools."""

    def __init__(
        self,
        *,
        policy: PolicyEvaluator | None = None,
        permissions: PermissionEngine | None = None,
        data_flow: DataFlowEngine | None = None,
        runtime_policy: RuntimePolicy | None = None,
        authenticator: Authenticator | None = None,
        approval_manager: ApprovalManager | None = None,
        approval_policy: ApprovalPolicy | None = None,
        data_release_policy: DataReleasePolicy | None = None,
        audit_logger: AuditLogger | None = None,
        require_missions: bool = True,
        require_tool_trust: bool = True,
    ) -> None:
        if audit_logger is not None and not isinstance(audit_logger, AuditLogger):
            raise TypeError("audit_logger must be an AuditLogger")
        self._audit_logger = audit_logger or AuditLogger()
        if data_release_policy is not None and not isinstance(data_release_policy, DataReleasePolicy):
            raise TypeError("data_release_policy must be a DataReleasePolicy")
        self._data_release_policy = data_release_policy
        effective_policy = policy if policy is not None else DefaultDenyPolicy()
        effective_permissions = permissions if permissions is not None else PermissionEngine()
        self._policy_integrity = PolicyIntegrityRegistry(
            audit_logger=self._audit_logger,
            _owner_token=_POLICY_INTEGRITY_CONTEXT_TOKEN,
        )
        if data_flow is not None and not isinstance(data_flow, DataFlowEngine):
            raise TypeError("data_flow must be a DataFlowEngine")
        self._data_flow = data_flow
        if runtime_policy is not None and not isinstance(runtime_policy, RuntimePolicy):
            raise TypeError("runtime_policy must be a RuntimePolicy")
        self._runtime_policy = runtime_policy
        if authenticator is not None and not isinstance(authenticator, Authenticator):
            raise TypeError("authenticator must be an Authenticator")
        self._authenticator = authenticator
        if approval_manager is not None and not isinstance(approval_manager, ApprovalManager):
            raise TypeError("approval_manager must be an ApprovalManager")
        if approval_policy is not None and not isinstance(approval_policy, ApprovalPolicy):
            raise TypeError("approval_policy must be an ApprovalPolicy")
        self._approval_manager = approval_manager
        self._approval_policy = approval_policy
        if require_missions is not True:
            raise ValueError("mission enforcement is mandatory")
        self._policy_integrity_initialization_failed = False
        try:
            self._policy_integrity.register(
                configuration_id="application-security",
                version=1,
                policy=effective_policy,
                permissions=effective_permissions,
                runtime=runtime_policy,
            )
        except Exception:
            self._policy_integrity_initialization_failed = True
        self._engine = SecurityEngine(
            effective_policy, effective_permissions, self._policy_integrity
        )
        self._lifecycle_registry = AgentLifecycleRegistry(audit_logger=self._audit_logger)
        from ..identity.mission import _APPLICATION_CONTEXT_TOKEN
        self._mission_registry = MissionRegistry(
            audit_logger=self._audit_logger,
            require_missions=require_missions,
            _owner_token=_APPLICATION_CONTEXT_TOKEN,
        )
        self._delegation_registry = DelegationRegistry()
        self._tool_trust_registry = ToolTrustRegistry(
            audit_logger=self._audit_logger,
            _owner_token=_TOOL_TRUST_CONTEXT_TOKEN,
        )
        self._admission_registry = AgentAdmissionRegistry(
            application_id=_stable_id("app", str(id(self))),
            audit_logger=self._audit_logger,
            _owner_token=_ADMISSION_CONTEXT_TOKEN,
        )
        # The application-owned registry is authoritative and never caller-created.
        self._require_tool_trust = True
        self._application_id = self._admission_registry.application_id
        self._runtime_controller = (
            RuntimeController(runtime_policy, application_id=self._application_id)
            if runtime_policy is not None
            else None
        )
        self._agents: dict[str, Agent] = {}
        self._state = AppState.NEW

    @property
    def policy_integrity(self) -> PolicyIntegrityRegistry:
        return self._policy_integrity

    def update_security_policy(
        self,
        *,
        policy: PolicyEvaluator,
        permissions: PermissionEngine,
        runtime: RuntimePolicy | None = None,
        version: int,
        activate: bool = True,
        expires_at: datetime | None = None,
    ):
        snapshot = self._policy_integrity.register(
            configuration_id="application-security",
            version=version,
            policy=policy,
            permissions=permissions,
            runtime=self._runtime_policy if runtime is None else runtime,
            activate=activate,
            expires_at=expires_at,
        )
        if activate:
            self._engine._policy = snapshot.policy
            self._engine._permissions = snapshot.permissions
        return snapshot

    @property
    def agents(self) -> Mapping[str, Agent]:
        return dict(self._agents)

    @property
    def state(self) -> AppState:
        return self._state

    @property
    def lifecycle(self) -> AgentLifecycleRegistry:
        """The shared lifecycle authority for identities in this application."""
        return self._lifecycle_registry

    @property
    def missions(self) -> MissionRegistry:
        """The shared mission authority for identities in this application."""
        return self._mission_registry

    @property
    def tool_trust(self) -> ToolTrustRegistry:
        return self._tool_trust_registry

    @property
    def admission(self) -> AgentAdmissionRegistry:
        return self._admission_registry

    def verify_agent_admission(
        self,
        agent: str | Agent,
        *,
        session_id: str | None = None,
        mission_id: str | None = None,
        mission_version: int | None = None,
        capabilities: tuple[str, ...] = (),
        tools: tuple[str, ...] = (),
        policy_version: str | None = None,
    ) -> Decision:
        agent_id = agent.agent_id if isinstance(agent, Agent) else agent
        return self._admission_registry.verify(
            agent_id=agent_id,
            application_id=self._application_id,
            session_id=session_id,
            mission_id=mission_id,
            mission_version=mission_version,
            capabilities=tuple(capabilities),
            tools=tuple(tools),
            policy_version=policy_version,
        )

    def register_tool_trust(
        self, tool: Tool, *, version: str | None = None,
        provenance: ToolProvenance | None = None,
        integrity_digest: str | None = None,
        expires_at: datetime | None = None, trusted: bool = False,
    ):
        if not isinstance(tool, Tool):
            raise TypeError("tool must be a Tool")
        current_integrity_digest = implementation_digest(tool._function)
        tool.integrity_digest = current_integrity_digest
        record = self._tool_trust_registry.register(
            tool_id=tool.tool_id, name=tool.name, version=version or tool.version,
            provenance=provenance or tool.provenance,
            integrity_digest=integrity_digest or current_integrity_digest,
            expires_at=expires_at,
        )
        return self._tool_trust_registry.trust(record.tool_id, record.version) if trusted else record

    def bind_mission(
        self,
        agent: str | Agent,
        mission_id: str,
        version: int,
        *,
        context: Mapping[str, Any] | None = None,
    ) -> MissionBinding:
        """Bind an agent to one explicit version in this application's mission registry."""
        agent_id = agent.agent_id if isinstance(agent, Agent) else agent
        if not isinstance(agent_id, str) or agent_id not in self._agents:
            raise ValueError("mission binding requires an agent registered with this application")
        return self._mission_registry.bind(
            agent_id, mission_id, version, context=context
        )

    def bind_agent_communication(self, boundary: object) -> object:
        """Bind a communication boundary to this application's security context."""
        from ..protocol.agent_communication import AgentCommunicationBoundary

        if not isinstance(boundary, AgentCommunicationBoundary):
            raise TypeError("boundary must be an AgentCommunicationBoundary")
        boundary._bind_application_context(
            self._lifecycle_registry,
            self._audit_logger,
            self._delegation_registry,
            self._mission_registry,
        )
        boundary._application_id = self._application_id
        boundary._release_issuer = self.authorize_data_release
        boundary._release_policy_version_provider = (
            lambda: self._data_release_policy.version
            if self._data_release_policy is not None
            else None
        )
        return boundary

    def bind_mcp(self, boundary: object) -> object:
        from ..protocol.mcp import MCPBoundary

        if not isinstance(boundary, MCPBoundary):
            raise TypeError("boundary must be an MCPBoundary")
        boundary.bind_mission_context(self._mission_registry, next(iter(self._agents), ""))
        boundary._application_id = self._application_id
        boundary._release_issuer = self.authorize_data_release
        boundary._release_policy_version_provider = (
            lambda: self._data_release_policy.version
            if self._data_release_policy is not None
            else None
        )
        return boundary

    def authorize_data_release(self, request: DataReleaseRequest) -> DataReleaseAuthorization:
        if self._data_release_policy is None:
            raise ToolExecutionDenied(_blocked_release("release_policy_missing"))
        decision = evaluate_data_release(request, self._data_release_policy)
        if decision.decision_type is DecisionType.APPROVAL_REQUIRED:
            raise ToolApprovalRequired(decision)
        if decision.decision_type is not DecisionType.ALLOW:
            raise ToolExecutionDenied(decision)
        return DataReleaseAuthorization.issue(
            request,
            application_id=self._application_id,
            policy_version=self._data_release_policy.version,
            _authority=_RELEASE_AUTHORITY,
        )

    def add_agent(self, agent: Agent) -> Agent:
        if self._state is AppState.RUNNING:
            raise RuntimeError("cannot register agents after application is running")
        if not isinstance(agent, Agent):
            raise TypeError("agent must be an Agent")
        if (
            agent._lifecycle_registry is not None
            and agent._lifecycle_registry is not self._lifecycle_registry
        ):
            raise ValueError("agent is already bound to another PreventApp")
        if agent._application_id != "preventai" and agent._application_id != self._application_id:
            raise ValueError("agent is already bound to another application")
        if agent.agent_id in self._agents and self._agents[agent.agent_id] is not agent:
            raise ValueError(f"agent identity already registered: {agent.agent_id}")
        if self._authenticator is not None:
            if agent._credential is None:
                raise ValueError("authenticated applications require an agent credential")
            if not self._authenticator.has_identity(agent.agent_id):
                self._authenticator.register(agent.identity, agent._credential)
        if agent.agent_id not in self._agents:
            self._lifecycle_registry.register(agent.identity)
            self._admission_registry.admit(
                agent.agent_id,
                application_id=self._application_id,
                session_id=agent.session_id,
                capabilities=tuple(),
                tools=tuple(sorted(agent._tools.keys())),
                version="1.0.0",
                provenance="framework",
                policy_version=(
                    self._policy_integrity.get_active_version("application-security")
                    if hasattr(self._policy_integrity, "get_active_version")
                    else "1"
                ),
                configuration_digest=_agent_admission_configuration_digest(
                    application_id=self._application_id,
                    agent_id=agent.agent_id,
                    version="1.0.0",
                    provenance="framework",
                    session_id=agent.session_id,
                    tools=tuple(sorted(agent._tools.keys())),
                    policy_version=(
                        self._policy_integrity.get_active_version("application-security")
                        if hasattr(self._policy_integrity, "get_active_version")
                        else "1"
                    ),
                ),
                integrity_digest=hashlib.sha256(b"framework-managed-agent-envelope").hexdigest(),
            )
        self._agents[agent.agent_id] = agent
        agent._application_id = self._application_id
        agent._admission_registry = self._admission_registry
        agent._policy_integrity = self._policy_integrity
        agent._bind(
            self._engine,
            self._data_flow,
            self._runtime_controller,
            self._application_id,
            self._authenticator,
            self._approval_manager,
            self._delegation_registry,
            self._agents,
            self._approval_policy,
            data_release_policy=self._data_release_policy,
            audit_logger=self._audit_logger,
            lifecycle_registry=self._lifecycle_registry,
        )
        agent._mission_registry = self._mission_registry
        agent._tool_trust_registry = self._tool_trust_registry
        agent._require_tool_trust = self._require_tool_trust
        for tool in agent.tools.values():
            tool._data_release_policy = self._data_release_policy
            if tool._runtime_policy is None:
                tool._runtime_controller = self._runtime_controller
            else:
                tool._runtime_controller = RuntimeController(
                    tool._runtime_policy,
                    application_id=self._application_id,
                )
            tool._delegation_registry = self._delegation_registry
            tool._mission_registry = self._mission_registry
            tool._bind_tool_trust(
                self._tool_trust_registry, required=self._require_tool_trust
            )
            for boundary in (
                tool._sandbox,
                tool._network,
                tool._filesystem,
                tool._database,
            ):
                if boundary is not None:
                    boundary.bind_mission_context(
                        self._mission_registry, agent.agent_id, tool=tool.name
                    )
            tool._approval_manager = self._approval_manager
            tool._audit_logger = self._audit_logger
            tool._lifecycle_registry = self._lifecycle_registry
        agent._app_state = self._state
        return agent

    def run(self) -> "PreventApp":
        """Validate registration and transition the app to its running state."""

        if self._state is AppState.RUNNING:
            return self
        for agent in self._agents.values():
            if agent._engine is not self._engine or not agent.tools:
                raise RuntimeError("application has invalid agent registration")
            for tool in agent.tools.values():
                if tool._engine is not self._engine or tool._agent is not agent:
                    raise RuntimeError("application has invalid tool registration")
                tool._mission_registry = self._mission_registry
                for boundary in (
                    tool._sandbox,
                    tool._network,
                    tool._filesystem,
                    tool._database,
                ):
                    if boundary is not None:
                        boundary.bind_mission_context(
                            self._mission_registry, agent.agent_id, tool=tool.name
                        )
        for agent in self._agents.values():
            admission = self._admission_registry.get(agent.agent_id)
            if admission is None or admission.application_id != self._application_id:
                raise RuntimeError("application contains an unadmitted agent")
            expected_tools = tuple(sorted(agent._tools.keys()))
            if admission.tool_set != expected_tools:
                policy_version = self._policy_integrity.get_active_version(
                    "application-security"
                )
                admission = self._admission_registry.admit(
                    agent.agent_id,
                    application_id=self._application_id,
                    session_id=agent.session_id,
                    capabilities=admission.capability_envelope,
                    tools=expected_tools,
                    version=admission.version,
                    provenance=admission.provenance,
                    policy_version=policy_version,
                    configuration_digest=_agent_admission_configuration_digest(
                        application_id=self._application_id,
                        agent_id=agent.agent_id,
                        version=admission.version,
                        provenance=admission.provenance,
                        session_id=agent.session_id,
                        parent_agent_id=admission.parent_agent_id,
                        mission_id=admission.mission_id,
                        mission_version=admission.mission_version,
                        capabilities=admission.capability_envelope,
                        tools=expected_tools,
                        policy_version=policy_version,
                    ),
                    integrity_digest=admission.integrity_digest,
                    expires_at=admission.expires_at,
                )
            if admission.state is AgentAdmissionState.EXPIRED or admission.state is AgentAdmissionState.REVOKED:
                raise RuntimeError("application contains a denied agent")
            lifecycle_state = self._lifecycle_registry.get_state(agent.agent_id)
            if lifecycle_state is LifecycleState.REGISTERED:
                self._lifecycle_registry.activate(agent.agent_id)
            elif lifecycle_state is not LifecycleState.ACTIVE:
                raise RuntimeError("application contains a non-executable agent")
        self._state = AppState.READY
        self._state = AppState.RUNNING
        for agent in self._agents.values():
            agent._app_state = self._state
        return self
