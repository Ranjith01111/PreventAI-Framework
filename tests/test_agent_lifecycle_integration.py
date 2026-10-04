from datetime import datetime, timedelta, timezone
import sys
from types import MappingProxyType

import pytest

from preventai import (
    Agent,
    AgentCommunicationBoundary,
    AgentCommunicationDenied,
    AgentMessagePolicy,
    AgentLifecycleRegistry,
    AgentSecurityContext,
    Authenticator,
    AuditEventType,
    AuditLogger,
    Capability,
    Credential,
    DecisionType,
    LifecycleError,
    LifecycleRecord,
    LifecycleState,
    MCPBoundary,
    MCPBoundaryAdapter,
    MCPPolicy,
    PreventApp,
    RegisteredAgentIdentity,
    ToolExecutionDenied,
)
from preventai.core import (
    PermissionCategory,
    PermissionEngine,
    PermissionRule,
    PolicyEngine,
    PolicyRule,
)
from preventai.tools import SandboxCommand, SandboxPolicy
from preventai.runtime import RuntimeController, RuntimePolicy
from tests._agent_communication_support import make_message


DENIED_STATES = (
    LifecycleState.SUSPENDED,
    LifecycleState.QUARANTINED,
    LifecycleState.REVOKED,
    LifecycleState.TERMINATED,
)


def make_app(*, authenticator=None, audit_logger=None, action="read-data"):
    permissions = PermissionEngine([
        PermissionRule(
            "read",
            PermissionCategory.READ,
            action="*",
            resource="*",
        ),
        PermissionRule(
            "execute",
            PermissionCategory.EXECUTE,
            action="*",
            resource="*",
        ),
    ])
    app = PreventApp(
        policy=PolicyEngine([PolicyRule("allow", DecisionType.ALLOW, action="*")]),
        permissions=permissions,
        authenticator=authenticator,
        audit_logger=audit_logger,
    )
    agent = app.add_agent(Agent(
        "lifecycle-agent",
        agent_id="lifecycle-agent" if authenticator else None,
        credential=Credential("lifecycle-agent", "local-secret") if authenticator else None,
    ))
    return app, agent


def test_prevent_app_owns_registry_and_activates_registered_identity():
    app, agent = make_app()

    @agent.tool("read-data")
    def read():
        return "safe"

    assert app.lifecycle.get_state(agent.agent_id) is LifecycleState.REGISTERED
    app.run()
    assert app.lifecycle.get_state(agent.agent_id) is LifecycleState.ACTIVE
    assert read() == "safe"
    assert app.lifecycle is agent._lifecycle_registry


def test_communication_boundary_uses_the_app_owned_registry():
    app, sender = make_app()
    receiver = app.add_agent(Agent("receiver", agent_id="receiver"))

    @sender.tool("read-data")
    def sender_tool():
        return "sender"

    @receiver.tool("read-data")
    def receiver_tool():
        return "receiver"

    app.run()
    sender_context = AgentSecurityContext(
        sender.agent_id, sender.session_id, "tenant-a", True
    )
    boundary = AgentCommunicationBoundary(
        AgentMessagePolicy(
            allowed_senders=(sender.agent_id,),
            allowed_receivers=(receiver.agent_id,),
            allowed_tenants=("tenant-a",),
        ),
        lambda _message: {"accepted": True},
        sender_contexts={sender.agent_id: sender_context},
        policy_check=lambda _message: True,
        permission_check=lambda _message: True,
        delegation_check=lambda _message: True,
        dataflow_check=lambda _message: True,
        runtime_check=lambda _message: True,
        approval_check=lambda _message: True,
        output_check=lambda _result: True,
        audit_check=lambda _message: True,
        runtime_controller=RuntimeController(
            RuntimePolicy(), application_id="lifecycle-integration"
        ),
    )
    app.bind_agent_communication(boundary)
    assert boundary._lifecycle is app.lifecycle
    message = make_message(
        sender_agent_id=sender.agent_id,
        receiver_agent_id=receiver.agent_id,
        sender_session_id=sender.session_id,
    )
    boundary.validate(message, sender_context)
    app.lifecycle.suspend(receiver.agent_id)
    with pytest.raises(AgentCommunicationDenied):
        boundary.validate(message, sender_context)


def test_app_context_automatically_gates_managed_mcp_adapter():
    app = PreventApp(
        policy=PolicyEngine([PolicyRule("allow", DecisionType.ALLOW, action="*")]),
        permissions=PermissionEngine([
            PermissionRule("read", PermissionCategory.READ, action="*", resource="*")
        ]),
    )
    sender = app.add_agent(Agent("sender", agent_id="agent-a"))
    receiver = app.add_agent(Agent("receiver", agent_id="agent-b"))

    @sender.tool("read-data")
    def sender_tool():
        return "sender"

    @receiver.tool("read-data")
    def receiver_tool():
        return "receiver"

    app.run()
    calls = []
    mcp = MCPBoundary(
        MCPPolicy(allowed_tools=("customer.read",), allowed_clients=("agent-a",)),
        lambda request: (calls.append(request), {"ok": True})[1],
        policy_check=lambda _request: True,
        permission_check=lambda _request: True,
        delegation_check=lambda _request: True,
        dataflow_check=lambda _request: True,
        runtime_check=lambda _request: True,
        approval_check=lambda _request: True,
        output_check=lambda _response: True,
    )
    adapter = MCPBoundaryAdapter(mcp, boundary_id="mcp-managed")
    sender_context = AgentSecurityContext(
        sender.agent_id, sender.session_id, "tenant-a", True
    )
    communication = AgentCommunicationBoundary(
        AgentMessagePolicy(
            allowed_senders=("agent-a",),
            allowed_receivers=("agent-b",),
            allowed_tenants=("tenant-a",),
        ),
        lambda _message: {"unused": True},
        sender_contexts={"agent-a": sender_context},
        policy_check=lambda _message: True,
        permission_check=lambda _message: True,
        delegation_check=lambda _message: True,
        dataflow_check=lambda _message: True,
        runtime_check=lambda _message: True,
        approval_check=lambda _message: True,
        output_check=lambda _response: True,
        audit_check=lambda _message: True,
        runtime_controller=RuntimeController(
            RuntimePolicy(), application_id="managed-mcp-test"
        ),
    )
    app.bind_agent_communication(communication)
    message = make_message(
        payload={
            "message": {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {"name": "customer.read", "arguments": {}},
            },
            "protocol_version": "2025-06-18",
            "client_id": sender.agent_id,
            "session_id": sender.session_id,
        },
        sender_session_id=sender.session_id,
    )
    authorization = communication.issue(
        message, sender_context,
        issuance_capability=communication._issuance_capability,
    )
    result = communication.dispatch_managed(
        message, authorization, sender_context, adapter
    )
    assert result.accepted and len(calls) == 1

    app.lifecycle.revoke(sender.agent_id)
    with pytest.raises(AgentCommunicationDenied):
        communication.dispatch_managed(
            message, authorization, sender_context, adapter
        )
    assert len(calls) == 1


def test_valid_capability_cannot_be_reused_after_agent_revocation():
    authenticator = Authenticator()
    credential = Credential("lifecycle-agent", "local-secret")
    authenticator.register(
        RegisteredAgentIdentity("lifecycle-agent"),
        credential,
    )
    app, agent = make_app(authenticator=authenticator)

    @agent.tool("read-data")
    def read():
        return "safe"

    app.run()
    grant = agent.delegate(
        agent,
        [Capability("read-data", resource=read.tool_id, tool="read-data")],
        expires_at=datetime.now(timezone.utc) + timedelta(minutes=5),
    )
    agent.use_delegation(grant)
    assert read() == "safe"

    app.lifecycle.revoke(agent.agent_id)
    with pytest.raises(ToolExecutionDenied) as error:
        read()
    assert error.value.decision.decision_type is DecisionType.BLOCK
    assert error.value.decision.reason.value == "lifecycle_invalid"


@pytest.mark.parametrize("state", DENIED_STATES)
def test_existing_authenticated_session_does_not_survive_lifecycle_change(state):
    authenticator = Authenticator()
    credential = Credential("lifecycle-agent", "local-secret")
    authenticator.register(
        RegisteredAgentIdentity("lifecycle-agent"),
        credential,
    )
    app, agent = make_app(authenticator=authenticator)

    @agent.tool("read-data")
    def read():
        return "safe"

    app.run()
    session = authenticator.authenticate(
        agent.agent_id, credential, session_id=agent.session_id
    )
    assert session.authenticated
    assert read() == "safe"

    app.lifecycle.transition(agent.agent_id, state)
    assert session.authenticated
    with pytest.raises(ToolExecutionDenied):
        read()


def test_tool_revalidates_lifecycle_at_final_sandbox_dispatch(tmp_path, monkeypatch):
    app, agent = make_app()
    process_starts = []
    monkeypatch.setattr(
        "preventai.tools.sandbox.subprocess.Popen",
        lambda *args, **kwargs: process_starts.append(args),
    )

    @agent.tool(
        "run-command",
        sandbox_policy=SandboxPolicy(
            allowed_commands=(sys.executable,),
            working_directory=str(tmp_path),
            allowed_paths=(str(tmp_path),),
        ),
    )
    def run():
        app.lifecycle.revoke(agent.agent_id)
        return SandboxCommand(sys.executable, ("-c", "print('should-not-run')"))

    app.run()
    with pytest.raises(ToolExecutionDenied) as error:
        run()
    assert error.value.decision.reason.value == "lifecycle_invalid"
    assert process_starts == []


def test_lifecycle_lookup_failure_and_malformed_record_fail_closed(monkeypatch):
    app, agent = make_app()

    @agent.tool("read-data")
    def read():
        return "unsafe"

    app.run()
    registry = app.lifecycle
    monkeypatch.setattr(
        AgentLifecycleRegistry,
        "get_state",
        lambda _self, _agent_id: (_ for _ in ()).throw(RuntimeError("lookup failure")),
    )
    with pytest.raises(ToolExecutionDenied):
        read()
    monkeypatch.undo()

    record = registry.get_record(agent.agent_id)
    assert record is not None
    malformed = LifecycleRecord(
        agent.agent_id,
        "active",  # type: ignore[arg-type]
        record.registered_at,
        record.updated_at,
    )
    registry._record_snapshot = MappingProxyType({agent.agent_id: malformed})
    with pytest.raises(ToolExecutionDenied):
        read()


def test_lifecycle_audit_failure_latches_execution_closed():
    app, agent = make_app()
    calls = []

    @agent.tool("read-data")
    def read():
        calls.append(True)
        return "unsafe"

    app.run()
    audit = app._audit_logger

    def fail_record(*args, **kwargs):
        raise OSError("audit unavailable")

    audit.record = fail_record
    with pytest.raises(OSError):
        app.lifecycle.revoke(agent.agent_id)
    assert app.lifecycle.get_state(agent.agent_id) is LifecycleState.ACTIVE
    with pytest.raises(ToolExecutionDenied) as error:
        read()
    assert error.value.decision.decision_type is DecisionType.BLOCK
    assert calls == []


def test_lifecycle_transitions_emit_structured_audit_and_invalid_transition():
    audit = AuditLogger()
    registry = AgentLifecycleRegistry(audit_logger=audit)
    registry.register("audited-agent")
    registry.activate("audited-agent")
    registry.suspend("audited-agent")
    registry.activate("audited-agent")
    registry.quarantine("audited-agent")
    registry.revoke("audited-agent")
    registry.terminate("audited-agent")
    with pytest.raises(LifecycleError):
        registry.activate("audited-agent")

    events = audit.events()
    assert [event.event_type for event in events].count(
        AuditEventType.LIFECYCLE_TRANSITION
    ) == 7
    assert events[-1].event_type is AuditEventType.LIFECYCLE_BLOCKED
    assert audit.verify().valid
