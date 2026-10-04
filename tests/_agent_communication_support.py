from datetime import datetime, timezone
import time

from preventai import (
    Agent, AgentCommunicationBoundary, AgentMessage, AgentMessagePolicy,
    AgentSecurityContext, PreventApp,
)
from preventai.runtime import RuntimeController, RuntimePolicy


def make_message(**changes):
    values = dict(
        message_id="m1", message_type="request", sender_agent_id="agent-a",
        receiver_agent_id="agent-b", sender_session_id="session-a",
        receiver_session_id="session-b", conversation_id="conversation-1",
        parent_message_id=None,         timestamp=datetime(2025, 1, 1, tzinfo=timezone.utc),
        expires_at=4102444800.0, payload={"value": "hello"},
        capability_scope=(), delegation_scope=(), classification="internal",
        correlation_id="corr-1", protocol_version="1.0", tenant_id="tenant-a",
        chain_depth=0,
    )
    values.update(changes)
    return AgentMessage(**values)


def make_boundary(**policy_changes):
    calls = []
    hooks = {name: (lambda message: True) for name in (
        "policy_check", "permission_check", "delegation_check",
        "dataflow_check", "runtime_check", "approval_check",
    )}
    hooks.update({k: policy_changes.pop(k) for k in list(policy_changes)
                  if k.endswith("_check")})
    policy = AgentMessagePolicy(
        allowed_senders=("agent-a",), allowed_receivers=("agent-b",),
        allowed_tenants=("tenant-a",), **policy_changes
    )
    context = AgentSecurityContext("agent-a", "session-a", "tenant-a", True)
    app = PreventApp()
    sender = app.add_agent(Agent("sender", agent_id="agent-a"))
    receiver = app.add_agent(Agent("receiver", agent_id="agent-b"))
    sender.session_id = "session-a"
    receiver.session_id = "session-b"
    @sender.tool("sender.noop")
    def sender_noop():
        return "unused"
    @receiver.tool("receiver.noop")
    def receiver_noop():
        return "unused"
    app.run()
    context = AgentSecurityContext("agent-a", "session-a", "tenant-a", True)
    boundary = AgentCommunicationBoundary(
        policy, lambda message: (calls.append(message), {"accepted": True})[1],
        sender_contexts={"agent-a": context}, **hooks,
        output_check=lambda result: True,
        audit_check=lambda message: True,
        runtime_controller=RuntimeController(RuntimePolicy(), application_id="tests"),
    )
    app.bind_agent_communication(boundary)
    implementation_issue = type(boundary).issue
    capability = boundary._issuance_capability
    boundary.issue = lambda message, context: implementation_issue(
        boundary, message, context, issuance_capability=capability
    )
    return boundary, context, calls
