from datetime import datetime, timedelta, timezone
import sys

import pytest

from preventai import (
    Agent,
    Authenticator,
    Capability,
    Credential,
    DataFlowDestination,
    DataReleasePolicy,
    DataSource,
    DecisionType,
    DestinationTrust,
    FlowClassification,
    FlowTrust,
    Mission,
    PermissionCategory,
    PermissionEngine,
    PermissionRule,
    PolicyEngine,
    PolicyRule,
    PreventApp,
    RuntimePolicy,
    ToolApprovalRequired,
    ToolExecutionDenied,
)
from preventai.core.engine import SecurityEngine
from preventai.tools import SandboxCommand, SandboxPolicy


def _mission(app, agent, tool, *, mission_id="phase32-mission"):
    mission = Mission(
        mission_id,
        agent.agent_id,
        action=tool.name,
        resource=tool.tool_id,
        tool=tool.name,
        expires_at=datetime.now(timezone.utc) + timedelta(minutes=5),
        allowed_actions=(tool.name,),
        allowed_resources=(tool.tool_id,),
        allowed_tools=(tool.name,),
    )
    app.missions.register(mission)
    app.bind_mission(agent, mission.mission_id, mission.version)
    return mission


def test_phase32_positive_authenticated_delegated_runtime_guard_and_audit():
    auth = Authenticator()
    issuer = Agent(
        "phase32-issuer",
        agent_id="phase32-issuer",
        credential=Credential("phase32-issuer", "issuer-secret"),
    )
    subject = Agent(
        "phase32-subject",
        agent_id="phase32-subject",
        credential=Credential("phase32-subject", "subject-secret"),
    )
    auth.register(issuer.identity, issuer._credential)
    auth.register(subject.identity, subject._credential)
    app = PreventApp(
        authenticator=auth,
        runtime_policy=RuntimePolicy(max_tool_calls=1),
        policy=PolicyEngine([PolicyRule("allow", DecisionType.ALLOW, action="customer.read")]),
        permissions=PermissionEngine([
            PermissionRule(
                "read",
                PermissionCategory.READ,
                action="customer.read",
                agent="phase32-subject",
                resource="*",
            )
        ]),
    )
    app.add_agent(issuer)
    app.add_agent(subject)
    calls = []

    @issuer.tool("issuer.noop")
    def issuer_noop():
        return "unused"

    @subject.tool("customer.read", input_schema={
        "type": "object",
        "properties": {"customer_id": {"type": "string", "min_length": 1}},
        "required": ["customer_id"],
        "additional_properties": False,
    })
    def read_customer(customer_id):
        calls.append(customer_id)
        return {"customer_id": customer_id}

    app.run()
    mission = _mission(app, subject, read_customer)
    grant = issuer.delegate(
        subject,
        [Capability("customer.read", read_customer.tool_id, "customer.read")],
        expires_at=datetime.now(timezone.utc) + timedelta(minutes=5),
    )
    subject.use_delegation(grant)

    assert read_customer("c-1") == {"customer_id": "c-1"}
    assert calls == ["c-1"]
    assert auth.authenticate(
        subject.agent_id, subject._credential, session_id=subject.session_id
    ).authenticated
    assert app.verify_agent_admission(
        subject,
        session_id=subject.session_id,
        policy_version=app.policy_integrity.get_active_version("application-security"),
    ).decision_type is DecisionType.ALLOW
    assert app.missions.binding_for(subject.agent_id).mission_id == mission.mission_id
    event_types = {event.event_type for event in app._audit_logger.events()}
    assert "tool_verified" in {event.value for event in event_types}
    assert "policy_evaluation" in {event.value for event in event_types}


def test_phase32_positive_managed_sandbox_boundary(tmp_path):
    app = PreventApp(
        runtime_policy=RuntimePolicy(max_tool_calls=1),
        policy=PolicyEngine([PolicyRule("allow", DecisionType.ALLOW, action="run")]),
        permissions=PermissionEngine([
            PermissionRule("execute", PermissionCategory.EXECUTE, action="run")
        ]),
    )
    agent = app.add_agent(Agent("phase32-sandbox"))

    @agent.tool(
        "run",
        sandbox_policy=SandboxPolicy(
            allowed_commands=(sys.executable,),
            working_directory=str(tmp_path),
            allowed_paths=(str(tmp_path),),
        ),
    )
    def run():
        return SandboxCommand(sys.executable, ("-c", "print('phase32-safe')"))

    app.run()
    assert run()["stdout"].strip() == "phase32-safe"


def test_phase32_unadmitted_agent_has_no_callable_side_effect():
    agent = Agent("phase32-unadmitted")
    calls = []

    @agent.tool("unadmitted")
    def unadmitted():
        calls.append(True)

    with pytest.raises(ToolExecutionDenied):
        unadmitted()
    assert calls == []


def test_phase32_lifecycle_and_mission_revocation_block_before_callable():
    app = PreventApp(
        policy=PolicyEngine([PolicyRule("allow", DecisionType.ALLOW, action="read")]),
        permissions=PermissionEngine([
            PermissionRule("read", PermissionCategory.READ, action="read")
        ]),
    )
    agent = app.add_agent(Agent("phase32-revocation"))
    calls = []

    @agent.tool("read")
    def read():
        calls.append(True)
        return "ok"

    app.run()
    mission = _mission(app, agent, read)
    app.missions.revoke(mission.mission_id, mission.version)
    with pytest.raises(ToolExecutionDenied):
        read()
    assert calls == []

    app.lifecycle.revoke(agent.agent_id)
    with pytest.raises(ToolExecutionDenied):
        read()
    assert calls == []


def test_phase32_revoked_delegation_and_tool_trust_block_before_callable():
    auth = Authenticator()
    issuer = Agent("phase32-issuer-2", agent_id="phase32-issuer-2",
                   credential=Credential("phase32-issuer-2", "issuer-secret"))
    subject = Agent("phase32-subject-2", agent_id="phase32-subject-2",
                    credential=Credential("phase32-subject-2", "subject-secret"))
    auth.register(issuer.identity, issuer._credential)
    auth.register(subject.identity, subject._credential)
    app = PreventApp(
        authenticator=auth,
        policy=PolicyEngine([PolicyRule("allow", DecisionType.ALLOW, action="read")]),
        permissions=PermissionEngine([
            PermissionRule("read", PermissionCategory.READ, action="read", resource="*")
        ]),
    )
    app.add_agent(issuer)
    app.add_agent(subject)
    calls = []

    @issuer.tool("issuer.noop")
    def issuer_noop():
        return "unused"

    @subject.tool("read")
    def read():
        calls.append(True)
        return "ok"

    app.run()
    _mission(app, subject, read)
    grant = issuer.delegate(
        subject,
        [Capability("read", read.tool_id, "read")],
        expires_at=datetime.now(timezone.utc) + timedelta(minutes=5),
    )
    subject.use_delegation(grant)
    app._delegation_registry.revoke(grant.grant_id)
    with pytest.raises(ToolExecutionDenied):
        read()
    assert calls == []

    trust_app = PreventApp(
        policy=PolicyEngine([PolicyRule("allow", DecisionType.ALLOW, action="read")]),
        permissions=PermissionEngine([
            PermissionRule("read", PermissionCategory.READ, action="read")
        ]),
    )
    trust_agent = trust_app.add_agent(Agent("phase32-untrusted"))
    trust_calls = []

    @trust_agent.tool("read")
    def untrusted_read():
        trust_calls.append(True)
        return "ok"

    trust_app.run()
    _mission(trust_app, trust_agent, untrusted_read)
    trust_app.tool_trust.revoke(untrusted_read.tool_id, untrusted_read.version)
    with pytest.raises(ToolExecutionDenied):
        untrusted_read()
    assert trust_calls == []


def test_phase32_revoked_policy_blocks_before_callable():
    app = PreventApp(
        policy=PolicyEngine([PolicyRule("allow", DecisionType.ALLOW, action="read")]),
        permissions=PermissionEngine([
            PermissionRule("read", PermissionCategory.READ, action="read")
        ]),
    )
    agent = app.add_agent(Agent("phase32-policy"))
    calls = []

    @agent.tool("read")
    def read():
        calls.append(True)
        return "ok"

    app.run()
    _mission(app, agent, read)
    app.policy_integrity.revoke("application-security", 1)
    with pytest.raises(ToolExecutionDenied):
        read()
    assert calls == []


def test_phase32_release_and_approval_denials_have_no_callable_side_effect():
    release_app = PreventApp(
        policy=PolicyEngine([PolicyRule("allow", DecisionType.ALLOW, action="export")]),
        permissions=PermissionEngine([
            PermissionRule("read", PermissionCategory.READ, action="export")
        ]),
        data_release_policy=DataReleasePolicy(
            allowed_destinations=("approved.sink",),
            allowed_destination_trust=(DestinationTrust.INTERNAL,),
            allowed_classifications=(FlowClassification.PUBLIC,),
        ),
    )
    release_agent = release_app.add_agent(Agent("phase32-release"))
    release_calls = []

    @release_agent.tool(
        "export",
        data_source=DataSource("private-db", FlowClassification.PRIVATE, FlowTrust.TRUSTED),
        data_destination=DataFlowDestination("unapproved.sink"),
    )
    def export():
        release_calls.append(True)
        return "secret"

    release_app.run()
    _mission(release_app, release_agent, export)
    with pytest.raises(ToolExecutionDenied):
        export()
    assert release_calls == []

    approval_app = PreventApp(
        policy=PolicyEngine([PolicyRule("approval", DecisionType.APPROVAL_REQUIRED, action="send")]),
        permissions=PermissionEngine([
            PermissionRule("network", PermissionCategory.NETWORK, action="send")
        ]),
    )
    approval_agent = approval_app.add_agent(Agent("phase32-approval"))
    approval_calls = []

    @approval_agent.tool("send")
    def send():
        approval_calls.append(True)
        return "sent"

    approval_app.run()
    _mission(approval_app, approval_agent, send)
    with pytest.raises(ToolApprovalRequired):
        send()
    assert approval_calls == []


def test_phase32_malformed_security_evaluation_fails_closed():
    decision = SecurityEngine().evaluate(object())
    assert decision.decision_type is DecisionType.BLOCK


def test_phase32_boundary_denial_prevents_sandbox_dispatch(tmp_path):
    app = PreventApp(
        policy=PolicyEngine([PolicyRule("deny", DecisionType.BLOCK, action="run")]),
        permissions=PermissionEngine([
            PermissionRule("execute", PermissionCategory.EXECUTE, action="run")
        ]),
    )
    agent = app.add_agent(Agent("phase32-boundary"))
    calls = []

    @agent.tool(
        "run",
        sandbox_policy=SandboxPolicy(
            allowed_commands=(sys.executable,),
            working_directory=str(tmp_path),
            allowed_paths=(str(tmp_path),),
        ),
    )
    def run():
        calls.append(True)
        return SandboxCommand(sys.executable, ("-c", "print('must-not-run')"))

    app.run()
    with pytest.raises(ToolExecutionDenied):
        run()
    assert calls == []
