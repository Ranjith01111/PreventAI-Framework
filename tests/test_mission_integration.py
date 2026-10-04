from datetime import datetime, timedelta, timezone

import pytest

from preventai import (
    Agent,
    ApprovalPolicy,
    ApprovalRule,
    Authenticator,
    Capability,
    Credential,
    DecisionType,
    Mission,
    PermissionCategory,
    PermissionEngine,
    PermissionRule,
    PolicyEngine,
    PolicyRule,
    PreventApp,
    ToolExecutionDenied,
)
from preventai.core.engine import SecurityEngine


def build_tool(*, authenticated=False):
    authenticator = Authenticator() if authenticated else None
    credential = Credential("mission-integration-agent", "local-test-secret") if authenticated else None
    app = PreventApp(
        policy=PolicyEngine([PolicyRule("allow", DecisionType.ALLOW, action="*")]),
        permissions=PermissionEngine([
            PermissionRule("read", PermissionCategory.READ, action="*"),
            PermissionRule("execute", PermissionCategory.EXECUTE, action="*"),
        ]),
        authenticator=authenticator,
    )
    agent = app.add_agent(
        Agent("mission-integration-agent", agent_id="mission-integration-agent", credential=credential)
    )
    calls = []

    @agent.tool("customer.read")
    def lookup():
        calls.append("backend")
        return {"ok": True}

    app.run()
    return app, agent, lookup, calls


def issue_tool_mission(app, agent, tool, *, mission_id="integration-mission", expires=None, version=1):
    mission = Mission(
        mission_id=mission_id,
        agent_id=agent.agent_id,
        action=tool.name,
        resource=tool.tool_id,
        tool=tool.name,
        version=version,
        expires_at=expires or datetime.now(timezone.utc) + timedelta(minutes=5),
    )
    app.missions.register(mission)
    app.bind_mission(agent, mission_id, version)
    return mission


def test_framework_requires_an_explicit_bound_mission_before_tool_execution():
    app, _, tool, calls = build_tool()
    with pytest.raises(ToolExecutionDenied):
        tool()
    assert calls == []
    assert app.missions.require_missions is True


def test_framework_rechecks_mission_after_request_starts_before_callable(monkeypatch):
    app, agent, tool, calls = build_tool()
    mission = issue_tool_mission(app, agent, tool)
    engine = agent.tools[tool.name]._engine
    evaluate = engine.evaluate

    def revoke_after_initial_mission_check(self, request):
        app.missions.revoke(mission.mission_id, mission.version)
        return evaluate(request)

    monkeypatch.setattr(SecurityEngine, "evaluate", revoke_after_initial_mission_check)
    with pytest.raises(ToolExecutionDenied):
        tool()
    assert calls == []


def test_existing_authenticated_session_does_not_survive_mission_revocation():
    app, agent, tool, calls = build_tool(authenticated=True)
    mission = issue_tool_mission(app, agent, tool)
    assert tool() == {"ok": True}
    app.missions.revoke(mission.mission_id, mission.version)
    with pytest.raises(ToolExecutionDenied):
        tool()
    assert calls == ["backend"]


def test_expired_mission_blocks_existing_tool_before_callable():
    app, agent, tool, calls = build_tool()
    now = datetime.now(timezone.utc)
    mission = Mission(
        "expired-integration-mission",
        agent.agent_id,
        action=tool.name,
        resource=tool.tool_id,
        tool=tool.name,
        issued_at=now - timedelta(minutes=2),
        expires_at=now - timedelta(minutes=1),
    )
    app.missions.register(mission)
    app.bind_mission(agent, mission.mission_id, mission.version)
    with pytest.raises(ToolExecutionDenied):
        tool()
    assert calls == []


def test_trusted_binding_context_is_authoritative_for_tool_scope():
    app, agent, tool, calls = build_tool()
    mission = Mission(
        "context-bound-mission",
        agent.agent_id,
        action=tool.name,
        resource=tool.tool_id,
        tool=tool.name,
        context_constraints={"tenant": "tenant-a"},
        expires_at=datetime.now(timezone.utc) + timedelta(minutes=5),
    )
    app.missions.register(mission)
    app.bind_mission(
        agent, mission.mission_id, mission.version,
        context={"tenant": "tenant-a", "environment": "production"},
    )
    assert tool() == {"ok": True}
    assert calls == ["backend"]


def test_approval_never_substitutes_for_a_missing_mission():
    app = PreventApp(
        policy=PolicyEngine([PolicyRule("allow", DecisionType.ALLOW, action="*")]),
        permissions=PermissionEngine([
            PermissionRule("read", PermissionCategory.READ, action="*"),
        ]),
        approval_policy=ApprovalPolicy([ApprovalRule("approval", action="sensitive")]),
    )
    agent = app.add_agent(Agent("approval-agent", agent_id="approval-agent"))
    calls = []

    @agent.tool("sensitive")
    def sensitive():
        calls.append(True)

    app.run()
    with pytest.raises(ToolExecutionDenied):
        sensitive()
    assert calls == []


def test_delegation_capability_and_mission_are_independent_gates():
    authenticator = Authenticator()
    app = PreventApp(
        policy=PolicyEngine([PolicyRule("allow", DecisionType.ALLOW, action="*")]),
        permissions=PermissionEngine([
            PermissionRule("read", PermissionCategory.READ, action="*", resource="*", tool="*"),
        ]),
        authenticator=authenticator,
    )
    issuer = app.add_agent(Agent(
        "issuer", agent_id="issuer", credential=Credential("issuer", "issuer-secret")
    ))
    subject = app.add_agent(Agent(
        "subject", agent_id="subject", credential=Credential("subject", "subject-secret")
    ))
    calls = []

    @subject.tool("customer.read")
    def read_customer():
        calls.append("backend")
        return "ok"

    @issuer.tool("issuer.noop")
    def issuer_noop():
        return "unused"

    app.run()
    grant = issuer.delegate(
        subject,
        [Capability("customer.read", read_customer.tool_id, "customer.read")],
        expires_at=datetime.now(timezone.utc) + timedelta(minutes=5),
    )
    subject.use_delegation(grant)
    mission = Mission(
        "subject-mission", subject.agent_id, action="customer.read",
        resource=read_customer.tool_id, tool="customer.read",
        expires_at=datetime.now(timezone.utc) + timedelta(minutes=5),
    )
    app.missions.register(mission)
    app.bind_mission(subject, mission.mission_id, 1)
    assert read_customer() == "ok"

    app.missions.revoke(mission.mission_id)
    with pytest.raises(ToolExecutionDenied):
        read_customer()
    assert calls == ["backend"]

    app.missions.register(Mission(
        "subject-mission", subject.agent_id, action="customer.read",
        resource=read_customer.tool_id, tool="customer.read", version=2,
        expires_at=datetime.now(timezone.utc) + timedelta(minutes=5),
    ))
    app.bind_mission(subject, "subject-mission", 2)
    app._delegation_registry.revoke(grant.grant_id)
    with pytest.raises(ToolExecutionDenied):
        read_customer()
    assert calls == ["backend"]
