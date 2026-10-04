from datetime import datetime, timedelta, timezone

import pytest

from preventai import (
    Agent,
    DecisionType,
    Mission,
    PermissionCategory,
    PermissionEngine,
    PermissionRule,
    PolicyEngine,
    PolicyRule,
    PreventApp,
    ToolExecutionDenied,
    ToolProvenance,
)


def _app():
    app = PreventApp(
        policy=PolicyEngine([PolicyRule("allow", DecisionType.ALLOW, action="*")]),
        permissions=PermissionEngine([
            PermissionRule("execute", PermissionCategory.EXECUTE, action="*", resource="*"),
            PermissionRule("read", PermissionCategory.READ, action="*", resource="*"),
        ]),
    )
    agent = app.add_agent(Agent("integration-agent", agent_id="integration-agent"))
    return app, agent


def _mission(app, agent, tool):
    now = datetime.now(timezone.utc)
    mission = Mission(
        "integration-mission", agent.agent_id, action=tool.name,
        resource=tool.tool_id, tool=tool.name, issued_at=now,
        expires_at=now + timedelta(minutes=5), allowed_actions=(tool.name,),
        allowed_resources=(tool.tool_id,), allowed_tools=(tool.name,),
    )
    app.missions.register(mission)
    app.bind_mission(agent, mission.mission_id, mission.version)


def test_default_app_blocks_unknown_security_tool():
    app, agent = _app()

    @agent.tool("unknown")
    def unknown():
        return "must not run"

    _mission(app, agent, unknown)
    app.run()
    with pytest.raises(ToolExecutionDenied):
        unknown()


def test_explicit_trust_binds_identity_version_provenance_and_digest():
    app, agent = _app()

    @agent.tool("lookup", version="1.0.0", provenance=ToolProvenance("pkg", "pkg://lookup"))
    def lookup():
        return "ok"

    _mission(app, agent, lookup)
    app.register_tool_trust(lookup, trusted=True)
    app.run()
    assert lookup() == "ok"


def test_integrity_mismatch_blocks_before_callable():
    app, agent = _app()
    calls = []

    @agent.tool("lookup")
    def lookup():
        calls.append(True)
        return "ok"

    _mission(app, agent, lookup)
    app.register_tool_trust(lookup, integrity_digest="0" * 64, trusted=True)
    app.run()
    with pytest.raises(ToolExecutionDenied):
        lookup()
    assert calls == []


def test_revocation_is_rechecked_on_reused_session():
    app, agent = _app()

    @agent.tool("lookup")
    def lookup():
        return "ok"

    _mission(app, agent, lookup)
    app.register_tool_trust(lookup, trusted=True)
    app.run()
    assert lookup() == "ok"
    app.tool_trust.revoke(lookup.tool_id, lookup.version)
    with pytest.raises(ToolExecutionDenied):
        lookup()


def test_implementation_change_is_rejected_at_execution_boundary():
    app, agent = _app()

    @agent.tool("lookup")
    def lookup():
        return "v1"

    _mission(app, agent, lookup)
    app.register_tool_trust(lookup, trusted=True)
    app.run()
    lookup._function = lambda: "v2"
    with pytest.raises(ToolExecutionDenied):
        lookup()
