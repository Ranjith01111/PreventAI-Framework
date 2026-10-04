from datetime import datetime, timedelta, timezone

import pytest

from preventai import (
    Agent, DecisionType, Mission, PolicyEngine, PolicyRule, PreventApp,
    PermissionCategory, PermissionEngine, PermissionRule,
    ToolExecutionDenied, ToolProvenance,
)


def make_app():
    return PreventApp(
        policy=PolicyEngine([PolicyRule("allow", DecisionType.ALLOW, action="*")]),
        permissions=PermissionEngine([
            PermissionRule("execute", PermissionCategory.EXECUTE, action="*", resource="*"),
            PermissionRule("read", PermissionCategory.READ, action="*", resource="*"),
        ]),
        require_tool_trust=True,
    )


def setup_tool():
    app = make_app()
    agent = app.add_agent(Agent("trust-agent", agent_id="trust-agent"))

    @agent.tool("lookup", version="1.2.0", provenance=ToolProvenance("acme", "pkg://lookup"))
    def lookup():
        return "ok"

    now = datetime.now(timezone.utc)
    mission = Mission(
        "trust-mission", agent.agent_id, action="lookup", resource=lookup.tool_id,
        tool="lookup", issued_at=now, expires_at=now + timedelta(minutes=5),
        allowed_actions=("lookup",), allowed_resources=(lookup.tool_id,),
        allowed_tools=("lookup",),
    )
    app.missions.register(mission)
    app.bind_mission(agent, mission.mission_id, 1)
    app.run()
    return app, agent, lookup


def test_unknown_tool_blocks_and_trusted_tool_runs():
    app, _, lookup = setup_tool()
    with pytest.raises(ToolExecutionDenied):
        lookup()
    app.register_tool_trust(lookup, trusted=True)
    assert lookup() == "ok"


def test_false_compatibility_argument_cannot_disable_managed_trust():
    app = PreventApp(require_tool_trust=False)
    agent = app.add_agent(Agent("strict-agent", agent_id="strict-agent"))

    @agent.tool("unknown")
    def unknown():
        return "must not run"

    app.run()
    with pytest.raises(ToolExecutionDenied):
        unknown()


def test_revocation_and_integrity_mismatch_block():
    app, _, lookup = setup_tool()
    app.register_tool_trust(lookup, trusted=True)
    assert lookup() == "ok"
    app.tool_trust.revoke(lookup.tool_id, lookup.version)
    with pytest.raises(ToolExecutionDenied):
        lookup()


def test_provenance_is_bound_exactly():
    app, _, lookup = setup_tool()
    app.tool_trust.register(
        tool_id=lookup.tool_id, name=lookup.name, version="2.0.0",
        provenance=ToolProvenance("other", "pkg://lookup"),
        integrity_digest=lookup.integrity_digest,
    )
    allowed, reason = app.tool_trust.check(
        tool_id=lookup.tool_id, name=lookup.name, version="2.0.0",
        provenance=lookup.provenance, integrity_digest=lookup.integrity_digest,
    )
    assert not allowed and reason == "provenance_mismatch"
