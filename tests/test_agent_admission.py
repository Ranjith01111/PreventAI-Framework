from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from preventai import (
    Agent,
    AgentAdmissionState,
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


def _app(agent_name: str = "admission-agent"):
    app = PreventApp(
        policy=PolicyEngine([PolicyRule("allow", DecisionType.ALLOW, action="*")]),
        permissions=PermissionEngine([
            PermissionRule("read", PermissionCategory.READ, action="*", resource="*"),
        ]),
    )
    return app, app.add_agent(Agent(agent_name, agent_id=agent_name))


def _register_mission(app, agent, tool):
    mission = Mission(
        "admission-mission",
        agent.agent_id,
        action=tool.name,
        resource=tool.tool_id,
        tool=tool.name,
        issued_at=datetime.now(timezone.utc),
        expires_at=datetime.now(timezone.utc) + timedelta(minutes=5),
    )
    app.missions.register(mission)
    app.bind_mission(agent, mission.mission_id, mission.version)


def test_unadmitted_agent_cannot_reach_callable():
    calls = []
    agent = Agent("unadmitted-agent")

    @agent.tool()
    def lookup():
        calls.append(True)

    with pytest.raises(ToolExecutionDenied):
        lookup()
    assert calls == []


def test_admission_envelope_is_finalized_with_registered_tools():
    app, agent = _app()

    @agent.tool("lookup")
    def lookup():
        return "ok"

    _register_mission(app, agent, lookup)
    app.register_tool_trust(lookup, trusted=True)
    app.run()

    record = app.admission.get(agent.agent_id)
    assert record is not None
    assert record.state is AgentAdmissionState.ADMITTED
    assert record.tool_set == ("lookup",)
    assert lookup() == "ok"


def test_cross_app_admission_and_session_substitution_block():
    app_a, agent = _app("agent-a")
    app_b, _ = _app("agent-b")
    assert app_b.verify_agent_admission(agent).decision_type is DecisionType.BLOCK

    @agent.tool("lookup")
    def lookup():
        return "ok"

    _register_mission(app_a, agent, lookup)
    app_a.register_tool_trust(lookup, trusted=True)
    app_a.run()
    assert app_a.verify_agent_admission(
        agent, session_id="wrong-session", tools=("lookup",)
    ).decision_type is DecisionType.BLOCK


def test_attestation_tampering_blocks_before_callable():
    calls = []
    app, agent = _app()

    @agent.tool("lookup")
    def lookup():
        calls.append(True)
        return "ok"

    _register_mission(app, agent, lookup)
    app.register_tool_trust(lookup, trusted=True)
    app.run()
    record = app.admission.get(agent.agent_id)
    assert record is not None
    app.admission._records[agent.agent_id] = replace(
        record,
        provenance="tampered",
    )

    with pytest.raises(ToolExecutionDenied):
        lookup()
    assert calls == []


@pytest.mark.parametrize("transition", ["suspend", "revoke", "expire"])
def test_admission_state_transitions_block_execution(transition):
    app, agent = _app()

    @agent.tool("lookup")
    def lookup():
        return "ok"

    _register_mission(app, agent, lookup)
    app.register_tool_trust(lookup, trusted=True)
    app.run()
    getattr(app.admission, transition)(agent.agent_id)

    with pytest.raises(ToolExecutionDenied):
        lookup()


def test_policy_version_change_blocks_stale_admission():
    app, agent = _app()

    @agent.tool("lookup")
    def lookup():
        return "ok"

    _register_mission(app, agent, lookup)
    app.register_tool_trust(lookup, trusted=True)
    app.run()
    app.update_security_policy(
        policy=PolicyEngine([PolicyRule("allow-v2", DecisionType.ALLOW, action="*")]),
        permissions=PermissionEngine([
            PermissionRule("read-v2", PermissionCategory.READ, action="*", resource="*"),
        ]),
        version=2,
    )

    with pytest.raises(ToolExecutionDenied):
        lookup()
