from datetime import datetime, timedelta, timezone

from preventai import Agent, Mission, PreventApp


def bind_test_boundary(boundary, *, agent_id="agent", tool_id="boundary-test"):
    app = PreventApp()
    agent = app.add_agent(Agent("boundary-test-agent", agent_id=agent_id))
    mission = Mission(
        "boundary-test-mission",
        agent.agent_id,
        action="*",
        resource="*",
        tool="*",
        expires_at=datetime.now(timezone.utc) + timedelta(minutes=5),
        allowed_actions=("*",),
        allowed_resources=("*",),
        allowed_tools=("*",),
        allowed_destinations=("*",),
    )
    app.missions.register(mission)
    app.bind_mission(agent, mission.mission_id, mission.version)
    try:
        boundary.bind_mission_context(app.missions, agent.agent_id, tool=tool_id)
    except TypeError:
        boundary.bind_mission_context(app.missions, agent.agent_id)
    return boundary
