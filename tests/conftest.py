from datetime import datetime, timedelta, timezone

import pytest

from preventai import Mission, PreventApp


_MISSION_TEST_MODULES = {
    "test_mission",
    "test_mission_integration",
    "test_mission_boundaries",
}
_TOOL_TRUST_TEST_MODULES = {
    "test_tool_trust",
    "test_tool_trust_integration",
    "test_tool_trust_boundaries",
    "test_tool_trust_concurrency",
}


@pytest.fixture(autouse=True)
def bind_explicit_test_missions(monkeypatch, request):
    module_name = request.module.__name__.rsplit(".", 1)[-1]
    run = PreventApp.run

    def run_with_test_missions(app):
        if module_name not in _TOOL_TRUST_TEST_MODULES:
            for agent in app._agents.values():
                for tool in agent.tools.values():
                    if app.tool_trust.get(tool.tool_id, tool.version) is None:
                        app.register_tool_trust(tool, trusted=True)
        if module_name not in _MISSION_TEST_MODULES:
            for agent in app._agents.values():
                if app.missions.binding_for(agent.agent_id) is not None:
                    continue
                mission_id = f"pytest-mission-{agent.agent_id}"
                app.missions.register(Mission(
                    mission_id=mission_id,
                    agent_id=agent.agent_id,
                    action="*",
                    resource="*",
                    tool="*",
                    expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
                ))
                app.bind_mission(agent, mission_id, 1)
        return run(app)

    monkeypatch.setattr(PreventApp, "run", run_with_test_missions)
