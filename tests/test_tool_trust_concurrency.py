from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

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
)


def test_concurrent_trust_checks_are_fail_closed_after_revocation():
    app = PreventApp(
        policy=PolicyEngine([PolicyRule("allow", DecisionType.ALLOW, action="*")]),
        permissions=PermissionEngine([
            PermissionRule("execute", PermissionCategory.EXECUTE, action="*", resource="*"),
        ]),
    )
    agent = app.add_agent(Agent("concurrent-agent", agent_id="concurrent-agent"))

    @agent.tool("work")
    def work():
        return "ok"

    now = datetime.now(timezone.utc)
    mission = Mission(
        "concurrent-mission", agent.agent_id, action="work", resource=work.tool_id,
        tool="work", issued_at=now, expires_at=now + timedelta(minutes=5),
        allowed_actions=("work",), allowed_resources=(work.tool_id,),
        allowed_tools=("work",),
    )
    app.missions.register(mission)
    app.bind_mission(agent, mission.mission_id, mission.version)
    app.register_tool_trust(work, trusted=True)
    app.run()
    app.tool_trust.revoke(work.tool_id, work.version)
    results = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(work) for _ in range(8)]
        for future in futures:
            try:
                results.append(future.result())
            except ToolExecutionDenied:
                results.append("blocked")
    assert results == ["blocked"] * 8
