from preventai import (
    Agent,
    AppState,
    PreventApp,
    ToolApprovalRequired,
    ToolExecutionDenied,
)
from preventai.core import (
    DecisionType,
    PermissionCategory,
    PermissionEngine,
    PermissionRule,
    PolicyEngine,
    PolicyRule,
)


def make_app(action="lookup"):
    policy = PolicyEngine([
        PolicyRule("allow-tool", DecisionType.ALLOW, action=action),
    ])
    permissions = PermissionEngine([
        PermissionRule(
            "tool-permission",
            PermissionCategory.READ,
            action=action,
        ),
    ])
    return PreventApp(policy=policy, permissions=permissions)


def test_application_creation():
    assert isinstance(PreventApp(), PreventApp)


def test_agent_creation_and_stable_identity():
    assert Agent("customer-agent").agent_id == Agent("customer-agent").agent_id


def test_multiple_agents():
    app = PreventApp()
    first = app.add_agent(Agent("one"))
    second = app.add_agent(Agent("two"))
    assert set(app.agents.values()) == {first, second}


def test_same_tool_name_has_distinct_stable_identity_per_agent():
    first = Agent("one")
    second = Agent("two")

    @first.tool("lookup")
    def first_lookup():
        return 1

    @second.tool("lookup")
    def second_lookup():
        return 2

    assert first_lookup.tool_id != second_lookup.tool_id


def test_tool_registration_and_stable_identity():
    agent = Agent("customer-agent")

    @agent.tool()
    def lookup(customer_id):
        return customer_id

    assert lookup.name == "lookup"
    assert lookup.tool_id == Agent("customer-agent").tool("lookup")(lambda: None).tool_id
    assert agent.tools["lookup"] is lookup


def test_tool_execution_passes_through_engine():
    app = make_app()
    agent = app.add_agent(Agent("customer-agent"))

    @agent.tool()
    def lookup(customer_id):
        return {"customer_id": customer_id}

    app.run()
    assert lookup("c-1") == {"customer_id": "c-1"}


def test_application_startup_and_repeated_run():
    app = make_app()
    agent = app.add_agent(Agent("customer-agent"))

    @agent.tool()
    def lookup():
        return "ok"

    assert app.run().run() is app
    assert app.state is AppState.RUNNING


def test_invalid_application_startup_is_rejected():
    app = PreventApp()
    app.add_agent(Agent("empty-agent"))
    try:
        app.run()
    except RuntimeError:
        pass
    else:
        raise AssertionError("invalid application started")


def test_tool_execution_before_run_is_blocked():
    app = make_app()
    agent = app.add_agent(Agent("customer-agent"))

    @agent.tool()
    def lookup():
        return "unsafe"

    try:
        lookup()
    except ToolExecutionDenied:
        pass
    else:
        raise AssertionError("tool executed before application startup")


def test_blocked_tool_execution_does_not_call_function():
    called = []
    app = PreventApp(
        policy=PolicyEngine([PolicyRule("block", DecisionType.BLOCK, action="delete")]),
        permissions=PermissionEngine([
            PermissionRule("delete", PermissionCategory.DELETE, action="delete"),
        ]),
    )
    agent = app.add_agent(Agent("customer-agent"))

    @agent.tool("delete")
    def delete():
        called.append(True)

    app.run()
    try:
        delete()
    except ToolExecutionDenied:
        pass
    assert called == []


def test_approval_required_tool_execution_does_not_call_function():
    called = []
    app = PreventApp(
        policy=PolicyEngine([
            PolicyRule("approval", DecisionType.APPROVAL_REQUIRED, action="send"),
        ]),
        permissions=PermissionEngine([
            PermissionRule("send", PermissionCategory.NETWORK, action="send"),
        ]),
    )
    agent = app.add_agent(Agent("customer-agent"))

    @agent.tool("send")
    def send():
        called.append(True)

    app.run()
    try:
        send()
    except ToolApprovalRequired:
        pass
    assert called == []
