import pytest
from preventai import (
    Agent, DataFlowEngine, DataFlowPolicy, DataSource, DataFlowDestination,
    DataReleasePolicy,
    DestinationTrust,
    DatabasePolicy, DatabaseRequest, DecisionType, PreventApp,
    ToolExecutionDenied, FlowClassification, FlowTrust,
)
from preventai.core import PermissionCategory, PermissionEngine, PermissionRule, PolicyEngine, PolicyRule


def make_tool(*, data_release_policy=None, **kwargs):
    app = PreventApp(
        policy=PolicyEngine([PolicyRule("allow", DecisionType.ALLOW, action="*")]),
        permissions=PermissionEngine([
            PermissionRule("operate", PermissionCategory.READ, action="operate"),
            PermissionRule("db", PermissionCategory.DATABASE, action="database.*", resource="app:public:users"),
        ]),
        data_flow=DataFlowEngine([DataFlowPolicy("flow", DecisionType.ALLOW)]),
        data_release_policy=data_release_policy,
    )
    agent = app.add_agent(Agent("db-agent"))
    policy = DatabasePolicy(
        allowed_databases=("app",), allowed_schemas=("public",), allowed_tables=("users",),
        allowed_columns=("id",), allowed_operations=("SELECT",),
    )
    @agent.tool("operate", database_policy=policy, data_source=DataSource("src", FlowClassification.INTERNAL, FlowTrust.TRUSTED), data_destination=DataFlowDestination("sink"), **kwargs)
    def operate():
        return DatabaseRequest("SELECT", "app", "public", "users", ("id",), (1,), "select users where id = ?")
    app.run()
    return app, agent, operate


def test_framework_executes_database_request():
    _, _, operate = make_tool()
    assert operate()["operation"] == "SELECT"


def test_authorized_data_release_reaches_database_backend():
    _, agent, operate = make_tool(
        data_release_policy=DataReleasePolicy(
            allowed_destinations=("sink", "app:public:users"),
            allowed_destination_trust=(DestinationTrust.INTERNAL,),
            allowed_classifications=(FlowClassification.INTERNAL,),
            allowed_field_paths=("id",),
        )
    )
    assert operate()["operation"] == "SELECT"
    assert agent._tools["operate"]._database._backend.execution_count == 1


def test_framework_requires_explicit_request():
    app, agent, operate = make_tool()
    agent._tools["operate"]._function = lambda: "arbitrary"
    with pytest.raises(ToolExecutionDenied): operate()


def test_policy_denial_has_zero_backend_execution():
    app, agent, operate = make_tool()
    agent._tools["operate"]._engine._policy = PolicyEngine([PolicyRule("deny", DecisionType.BLOCK, action="*")])
    with pytest.raises(ToolExecutionDenied): operate()
    assert agent._tools["operate"]._database._backend.execution_count == 0


def test_permission_denial_has_zero_backend_execution():
    app, agent, operate = make_tool()
    agent._tools["operate"]._engine._permissions = PermissionEngine([])
    with pytest.raises(ToolExecutionDenied): operate()
    assert agent._tools["operate"]._database._backend.execution_count == 0


def test_dataflow_denial_has_zero_backend_execution():
    app, agent, operate = make_tool()
    agent._tools["operate"]._data_flow_engine = DataFlowEngine([DataFlowPolicy("deny", DecisionType.BLOCK, destination="sink")])
    with pytest.raises(ToolExecutionDenied): operate()
    assert agent._tools["operate"]._database._backend.execution_count == 0


@pytest.mark.parametrize("value", [None, b"sql", {"operation": "SELECT"}])
def test_framework_no_fallback(value):
    app, agent, operate = make_tool()
    agent._tools["operate"]._function = lambda: value
    with pytest.raises(ToolExecutionDenied): operate()
    assert agent._tools["operate"]._database._backend.execution_count == 0
