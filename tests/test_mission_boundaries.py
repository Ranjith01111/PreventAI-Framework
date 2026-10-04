import sys
from datetime import datetime, timedelta, timezone

import pytest

from preventai import (
    AgentCommunicationBoundary,
    AgentMessage,
    AgentMessagePolicy,
    AgentSecurityContext,
    Agent,
    DataFlowDestination,
    DataFlowEngine,
    DataFlowPolicy,
    DataSource,
    DatabasePolicy,
    DatabaseRequest,
    DecisionType,
    FileSystemPolicy,
    FileSystemRequest,
    FlowClassification,
    FlowTrust,
    Mission,
    MCPBoundary,
    MCPExecutionDenied,
    MCPBoundaryAdapter,
    MCPPolicy,
    NetworkPolicy,
    NetworkRequest,
    PermissionCategory,
    PermissionEngine,
    PermissionRule,
    PolicyEngine,
    PolicyRule,
    PreventApp,
    SandboxCommand,
    SandboxPolicy,
    ToolExecutionDenied,
)
from preventai.tools.database import Database, DatabaseExecutionDenied, InstrumentedDatabaseBackend
from preventai.tools.filesystem import FileSystem, FileSystemExecutionDenied
from preventai.tools.network import Network, NetworkExecutionDenied
from preventai.tools.sandbox import Sandbox
from preventai.runtime import RuntimeController, RuntimePolicy


def app_for_tool(name, *, data_flow=False, **tool_options):
    app = PreventApp(
        policy=PolicyEngine([PolicyRule("allow", DecisionType.ALLOW, action="*")]),
        permissions=PermissionEngine([
            PermissionRule("read", PermissionCategory.READ, action="*"),
            PermissionRule("execute", PermissionCategory.EXECUTE, action="*"),
            PermissionRule("filesystem", PermissionCategory.FILE, action="filesystem.*", resource="*"),
            PermissionRule("network", PermissionCategory.NETWORK, action="http.*", resource="*"),
            PermissionRule("database", PermissionCategory.DATABASE, action="database.*", resource="*"),
        ]),
        data_flow=(
            DataFlowEngine([DataFlowPolicy("allow", DecisionType.ALLOW)])
            if data_flow else None
        ),
    )
    agent = app.add_agent(Agent("boundary-agent", agent_id="boundary-agent"))
    return app, agent, tool_options


def test_unbound_security_boundaries_fail_closed_before_downstream(tmp_path):
    sandbox = Sandbox(SandboxPolicy(
        allowed_commands=(sys.executable,),
        working_directory=str(tmp_path),
        allowed_paths=(str(tmp_path),),
    ))
    result = sandbox.execute(SandboxCommand(sys.executable, ("-c", "print('no')")))
    assert result.decision.decision_type is DecisionType.BLOCK

    network = Network(NetworkPolicy(allowed_hosts=("example.com",)))
    with pytest.raises(NetworkExecutionDenied):
        network.execute(NetworkRequest(
            "https://example.com", method="GET",
            agent_id="a", session_id="s", tool_id="t",
        ))

    filesystem = FileSystem(FileSystemPolicy(allowed_roots=(str(tmp_path),)))
    with pytest.raises(FileSystemExecutionDenied):
        filesystem.execute(FileSystemRequest("read", str(tmp_path / "missing")), authorization=None)

    backend = InstrumentedDatabaseBackend()
    database = Database(DatabasePolicy(
        allowed_databases=("db",), allowed_schemas=("public",),
        allowed_tables=("items",), allowed_columns=("id",),
    ), backend=backend)
    request = DatabaseRequest(
        "SELECT", "db", "public", "items", ("id",), (1,),
        "SELECT id FROM items WHERE id = ?",
    )
    with pytest.raises(DatabaseExecutionDenied):
        database.execute(request)
    assert backend.execution_count == 0

    calls = []
    mcp = MCPBoundary(
        MCPPolicy(allowed_tools=("echo",)),
        lambda request: (calls.append(request), {"ok": True})[1],
    )
    mcp_request = mcp.parse(
        {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
         "params": {"name": "echo", "arguments": {}}},
        protocol_version="2025-06-18", client_id="client", session_id="session",
    )
    with pytest.raises(MCPExecutionDenied):
        mcp.dispatch(mcp_request, None)
    assert calls == []


def test_standalone_registry_cannot_be_presented_as_app_context(tmp_path):
    from preventai.identity.mission import MissionRegistry

    boundary = Sandbox(SandboxPolicy(
        allowed_commands=(sys.executable,),
        working_directory=str(tmp_path),
        allowed_paths=(str(tmp_path),),
    ))
    with pytest.raises(TypeError, match="authoritative mission context"):
        boundary.bind_mission_context(MissionRegistry(), "agent")


def bind_mission(
    app,
    agent,
    tool,
    *,
    mission_id,
    actions,
    resources,
    tools=None,
    destinations=(),
    data_classes=(),
    context=None,
):
    mission = Mission(
        mission_id=mission_id,
        agent_id=agent.agent_id,
        action=actions[0] if len(actions) == 1 else "*",
        resource=resources[0] if len(resources) == 1 else "*",
        tool=(tools or (tool.name,))[0] if len(tools or (tool.name,)) == 1 else "*",
        expires_at=datetime.now(timezone.utc) + timedelta(minutes=5),
        allowed_actions=tuple(actions),
        allowed_resources=tuple(resources),
        allowed_tools=tuple(tools or (tool.name,)),
        allowed_destinations=tuple(destinations),
        allowed_data_classes=tuple(data_classes),
        context_constraints=context or {},
    )
    app.missions.register(mission)
    app.bind_mission(agent, mission_id, 1)
    return mission


def test_sandbox_mission_denial_prevents_executor_start(tmp_path, monkeypatch):
    app, agent, _ = app_for_tool("run")
    starts = []

    @agent.tool(
        "run",
        sandbox_policy=SandboxPolicy(
            allowed_commands=(sys.executable,),
            working_directory=str(tmp_path),
            allowed_paths=(str(tmp_path),),
        ),
    )
    def run_command():
        return SandboxCommand(sys.executable, ("-c", "print('ok')"))

    app.run()
    bind_mission(app, agent, run_command, mission_id="sandbox-mission",
                 actions=("run", "run"), resources=(run_command.tool_id,),
                 tools=("run",))
    assert run_command()["stdout"].strip() == "ok"

    monkeypatch.setattr(
        "preventai.tools.sandbox.subprocess.Popen",
        lambda *args, **kwargs: starts.append(True),
    )
    app.missions.revoke("sandbox-mission")
    with pytest.raises(ToolExecutionDenied):
        run_command()
    assert starts == []


def test_filesystem_mission_denial_prevents_backend_call(tmp_path):
    path = tmp_path / "item.txt"
    path.write_text("data", encoding="utf-8")
    app, agent, _ = app_for_tool("operate", data_flow=True)

    @agent.tool(
        "operate",
        filesystem_policy=FileSystemPolicy(
            allowed_roots=(str(tmp_path),), allowed_operations=("read", "write"),
        ),
        data_source=DataSource("filesystem-source", FlowClassification.INTERNAL, FlowTrust.TRUSTED),
        data_destination=DataFlowDestination("filesystem-sink"),
    )
    def operate(path, operation):
        from preventai import FileSystemRequest
        return FileSystemRequest(operation, str(path))

    app.run()
    tool = agent.tools["operate"]
    bind_mission(app, agent, tool, mission_id="filesystem-mission",
                 actions=("operate", "read"), resources=(tool.tool_id, str(path)),
                 tools=("operate",))
    calls = []
    backend = tool._filesystem._backend
    original_read = backend.read

    def read(*args, **kwargs):
        calls.append("read")
        return original_read(*args, **kwargs)

    backend.read = read
    assert operate(str(path), "read")["operation"] == "read"
    with pytest.raises(ToolExecutionDenied):
        operate(str(path), "write")
    with pytest.raises(ToolExecutionDenied):
        operate(str(tmp_path / "other.txt"), "read")
    assert calls == ["read"]
    app.missions.revoke("filesystem-mission")
    with pytest.raises(ToolExecutionDenied):
        operate(str(path), "read")
    assert calls == ["read"]


def test_network_mission_destination_denial_prevents_http_call(monkeypatch):
    app, agent, _ = app_for_tool("fetch", data_flow=True)

    @agent.tool(
        "fetch",
        network_policy=NetworkPolicy(allowed_hosts=("example.com",)),
        data_source=DataSource("network-source", FlowClassification.INTERNAL, FlowTrust.TRUSTED),
        data_destination=DataFlowDestination("network-sink"),
    )
    def fetch():
        return NetworkRequest("https://example.com/data")

    app.run()
    tool = agent.tools["fetch"]
    bind_mission(
        app, agent, tool, mission_id="network-mission",
        actions=("fetch", "http.get"),
        resources=(tool.tool_id, "https://example.com:443"),
        tools=("fetch",), destinations=("approved-service",),
    )
    calls = []
    monkeypatch.setattr(tool._network, "_resolve", lambda host, port: ("93.184.216.34",))
    monkeypatch.setattr(
        tool._network, "_request_once",
        lambda *args: (calls.append(True), (200, {}, b"ok"))[1],
    )
    with pytest.raises(ToolExecutionDenied):
        fetch()
    assert calls == []


def test_network_actual_destination_and_data_class_allow_only_declared_scope(monkeypatch):
    app, agent, _ = app_for_tool("fetch", data_flow=True)

    @agent.tool(
        "fetch",
        network_policy=NetworkPolicy(allowed_hosts=("example.com",)),
        data_source=DataSource("network-source", FlowClassification.INTERNAL, FlowTrust.TRUSTED),
        data_destination=DataFlowDestination("network-sink"),
    )
    def fetch():
        from preventai import NetworkRequest
        return NetworkRequest(
            "https://example.com/data",
            classification=FlowClassification.PUBLIC,
        )

    app.run()
    tool = agent.tools["fetch"]
    bind_mission(
        app, agent, tool, mission_id="network-public-mission",
        actions=("fetch", "http.get"),
        resources=(tool.tool_id, "https://example.com:443"),
        tools=("fetch",), destinations=("example.com",), data_classes=("public",),
    )
    calls = []
    monkeypatch.setattr(tool._network, "_resolve", lambda host, port: ("93.184.216.34",))
    monkeypatch.setattr(
        tool._network, "_request_once",
        lambda *args: (calls.append(True), (200, {}, b"ok"))[1],
    )
    assert fetch()["status"] == 200
    assert calls == [True]


def test_database_mission_table_and_context_denial_prevents_backend_call():
    app, agent, _ = app_for_tool("operate", data_flow=True)

    @agent.tool(
        "operate",
        database_policy=DatabasePolicy(
            allowed_databases=("customer_db",), allowed_schemas=("public",),
            allowed_tables=("customer",), allowed_columns=("id",),
            allowed_operations=("SELECT",), allowed_tenants=("tenant-a",),
        ),
        data_source=DataSource("db-source", FlowClassification.INTERNAL, FlowTrust.TRUSTED),
        data_destination=DataFlowDestination("db-sink"),
    )
    def operate():
        return DatabaseRequest(
            "SELECT", "customer_db", "public", "customer", ("id",), (1,),
            "select id from customer where id = ? and tenant_id = ?", tenant_id="tenant-b",
        )

    app.run()
    tool = agent.tools["operate"]
    bind_mission(
        app, agent, tool, mission_id="database-mission",
        actions=("operate", "database.select"),
        resources=(tool.tool_id, "customer_db:public:customer"),
        tools=("operate",),
        context={"tenant": "tenant-a", "table": "customer"},
    )
    with pytest.raises(ToolExecutionDenied):
        operate()
    assert isinstance(tool._database._backend, InstrumentedDatabaseBackend)
    assert tool._database._backend.execution_count == 0


def test_database_mission_scope_allows_read_but_denies_write_before_backend():
    app, agent, _ = app_for_tool("operate", data_flow=True)

    @agent.tool(
        "operate",
        database_policy=DatabasePolicy(
            allowed_databases=("customer_db",), allowed_schemas=("public",),
            allowed_tables=("customer",), allowed_columns=("id",),
            allowed_operations=("SELECT", "UPDATE"), allowed_tenants=("tenant-a",),
        ),
        data_source=DataSource("db-source", FlowClassification.INTERNAL, FlowTrust.TRUSTED),
        data_destination=DataFlowDestination("db-sink"),
    )
    def operate(operation):
        if operation == "SELECT":
            return DatabaseRequest(
                operation, "customer_db", "public", "customer", ("id",), (1,),
                "select id from customer where id = ? and tenant_id = ?", tenant_id="tenant-a",
            )
        return DatabaseRequest(
            operation, "customer_db", "public", "customer", ("id",), (1, 1),
            "update customer set id = ? where id = ?", tenant_id="tenant-a",
        )

    app.run()
    tool = agent.tools["operate"]
    bind_mission(
        app, agent, tool, mission_id="database-read-only",
        actions=("operate", "database.select"),
        resources=(tool.tool_id, "customer_db:public:customer"),
        tools=("operate",),
        context={"tenant": "tenant-a", "table": "customer", "operation": "SELECT"},
    )
    assert operate("SELECT")["operation"] == "SELECT"
    backend = tool._database._backend
    assert backend.execution_count == 1
    with pytest.raises(ToolExecutionDenied):
        operate("UPDATE")
    assert backend.execution_count == 1


def test_database_mission_uses_request_classification_and_fails_closed_when_required():
    app, agent, _ = app_for_tool("operate", data_flow=True)

    @agent.tool(
        "operate",
        database_policy=DatabasePolicy(
            allowed_databases=("customer_db",), allowed_schemas=("public",),
            allowed_tables=("customer",), allowed_columns=("id",),
            allowed_operations=("SELECT",), allowed_tenants=("tenant-a",),
        ),
        data_source=DataSource("db-source", FlowClassification.INTERNAL, FlowTrust.TRUSTED),
        data_destination=DataFlowDestination("db-sink"),
    )
    def operate(classification=None):
        return DatabaseRequest(
            "SELECT", "customer_db", "public", "customer", ("id",), (1,),
            "select id from customer where id = ? and tenant_id = ?",
            tenant_id="tenant-a", classification=classification,
        )

    app.run()
    tool = agent.tools["operate"]
    bind_mission(
        app, agent, tool, mission_id="database-classification-mission",
        actions=("operate", "database.select"),
        resources=(tool.tool_id, "customer_db:public:customer"),
        tools=("operate",), data_classes=("internal",),
        context={"tenant": "tenant-a", "table": "customer", "operation": "SELECT"},
    )
    backend = tool._database._backend
    assert operate(FlowClassification.INTERNAL)["operation"] == "SELECT"
    assert backend.execution_count == 1
    with pytest.raises(ToolExecutionDenied):
        operate(FlowClassification.PRIVATE)
    assert backend.execution_count == 1
    with pytest.raises(ToolExecutionDenied):
        operate()
    assert backend.execution_count == 1


def _communication_fixture():
    app = PreventApp(
        policy=PolicyEngine([PolicyRule("allow", DecisionType.ALLOW, action="*")]),
        permissions=PermissionEngine([
            PermissionRule("read", PermissionCategory.READ, action="*"),
        ]),
    )
    sender = app.add_agent(Agent("sender", agent_id="agent-a"))
    receiver = app.add_agent(Agent("receiver", agent_id="agent-b"))

    @sender.tool("sender.noop")
    def sender_noop():
        return "unused"

    @receiver.tool("receiver.noop")
    def receiver_noop():
        return "unused"

    app.run()
    sender_context = AgentSecurityContext(sender.agent_id, sender.session_id, "tenant-a", True)
    boundary = AgentCommunicationBoundary(
        AgentMessagePolicy(
            allowed_senders=(sender.agent_id,),
            allowed_receivers=(receiver.agent_id,),
            allowed_tenants=("tenant-a",),
        ),
        lambda _: {"ok": True},
        sender_contexts={sender.agent_id: sender_context},
        policy_check=lambda _: True,
        permission_check=lambda _: True,
        delegation_check=lambda _: True,
        dataflow_check=lambda _: True,
        runtime_check=lambda _: True,
        approval_check=lambda _: True,
        output_check=lambda _: True,
        audit_check=lambda _: True,
        runtime_controller=RuntimeController(RuntimePolicy(), application_id="mission-boundaries"),
    )
    app.bind_agent_communication(boundary)
    issue = type(boundary).issue
    boundary.issue = lambda message, context: issue(
        boundary,
        message,
        context,
        issuance_capability=boundary._issuance_capability,
    )
    return app, sender, receiver, sender_context, boundary


def _bind_communication_missions(
    app,
    sender,
    receiver,
    *,
    receiver_actions=("request",),
    sender_actions=("request", "tools/call"),
    sender_resources=None,
    sender_tools=("agent-communication", "customer.read"),
    sender_destinations=("preventai-server",),
):
    sender_mission = Mission(
        "sender-communication",
        sender.agent_id,
        action="*",
        resource="*",
        tool="*",
        expires_at=datetime.now(timezone.utc) + timedelta(minutes=5),
        allowed_actions=sender_actions,
        allowed_resources=sender_resources or (receiver.agent_id, "customer.read"),
        allowed_tools=sender_tools,
        allowed_destinations=sender_destinations,
    )
    receiver_mission = Mission(
        "receiver-communication",
        receiver.agent_id,
        action="request",
        resource=sender.agent_id,
        tool="agent-communication",
        expires_at=datetime.now(timezone.utc) + timedelta(minutes=5),
        allowed_actions=receiver_actions,
        allowed_resources=(sender.agent_id,),
        allowed_tools=("agent-communication",),
    )
    app.missions.register(sender_mission)
    app.missions.register(receiver_mission)
    app.bind_mission(sender, sender_mission.mission_id, 1)
    app.bind_mission(receiver, receiver_mission.mission_id, 1)
    return sender_mission, receiver_mission


def _mcp_message(sender, receiver, context):
    return AgentMessage(
        "mission-mcp-1",
        "request",
        sender.agent_id,
        receiver.agent_id,
        sender.session_id,
        receiver.session_id,
        "conversation",
        None,
        datetime.now(timezone.utc),
        4102444800.0,
        {
            "message": {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {"name": "customer.read", "arguments": {}},
            },
            "protocol_version": "2025-06-18",
            "client_id": sender.agent_id,
            "session_id": sender.session_id,
        },
        tenant_id="tenant-a",
    )


def test_app_bound_mcp_adapter_checks_mission_before_downstream():
    app, sender, receiver, context, communication = _communication_fixture()
    _bind_communication_missions(app, sender, receiver)
    calls = []
    mcp = MCPBoundary(
        MCPPolicy(
            allowed_tools=("customer.read",),
            allowed_clients=(sender.agent_id,),
        ),
        lambda request: (calls.append(request), {"ok": True})[1],
        policy_check=lambda _: True,
        permission_check=lambda _: True,
        delegation_check=lambda _: True,
        dataflow_check=lambda _: True,
        runtime_check=lambda _: True,
        approval_check=lambda _: True,
        output_check=lambda _: True,
    )
    adapter = MCPBoundaryAdapter(mcp, boundary_id="mcp-boundary")
    message = _mcp_message(sender, receiver, context)
    result = communication.dispatch(
        message,
        communication.issue(message, context),
        context,
        managed_adapter=adapter,
    )
    assert result.payload["result"] == {"ok": True}
    assert len(calls) == 1

    app.missions.revoke("sender-communication", 1)
    message = _mcp_message(sender, receiver, context)
    with pytest.raises(Exception):
        communication.dispatch(
            message,
            communication.issue(message, context),
            context,
            managed_adapter=adapter,
        )
    assert len(calls) == 1


def test_receiver_mission_denial_and_final_revalidation_block_mcp_dispatch(monkeypatch):
    app, sender, receiver, context, communication = _communication_fixture()
    sender_mission, receiver_mission = _bind_communication_missions(
        app, sender, receiver, receiver_actions=("response",)
    )
    calls = []
    mcp = MCPBoundary(
        MCPPolicy(allowed_tools=("customer.read",), allowed_clients=(sender.agent_id,)),
        lambda request: (calls.append(request), {"ok": True})[1],
        policy_check=lambda _: True,
        permission_check=lambda _: True,
        delegation_check=lambda _: True,
        dataflow_check=lambda _: True,
        runtime_check=lambda _: True,
        approval_check=lambda _: True,
        output_check=lambda _: True,
    )
    adapter = MCPBoundaryAdapter(mcp, boundary_id="mcp-boundary")
    message = _mcp_message(sender, receiver, context)
    with pytest.raises(Exception):
        communication.dispatch(
            message,
            communication.issue(message, context),
            context,
            managed_adapter=adapter,
        )
    assert calls == []

    app.missions.bind(receiver.agent_id, receiver_mission.mission_id, 1)
    authorize = adapter.authorize

    def revoke_after_authorization(msg):
        result = authorize(msg)
        app.missions.revoke(sender_mission.mission_id, sender_mission.version)
        return result

    monkeypatch.setattr(adapter, "authorize", revoke_after_authorization)
    message = _mcp_message(sender, receiver, context)
    with pytest.raises(Exception):
        communication.dispatch(
            message,
            communication.issue(message, context),
            context,
            managed_adapter=adapter,
        )
    assert calls == []


def test_communication_token_cannot_switch_to_a_new_mission_version():
    app, sender, receiver, context, communication = _communication_fixture()
    sender_mission, _ = _bind_communication_missions(app, sender, receiver)
    calls = []
    mcp = MCPBoundary(
        MCPPolicy(allowed_tools=("customer.read",), allowed_clients=(sender.agent_id,)),
        lambda request: (calls.append(request), {"ok": True})[1],
        policy_check=lambda _: True,
        permission_check=lambda _: True,
        delegation_check=lambda _: True,
        dataflow_check=lambda _: True,
        runtime_check=lambda _: True,
        approval_check=lambda _: True,
        output_check=lambda _: True,
    )
    adapter = MCPBoundaryAdapter(mcp, boundary_id="mcp-boundary")
    message = _mcp_message(sender, receiver, context)
    authorization = communication.issue(message, context)
    app.missions.register(Mission(
        sender_mission.mission_id,
        sender.agent_id,
        action="*",
        resource="*",
        tool="*",
        version=2,
        expires_at=datetime.now(timezone.utc) + timedelta(minutes=5),
    ))
    app.bind_mission(sender, sender_mission.mission_id, 2)
    with pytest.raises(Exception):
        communication.dispatch(
            message, authorization, context, managed_adapter=adapter
        )
    assert calls == []


@pytest.mark.parametrize("agent", ["sender", "receiver"])
def test_either_participant_mission_revocation_blocks_communication(agent):
    app, sender, receiver, context, communication = _communication_fixture()
    sender_mission, receiver_mission = _bind_communication_missions(app, sender, receiver)
    calls = []
    mcp = MCPBoundary(
        MCPPolicy(allowed_tools=("customer.read",), allowed_clients=(sender.agent_id,)),
        lambda request: (calls.append(request), {"ok": True})[1],
        policy_check=lambda _: True,
        permission_check=lambda _: True,
        delegation_check=lambda _: True,
        dataflow_check=lambda _: True,
        runtime_check=lambda _: True,
        approval_check=lambda _: True,
        output_check=lambda _: True,
    )
    adapter = MCPBoundaryAdapter(mcp, boundary_id="mcp-boundary")
    message = _mcp_message(sender, receiver, context)
    authorization = communication.issue(message, context)
    mission = sender_mission if agent == "sender" else receiver_mission
    app.missions.revoke(mission.mission_id, mission.version)
    with pytest.raises(Exception):
        communication.dispatch(
            message, authorization, context, managed_adapter=adapter
        )
    assert calls == []


@pytest.mark.parametrize(
    ("method", "params", "identity", "allowed_tools", "allowed_resources", "allowed_prompts"),
    [
        (
            "resources/read",
            {"uri": "mcp://server/customer/1"},
            "mcp://server/customer/1",
            (),
            ("mcp://server/customer/1",),
            (),
        ),
        (
            "prompts/get",
            {"name": "customer-summary"},
            "customer-summary",
            (),
            (),
            ("customer-summary",),
        ),
    ],
)
def test_mcp_resource_and_prompt_scopes_are_checked_by_mission(
    method, params, identity, allowed_tools, allowed_resources, allowed_prompts
):
    app, sender, receiver, context, communication = _communication_fixture()
    _bind_communication_missions(
        app,
        sender,
        receiver,
        sender_actions=("request", method),
        sender_resources=(receiver.agent_id,),
        sender_tools=("agent-communication", method),
    )
    calls = []
    mcp = MCPBoundary(
        MCPPolicy(
            allowed_tools=allowed_tools,
            allowed_resources=allowed_resources,
            allowed_prompts=allowed_prompts,
            allowed_clients=(sender.agent_id,),
        ),
        lambda request: (calls.append(request), {"ok": True})[1],
        policy_check=lambda _: True,
        permission_check=lambda _: True,
        delegation_check=lambda _: True,
        dataflow_check=lambda _: True,
        runtime_check=lambda _: True,
        approval_check=lambda _: True,
        output_check=lambda _: True,
    )
    adapter = MCPBoundaryAdapter(mcp, boundary_id="mcp-boundary")
    message = _mcp_message(sender, receiver, context)
    payload = dict(message.payload)
    payload["message"] = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": method,
        "params": params,
    }
    message = AgentMessage(
        message.message_id,
        message.message_type,
        message.sender_agent_id,
        message.receiver_agent_id,
        message.sender_session_id,
        message.receiver_session_id,
        message.conversation_id,
        message.parent_message_id,
        message.timestamp,
        message.expires_at,
        payload,
        tenant_id=message.tenant_id,
    )
    assert identity not in app.missions.get("sender-communication").allowed_resources
    with pytest.raises(Exception):
        communication.dispatch(
            message,
            communication.issue(message, context),
            context,
            managed_adapter=adapter,
        )
    assert calls == []
