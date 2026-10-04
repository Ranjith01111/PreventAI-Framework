from datetime import datetime, timezone
from pathlib import Path

import pytest

from preventai import (
    Agent,
    AgentCommunicationBoundary,
    AgentMessage,
    AgentMessagePolicy,
    AgentSecurityContext,
    Database,
    DatabaseBoundaryAdapter,
    DatabasePolicy,
    FileSystem,
    FileSystemBoundaryAdapter,
    FileSystemPolicy,
    MCPBoundary,
    MCPBoundaryAdapter,
    MCPPolicy,
    Network,
    NetworkBoundaryAdapter,
    NetworkPolicy,
    Sandbox,
    SandboxBoundaryAdapter,
    SandboxPolicy,
    Decision,
    DecisionReason,
    DecisionType,
    PreventApp,
)
from preventai.runtime import RuntimeController, RuntimePolicy
from preventai.tools.database import InstrumentedDatabaseBackend
from preventai.tools.network import NetworkResponse


def message(payload, *, message_id="adapter-1"):
    return AgentMessage(
        message_id, "request", "agent-a", "agent-b", "session-a", "session-b",
        "conversation", None, datetime.now(timezone.utc), 4102444800.0,
        payload, tenant_id="tenant-a",
    )


def communication():
    app = PreventApp()
    sender = app.add_agent(Agent("sender", agent_id="agent-a"))
    receiver = app.add_agent(Agent("receiver", agent_id="agent-b"))
    sender.session_id = "session-a"
    receiver.session_id = "session-b"
    @sender.tool("sender.noop")
    def sender_noop():
        return "unused"
    @receiver.tool("receiver.noop")
    def receiver_noop():
        return "unused"
    app.run()
    context = AgentSecurityContext("agent-a", "session-a", "tenant-a", True)
    checks = [lambda _: True] * 6
    boundary = AgentCommunicationBoundary(
        AgentMessagePolicy(
            allowed_senders=("agent-a",), allowed_receivers=("agent-b",),
            allowed_tenants=("tenant-a",),
        ),
        lambda _: {"unused": True},
        sender_contexts={"agent-a": context},
        policy_check=checks[0], permission_check=checks[1],
        delegation_check=checks[2], dataflow_check=checks[3],
        runtime_check=checks[4], approval_check=checks[5],
        output_check=lambda _: True, audit_check=lambda _: True,
        runtime_controller=RuntimeController(RuntimePolicy(), application_id="adapter-tests"),
    )
    app.bind_agent_communication(boundary)
    issue = type(boundary).issue
    boundary.issue = lambda msg, ctx: issue(
        boundary, msg, ctx, issuance_capability=boundary._issuance_capability
    )
    return boundary, context


def dispatch(boundary, context, adapter, payload, *, message_id="adapter-1"):
    msg = message(payload, message_id=message_id)
    return boundary.dispatch(msg, boundary.issue(msg, context), context, managed_adapter=adapter)


def test_mcp_adapter_executes_only_through_mcp_boundary():
    calls = []
    mcp = MCPBoundary(
        MCPPolicy(allowed_tools=("customer.read",), allowed_clients=("agent-a",)),
        lambda request: (calls.append(request), {"ok": True})[1],
        policy_check=lambda _: True, permission_check=lambda _: True,
        delegation_check=lambda _: True, dataflow_check=lambda _: True,
        runtime_check=lambda _: True, approval_check=lambda _: True,
        output_check=lambda _: True,
    )
    adapter = MCPBoundaryAdapter(mcp, boundary_id="mcp-a")
    boundary, context = communication()
    payload = {
        "message": {
            "jsonrpc": "2.0", "id": 1, "method": "tools/call",
            "params": {"name": "customer.read", "arguments": {}},
        },
        "protocol_version": "2025-06-18", "client_id": "agent-a",
        "session_id": "session-a",
    }
    result = dispatch(boundary, context, adapter, payload)
    assert result.payload["result"] == {"ok": True}
    assert len(calls) == 1


def test_mcp_inner_policy_denial_has_zero_execution():
    calls = []
    mcp = MCPBoundary(MCPPolicy(allowed_tools=("customer.read",)), lambda _: calls.append(1))
    adapter = MCPBoundaryAdapter(mcp, boundary_id="mcp-a")
    boundary, context = communication()
    payload = {
        "message": {
            "jsonrpc": "2.0", "id": 1, "method": "tools/call",
            "params": {"name": "customer.delete", "arguments": {}},
        },
        "protocol_version": "2025-06-18", "client_id": "agent-a",
        "session_id": "session-a",
    }
    with pytest.raises(Exception):
        dispatch(boundary, context, adapter, payload)
    assert calls == []


def test_mcp_replay_has_zero_execution():
    calls = []
    mcp = MCPBoundary(
        MCPPolicy(allowed_tools=("customer.read",), allowed_clients=("agent-a",)),
        lambda request: (calls.append(request), {"ok": True})[1],
        policy_check=lambda _: True, permission_check=lambda _: True,
        delegation_check=lambda _: True, dataflow_check=lambda _: True,
        runtime_check=lambda _: True, approval_check=lambda _: True,
        output_check=lambda _: True,
    )
    adapter = MCPBoundaryAdapter(mcp, boundary_id="mcp-a")
    boundary, context = communication()
    payload = {
        "message": {
            "jsonrpc": "2.0", "id": 1, "method": "tools/call",
            "params": {"name": "customer.read", "arguments": {}},
        },
        "protocol_version": "2025-06-18", "client_id": "agent-a",
        "session_id": "session-a",
    }
    msg = message(payload)
    token = boundary.issue(msg, context)
    boundary.dispatch(msg, token, context, managed_adapter=adapter)
    with pytest.raises(Exception):
        boundary.dispatch(msg, token, context, managed_adapter=adapter)
    assert len(calls) == 1


def test_sandbox_adapter_reaches_existing_boundary(tmp_path):
    sandbox = Sandbox(SandboxPolicy(
        allowed_commands=("cmd.exe",), working_directory=str(tmp_path),
        allowed_paths=(str(tmp_path),),
    ))
    adapter = SandboxBoundaryAdapter(sandbox, boundary_id="sandbox-a")
    boundary, context = communication()
    result = dispatch(boundary, context, adapter, {
        "executable": "cmd.exe", "args": ["/c", "echo", "ok"], "cwd": str(tmp_path),
        "action": "process.execute", "resource": "sandbox-a",
    })
    assert result.payload["returncode"] == 0


def test_sandbox_inner_policy_denial_has_zero_process_starts(tmp_path, monkeypatch):
    sandbox = Sandbox(SandboxPolicy(
        allowed_commands=("cmd.exe",), working_directory=str(tmp_path),
        allowed_paths=(str(tmp_path),),
    ), permission_checker=lambda _: Decision(
        DecisionType.BLOCK, DecisionReason.PERMISSION_DENIED
    ))
    starts = []
    monkeypatch.setattr(
        "preventai.tools.sandbox.subprocess.Popen",
        lambda *args, **kwargs: starts.append(1),
    )
    adapter = SandboxBoundaryAdapter(sandbox, boundary_id="sandbox-a")
    boundary, context = communication()
    result = dispatch(boundary, context, adapter, {
        "executable": "cmd.exe", "args": ["/c", "echo", "ok"],
        "cwd": str(tmp_path), "action": "process.execute", "resource": "sandbox-a",
    })
    assert result.payload["decision"] == "block"
    assert starts == []


def test_network_adapter_reaches_existing_boundary(monkeypatch):
    network = Network(NetworkPolicy(allowed_hosts=("example.com",)))
    calls = []
    monkeypatch.setattr(
        network, "_request_once",
        lambda request, addresses, timeout: (calls.append(request), (200, {}, b"ok"))[1],
    )
    adapter = NetworkBoundaryAdapter(network, boundary_id="network-a")
    boundary, context = communication()
    result = dispatch(boundary, context, adapter, {
        "url": "https://example.com/data", "method": "GET",
    })
    assert result.payload["status"] == 200
    assert len(calls) == 1


def test_network_inner_policy_denial_has_zero_backend_calls(monkeypatch):
    network = Network(NetworkPolicy(allowed_hosts=("example.com",)))
    calls = []
    monkeypatch.setattr(
        network, "_request_once",
        lambda *args: (calls.append(1), (200, {}, b"ok"))[1],
    )
    adapter = NetworkBoundaryAdapter(network, boundary_id="network-a")
    boundary, context = communication()
    with pytest.raises(Exception):
        dispatch(boundary, context, adapter, {"url": "https://other.example/data"})
    assert calls == []


def test_filesystem_adapter_reaches_existing_boundary(tmp_path):
    path = Path(tmp_path) / "item"
    path.write_text("ok", encoding="utf-8")
    filesystem = FileSystem(FileSystemPolicy(
        allowed_roots=(str(tmp_path),), allowed_operations=("read",),
    ))
    adapter = FileSystemBoundaryAdapter(filesystem, boundary_id="filesystem-a")
    boundary, context = communication()
    result = dispatch(boundary, context, adapter, {
        "operation": "read", "path": str(path),
    })
    assert result.payload["content"] == "ok"


def test_filesystem_path_denial_has_zero_backend_calls(tmp_path, monkeypatch):
    filesystem = FileSystem(FileSystemPolicy(
        allowed_roots=(str(tmp_path),), allowed_operations=("read",),
    ))
    adapter = FileSystemBoundaryAdapter(filesystem, boundary_id="filesystem-a")
    boundary, context = communication()
    calls = []
    monkeypatch.setattr(filesystem, "_perform", lambda *args: calls.append(1))
    with pytest.raises(Exception):
        dispatch(boundary, context, adapter, {
            "operation": "read", "path": str(Path(tmp_path).parent / "outside"),
        })
    assert calls == []


def test_database_adapter_reaches_existing_boundary():
    backend = InstrumentedDatabaseBackend()
    database = Database(DatabasePolicy(
        allowed_databases=("db",), allowed_schemas=("public",),
        allowed_tables=("customers",), allowed_columns=("id",),
    ), backend=backend)
    adapter = DatabaseBoundaryAdapter(database, boundary_id="database-a")
    boundary, context = communication()
    result = dispatch(boundary, context, adapter, {
        "operation": "SELECT", "database": "db", "schema": "public",
        "table": "customers", "columns": ["id"],
        "template": "SELECT id FROM db.public.customers WHERE id = ?",
        "parameters": [1],
    })
    assert result.payload["operation"] == "SELECT"
    assert backend.execution_count == 1


def test_database_operation_escalation_has_zero_backend_calls():
    backend = InstrumentedDatabaseBackend()
    database = Database(DatabasePolicy(
        allowed_databases=("db",), allowed_schemas=("public",),
        allowed_tables=("customers",), allowed_columns=("id",),
        allowed_operations=("SELECT",),
    ), backend=backend)
    adapter = DatabaseBoundaryAdapter(database, boundary_id="database-a")
    boundary, context = communication()
    with pytest.raises(Exception):
        dispatch(boundary, context, adapter, {
            "operation": "DELETE", "database": "db", "schema": "public",
            "table": "customers", "columns": ["id"],
            "template": "DELETE FROM db.public.customers WHERE id = ?",
            "parameters": [1],
        })
    assert backend.execution_count == 0


@pytest.mark.parametrize("kind", ["mcp", "sandbox", "network", "filesystem", "database"])
def test_wrong_adapter_instance_is_rejected(kind):
    from preventai import ManagedBoundaryRegistry, ManagedBoundaryRegistration
    boundary = object()
    with pytest.raises(Exception):
        ManagedBoundaryRegistration(kind, "wrong", boundary)  # type: ignore[arg-type]
