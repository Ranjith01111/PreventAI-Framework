from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType

import pytest

from preventai import (
    AgentCommunicationDenied,
    AuditEventType,
    Database,
    DatabaseBoundaryAdapter,
    DatabasePolicy,
    FileSystem,
    FileSystemBoundaryAdapter,
    FileSystemPolicy,
    LifecycleRecord,
    LifecycleState,
    MCPBoundary,
    MCPBoundaryAdapter,
    MCPPolicy,
    Network,
    NetworkBoundaryAdapter,
    NetworkPolicy,
    Sandbox,
    SandboxBoundaryAdapter,
    SandboxPolicy,
)
from preventai.tools.database import InstrumentedDatabaseBackend
from tests._agent_communication_support import make_boundary, make_message


DENIED_STATES = (
    LifecycleState.SUSPENDED,
    LifecycleState.QUARANTINED,
    LifecycleState.REVOKED,
    LifecycleState.TERMINATED,
)


def make_managed_adapter(kind, tmp_path, backend_calls):
    if kind == "mcp":
        boundary = MCPBoundary(
            MCPPolicy(allowed_tools=("customer.read",)),
            lambda request: (backend_calls.append(request), {"ok": True})[1],
        )
        adapter = MCPBoundaryAdapter(boundary, boundary_id="mcp-test")
        payload = {
            "message": {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {"name": "customer.read", "arguments": {}},
            },
            "protocol_version": "2025-06-18",
            "client_id": "agent-a",
            "session_id": "session-a",
        }
    elif kind == "sandbox":
        boundary = Sandbox(SandboxPolicy(
            allowed_commands=("cmd.exe",),
            working_directory=str(tmp_path),
            allowed_paths=(str(tmp_path),),
        ))
        adapter = SandboxBoundaryAdapter(boundary, boundary_id="sandbox-test")
        payload = {
            "executable": "cmd.exe",
            "args": ["/c", "echo", "ok"],
            "cwd": str(tmp_path),
            "action": "process.execute",
            "resource": "sandbox-test",
        }
    elif kind == "network":
        boundary = Network(NetworkPolicy(allowed_hosts=("example.com",)))
        boundary._resolve = lambda *_: ("93.184.216.34",)
        boundary._request_once = lambda *args: (
            backend_calls.append(args),
            (200, {}, b"ok"),
        )[1]
        adapter = NetworkBoundaryAdapter(boundary, boundary_id="network-test")
        payload = {"url": "https://example.com/data", "method": "GET"}
    elif kind == "filesystem":
        path = Path(tmp_path) / "item"
        path.write_text("ok", encoding="utf-8")
        boundary = FileSystem(FileSystemPolicy(
            allowed_roots=(str(tmp_path),),
            allowed_operations=("read",),
        ))
        boundary._perform = lambda *args: backend_calls.append(args)
        adapter = FileSystemBoundaryAdapter(boundary, boundary_id="filesystem-test")
        payload = {"operation": "read", "path": str(path)}
    else:
        backend = InstrumentedDatabaseBackend()
        boundary = Database(DatabasePolicy(
            allowed_databases=("db",),
            allowed_schemas=("public",),
            allowed_tables=("customers",),
            allowed_columns=("id",),
        ), backend=backend)
        adapter = DatabaseBoundaryAdapter(boundary, boundary_id="database-test")
        backend_calls.append(backend)
        payload = {
            "operation": "SELECT",
            "database": "db",
            "schema": "public",
            "table": "customers",
            "columns": ["id"],
            "template": "SELECT id FROM db.public.customers WHERE id = ?",
            "parameters": [1],
        }
    return adapter, payload


@pytest.mark.parametrize("kind", ["mcp", "sandbox", "network", "filesystem", "database"])
@pytest.mark.parametrize("state", DENIED_STATES)
def test_managed_boundary_denies_non_active_sender_before_backend(kind, state, tmp_path):
    boundary, context, _ = make_boundary()
    lifecycle = boundary._lifecycle
    calls = []
    adapter, payload = make_managed_adapter(kind, tmp_path, calls)
    message = make_message(payload=payload, message_id=f"{kind}-{state.value}")
    authorization = boundary.issue(message, context)
    lifecycle.transition("agent-a", state)

    with pytest.raises(AgentCommunicationDenied):
        boundary.dispatch_managed(message, authorization, context, adapter)
    if kind == "database":
        assert calls[0].execution_count == 0
    else:
        assert calls == []
    assert any(event.event_type is AuditEventType.LIFECYCLE_BLOCKED
               for event in lifecycle._audit.events())


@pytest.mark.parametrize("state", DENIED_STATES)
def test_agent_communication_denies_non_active_receiver(state):
    boundary, context, downstream_calls = make_boundary()
    lifecycle = boundary._lifecycle
    message = make_message()
    authorization = boundary.issue(message, context)
    lifecycle.transition("agent-b", state)

    with pytest.raises(AgentCommunicationDenied):
        boundary.dispatch(message, authorization, context)
    assert downstream_calls == []


@pytest.mark.parametrize("agent_id", ["agent-a", "agent-b"])
def test_agent_communication_unknown_participant_fails_closed(agent_id):
    boundary, context, downstream_calls = make_boundary()
    message = make_message()
    authorization = boundary.issue(message, context)
    records = dict(boundary._lifecycle._record_snapshot)
    records.pop(agent_id)
    boundary._lifecycle._record_snapshot = MappingProxyType(records)

    with pytest.raises(AgentCommunicationDenied):
        boundary.dispatch(message, authorization, context)
    assert downstream_calls == []


def test_adapter_cannot_be_used_without_managed_lifecycle_context(tmp_path):
    calls = []
    adapter, payload = make_managed_adapter("mcp", tmp_path, calls)
    message = make_message(payload=payload)

    with pytest.raises(AgentCommunicationDenied):
        adapter.authorize(message)
    assert calls == []


def test_unbound_communication_boundary_fails_closed():
    boundary, context, downstream_calls = make_boundary()
    boundary._lifecycle = None

    with pytest.raises(AgentCommunicationDenied):
        boundary.validate(make_message(), context)
    assert downstream_calls == []


def test_adapter_final_check_closes_lifecycle_toctou(tmp_path):
    boundary, context, _ = make_boundary()
    registry = boundary._lifecycle
    calls = []

    adapter, payload = make_managed_adapter("mcp", tmp_path, calls)
    message = make_message(payload=payload, message_id="lifecycle-toctou")
    authorization = boundary.issue(message, context)
    original_authorize = adapter.authorize

    def authorize_then_revoke(request):
        tool_authorization = original_authorize(request)
        registry.revoke("agent-a")
        return tool_authorization

    adapter.authorize = authorize_then_revoke

    with pytest.raises(AgentCommunicationDenied):
        boundary.dispatch_managed(message, authorization, context, adapter)
    assert calls == []
