import pytest
from preventai import MCPExecutionDenied, MCPPolicy
from _mcp_test_support import MCPBoundary
from test_mcp import message, valid


@pytest.mark.parametrize("payload", [
    {"jsonrpc": "1.0", "id": 1, "method": "tools/call", "params": {}},
    {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": []},
    {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "echo", "arguments": {"x": object()}}},
])
def test_malformed_protocol_payloads_do_not_execute(payload):
    calls = []
    b = MCPBoundary(MCPPolicy(allowed_tools=("echo",)), lambda r: (calls.append(r), {})[1])
    with pytest.raises(Exception):
        b.parse(payload, protocol_version="2025-06-18", client_id="c", session_id="s")
    assert calls == []


@pytest.mark.parametrize("version", ["", "2024-01-01", "9999", "1.0"])
def test_version_confusion_denied(version):
    b = MCPBoundary(MCPPolicy(allowed_tools=("echo",)), lambda r: {})
    with pytest.raises(MCPExecutionDenied):
        b.parse(message(), protocol_version=version, client_id="c", session_id="s")


@pytest.mark.parametrize("server,client,session", [("other", "c", "s"), ("preventai-server", "other", "s"), ("preventai-server", "c", "other")])
def test_context_isolation(server, client, session):
    b = MCPBoundary(MCPPolicy(allowed_tools=("echo",), server_id="preventai-server"), lambda r: {})
    request = b.parse(message(), protocol_version="2025-06-18", client_id="c", session_id="s")
    changed = b.parse(message(), protocol_version="2025-06-18", client_id=client, session_id=session)
    if server != "preventai-server":
        changed = type(changed)(changed.method, changed.request_id, changed.params, changed.protocol_version, server, changed.client_id, changed.session_id)
    with pytest.raises(MCPExecutionDenied): b.dispatch(changed, b.issue(request))


def test_untrusted_arguments_cannot_change_policy():
    b = MCPBoundary(MCPPolicy(allowed_tools=("echo",)), lambda r: {"ok": True})
    request = b.parse(message(params={"name": "echo", "arguments": {"allowed_tools": ["admin"]}}),
                      protocol_version="2025-06-18", client_id="c", session_id="s")
    assert b.dispatch(request, b.issue(request)).result == {"ok": True}
    assert b.policy.allowed_tools == ("echo",)


@pytest.mark.parametrize("name", ["echo", "echo ", "ECHO", "echo/other", "admin"])
def test_tool_identity_canonical_scope(name):
    b = MCPBoundary(MCPPolicy(allowed_tools=("echo",)), lambda r: {})
    if name == "echo":
        assert valid(b, params={"name": name, "arguments": {}}).identity == "echo"
    else:
        with pytest.raises(MCPExecutionDenied): valid(b, params={"name": name, "arguments": {}})


def test_no_raw_method_dispatch():
    calls = []
    b = MCPBoundary(MCPPolicy(allowed_tools=("echo",)), lambda r: (calls.append(r), {})[1])
    with pytest.raises(Exception):
        b.dispatch("tools/call", None)  # type: ignore[arg-type]
    assert not calls


def test_forged_response_is_not_created_by_boundary():
    b = MCPBoundary(MCPPolicy(allowed_tools=("echo",)), lambda r: {"ok": True})
    r = valid(b)
    response = b.dispatch(r, b.issue(r))
    assert response.request_id == r.request_id
    assert response.error is None


@pytest.mark.parametrize("method", ["tools/list", "resources/list", "prompts/list"])
def test_discovery_methods_do_not_become_tool_calls(method):
    calls = []
    b = MCPBoundary(MCPPolicy(allowed_tools=("echo",)), lambda r: (calls.append(r), {})[1])
    r = valid(b, method=method, params={})
    response = b.dispatch(r, b.issue(r))
    assert response.result is not None and not calls


def test_capability_change_invalidates_digest():
    b = MCPBoundary(MCPPolicy(allowed_tools=("echo",), allowed_capabilities=("safe",)), lambda r: {})
    r = b.parse({**message(), "params": {"name": "echo", "arguments": {}, "capabilities": ["safe"]}},
                protocol_version="2025-06-18", client_id="c", session_id="s")
    token = b.issue(r)
    changed = b.parse(message(), protocol_version="2025-06-18", client_id="c", session_id="s")
    with pytest.raises(MCPExecutionDenied): b.dispatch(changed, token)
