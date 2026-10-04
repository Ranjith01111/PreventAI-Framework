import pytest
from preventai import MCPExecutionDenied, MCPPolicy
from _mcp_test_support import MCPBoundary


def make_boundary(**kwargs):
    calls = []
    policy = MCPPolicy(
        allowed_tools=("echo",), allowed_resources=("mcp:docs/readme",),
        allowed_prompts=("welcome",), **{k: v for k, v in kwargs.items() if k in {
            "allowed_methods", "allowed_capabilities", "allowed_clients",
            "protocol_versions", "max_message_bytes", "max_arguments",
            "max_string_length", "max_response_bytes", "server_id",
        }},
    )
    boundary = MCPBoundary(policy, lambda request: (calls.append(request), {"ok": True})[1])
    return boundary, calls


def message(method="tools/call", request_id=1, params=None):
    return {"jsonrpc": "2.0", "id": request_id, "method": method,
            "params": params if params is not None else {"name": "echo", "arguments": {"value": "x"}}}


def valid(boundary, **kwargs):
    return boundary.parse(message(**kwargs), protocol_version="2025-06-18", client_id="client-a", session_id="session-a")


def test_tools_call_dispatches():
    boundary, calls = make_boundary()
    request = valid(boundary)
    assert boundary.dispatch(request, boundary.issue(request)).result == {"ok": True}
    assert len(calls) == 1


@pytest.mark.parametrize("method", ["tools/list", "resources/list", "prompts/list", "initialize"])
def test_supported_methods_parse(method):
    boundary, _ = make_boundary()
    params = {} if method != "initialize" else {"protocolVersion": "2025-06-18"}
    assert valid(boundary, method=method, params=params).method == method


@pytest.mark.parametrize("method", ["unknown", "tools/callx", "", "admin/run"])
def test_unknown_methods_fail_closed(method):
    boundary, calls = make_boundary()
    with pytest.raises(MCPExecutionDenied):
        valid(boundary, method=method)
    assert not calls


@pytest.mark.parametrize("request_id", [True, [], {}, ""])
def test_invalid_request_ids_fail(request_id):
    boundary, _ = make_boundary()
    with pytest.raises(Exception):
        valid(boundary, request_id=request_id)


def test_malformed_jsonrpc_fails():
    boundary, calls = make_boundary()
    with pytest.raises(Exception):
        boundary.parse({"method": "tools/call"}, protocol_version="2025-06-18", client_id="client-a", session_id="s")
    assert not calls


def test_invalid_params_fail():
    boundary, _ = make_boundary()
    with pytest.raises(Exception):
        boundary.parse(message(params=[]), protocol_version="2025-06-18", client_id="client-a", session_id="s")


def test_unknown_tool_denied():
    boundary, calls = make_boundary()
    with pytest.raises(MCPExecutionDenied):
        valid(boundary, params={"name": "admin", "arguments": {}})
    assert not calls


def test_missing_authorization_denied():
    boundary, calls = make_boundary()
    with pytest.raises(MCPExecutionDenied):
        boundary.dispatch(valid(boundary), None)
    assert not calls


def test_request_digest_binding():
    boundary, calls = make_boundary()
    request = valid(boundary)
    authorization = boundary.issue(request)
    changed = valid(boundary, request_id=2)
    with pytest.raises(MCPExecutionDenied):
        boundary.dispatch(changed, authorization)
    assert not calls


def test_cross_server_authorization_denied():
    first, _ = make_boundary(server_id="server-a")
    second, calls = make_boundary(server_id="server-b")
    request = valid(first)
    with pytest.raises(MCPExecutionDenied):
        second.dispatch(request, first.issue(request))
    assert not calls


def test_cross_client_authorization_denied():
    boundary, calls = make_boundary()
    request = valid(boundary)
    changed = boundary.parse(message(), protocol_version="2025-06-18", client_id="client-b", session_id="session-a")
    with pytest.raises(MCPExecutionDenied):
        boundary.dispatch(changed, boundary.issue(request))
    assert not calls


def test_cross_session_authorization_denied():
    boundary, calls = make_boundary()
    request = valid(boundary)
    changed = boundary.parse(message(), protocol_version="2025-06-18", client_id="client-a", session_id="session-b")
    with pytest.raises(MCPExecutionDenied):
        boundary.dispatch(changed, boundary.issue(request))
    assert not calls
