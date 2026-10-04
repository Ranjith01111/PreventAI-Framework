import pytest
from preventai import MCPExecutionDenied, MCPPolicy
from _mcp_test_support import MCPBoundary
from preventai import MCPBoundary as RealMCPBoundary
from test_mcp import message, valid


@pytest.mark.parametrize("name", ["other", "filesystem.write", "database.delete"])
def test_tool_allowlist_is_explicit(name):
    b = MCPBoundary(MCPPolicy(allowed_tools=("echo",)), lambda r: {})
    with pytest.raises(MCPExecutionDenied):
        valid(b, params={"name": name, "arguments": {}})


def test_capability_spoofing_denied():
    b = MCPBoundary(MCPPolicy(allowed_tools=("echo",), allowed_capabilities=("safe",)), lambda r: {})
    with pytest.raises(MCPExecutionDenied):
        b.parse({**message(), "params": {"name": "echo", "capabilities": ["admin"], "arguments": {}}},
                protocol_version="2025-06-18", client_id="c", session_id="s")


def test_capability_allowlist():
    b = MCPBoundary(MCPPolicy(allowed_tools=("echo",), allowed_capabilities=("safe",)), lambda r: {})
    request = b.parse({**message(), "params": {"name": "echo", "capabilities": ["safe"], "arguments": {}}},
                      protocol_version="2025-06-18", client_id="c", session_id="s")
    assert request.method == "tools/call"


@pytest.mark.parametrize("method", ["notifications/initialized", "notifications/cancelled"])
def test_notifications_never_dispatch(method):
    b = MCPBoundary(MCPPolicy(allowed_tools=("echo",)), lambda r: pytest.fail("downstream"))
    request = b.parse({"jsonrpc": "2.0", "method": method, "params": {}},
                      protocol_version="2025-06-18", client_id="c", session_id="s")
    with pytest.raises(MCPExecutionDenied): b.dispatch(request, None)


def test_mcp_does_not_fallback_to_generic_method():
    calls = []
    b = MCPBoundary(MCPPolicy(allowed_tools=("echo",)), lambda r: (calls.append(r), {})[1])
    with pytest.raises(MCPExecutionDenied):
        b.parse(message(method="arbitrary_callable"), protocol_version="2025-06-18", client_id="c", session_id="s")
    assert calls == []


def test_delegation_hook_is_authoritative():
    calls = []
    b = MCPBoundary(MCPPolicy(allowed_tools=("echo",)), lambda r: calls.append(r), delegation_check=lambda r: False)
    r = valid(b)
    with pytest.raises(MCPExecutionDenied): b.dispatch(r, b.issue(r))
    assert calls == []


def test_permission_hook_is_authoritative():
    calls = []
    b = MCPBoundary(MCPPolicy(allowed_tools=("echo",)), lambda r: calls.append(r), permission_check=lambda r: False)
    r = valid(b)
    with pytest.raises(MCPExecutionDenied): b.dispatch(r, b.issue(r))
    assert calls == []


def test_dataflow_hook_is_authoritative():
    calls = []
    b = MCPBoundary(MCPPolicy(allowed_tools=("echo",)), lambda r: calls.append(r), dataflow_check=lambda r: False)
    r = valid(b)
    with pytest.raises(MCPExecutionDenied): b.dispatch(r, b.issue(r))
    assert calls == []


def test_runtime_hook_is_authoritative():
    calls = []
    b = MCPBoundary(MCPPolicy(allowed_tools=("echo",)), lambda r: calls.append(r), runtime_check=lambda r: False)
    r = valid(b)
    with pytest.raises(MCPExecutionDenied): b.dispatch(r, b.issue(r))
    assert calls == []


def test_approval_hook_is_authoritative():
    calls = []
    b = MCPBoundary(MCPPolicy(allowed_tools=("echo",)), lambda r: calls.append(r), approval_check=lambda r: False)
    r = valid(b)
    with pytest.raises(MCPExecutionDenied): b.dispatch(r, b.issue(r))
    assert calls == []


@pytest.mark.parametrize("hook", [
    "policy_check", "permission_check", "delegation_check",
    "dataflow_check", "runtime_check", "approval_check",
])
def test_missing_security_hook_blocks_before_downstream(hook):
    calls = []
    kwargs = {
        "policy_check": lambda r: True,
        "permission_check": lambda r: True,
        "delegation_check": lambda r: True,
        "dataflow_check": lambda r: True,
        "runtime_check": lambda r: True,
        "approval_check": lambda r: True,
    }
    kwargs[hook] = None
    boundary = RealMCPBoundary(MCPPolicy(allowed_tools=("echo",)), lambda r: (calls.append(r), {})[1], **kwargs)
    request = boundary.parse(message(), protocol_version="2025-06-18", client_id="c", session_id="s")
    with pytest.raises(MCPExecutionDenied):
        boundary.dispatch(request, boundary.issue(request))
    assert calls == []


def test_issue_revalidates_changed_policy():
    boundary = RealMCPBoundary(
        MCPPolicy(allowed_tools=("echo",)),
        lambda r: {},
        policy_check=lambda r: True, permission_check=lambda r: True,
        delegation_check=lambda r: True, dataflow_check=lambda r: True,
        runtime_check=lambda r: True, approval_check=lambda r: True,
    )
    request = boundary.parse(message(), protocol_version="2025-06-18", client_id="c", session_id="s")
    boundary.policy = MCPPolicy(allowed_tools=("other",))
    with pytest.raises(MCPExecutionDenied):
        boundary.issue(request)
