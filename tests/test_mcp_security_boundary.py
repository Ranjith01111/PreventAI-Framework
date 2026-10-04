import pytest
from preventai import MCPExecutionDenied, MCPPolicy, canonical_resource_uri
from _mcp_test_support import MCPBoundary
from test_mcp import message, valid


@pytest.mark.parametrize("uri", ["mcp:/docs/readme", "urn:resource:one"])
def test_resource_uri_allowlist(uri):
    assert canonical_resource_uri(uri) == uri


@pytest.mark.parametrize("uri", ["../secret", "mcp:/../secret", "mcp://host/path", "file:/secret", "mcp:\\secret", "mcp:/%2e%2e/secret"])
def test_resource_uri_abuse_denied(uri):
    with pytest.raises(Exception): canonical_resource_uri(uri)


def test_resource_identity_binds_authorization():
    b = MCPBoundary(MCPPolicy(allowed_resources=("mcp:/docs/readme", "mcp:/docs/other")), lambda r: {"ok": True})
    r = b.parse(message(method="resources/read", params={"uri": "mcp:/docs/readme"}),
                protocol_version="2025-06-18", client_id="c", session_id="s")
    token = b.issue(r)
    other = b.parse(message(method="resources/read", params={"uri": "mcp:/docs/other"}),
                    protocol_version="2025-06-18", client_id="c", session_id="s")
    with pytest.raises(MCPExecutionDenied): b.dispatch(other, token)


def test_prompt_identity_binds_authorization():
    b = MCPBoundary(MCPPolicy(allowed_prompts=("welcome", "other")), lambda r: {"ok": True})
    r = b.parse(message(method="prompts/get", params={"name": "welcome", "arguments": {}}),
                protocol_version="2025-06-18", client_id="c", session_id="s")
    token = b.issue(r)
    other = b.parse(message(method="prompts/get", params={"name": "other", "arguments": {}}),
                    protocol_version="2025-06-18", client_id="c", session_id="s")
    with pytest.raises(MCPExecutionDenied): b.dispatch(other, token)


def test_response_size_limit():
    calls = []
    b = MCPBoundary(MCPPolicy(allowed_tools=("echo",), max_response_bytes=10),
                    lambda r: (calls.append(r), {"large": "x" * 100})[1])
    r = valid(b)
    with pytest.raises(MCPExecutionDenied): b.dispatch(r, b.issue(r))
    assert len(calls) == 1


def test_response_correlation():
    b = MCPBoundary(MCPPolicy(allowed_tools=("echo",)), lambda r: {"id": r.request_id})
    r = valid(b, request_id=7)
    response = b.dispatch(r, b.issue(r))
    assert response.request_id == 7


def test_duplicate_request_id_denied():
    b = MCPBoundary(MCPPolicy(allowed_tools=("echo",)), lambda r: {"ok": True})
    r = valid(b)
    b.dispatch(r, b.issue(r))
    r2 = valid(b)
    with pytest.raises(MCPExecutionDenied): b.dispatch(r2, b.issue(r2))


def test_invalid_downstream_result_denied():
    b = MCPBoundary(MCPPolicy(allowed_tools=("echo",)), lambda r: "not-an-object")
    r = valid(b)
    with pytest.raises(MCPExecutionDenied): b.dispatch(r, b.issue(r))


def test_audit_receives_safe_event_only():
    events = []
    b = MCPBoundary(MCPPolicy(allowed_tools=("echo",)), lambda r: {"ok": True},
                    audit=lambda event, request: events.append((event, request.method, request.identity)))
    r = valid(b)
    b.dispatch(r, b.issue(r))
    assert events == [("accepted", "tools/call", "echo")]


@pytest.mark.parametrize("method", ["resources/read", "prompts/get"])
def test_missing_identity_denied(method):
    b = MCPBoundary(MCPPolicy(allowed_resources=("mcp:/docs/readme",), allowed_prompts=("welcome",)), lambda r: {})
    params = {"uri": "mcp:/docs/readme"} if method == "resources/read" else {"name": "welcome"}
    request = b.parse(message(method=method, params=params), protocol_version="2025-06-18", client_id="c", session_id="s")
    assert request.identity


def test_empty_resource_allowlist_denies():
    boundary = MCPBoundary(MCPPolicy(allowed_tools=("echo",)), lambda r: {})
    with pytest.raises(MCPExecutionDenied):
        boundary.parse(message(method="resources/read", params={"uri": "mcp:/public/a"}),
                       protocol_version="2025-06-18", client_id="c", session_id="s")


def test_empty_prompt_allowlist_denies():
    boundary = MCPBoundary(MCPPolicy(allowed_tools=("echo",)), lambda r: {})
    with pytest.raises(MCPExecutionDenied):
        boundary.parse(message(method="prompts/get", params={"name": "safe-prompt"}),
                       protocol_version="2025-06-18", client_id="c", session_id="s")
