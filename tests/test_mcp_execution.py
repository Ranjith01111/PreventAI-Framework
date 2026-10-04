import time
import pytest
from preventai import MCPExecutionDenied, MCPPolicy
from _mcp_test_support import MCPBoundary
from test_mcp import message, valid
from _boundary_test_support import bind_test_boundary


def boundary(**kwargs):
    calls = []
    policy = MCPPolicy(allowed_tools=("echo",), **kwargs)
    return bind_test_boundary(
        MCPBoundary(
            policy,
            lambda r: (calls.append(r), {"value": r.params["arguments"]["value"]})[1],
        )
    ), calls


def test_expired_authorization():
    b, calls = boundary(authorization_ttl=0.001)
    r = valid(b); token = b.issue(r); time.sleep(0.01)
    with pytest.raises(MCPExecutionDenied): b.dispatch(r, token)
    assert not calls


def test_revoked_authorization():
    b, calls = boundary(); r = valid(b); token = b.issue(r); b.revoke(token)
    with pytest.raises(MCPExecutionDenied): b.dispatch(r, token)
    assert not calls


def test_replay_authorization():
    b, calls = boundary(); r = valid(b); token = b.issue(r)
    b.dispatch(r, token)
    with pytest.raises(MCPExecutionDenied): b.dispatch(r, token)
    assert len(calls) == 1


def test_forged_authorization():
    b, calls = boundary(); r = valid(b); token = b.issue(r)
    object.__setattr__(token, "signature", "forged")
    with pytest.raises(MCPExecutionDenied): b.dispatch(r, token)
    assert not calls


@pytest.mark.parametrize("params", [
    {"name": "echo", "arguments": {"value": "x", "extra": "y"}},
    {"name": "echo", "arguments": {"value": ["nested"]}},
    {"name": "echo", "arguments": {"value": None}},
])
def test_typed_json_arguments(params):
    b, calls = boundary()
    r = b.parse(message(params=params), protocol_version="2025-06-18", client_id="c", session_id="s")
    assert b.dispatch(r, b.issue(r)).result is not None


def test_argument_limit():
    b, calls = boundary(max_arguments=1)
    with pytest.raises(MCPExecutionDenied): valid(b)
    assert not calls


def test_message_limit():
    b, calls = boundary(max_message_bytes=20)
    with pytest.raises(MCPExecutionDenied): valid(b)
    assert not calls


def test_version_denial():
    b, calls = boundary(protocol_versions=("2025-06-18",))
    with pytest.raises(MCPExecutionDenied):
        b.parse(message(), protocol_version="old", client_id="c", session_id="s")
    assert not calls


def test_client_allowlist():
    b, calls = boundary(allowed_clients=("trusted",))
    with pytest.raises(MCPExecutionDenied):
        b.parse(message(), protocol_version="2025-06-18", client_id="untrusted", session_id="s")
    assert not calls


def test_framework_hook_denial_has_zero_downstream():
    b, calls = boundary()
    b._checks = (lambda r: False,)
    r = valid(b)
    with pytest.raises(MCPExecutionDenied): b.dispatch(r, b.issue(r))
    assert not calls


def test_output_hook_denial():
    calls = []
    b = bind_test_boundary(MCPBoundary(
        MCPPolicy(allowed_tools=("echo",)),
        lambda r: (calls.append(r), {"secret": "x"})[1],
        output_check=lambda r: False,
    ))
    r = valid(b)
    with pytest.raises(MCPExecutionDenied): b.dispatch(r, b.issue(r))
    assert len(calls) == 1
