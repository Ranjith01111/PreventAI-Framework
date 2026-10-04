import pytest
from dataclasses import replace
from _agent_communication_support import make_boundary, make_message
from preventai import AgentCommunicationDenied, AgentSecurityContext


def test_forged_signature():
    boundary, context, calls = make_boundary()
    message = make_message()
    authorization = boundary.issue(message, context)
    object.__setattr__(authorization, "signature", "forged")
    with pytest.raises(AgentCommunicationDenied):
        boundary.dispatch(message, authorization, context)
    assert calls == []


def test_replay_is_denied():
    boundary, context, calls = make_boundary()
    message = make_message()
    authorization = boundary.issue(message, context)
    boundary.dispatch(message, authorization, context)
    with pytest.raises(AgentCommunicationDenied):
        boundary.dispatch(message, authorization, context)
    assert len(calls) == 1


def test_cross_agent_replay():
    boundary, context, calls = make_boundary()
    message = make_message()
    authorization = boundary.issue(message, context)
    changed = replace(message, receiver_agent_id="agent-c")
    with pytest.raises(AgentCommunicationDenied):
        boundary.dispatch(changed, authorization, context)
    assert calls == []


def test_cross_session_replay():
    boundary, context, calls = make_boundary()
    message = make_message()
    authorization = boundary.issue(message, context)
    changed_context = AgentSecurityContext("agent-a", "session-other", "tenant-a", True)
    with pytest.raises(AgentCommunicationDenied):
        boundary.dispatch(message, authorization, changed_context)
    assert calls == []


def test_cross_tenant_denied():
    boundary, context, calls = make_boundary()
    message = make_message(tenant_id="tenant-b")
    with pytest.raises(AgentCommunicationDenied):
        boundary.issue(message, context)
    assert calls == []


def test_unknown_sender_denied():
    boundary, calls_context, calls = make_boundary()
    context = AgentSecurityContext("agent-d", "s", "tenant-a", True)
    message = make_message(sender_agent_id="agent-d", sender_session_id="s")
    with pytest.raises(AgentCommunicationDenied):
        boundary.issue(message, context)
    assert calls == []


@pytest.mark.parametrize("hook", [
    "policy_check", "permission_check", "delegation_check",
    "dataflow_check", "runtime_check", "approval_check",
])
def test_security_hook_denial_has_zero_downstream(hook):
    boundary, context, calls = make_boundary()
    boundary._checks = tuple((lambda message: False) if i == ("policy_check", "permission_check",
        "delegation_check", "dataflow_check", "runtime_check", "approval_check").index(hook)
        else check for i, check in enumerate(boundary._checks))
    message = make_message()
    with pytest.raises(AgentCommunicationDenied):
        boundary.dispatch(message, boundary.issue(message, context), context)
    assert calls == []


def test_missing_security_hook_blocks():
    boundary, context, calls = make_boundary()
    boundary._checks = (None, *boundary._checks[1:])
    message = make_message()
    with pytest.raises(AgentCommunicationDenied):
        boundary.issue(message, context)
    assert calls == []


def test_receiver_substitution_blocks():
    boundary, context, calls = make_boundary()
    with pytest.raises(AgentCommunicationDenied):
        boundary.forward(make_message(), context, "agent-c")
    assert calls == []


def test_chain_depth_limit():
    boundary, context, calls = make_boundary(max_chain_depth=1)
    message = make_message(chain_depth=2)
    with pytest.raises(AgentCommunicationDenied):
        boundary.issue(message, context)
    assert calls == []


def test_payload_limit():
    boundary, context, calls = make_boundary(max_payload_bytes=8)
    message = make_message(payload={"large": "x" * 100})
    with pytest.raises(AgentCommunicationDenied):
        boundary.issue(message, context)
    assert calls == []


def test_capability_limit():
    boundary, context, calls = make_boundary(allowed_capabilities=("read",))
    message = make_message(capability_scope=("write",))
    with pytest.raises(AgentCommunicationDenied):
        boundary.issue(message, context)
    assert calls == []


def test_nonce_tampering_is_denied_when_replay_is_disabled():
    boundary, context, calls = make_boundary(one_time=False)
    message = make_message()
    authorization = boundary.issue(message, context)
    object.__setattr__(authorization, "nonce", "tampered")
    with pytest.raises(AgentCommunicationDenied):
        boundary.dispatch(message, authorization, context)
    assert calls == []


def test_missing_output_guard_blocks_before_downstream():
    boundary, context, calls = make_boundary()
    boundary._output_check = None
    message = make_message()
    with pytest.raises(AgentCommunicationDenied):
        boundary.dispatch(message, boundary.issue(message, context), context)
    assert calls == []


def test_missing_audit_integration_blocks_before_downstream():
    boundary, context, calls = make_boundary()
    boundary._audit_check = None
    message = make_message()
    with pytest.raises(AgentCommunicationDenied):
        boundary.dispatch(message, boundary.issue(message, context), context)
    assert calls == []
