import pytest
from _agent_communication_support import make_boundary, make_message
from preventai import AgentCommunicationDenied


def test_permission_boundary_is_required():
    boundary, context, calls = make_boundary(permission_check=None)
    with pytest.raises(AgentCommunicationDenied):
        boundary.issue(make_message(), context)
    assert calls == []


def test_delegation_cannot_expand_scope():
    boundary, context, calls = make_boundary(delegation_check=lambda message: False)
    with pytest.raises(AgentCommunicationDenied):
        boundary.dispatch(make_message(), boundary.issue(make_message(), context), context)
    assert calls == []


def test_policy_denial_is_pre_execution():
    boundary, context, calls = make_boundary(policy_check=lambda message: False)
    with pytest.raises(AgentCommunicationDenied):
        boundary.dispatch(make_message(), boundary.issue(make_message(), context), context)
    assert calls == []


def test_dataflow_denial_is_pre_execution():
    boundary, context, calls = make_boundary(dataflow_check=lambda message: False)
    with pytest.raises(AgentCommunicationDenied):
        boundary.dispatch(make_message(), boundary.issue(make_message(), context), context)
    assert calls == []


def test_runtime_denial_is_pre_execution():
    boundary, context, calls = make_boundary(runtime_check=lambda message: False)
    with pytest.raises(AgentCommunicationDenied):
        boundary.dispatch(make_message(), boundary.issue(make_message(), context), context)
    assert calls == []


def test_approval_denial_is_pre_execution():
    boundary, context, calls = make_boundary(approval_check=lambda message: False)
    with pytest.raises(AgentCommunicationDenied):
        boundary.dispatch(make_message(), boundary.issue(make_message(), context), context)
    assert calls == []


def test_unauthenticated_context_denied():
    boundary, context, calls = make_boundary()
    unauthenticated = type(context)(context.agent_id, context.session_id, context.tenant_id, False)
    with pytest.raises(AgentCommunicationDenied):
        boundary.issue(make_message(), unauthenticated)
    assert calls == []


def test_wrong_receiver_policy_denied():
    boundary, context, calls = make_boundary()
    message = make_message(receiver_agent_id="agent-c")
    with pytest.raises(AgentCommunicationDenied):
        boundary.issue(message, context)
    assert calls == []


def test_forwarding_without_child_authority_is_denied():
    boundary, context, _ = make_boundary()
    message = make_message()
    authorization = boundary.issue(message, context)
    with pytest.raises(Exception):
        boundary.forward(message, context, "agent-b", authorization)


def test_response_is_structured():
    boundary, context, _ = make_boundary()
    message = make_message()
    result = boundary.dispatch(message, boundary.issue(message, context), context)
    assert result.message_id == "m1" and result.payload == {"accepted": True}
