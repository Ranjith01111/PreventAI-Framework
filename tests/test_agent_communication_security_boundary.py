import pytest
from _agent_communication_support import make_boundary, make_message
from preventai import AgentCommunicationDenied, AgentMessagePolicy, AgentSecurityContext


@pytest.mark.parametrize("tenant", ["tenant-b", "", None])
def test_tenant_isolation(tenant):
    boundary, context, calls = make_boundary()
    with pytest.raises(Exception):
        message = make_message(tenant_id=tenant)
        boundary.issue(message, context)
    assert calls == []


@pytest.mark.parametrize("receiver", ["agent-c", "agent-a", ""])
def test_receiver_allowlist(receiver):
    boundary, context, calls = make_boundary()
    with pytest.raises(Exception):
        message = make_message(receiver_agent_id=receiver)
        boundary.issue(message, context)
    assert calls == []


def test_cross_boundary_authorization_fails():
    first, context, _ = make_boundary()
    second, _, calls = make_boundary()
    message = make_message()
    with pytest.raises(AgentCommunicationDenied):
        second.dispatch(message, first.issue(message, context), context)
    assert calls == []


def test_capability_grant_is_attenuated():
    boundary, context, calls = make_boundary(allowed_capabilities=("read",))
    message = make_message(message_type="capability_grant", capability_scope=("write",))
    with pytest.raises(AgentCommunicationDenied):
        boundary.issue(message, context)
    assert calls == []


def test_delegation_scope_is_bound():
    boundary, context, calls = make_boundary()
    context = type(context)(
        context.agent_id, context.session_id, context.tenant_id, True,
        ("agent-b:read",),
    )
    boundary._contexts[context.agent_id] = context
    message = make_message(delegation_scope=("agent-b:read",))
    authorization = boundary.issue(message, context)
    changed = make_message(delegation_scope=("agent-b:write",))
    with pytest.raises(AgentCommunicationDenied):
        boundary.dispatch(changed, authorization, context)
    assert calls == []


def test_parent_message_binding():
    boundary, context, calls = make_boundary()
    message = make_message(parent_message_id="parent-a")
    authorization = boundary.issue(message, context)
    changed = make_message(parent_message_id="parent-b")
    with pytest.raises(AgentCommunicationDenied):
        boundary.dispatch(changed, authorization, context)
    assert calls == []


def test_protocol_version_binding():
    boundary, context, calls = make_boundary()
    message = make_message(protocol_version="1.0")
    authorization = boundary.issue(message, context)
    changed = make_message(protocol_version="2.0")
    with pytest.raises(AgentCommunicationDenied):
        boundary.dispatch(changed, authorization, context)
    assert calls == []


def test_correlation_binding():
    boundary, context, calls = make_boundary()
    message = make_message(correlation_id="one")
    authorization = boundary.issue(message, context)
    changed = make_message(correlation_id="two")
    with pytest.raises(AgentCommunicationDenied):
        boundary.dispatch(changed, authorization, context)
    assert calls == []


def test_classification_binding():
    boundary, context, calls = make_boundary()
    message = make_message(classification="restricted")
    authorization = boundary.issue(message, context)
    changed = make_message(classification="public")
    with pytest.raises(AgentCommunicationDenied):
        boundary.dispatch(changed, authorization, context)
    assert calls == []


def test_oversized_message_denied():
    boundary, context, calls = make_boundary(max_message_bytes=20)
    with pytest.raises(AgentCommunicationDenied):
        boundary.issue(make_message(), context)
    assert calls == []


def test_malformed_context_denied():
    boundary, calls_context, calls = make_boundary()
    with pytest.raises(AgentCommunicationDenied):
        boundary.issue(make_message(), object())  # type: ignore[arg-type]
    assert calls == []


def test_no_payload_fallback():
    boundary, context, calls = make_boundary()
    with pytest.raises(Exception):
        boundary.dispatch("message", None, context)  # type: ignore[arg-type]
    assert calls == []


def test_forged_equal_identity_context_is_denied():
    boundary, context, calls = make_boundary()
    forged = AgentSecurityContext(
        context.agent_id, context.session_id, context.tenant_id, True,
    )
    with pytest.raises(AgentCommunicationDenied):
        boundary.issue(make_message(), forged)
    assert calls == []


def test_message_tenant_must_match_authenticated_context():
    boundary, context, calls = make_boundary()
    message = make_message(tenant_id="tenant-b")
    with pytest.raises(AgentCommunicationDenied):
        boundary.issue(message, context)
    assert calls == []


def test_message_capability_must_be_owned_by_sender():
    boundary, context, calls = make_boundary(allowed_capabilities=("read",))
    message = make_message(capability_scope=("read",))
    with pytest.raises(AgentCommunicationDenied):
        boundary.issue(message, context)
    assert calls == []


def test_direct_issuance_without_framework_capability_is_denied():
    boundary, context, calls = make_boundary()
    implementation_issue = type(boundary).issue
    with pytest.raises(AgentCommunicationDenied):
        implementation_issue(boundary, make_message(), context)
    assert calls == []
