import pytest
from _agent_communication_support import make_boundary, make_message
from preventai import AgentCommunicationDenied


@pytest.mark.parametrize("payload", [
    {"command": "delete"},
    {"grant": "*"},
    {"sender_agent_id": "agent-admin"},
    {"receiver_agent_id": "agent-c"},
    {"capability_scope": ["admin"]},
])
def test_untrusted_payload_cannot_change_security_scope(payload):
    boundary, context, calls = make_boundary()
    message = make_message(payload=payload)
    authorization = boundary.issue(message, context)
    boundary.dispatch(message, authorization, context)
    assert len(calls) == 1


def test_message_id_replay_is_denied_with_new_signature():
    boundary, context, calls = make_boundary()
    first = make_message()
    boundary.dispatch(first, boundary.issue(first, context), context)
    second = make_message(payload={"new": True})
    with pytest.raises(AgentCommunicationDenied):
        boundary.dispatch(second, boundary.issue(second, context), context)
    assert len(calls) == 1


def test_tampered_payload_denied():
    boundary, context, calls = make_boundary()
    message = make_message(payload={"value": "a"})
    authorization = boundary.issue(message, context)
    changed = make_message(payload={"value": "b"})
    with pytest.raises(AgentCommunicationDenied):
        boundary.dispatch(changed, authorization, context)
    assert calls == []


def test_tampered_sender_denied():
    boundary, context, calls = make_boundary()
    message = make_message()
    authorization = boundary.issue(message, context)
    changed = make_message(sender_agent_id="agent-d")
    with pytest.raises(AgentCommunicationDenied):
        boundary.dispatch(changed, authorization, context)
    assert calls == []


def test_error_messages_are_categories_only():
    boundary, context, calls = make_boundary()
    with pytest.raises(AgentCommunicationDenied) as exc:
        boundary.dispatch(make_message(), None, context)
    assert "hello" not in str(exc.value)
    assert calls == []


def test_parent_chain_tamper_denied():
    boundary, context, calls = make_boundary()
    message = make_message(parent_message_id="p1")
    token = boundary.issue(message, context)
    with pytest.raises(AgentCommunicationDenied):
        boundary.dispatch(make_message(parent_message_id="p2"), token, context)
    assert calls == []


def test_session_binding_in_authorization():
    boundary, context, calls = make_boundary()
    message = make_message()
    token = boundary.issue(message, context)
    changed = type(context)("agent-a", "session-c", "tenant-a", True)
    with pytest.raises(AgentCommunicationDenied):
        boundary.dispatch(message, token, changed)
    assert calls == []


def test_sender_cannot_self_assert_identity():
    boundary, context, calls = make_boundary()
    message = make_message(sender_agent_id="agent-admin", sender_session_id="admin-session")
    fake = type(context)("agent-admin", "admin-session", "tenant-a", True)
    with pytest.raises(AgentCommunicationDenied):
        boundary.issue(message, fake)
    assert calls == []


def test_capability_change_changes_digest():
    assert make_message(capability_scope=("read",)).digest() != make_message(capability_scope=("write",)).digest()


def test_message_type_change_changes_digest():
    assert make_message(message_type="request").digest() != make_message(message_type="response").digest()


def test_receiver_session_change_changes_digest():
    assert make_message(receiver_session_id="s1").digest() != make_message(receiver_session_id="s2").digest()


def test_explicit_revocation():
    boundary, context, calls = make_boundary()
    message = make_message()
    token = boundary.issue(message, context)
    boundary.revoke(token)
    with pytest.raises(AgentCommunicationDenied):
        boundary.dispatch(message, token, context)
    assert calls == []


@pytest.mark.parametrize("expiry", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_expiry_is_rejected(expiry):
    with pytest.raises(Exception):
        make_message(expires_at=expiry)
