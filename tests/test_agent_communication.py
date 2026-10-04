import pytest
from preventai import AgentCommunicationDenied
from _agent_communication_support import make_boundary, make_message


def test_request_round_trip():
    boundary, context, calls = make_boundary()
    message = make_message()
    response = boundary.dispatch(message, boundary.issue(message, context), context)
    assert response.accepted and len(calls) == 1


@pytest.mark.parametrize("message_type", [
    "request", "response", "event", "capability_request", "error",
])
def test_supported_message_types(message_type):
    boundary, context, calls = make_boundary()
    message = make_message(message_type=message_type)
    boundary.dispatch(message, boundary.issue(message, context), context)
    assert len(calls) == 1


@pytest.mark.parametrize("message_type", ["unknown", "tool_call", "", "admin", "request/privileged"])
def test_unsupported_message_types_fail(message_type):
    with pytest.raises(Exception):
        make_message(message_type=message_type)


@pytest.mark.parametrize("field", [
    "message_id", "sender_agent_id", "receiver_agent_id", "sender_session_id",
    "conversation_id", "classification", "protocol_version",
])
def test_identity_fields_are_required(field):
    values = {field: ""}
    with pytest.raises(Exception):
        make_message(**values)


def test_arbitrary_object_payload_rejected():
    with pytest.raises(Exception):
        make_message(payload={"object": object()})


def test_expired_message_rejected_before_dispatch():
    boundary, context, calls = make_boundary()
    message = make_message(expires_at=0.1)
    with pytest.raises(AgentCommunicationDenied):
        boundary.issue(message, context)
    assert calls == []


def test_message_digest_binds_all_security_fields():
    original = make_message()
    for field in ("message_id", "sender_agent_id", "receiver_agent_id",
                  "sender_session_id", "conversation_id", "classification", "tenant_id"):
        changed = make_message(**{field: "changed"})
        assert changed.digest() != original.digest()


def test_payload_key_order_canonicalizes():
    assert make_message(payload={"a": 1, "b": 2}).digest() == make_message(payload={"b": 2, "a": 1}).digest()


def test_missing_authorization_blocks():
    boundary, context, calls = make_boundary()
    with pytest.raises(AgentCommunicationDenied):
        boundary.dispatch(make_message(), None, context)
    assert calls == []


def test_wrong_sender_context_blocks():
    boundary, _, calls = make_boundary()
    message = make_message()
    wrong = type(make_boundary()[1])("agent-d", "session-a", "tenant-a", True)
    with pytest.raises(AgentCommunicationDenied):
        boundary.dispatch(message, boundary.issue(message, make_boundary()[1]), wrong)
    assert calls == []
