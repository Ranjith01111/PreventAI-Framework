from datetime import datetime, timezone

import pytest

from preventai.protocol.security_request import (
    Action,
    AgentIdentity,
    ContractValidationError,
    DataClassification,
    DataMetadata,
    RequestedBy,
    Resource,
    SecurityRequest,
    SessionIdentity,
    SourceTrust,
    Tool,
)


def make_request(**changes):
    values = {
        "protocol_version": "1.0",
        "request_id": "req-001",
        "timestamp": datetime(2026, 1, 1, tzinfo=timezone.utc),
        "agent": AgentIdentity("agent-001", "autonomous"),
        "session": SessionIdentity("session-001"),
        "requested_by": RequestedBy.AGENT,
        "action": Action("read", {"path": "/docs/report.txt"}),
        "resource": Resource("report.txt", "file"),
        "tool": Tool("filesystem", "1.0"),
        "data": DataMetadata(DataClassification.INTERNAL, SourceTrust.TRUSTED),
        "context": {"purpose": "summarize"},
        "metadata": {"trace": "abc"},
    }
    values.update(changes)
    return SecurityRequest(**values)


def test_valid_request():
    request = make_request()
    assert request.action.name == "read"
    assert request.data.classification is DataClassification.INTERNAL


@pytest.mark.parametrize(
    "field,value",
    [
        ("agent", None),
        ("action", None),
    ],
)
def test_missing_security_critical_object(field, value):
    with pytest.raises(ContractValidationError):
        make_request(**{field: value})


def test_invalid_action():
    with pytest.raises(ContractValidationError):
        Action("", {})


def test_invalid_classification():
    with pytest.raises(ContractValidationError):
        DataMetadata("secret", SourceTrust.TRUSTED)


def test_invalid_trust_level():
    with pytest.raises(ContractValidationError):
        DataMetadata(DataClassification.INTERNAL, "trusted")


def test_invalid_context():
    with pytest.raises(ContractValidationError):
        make_request(context={"bad": object()})


def test_serialization_and_deserialization_are_deterministic():
    request = make_request()
    encoded = request.to_json()
    assert encoded == request.to_json()
    assert SecurityRequest.from_json(encoded) == request


def test_deserialization_is_strict():
    payload = make_request().to_dict()
    payload["unexpected"] = True
    with pytest.raises(ContractValidationError):
        SecurityRequest.from_dict(payload)


def test_protocol_version_validation():
    with pytest.raises(ContractValidationError):
        make_request(protocol_version="2.0")


def test_timestamp_must_be_timezone_aware():
    with pytest.raises(ContractValidationError):
        make_request(timestamp=datetime(2026, 1, 1))
