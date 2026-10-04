import pytest

from preventai.core.decision import (
    Decision,
    DecisionReason,
    DecisionType,
    Severity,
)
from preventai.protocol.security_request import ContractValidationError


@pytest.mark.parametrize(
    ("decision_type", "reason"),
    [
        (DecisionType.ALLOW, DecisionReason.REQUEST_ALLOWED),
        (DecisionType.BLOCK, DecisionReason.REQUEST_BLOCKED),
        (DecisionType.WARN, DecisionReason.SECURITY_WARNING),
        (DecisionType.APPROVAL_REQUIRED, DecisionReason.APPROVAL_REQUIRED),
    ],
)
def test_all_decision_types_have_deterministic_semantics(decision_type, reason):
    decision = Decision(
        decision_type=decision_type,
        reason=reason,
        rule_id="rule-001",
        severity=Severity.MEDIUM,
        metadata={"source": "test"},
    )
    assert decision.decision_type.value in {
        "allow",
        "block",
        "warn",
        "approval_required",
    }
    assert decision.reason is reason


def test_decision_serialization_and_deserialization():
    decision = Decision(
        DecisionType.WARN,
        DecisionReason.SECURITY_WARNING,
        rule_id="rule-001",
        severity=Severity.HIGH,
        metadata={"z": 1, "a": ["safe"]},
    )
    encoded = decision.to_json()
    assert encoded == decision.to_json()
    assert Decision.from_json(encoded) == decision


def test_decision_is_immutable():
    decision = Decision(DecisionType.ALLOW, DecisionReason.REQUEST_ALLOWED)
    with pytest.raises(AttributeError):
        decision.decision_type = DecisionType.BLOCK
    with pytest.raises(TypeError):
        decision.metadata["new"] = "value"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"decision_type": "allow", "reason": DecisionReason.REQUEST_ALLOWED},
        {"decision_type": DecisionType.ALLOW, "reason": "request_allowed"},
        {"decision_type": DecisionType.ALLOW, "reason": DecisionReason.REQUEST_ALLOWED, "severity": "high"},
        {"decision_type": DecisionType.ALLOW, "reason": DecisionReason.REQUEST_ALLOWED, "metadata": {"bad": object()}},
    ],
)
def test_invalid_decision_data_fails_closed(kwargs):
    with pytest.raises(ContractValidationError):
        Decision(**kwargs)


def test_decision_deserialization_is_strict():
    payload = Decision(DecisionType.BLOCK, DecisionReason.REQUEST_BLOCKED).to_dict()
    payload["unexpected"] = True
    with pytest.raises(ContractValidationError):
        Decision.from_dict(payload)
