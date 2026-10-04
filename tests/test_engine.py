from datetime import datetime, timezone

from preventai.core import (
    Decision,
    DecisionReason,
    DecisionType,
    SecurityEngine,
    Severity,
)
from preventai.protocol import (
    Action,
    AgentIdentity,
    DataClassification,
    DataMetadata,
    RequestedBy,
    Resource,
    SecurityRequest,
    SessionIdentity,
    SourceTrust,
    Tool,
)


def make_request() -> SecurityRequest:
    return SecurityRequest(
        protocol_version="1.0",
        request_id="req-engine-001",
        timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
        agent=AgentIdentity("agent-001", "autonomous"),
        session=SessionIdentity("session-001"),
        requested_by=RequestedBy.AGENT,
        action=Action("read", {"path": "/docs/report.txt"}),
        resource=Resource("report.txt", "file"),
        tool=Tool("filesystem", "1.0"),
        data=DataMetadata(DataClassification.INTERNAL, SourceTrust.TRUSTED),
        context={"purpose": "test"},
        metadata={"trace": "test"},
    )


def test_valid_request_uses_default_fail_closed_decision():
    decision = SecurityEngine().evaluate(make_request())
    assert decision.decision_type is DecisionType.BLOCK
    assert decision.reason is DecisionReason.NO_POLICY
    assert decision.severity is Severity.HIGH


def test_invalid_request_fails_closed():
    decision = SecurityEngine().evaluate(object())
    assert decision.decision_type is DecisionType.BLOCK
    assert decision.reason is DecisionReason.INVALID_REQUEST
    assert decision.severity is Severity.CRITICAL


def test_repeated_evaluation_is_deterministic():
    engine = SecurityEngine()
    request = make_request()
    assert engine.evaluate(request).to_json() == engine.evaluate(request).to_json()


def test_policy_evaluator_is_pluggable_without_engine_side_effects():
    class StaticPolicy:
        def evaluate(self, context):
            assert context.action_name == "read"
            return Decision(DecisionType.ALLOW, DecisionReason.POLICY_MATCH)

    decision = SecurityEngine(StaticPolicy()).evaluate(make_request())
    assert decision.decision_type is DecisionType.BLOCK
    assert decision.reason is DecisionReason.MISSING_PERMISSION


def test_policy_and_permission_allow():
    class StaticPolicy:
        def evaluate(self, context):
            return Decision(DecisionType.ALLOW, DecisionReason.POLICY_MATCH)

    from preventai.core import PermissionCategory, PermissionEngine, PermissionRule

    permissions = PermissionEngine([
        PermissionRule("read", PermissionCategory.READ, action="read")
    ])
    decision = SecurityEngine(StaticPolicy(), permissions).evaluate(make_request())
    assert decision.decision_type is DecisionType.ALLOW


def test_policy_exception_fails_closed():
    class BrokenPolicy:
        def evaluate(self, context):
            raise RuntimeError("internal detail")

    decision = SecurityEngine(BrokenPolicy()).evaluate(make_request())
    assert decision.decision_type is DecisionType.BLOCK
    assert decision.reason is DecisionReason.SECURITY_EVALUATION_ERROR
    assert "internal detail" not in decision.to_json()


def test_permission_exception_fails_closed():
    class AllowPolicy:
        def evaluate(self, context):
            return Decision(DecisionType.ALLOW, DecisionReason.POLICY_MATCH)

    class BrokenPermissions:
        def evaluate(self, context):
            raise RuntimeError("internal detail")

    decision = SecurityEngine(AllowPolicy(), BrokenPermissions()).evaluate(make_request())
    assert decision.decision_type is DecisionType.BLOCK
    assert decision.reason is DecisionReason.SECURITY_EVALUATION_ERROR


def test_repeated_evaluation_after_error_remains_blocked():
    class BrokenPolicy:
        def evaluate(self, context):
            raise ValueError("broken")

    engine = SecurityEngine(BrokenPolicy())
    first = engine.evaluate(make_request())
    second = engine.evaluate(make_request())
    assert first.to_json() == second.to_json()


def test_policy_rule_matching_error_fails_closed():
    from preventai.core import PolicyEngine, PolicyRule

    policy = PolicyEngine([
        PolicyRule("bad-condition", DecisionType.ALLOW, conditions={"unsupported": "x"})
    ])
    decision = SecurityEngine(policy).evaluate(make_request())
    assert decision.decision_type is DecisionType.BLOCK
    assert decision.reason is DecisionReason.SECURITY_EVALUATION_ERROR
