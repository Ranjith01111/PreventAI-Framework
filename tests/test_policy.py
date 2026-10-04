from datetime import datetime, timezone

import pytest

from preventai.core import (
    DecisionType,
    EvaluationContext,
    PolicyEngine,
    PolicyRule,
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
from preventai.protocol.security_request import ContractValidationError


def request(action="database.read", resource="db", agent="agent-1", tool="database", context=None):
    return SecurityRequest(
        protocol_version="1.0",
        request_id="policy-1",
        timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
        agent=AgentIdentity(agent, "autonomous"),
        session=SessionIdentity("session-1"),
        requested_by=RequestedBy.AGENT,
        action=Action(action, {}),
        resource=Resource(resource, "database"),
        tool=Tool(tool, "1.0"),
        data=DataMetadata(DataClassification.INTERNAL, SourceTrust.TRUSTED),
        context=context or {},
        metadata={},
    )


def evaluate(rules, **kwargs):
    request_kwargs = kwargs.pop("request_kwargs", {})
    return PolicyEngine(rules, **kwargs).evaluate(
        EvaluationContext.from_request(request(**request_kwargs))
    )


def rule(rule_id, decision, **kwargs):
    return PolicyRule(rule_id, decision, **kwargs)


def test_exact_allow_match():
    decision = evaluate([rule("allow-read", DecisionType.ALLOW, action="database.read")])
    assert decision.decision_type is DecisionType.ALLOW
    assert decision.rule_id == "allow-read"


def test_wildcard_match():
    decision = evaluate([rule("db-all", DecisionType.WARN, action="database.*")])
    assert decision.decision_type is DecisionType.WARN


def test_deny_rule():
    decision = evaluate([rule("deny-delete", DecisionType.BLOCK, action="database.delete")])
    assert decision.decision_type is DecisionType.BLOCK


def test_approval_rule():
    decision = evaluate(
        [rule("approve-send", DecisionType.APPROVAL_REQUIRED, action="email.send")],
        request_kwargs={"action": "email.send"},
    )
    assert decision.decision_type is DecisionType.APPROVAL_REQUIRED


def test_missing_policy_uses_secure_default():
    decision = evaluate([])
    assert decision.decision_type is DecisionType.BLOCK


def test_precedence_uses_highest_priority():
    decision = evaluate([
        rule("broad", DecisionType.BLOCK, action="database.*", priority=1),
        rule("specific", DecisionType.ALLOW, action="database.read", priority=10),
    ])
    assert decision.rule_id == "specific"


def test_conflicting_same_priority_rules_block():
    decision = evaluate([
        rule("allow", DecisionType.ALLOW, action="database.read", priority=5),
        rule("block", DecisionType.BLOCK, action="database.read", priority=5),
    ])
    assert decision.decision_type is DecisionType.BLOCK
    assert decision.metadata["conflict"] is True
    assert decision.reason.value == "conflicting_policies"


def test_multiple_matchers_and_conditions():
    decision = evaluate(
        [rule(
            "trusted-read",
            DecisionType.ALLOW,
            action="database.*",
            resource="db-*",
            agent="agent-*",
            tool="database",
            conditions={"context.environment": "prod"},
        )],
        request_kwargs={"resource": "db-main", "context": {"environment": "prod"}},
    )
    assert decision.decision_type is DecisionType.ALLOW


def test_unsupported_condition_fails_closed():
    with pytest.raises(ContractValidationError):
        evaluate([rule("bad", DecisionType.ALLOW, conditions={"unknown": "x"})])


def test_rule_ids_are_unique():
    with pytest.raises(ContractValidationError):
        PolicyEngine([rule("same", DecisionType.ALLOW), rule("same", DecisionType.BLOCK)])


def test_policy_serialization_round_trip():
    policy = PolicyEngine(
        [rule("read", DecisionType.ALLOW, action="database.read", priority=3)],
        default_decision=DecisionType.BLOCK,
    )
    assert PolicyEngine.from_json(policy.to_json()).to_json() == policy.to_json()
