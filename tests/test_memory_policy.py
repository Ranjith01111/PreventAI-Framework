import pytest

from preventai import DecisionReason, DecisionType
from preventai.core.dataflow import FlowClassification, FlowTrust
from preventai.memory import MemoryAccessContext, MemoryOperation, MemoryPolicy, MemoryPolicyRule


def ctx(agent="agent-a", session="session-a", namespace="customer"):
    return MemoryAccessContext(agent, agent, session, namespace)


@pytest.mark.parametrize("operation", list(MemoryOperation))
def test_operation_specific_allow_rules(operation):
    policy = MemoryPolicy([
        MemoryPolicyRule(operation.value, operation, DecisionType.ALLOW)
    ])
    decision = policy.evaluate(operation, ctx(), FlowClassification.PRIVATE, FlowTrust.TRUSTED)
    assert decision.decision_type is DecisionType.ALLOW
    assert decision.reason is DecisionReason.POLICY_MATCH


def test_deny_rule_and_missing_policy_fail_closed():
    deny = MemoryPolicy([
        MemoryPolicyRule("deny-read", MemoryOperation.READ, DecisionType.BLOCK)
    ])
    decision = deny.evaluate(MemoryOperation.READ, ctx(), FlowClassification.PRIVATE, FlowTrust.TRUSTED)
    assert decision.decision_type is DecisionType.BLOCK
    assert MemoryPolicy().evaluate(
        MemoryOperation.READ, ctx(), FlowClassification.PRIVATE, FlowTrust.TRUSTED
    ).reason is DecisionReason.NO_POLICY


def test_identity_session_namespace_classification_and_trust_matching():
    rule = MemoryPolicyRule(
        "specific", MemoryOperation.READ, DecisionType.ALLOW,
        identity="agent-a", agent="agent-a", session="session-a",
        namespace="customer", classification=FlowClassification.SECRET,
        trust=FlowTrust.UNTRUSTED,
    )
    policy = MemoryPolicy([rule])
    assert policy.evaluate(
        MemoryOperation.READ, ctx(), FlowClassification.SECRET, FlowTrust.UNTRUSTED
    ).decision_type is DecisionType.ALLOW
    assert policy.evaluate(
        MemoryOperation.READ, ctx(), FlowClassification.PRIVATE, FlowTrust.UNTRUSTED
    ).decision_type is DecisionType.BLOCK
    assert policy.evaluate(
        MemoryOperation.READ, ctx("agent-b"), FlowClassification.SECRET, FlowTrust.UNTRUSTED
    ).decision_type is DecisionType.BLOCK


def test_priority_conflict_is_blocked():
    policy = MemoryPolicy([
        MemoryPolicyRule("allow", MemoryOperation.READ, DecisionType.ALLOW, priority=5),
        MemoryPolicyRule("deny", MemoryOperation.READ, DecisionType.BLOCK, priority=5),
    ])
    decision = policy.evaluate(MemoryOperation.READ, ctx(), FlowClassification.PRIVATE, FlowTrust.TRUSTED)
    assert decision.decision_type is DecisionType.BLOCK
    assert decision.reason is DecisionReason.CONFLICTING_POLICIES


def test_higher_priority_rule_wins_deterministically():
    policy = MemoryPolicy([
        MemoryPolicyRule("default", MemoryOperation.READ, DecisionType.BLOCK, priority=1),
        MemoryPolicyRule("specific", MemoryOperation.READ, DecisionType.ALLOW, namespace="customer", priority=2),
    ])
    assert policy.evaluate(
        MemoryOperation.READ, ctx(), FlowClassification.PRIVATE, FlowTrust.TRUSTED
    ).rule_id == "specific"


def test_invalid_policy_configuration_is_rejected():
    with pytest.raises(ValueError):
        MemoryPolicy([MemoryPolicyRule("same", MemoryOperation.READ, DecisionType.ALLOW),
                      MemoryPolicyRule("same", MemoryOperation.WRITE, DecisionType.ALLOW)])
    with pytest.raises(ValueError):
        MemoryPolicyRule("bad", "read", DecisionType.ALLOW)


def test_policy_evaluation_exception_fails_closed():
    class BrokenPolicy(MemoryPolicy):
        def evaluate(self, *args, **kwargs):
            raise RuntimeError("synthetic evaluator failure")

    from preventai.memory import MemoryStore, MemoryAccessDenied
    memory = MemoryStore(policy=BrokenPolicy([
        MemoryPolicyRule("write", MemoryOperation.WRITE, DecisionType.ALLOW)
    ]))
    with pytest.raises(MemoryAccessDenied) as error:
        memory.write(ctx(), {"value": 1})
    assert error.value.decision.decision_type is DecisionType.BLOCK
