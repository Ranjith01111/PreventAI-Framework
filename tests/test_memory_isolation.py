from datetime import datetime, timedelta, timezone

import pytest

from preventai import Decision, DecisionReason, DecisionType
from preventai.memory import (
    MemoryAccessContext,
    MemoryAccessDenied,
    MemoryIsolation,
    MemoryOperation,
    MemoryPolicy,
    MemoryPolicyRule,
    MemoryStore,
)


def context(agent="agent-a", session="session-a", namespace="customer"):
    return MemoryAccessContext(agent, agent, session, namespace)


def memory_store(**kwargs):
    rules = [
        MemoryPolicyRule(operation.value, operation, DecisionType.ALLOW)
        for operation in MemoryOperation
    ]
    return MemoryStore(policy=MemoryPolicy(rules), **kwargs)


def test_agent_session_and_namespace_isolation():
    memory = memory_store()
    owner = context()
    entry = memory.write(owner, {"value": "private"})
    cases = [
        context("agent-b", "session-a", "customer"),
        context("agent-a", "session-b", "customer"),
        context("agent-a", "session-a", "other"),
    ]
    for foreign in cases:
        with pytest.raises(MemoryAccessDenied):
            memory.read(foreign, entry.memory_id)
        with pytest.raises(MemoryAccessDenied):
            memory.update(foreign, entry.memory_id, {"value": "tampered"}, expected_version=1)
        with pytest.raises(MemoryAccessDenied):
            memory.delete(foreign, entry.memory_id)


def test_cross_agent_same_namespace_requires_sharing():
    memory = memory_store()
    owner = context()
    recipient = context("agent-b", "session-b", "customer")
    entry = memory.write(owner, {"value": 1})
    with pytest.raises(MemoryAccessDenied):
        memory.read(recipient, entry.memory_id)
    share = memory.share(owner, entry.memory_id, recipient.agent_id, grantee_session_id=recipient.session_id)
    assert memory.read(recipient, entry.memory_id).payload == {"value": 1}
    memory.revoke_share(share.share_id)
    with pytest.raises(MemoryAccessDenied):
        memory.read(recipient, entry.memory_id)


def test_share_is_session_scoped_when_configured():
    memory = memory_store()
    owner = context()
    entry = memory.write(owner, {"value": 1})
    memory.share(owner, entry.memory_id, "agent-b", grantee_session_id="session-b")
    with pytest.raises(MemoryAccessDenied):
        memory.read(context("agent-b", "session-other", "customer"), entry.memory_id)


def test_expired_share_is_rejected():
    memory = memory_store()
    owner = context()
    recipient = context("agent-b", "session-b", "customer")
    entry = memory.write(owner, {"value": 1})
    memory.share(
        owner, entry.memory_id, recipient.agent_id,
        grantee_session_id=recipient.session_id,
        expires_at=datetime.now(timezone.utc) - timedelta(seconds=1),
    )
    with pytest.raises(MemoryAccessDenied):
        memory.read(recipient, entry.memory_id)


def test_authentication_callback_is_authoritative():
    calls = []
    memory = memory_store(authentication_checker=lambda value: calls.append(value) or False)
    with pytest.raises(MemoryAccessDenied) as error:
        memory.write(context(), {"value": 1})
    assert error.value.decision.decision_type is DecisionType.BLOCK
    assert calls


def test_mismatched_authenticated_identity_cannot_be_constructed():
    with pytest.raises(ValueError):
        MemoryAccessContext("identity-a", "agent-b", "session", "namespace")


def test_share_requires_share_policy_and_does_not_bypass_permission():
    owner = context()
    recipient = context("agent-b", "session-b", "customer")
    no_share_policy = MemoryPolicy([
        MemoryPolicyRule("write", MemoryOperation.WRITE, DecisionType.ALLOW),
        MemoryPolicyRule("read", MemoryOperation.READ, DecisionType.ALLOW),
    ])
    memory = MemoryStore(policy=no_share_policy)
    entry = memory.write(owner, {"value": 1})
    with pytest.raises(MemoryAccessDenied):
        memory.share(owner, entry.memory_id, recipient.agent_id, grantee_session_id=recipient.session_id)

    def deny_permission(operation, context, entry):
        if operation is MemoryOperation.WRITE:
            return Decision(DecisionType.ALLOW, DecisionReason.REQUEST_ALLOWED)
        return MemoryPolicyRule  # malformed callback result must fail closed

    allowed = memory_store(permission_checker=deny_permission)
    entry = allowed.write(owner, {"value": 1})
    with pytest.raises(MemoryAccessDenied):
        allowed.read(owner, entry.memory_id)


def test_concurrent_share_and_revoke_is_consistent():
    memory = memory_store()
    owner = context()
    recipient = context("agent-b", "session-b", "customer")
    entry = memory.write(owner, {"value": 1})
    share = memory.share(owner, entry.memory_id, recipient.agent_id, grantee_session_id=recipient.session_id)
    memory.revoke_share(share.share_id)
    with pytest.raises(MemoryAccessDenied):
        memory.read(recipient, entry.memory_id)


def test_isolation_direct_check_blocks_mismatches():
    isolation = MemoryIsolation()
    decision = isolation.check(context("agent-b", "session-b", "customer"), "agent-a", "session-a", "customer")
    assert decision.decision_type is DecisionType.BLOCK
