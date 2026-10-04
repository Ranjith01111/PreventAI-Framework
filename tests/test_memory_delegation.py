from datetime import datetime, timedelta, timezone
from dataclasses import replace

import pytest

from preventai import (
    Capability,
    DecisionType,
    MemoryAccessContext,
    MemoryAccessDenied,
    MemoryOperation,
    MemoryPolicy,
    MemoryPolicyRule,
    MemoryStore,
)
from preventai.core.permissions import PermissionCategory, PermissionEngine, PermissionRule
from preventai.identity import DelegationRegistry


def policy(*operations):
    return MemoryPolicy([
        MemoryPolicyRule(operation.value, operation, DecisionType.ALLOW)
        for operation in (operations or tuple(MemoryOperation))
    ])


def context(agent="agent-b", session="session-b", namespace="customer", **kwargs):
    return MemoryAccessContext(agent, agent, session, namespace, **kwargs)


def registry_for(*capabilities):
    permissions = PermissionEngine([
        PermissionRule(
            "issuer", PermissionCategory.READ,
            action="memory.*", agent="agent-a", tool="memory", resource="*",
        )
    ])
    registry = DelegationRegistry("memory-delegation-secret")
    grant = registry.create(
        "agent-a", "agent-b", capabilities,
        expires_at=datetime.now(timezone.utc) + timedelta(minutes=5),
        issuer_permissions=permissions,
    )
    return registry, grant


def delegated_context(grant, registry, namespace="customer", session="session-b"):
    return context(namespace=namespace, session=session, delegation=grant, delegation_registry=registry)


def setup_memory(namespace="customer"):
    memory = MemoryStore(policy=policy())
    owner = MemoryAccessContext("agent-a", "agent-a", "session-a", namespace)
    entry = memory.write(owner, {"value": "synthetic"})
    return memory, owner, entry


def grant_for(*, action="memory.read", resource="customer", session_id=None):
    return registry_for(Capability(action, resource=resource, tool="memory")) + (session_id,)


def test_valid_delegated_read_requires_explicit_sharing():
    memory, owner, entry = setup_memory()
    registry, grant = registry_for(Capability("memory.read", resource="customer", tool="memory"))
    memory.share(owner, entry.memory_id, "agent-b", grantee_session_id="session-b")
    assert memory.read(delegated_context(grant, registry), entry.memory_id).payload == {"value": "synthetic"}


@pytest.mark.parametrize("operation", [MemoryOperation.UPDATE, MemoryOperation.DELETE])
def test_read_only_delegation_cannot_update_or_delete(operation):
    memory, owner, entry = setup_memory()
    registry, grant = registry_for(Capability("memory.read", resource="customer", tool="memory"))
    memory.share(owner, entry.memory_id, "agent-b", grantee_session_id="session-b")
    with pytest.raises(MemoryAccessDenied) as error:
        if operation is MemoryOperation.UPDATE:
            memory.update(delegated_context(grant, registry), entry.memory_id, {"value": "changed"}, expected_version=1)
        else:
            memory.delete(delegated_context(grant, registry), entry.memory_id)
    assert error.value.decision.decision_type is DecisionType.BLOCK


def test_delegated_scope_and_namespace_widening_are_blocked():
    memory, owner, entry = setup_memory("customer.region_1")
    registry, grant = registry_for(Capability("memory.read", resource="customer.region_1", tool="memory"))
    memory.share(owner, entry.memory_id, "agent-b", grantee_session_id="session-b")
    with pytest.raises(MemoryAccessDenied):
        memory.read(delegated_context(grant, registry, namespace="customer.region_2"), entry.memory_id)

    with pytest.raises(MemoryAccessDenied):
        memory.read(delegated_context(grant, registry, namespace="customer.*"), entry.memory_id)


def test_child_delegation_cannot_widen_parent_scope():
    permissions = PermissionEngine([
        PermissionRule("issuer", PermissionCategory.READ, action="memory.*", agent="agent-a", tool="memory", resource="*"),
    ])
    registry = DelegationRegistry("chain-secret")
    parent = registry.create(
        "agent-a", "agent-b", [Capability("memory.read", resource="customer.region_1", tool="memory")],
        expires_at=datetime.now(timezone.utc) + timedelta(minutes=5),
        issuer_permissions=permissions,
    )
    with pytest.raises(ValueError):
        registry.create(
            "agent-b", "agent-c", [Capability("memory.read", resource="customer.*", tool="memory")],
            expires_at=datetime.now(timezone.utc) + timedelta(minutes=5),
            parent=parent,
            issuer_permissions=permissions,
        )


def test_session_bound_delegation_rejects_other_session():
    memory, owner, entry = setup_memory()
    registry, grant = registry_for(Capability("memory.read", resource="customer", tool="memory"))
    grant = replace(grant, session_id="session-b")
    # A modified grant is rejected by the signature check.
    memory.share(owner, entry.memory_id, "agent-b", grantee_session_id="session-c")
    with pytest.raises(MemoryAccessDenied):
        memory.read(delegated_context(grant, registry, session="session-c"), entry.memory_id)


def test_expired_revoked_and_tampered_delegations_fail_closed():
    memory, owner, entry = setup_memory()
    registry, grant = registry_for(Capability("memory.read", resource="customer", tool="memory"))
    memory.share(owner, entry.memory_id, "agent-b", grantee_session_id="session-b")

    expired = replace(
        grant,
        issued_at=datetime.now(timezone.utc) - timedelta(minutes=2),
        expires_at=datetime.now(timezone.utc) - timedelta(seconds=1),
    )
    object.__setattr__(expired, "signature", registry._sign(expired.unsigned_dict()))
    with pytest.raises(MemoryAccessDenied):
        memory.read(delegated_context(expired, registry), entry.memory_id)

    registry.revoke(grant.grant_id)
    with pytest.raises(MemoryAccessDenied):
        memory.read(delegated_context(grant, registry), entry.memory_id)


def test_missing_registry_and_subject_mismatch_fail_closed():
    memory, owner, entry = setup_memory()
    registry, grant = registry_for(Capability("memory.read", resource="customer", tool="memory"))
    memory.share(owner, entry.memory_id, "agent-b", grantee_session_id="session-b")
    with pytest.raises(MemoryAccessDenied):
        memory.read(context(delegation=grant), entry.memory_id)

    mismatched = replace(grant, subject_identity="agent-c")
    with pytest.raises(MemoryAccessDenied):
        memory.read(delegated_context(mismatched, registry), entry.memory_id)


def test_policy_and_permission_still_intersect_with_delegation():
    memory, owner, entry = setup_memory()
    registry, grant = registry_for(Capability("memory.read", resource="customer", tool="memory"))
    memory.share(owner, entry.memory_id, "agent-b", grantee_session_id="session-b")
    denied_policy = memory
    denied_policy.policy = MemoryPolicy()
    with pytest.raises(MemoryAccessDenied):
        denied_policy.read(delegated_context(grant, registry), entry.memory_id)

    permission_denied = memory
    permission_denied.policy = policy()
    permission_denied.permission_checker = lambda *args: __import__("preventai").Decision(
            DecisionType.BLOCK, __import__("preventai").DecisionReason.MISSING_PERMISSION,
        )
    with pytest.raises(MemoryAccessDenied):
        permission_denied.read(delegated_context(grant, registry), entry.memory_id)


def test_delegation_metadata_is_safe_in_audit():
    from preventai import AuditLogger
    logger = AuditLogger(secret="audit-secret")
    memory, owner, entry = setup_memory()
    memory.audit_logger = logger
    registry, grant = registry_for(Capability("memory.read", resource="customer", tool="memory"))
    memory.share(owner, entry.memory_id, "agent-b", grantee_session_id="session-b")
    memory.read(delegated_context(grant, registry), entry.memory_id)
    serialized = logger.events()[-1].to_json()
    assert grant.grant_id in serialized
    assert "synthetic" not in serialized
    assert "audit-secret" not in serialized
