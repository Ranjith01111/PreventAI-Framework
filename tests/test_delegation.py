from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor

import pytest

from preventai import (
    Agent,
    Authenticator,
    Capability,
    Credential,
    DecisionReason,
    DecisionType,
    DelegationGrant,
    DelegationRegistry,
    Mission,
    PreventApp,
    ToolExecutionDenied,
)
from preventai.core import (
    PermissionCategory,
    PermissionEngine,
    PermissionRule,
    PolicyEngine,
    PolicyRule,
)
from preventai.identity import AgentIdentity


def registry():
    auth = Authenticator()
    for identity_id in ("agent-a", "agent-b"):
        credential = Credential(identity_id, f"secret-{identity_id}")
        auth.register(AgentIdentity(identity_id), credential)
    permissions = PermissionEngine(
        [
            PermissionRule(
                "parent-read",
                PermissionCategory.READ,
                action="customer.*",
                agent="agent-a",
                resource="customer_db.*",
            )
        ]
    )
    return DelegationRegistry("delegation-test-secret"), auth, permissions


def test_capability_attenuation_and_signed_serialization():
    delegator, _, permissions = registry()
    grant = delegator.create(
        "agent-a",
        "agent-b",
        [Capability("customer.read", "customer_db.region_1")],
        expires_at=datetime.now(timezone.utc) + timedelta(minutes=5),
        issuer_permissions=permissions,
    )
    restored = DelegationGrant.from_dict(grant.to_dict())
    assert delegator.verify(restored)
    assert delegator.evaluate(
        restored,
        authenticated_identity="agent-b",
        action="customer.read",
        resource="customer_db.region_1",
        tool="customer.read",
    ).decision_type.value == "allow"
    denied = delegator.evaluate(
        restored,
        authenticated_identity="agent-b",
        action="customer.update",
        resource="customer_db.region_1",
        tool="customer.update",
    )
    assert denied.reason is DecisionReason.DELEGATION_SCOPE_EXCEEDED
    assert "secret" not in grant.to_json().lower()


def test_tamper_expiry_revocation_and_subject_binding_block():
    delegator, _, permissions = registry()
    grant = delegator.create(
        "agent-a",
        "agent-b",
        [Capability("customer.read", "customer_db.region_1")],
        expires_at=datetime.now(timezone.utc) + timedelta(minutes=5),
        issuer_permissions=permissions,
    )
    tampered = DelegationGrant(
        grant.grant_id,
        grant.issuer_identity,
        grant.subject_identity,
        grant.capabilities,
        grant.issued_at,
        grant.expires_at,
        grant.depth,
        grant.parent_grant_id,
        grant.session_id,
        grant.enabled,
        "tampered",
    )
    assert delegator.evaluate(
        tampered,
        authenticated_identity="agent-b",
        action="customer.read",
        resource="customer_db.region_1",
        tool="customer.read",
    ).reason is DecisionReason.DELEGATION_SIGNATURE_INVALID
    assert delegator.evaluate(
        grant,
        authenticated_identity="agent-c",
        action="customer.read",
        resource="customer_db.region_1",
        tool="customer.read",
    ).reason is DecisionReason.DELEGATION_SUBJECT_MISMATCH
    delegator.revoke(grant.grant_id)
    assert delegator.evaluate(
        grant,
        authenticated_identity="agent-b",
        action="customer.read",
        resource="customer_db.region_1",
        tool="customer.read",
    ).reason is DecisionReason.DELEGATION_REVOKED


def test_child_grant_cannot_exceed_parent_or_depth():
    delegator, _, permissions = registry()
    parent = delegator.create(
        "agent-a",
        "agent-b",
        [Capability("customer.read", "customer_db.*")],
        expires_at=datetime.now(timezone.utc) + timedelta(minutes=5),
        issuer_permissions=permissions,
    )
    child = delegator.create(
        "agent-b",
        "agent-c",
        [Capability("customer.read", "customer_db.region_1")],
        expires_at=datetime.now(timezone.utc) + timedelta(minutes=5),
        parent=parent,
        issuer_permissions=PermissionEngine(
            [PermissionRule("child", PermissionCategory.READ, action="customer.read", resource="customer_db.region_1")]
        ),
    )
    assert child.depth == 2
    with pytest.raises(ValueError):
        delegator.create(
            "agent-b",
            "agent-c",
            [Capability("customer.update")],
            expires_at=datetime.now(timezone.utc) + timedelta(minutes=5),
            parent=parent,
            issuer_permissions=permissions,
        )


def test_framework_delegation_blocks_before_callable():
    registry_obj, auth, _ = registry()
    agent_a = Agent("A", agent_id="agent-a", credential=Credential("agent-a", "secret-agent-a"))
    agent_b = Agent("B", agent_id="agent-b", credential=Credential("agent-b", "secret-agent-b"))
    @agent_a.tool(name="issuer.noop")
    def noop():
        return "ok"
    calls = []

    @agent_b.tool(name="customer.read")
    def read():
        calls.append(True)
        return "ok"

    permissions = PermissionEngine(
        [
            PermissionRule(
                "a-read",
                PermissionCategory.READ,
                action="customer.read",
                agent="agent-a",
                resource="*",
            ),
            PermissionRule(
                "b-read",
                PermissionCategory.READ,
                action="customer.read",
                agent="agent-b",
                resource="*",
            )
        ]
    )
    app = PreventApp(
        authenticator=auth,
        permissions=permissions,
        policy=PolicyEngine(
            [
                PolicyRule(
                    "b-read-policy",
                    DecisionType.ALLOW,
                    action="customer.read",
                    agent="agent-b",
                    resource=read.tool_id,
                )
            ]
        ),
    )
    app.add_agent(agent_a)
    app.add_agent(agent_b)
    app.run()
    mission = Mission(
        "delegation-tool-mission",
        agent_b.agent_id,
        action="customer.read",
        resource=read.tool_id,
        tool="customer.read",
        expires_at=datetime.now(timezone.utc) + timedelta(minutes=5),
    )
    app.missions.register(mission)
    app.bind_mission(agent_b, mission.mission_id, mission.version)
    grant = agent_a.delegate(
        agent_b,
        [Capability("customer.read", resource=read.tool_id, tool="customer.read")],
        expires_at=datetime.now(timezone.utc) + timedelta(minutes=5),
    )
    agent_b.use_delegation(grant)
    assert read() == "ok"
    assert calls == [True]


def test_delegation_attenuates_every_mission_scope_dimension():
    delegation, _, issuer_permissions = registry()
    expires = datetime.now(timezone.utc) + timedelta(minutes=5)
    parent = delegation.create(
        "agent-a",
        "agent-b",
        [Capability(
            "customer.read",
            "customer_db.users",
            "lookup",
            PermissionCategory.READ,
            ("approved-service",),
            ("internal",),
            {"tenant": "tenant-a"},
        )],
        expires_at=expires,
        issuer_permissions=issuer_permissions,
    )
    child = delegation.create(
        "agent-b",
        "agent-c",
        [Capability(
            "customer.read",
            "customer_db.users",
            "lookup",
            PermissionCategory.READ,
            ("approved-service",),
            ("internal",),
            {"tenant": "tenant-a", "region": "region-a"},
        )],
        expires_at=expires,
        parent=parent,
        issuer_permissions=issuer_permissions,
    )
    assert delegation.evaluate(
        child,
        authenticated_identity="agent-c",
        action="customer.read",
        resource="customer_db.users",
        tool="lookup",
        category=PermissionCategory.READ,
        destination="approved-service",
        data_class="internal",
        context={"tenant": "tenant-a", "region": "region-a"},
    ).decision_type is DecisionType.ALLOW

    wider_grants = (
        Capability(
            "customer.read", "customer_db.users", "lookup", PermissionCategory.READ,
            ("approved-service", "unknown-service"), ("internal",), {"tenant": "tenant-a"},
        ),
        Capability(
            "customer.read", "customer_db.users", "lookup", PermissionCategory.READ,
            ("approved-service",), ("internal", "private"), {"tenant": "tenant-a"},
        ),
        Capability(
            "customer.read", "customer_db.users", "lookup", PermissionCategory.READ,
            ("approved-service",), ("internal",), {},
        ),
        Capability(
            "customer.read", "customer_db.users", "lookup", PermissionCategory.READ,
            ("approved-service",), ("internal",), {"tenant": "tenant-b"},
        ),
        Capability(
            "customer.read", "*", "lookup", PermissionCategory.READ,
            ("approved-service",), ("internal",), {"tenant": "tenant-a"},
        ),
    )
    for capability in wider_grants:
        with pytest.raises(ValueError, match="exceeds parent scope"):
            delegation.create(
                "agent-b",
                "agent-c",
                [capability],
                expires_at=expires,
                parent=parent,
                issuer_permissions=issuer_permissions,
            )

    assert delegation.evaluate(
        child,
        authenticated_identity="agent-c",
        action="customer.read",
        resource="customer_db.users",
        tool="lookup",
        category=PermissionCategory.READ,
        destination="unknown-service",
        data_class="internal",
        context={"tenant": "tenant-a"},
    ).decision_type is DecisionType.BLOCK


def test_simultaneous_delegation_checks_and_revocation_are_serialized():
    delegation, _, issuer_permissions = registry()
    grant = delegation.create(
        "agent-a",
        "agent-b",
        [Capability("customer.read", "customer_db.users", "lookup")],
        expires_at=datetime.now(timezone.utc) + timedelta(minutes=5),
        issuer_permissions=issuer_permissions,
    )

    def evaluate(_):
        return delegation.evaluate(
            grant,
            authenticated_identity="agent-b",
            action="customer.read",
            resource="customer_db.users",
            tool="lookup",
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(evaluate, range(32)))
    assert all(result.decision_type is DecisionType.ALLOW for result in results)
    delegation.revoke(grant.grant_id)
    assert evaluate(0).decision_type is DecisionType.BLOCK
