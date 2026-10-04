from datetime import datetime, timedelta, timezone
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from preventai import (
    Agent,
    AuditEventType,
    AuditLogger,
    DecisionType,
    Mission,
    MissionError,
    MissionRegistry,
    MissionState,
    PermissionCategory,
    PermissionEngine,
    PermissionRule,
    PolicyEngine,
    PolicyRule,
    PreventApp,
    ToolExecutionDenied,
)
from preventai.core.dataflow import FlowClassification


def make_app(*, require_missions=True):
    permissions = PermissionEngine([
        PermissionRule("read", PermissionCategory.READ, action="*", resource="*"),
        PermissionRule("execute", PermissionCategory.EXECUTE, action="*", resource="*"),
    ])
    app = PreventApp(
        policy=PolicyEngine([PolicyRule("allow", DecisionType.ALLOW, action="*")]),
        permissions=permissions,
        require_missions=require_missions,
    )
    agent = app.add_agent(Agent("mission-agent", agent_id="mission-agent"))
    return app, agent


def test_missing_mission_blocks_when_required():
    app, agent = make_app(require_missions=True)

    @agent.tool("read-data")
    def read():
        return "safe"

    app.run()
    with pytest.raises(ToolExecutionDenied):
        read()


def test_revoked_mission_blocks_reuse():
    app, agent = make_app(require_missions=True)

    @agent.tool("read-data")
    def read():
        return "safe"

    app.run()
    mission = Mission(
        mission_id="mission-revoked",
        agent_id=agent.agent_id,
        action="read-data",
        resource=read.tool_id,
        tool="read-data",
        expires_at=datetime.now(timezone.utc) + timedelta(minutes=5),
    )
    app.missions.register(mission)
    app.bind_mission(agent, mission.mission_id, mission.version)
    assert read() == "safe"
    app.missions.revoke(mission.mission_id)
    with pytest.raises(ToolExecutionDenied):
        read()


def test_mission_scope_denies_unmatched_action():
    app, agent = make_app(require_missions=True)

    @agent.tool("read-data")
    def read():
        return "safe"

    app.run()
    app.missions.register(
        Mission(
            mission_id="mission-write",
            agent_id=agent.agent_id,
            action="write-data",
            resource=read.tool_id,
            tool="write-data",
            expires_at=datetime.now(timezone.utc) + timedelta(minutes=5),
        )
    )
    app.bind_mission(agent, "mission-write", 1)
    with pytest.raises(ToolExecutionDenied):
        read()


def test_mission_contract_is_deeply_immutable_and_version_bound():
    context = {"tenant": "tenant-a", "attributes": {"region": "approved-region"}}
    mission = Mission(
        "mission-immutable", "mission-agent", action="customer.read",
        resource="customer_db", tool="customer_lookup",
        allowed_data_classes=("public", "internal"),
        context_constraints=context,
        expires_at=datetime.now(timezone.utc) + timedelta(minutes=5),
    )
    context["tenant"] = "tenant-b"
    context["attributes"]["region"] = "unknown-region"
    with pytest.raises((AttributeError, TypeError)):
        mission.allowed_actions += ("customer.write",)
    with pytest.raises(TypeError):
        mission.context_constraints["tenant"] = "tenant-b"
    with pytest.raises(TypeError):
        mission.context_constraints["attributes"]["region"] = "unknown-region"

    registry = MissionRegistry()
    registry.register(mission)
    registry.bind("mission-agent", "mission-immutable", 1)
    assert registry.check(
        "mission-agent", action="customer.read", resource="customer_db",
        tool="customer_lookup", data="internal",
        context={"tenant": "tenant-a", "attributes": {"region": "approved-region"}},
    ).decision_type is DecisionType.ALLOW
    registry.bind(
        "mission-agent",
        "mission-immutable",
        1,
        context={"tenant": "tenant-a", "environment": "production"},
    )
    assert registry.check(
        "mission-agent", action="customer.read", resource="customer_db",
        tool="customer_lookup", data="internal",
        context={"tenant": "tenant-b"},
    ).decision_type is DecisionType.BLOCK
    assert registry.check(
        "mission-agent", action="customer.write", resource="customer_db",
        tool="customer_lookup", data="internal",
    ).decision_type is DecisionType.BLOCK


@pytest.mark.parametrize(
    ("allowed", "requested", "expected"),
    [
        ("customer.read", "customer.read", DecisionType.ALLOW),
        ("customer.read", "customer.write", DecisionType.BLOCK),
        ("customer.read", "customer.delete", DecisionType.BLOCK),
        ("customer.read", "shell.execute", DecisionType.BLOCK),
    ],
)
def test_action_authorization_is_exact(allowed, requested, expected):
    registry = MissionRegistry()
    registry.register(Mission(
        "mission-actions", "agent-a", action=allowed, resource="customer_db",
        tool="customer_lookup", expires_at=datetime.now(timezone.utc) + timedelta(minutes=5),
    ))
    registry.bind("agent-a", "mission-actions", 1)
    assert registry.check(
        "agent-a", action=requested, resource="customer_db", tool="customer_lookup",
    ).decision_type is expected


def test_data_class_scope_enforces_explicit_wildcard_rules_and_enum_values():
    registry = MissionRegistry()
    registry.register(Mission(
        "mission-data", "agent-a", action="customer.read", resource="customer_db",
        tool="customer_lookup", allowed_data_classes=("public", "internal"),
        expires_at=datetime.now(timezone.utc) + timedelta(minutes=5),
    ))
    registry.bind("agent-a", "mission-data", 1)
    assert registry.check(
        "agent-a", action="customer.read", resource="customer_db", tool="customer_lookup",
        data={"classification": FlowClassification.PUBLIC},
    ).decision_type is DecisionType.ALLOW
    assert registry.check(
        "agent-a", action="customer.read", resource="customer_db", tool="customer_lookup",
        data={"classification": FlowClassification.PRIVATE},
    ).decision_type is DecisionType.BLOCK
    with pytest.raises(MissionError):
        Mission(
            "mission-data-bad", "agent-a", action="customer.read", resource="customer_db",
            tool="customer_lookup", allowed_data_classes=("public", "*"),
            expires_at=datetime.now(timezone.utc) + timedelta(minutes=5),
        )


def test_only_an_explicit_full_wildcard_grants_every_action():
    registry = MissionRegistry()
    with pytest.raises(MissionError, match="full-scope wildcard"):
        Mission(
            "mission-wildcard", "agent-a", action="customer.*", resource="customer_db",
            tool="customer_lookup", expires_at=datetime.now(timezone.utc) + timedelta(minutes=5),
        )

    registry.register(Mission(
        "mission-explicit-wildcard", "agent-b", action="*", resource="customer_db",
        tool="customer_lookup", expires_at=datetime.now(timezone.utc) + timedelta(minutes=5),
    ))
    registry.bind("agent-b", "mission-explicit-wildcard", 1)
    assert registry.check(
        "agent-b", action="customer.read", resource="customer_db", tool="customer_lookup",
    ).decision_type is DecisionType.ALLOW


@pytest.mark.parametrize("resource", ["customer_db_backup", "customer_db_extra", "../admin", "customer_db/../admin_db"])
def test_resource_scope_rejects_prefix_and_traversal_confusion(resource):
    registry = MissionRegistry()
    registry.register(Mission(
        "mission-resource", "agent-a", action="customer.read", resource="customer_db",
        tool="customer_lookup", expires_at=datetime.now(timezone.utc) + timedelta(minutes=5),
    ))
    registry.bind("agent-a", "mission-resource", 1)
    assert registry.check(
        "agent-a", action="customer.read", resource=resource, tool="customer_lookup",
    ).decision_type is DecisionType.BLOCK


def test_mission_version_update_is_explicit_and_old_version_remains_narrow():
    registry = MissionRegistry()
    now = datetime.now(timezone.utc)
    registry.register(Mission(
        "mission-versioned", "agent-a", action="customer.read", resource="customer_db",
        tool="customer_lookup", version=1, issued_at=now,
        expires_at=now + timedelta(minutes=5),
    ))
    registry.register(Mission(
        "mission-versioned", "agent-a", action="customer.write", resource="customer_db",
        tool="customer_lookup", version=2, issued_at=now,
        expires_at=now + timedelta(minutes=5),
    ))
    registry.bind("agent-a", "mission-versioned", 1)
    assert registry.check(
        "agent-a", action="customer.write", resource="customer_db", tool="customer_lookup",
    ).decision_type is DecisionType.BLOCK
    registry.bind("agent-a", "mission-versioned", 2)
    assert registry.check(
        "agent-a", action="customer.write", resource="customer_db", tool="customer_lookup",
    ).decision_type is DecisionType.ALLOW
    with pytest.raises(MissionError):
        registry.register(Mission(
            "mission-versioned", "agent-a", action="*", resource="*",
            tool="*", version=4, issued_at=now, expires_at=now + timedelta(minutes=5),
        ))


def test_broader_version_requires_explicit_binding_and_is_audited():
    audit = AuditLogger()
    registry = MissionRegistry(audit_logger=audit)
    now = datetime.now(timezone.utc)
    registry.register(Mission(
        "mission-expansion", "agent-a", allowed_actions=("customer.read",),
        allowed_resources=("customer_db",), allowed_tools=("customer_lookup",),
        issued_at=now, expires_at=now + timedelta(minutes=5),
    ))
    registry.register(Mission(
        "mission-expansion", "agent-a",
        allowed_actions=("customer.read", "customer.update"),
        allowed_resources=("customer_db",), allowed_tools=("customer_lookup",),
        version=2, issued_at=now, expires_at=now + timedelta(minutes=5),
    ))
    assert any(
        event.event_type is AuditEventType.MISSION_SCOPE_ESCALATION
        for event in audit.events()
    )
    registry.bind("agent-a", "mission-expansion", 1)
    assert registry.check(
        "agent-a", action="customer.update", resource="customer_db",
        tool="customer_lookup",
    ).decision_type is DecisionType.BLOCK
    registry.bind("agent-a", "mission-expansion", 2)
    assert registry.check(
        "agent-a", action="customer.update", resource="customer_db",
        tool="customer_lookup",
    ).decision_type is DecisionType.ALLOW
    registry.revoke("mission-expansion")
    assert registry.check(
        "agent-a", action="customer.update", resource="customer_db",
        tool="customer_lookup",
    ).decision_type is DecisionType.BLOCK


def test_mission_enforcement_cannot_be_disabled():
    with pytest.raises(MissionError, match="mandatory"):
        MissionRegistry(require_missions=False)
    with pytest.raises(ValueError, match="mandatory"):
        PreventApp(require_missions=False)


def test_expired_and_revoked_bound_versions_fail_closed():
    now = datetime.now(timezone.utc)
    registry = MissionRegistry()
    registry.register(Mission(
        "mission-expired", "agent-a", action="read", resource="db", tool="lookup",
        issued_at=now - timedelta(minutes=2), expires_at=now - timedelta(minutes=1),
    ))
    registry.bind("agent-a", "mission-expired", 1)
    assert registry.check(
        "agent-a", action="read", resource="db", tool="lookup",
    ).decision_type is DecisionType.BLOCK
    registry.register(Mission(
        "mission-revocation", "agent-b", action="read", resource="db", tool="lookup",
        issued_at=now, expires_at=now + timedelta(minutes=5),
    ))
    registry.bind("agent-b", "mission-revocation", 1)
    registry.revoke("mission-revocation")
    assert registry.state_of("mission-revocation", 1) is MissionState.REVOKED
    assert registry.check(
        "agent-b", action="read", resource="db", tool="lookup",
    ).decision_type is DecisionType.BLOCK


def test_registry_isolation_and_structured_audit():
    audit_a = AuditLogger()
    audit_b = AuditLogger()
    app_a, _ = make_app(require_missions=True)
    app_b, _ = make_app(require_missions=True)
    assert app_a.missions is not app_b.missions

    for registry, audit, agent_id, mission_id in (
        (MissionRegistry(audit_logger=audit_a), audit_a, "agent-a", "mission-a"),
        (MissionRegistry(audit_logger=audit_b), audit_b, "agent-a", "mission-b"),
    ):
        registry.register(Mission(
            mission_id, agent_id, action="read", resource="db", tool="lookup",
            expires_at=datetime.now(timezone.utc) + timedelta(minutes=5),
        ))
        registry.bind(agent_id, mission_id, 1)
        registry.check(agent_id, action="write", resource="db", tool="lookup")
    assert any(event.event_type is AuditEventType.MISSION_SCOPE_DENIED for event in audit_a.events())
    assert any(event.event_type is AuditEventType.MISSION_SCOPE_DENIED for event in audit_b.events())
    registry_a = app_a.missions
    registry_b = app_b.missions
    registry_a.register(Mission(
        "isolated", "mission-agent", action="read", resource="db", tool="lookup",
        expires_at=datetime.now(timezone.utc) + timedelta(minutes=5),
    ))
    registry_b.register(Mission(
        "isolated", "mission-agent", action="read", resource="db", tool="lookup",
        expires_at=datetime.now(timezone.utc) + timedelta(minutes=5),
    ))
    registry_a.bind("mission-agent", "isolated", 1)
    registry_b.bind("mission-agent", "isolated", 1)
    registry_a.revoke("isolated")
    assert registry_a.check("mission-agent", action="read", resource="db", tool="lookup").decision_type is DecisionType.BLOCK
    assert registry_b.check("mission-agent", action="read", resource="db", tool="lookup").decision_type is DecisionType.ALLOW


def test_audit_failure_blocks_current_and_future_checks(monkeypatch):
    audit = AuditLogger()
    registry = MissionRegistry(audit_logger=audit)
    registry.register(Mission(
        "mission-audit", "agent-a", action="read", resource="db", tool="lookup",
        expires_at=datetime.now(timezone.utc) + timedelta(minutes=5),
    ))
    registry.bind("agent-a", "mission-audit", 1)
    monkeypatch.setattr(audit, "record", lambda *args, **kwargs: None)
    assert registry.check(
        "agent-a", action="read", resource="db", tool="lookup",
    ).decision_type is DecisionType.BLOCK
    assert registry.check(
        "agent-a", action="read", resource="db", tool="lookup",
    ).decision_type is DecisionType.BLOCK


def test_mission_audit_contains_safe_resource_fingerprint():
    audit = AuditLogger()
    registry = MissionRegistry(audit_logger=audit)
    registry.register(Mission(
        "mission-audit-safe", "agent-a", action="read", resource="db",
        tool="lookup", expires_at=datetime.now(timezone.utc) + timedelta(minutes=5),
    ))
    registry.bind("agent-a", "mission-audit-safe", 1)
    registry.check(
        "agent-a", action="read?token=not-for-audit",
        resource="private/path?secret=not-for-audit", tool="lookup",
    )
    event = audit.events()[-1]
    assert event.action.startswith("sha256:")
    assert event.resource.startswith("sha256:")
    assert "not-for-audit" not in event.to_json()


def test_mission_rejects_executable_context_predicates():
    with pytest.raises(MissionError):
        Mission(
            "mission-no-callback",
            "agent-a",
            action="read",
            resource="db",
            tool="lookup",
            context_constraints={"tenant": lambda _: True},
            expires_at=datetime.now(timezone.utc) + timedelta(minutes=5),
        )


def test_mission_revocation_serializes_with_execution_and_concurrent_reads():
    registry = MissionRegistry()
    registry.register(Mission(
        "mission-concurrent", "agent-a", action="read", resource="db", tool="lookup",
        expires_at=datetime.now(timezone.utc) + timedelta(minutes=5),
    ))
    registry.bind("agent-a", "mission-concurrent", 1)
    entered = threading.Event()
    release = threading.Event()
    revoked = threading.Event()
    request = {
        "action": "read", "resource": "db", "tool": "lookup",
        "operation": "concurrent_read",
    }

    def execute_under_guard():
        with registry.execution_guard("agent-a", **request) as decision:
            assert decision.decision_type is DecisionType.ALLOW
            entered.set()
            assert release.wait(timeout=2)

    def revoke():
        assert entered.wait(timeout=2)
        registry.revoke("mission-concurrent", 1)
        revoked.set()

    with ThreadPoolExecutor(max_workers=8) as pool:
        execution = pool.submit(execute_under_guard)
        revocation = pool.submit(revoke)
        assert entered.wait(timeout=2)
        assert not revoked.wait(timeout=0.05)
        with ThreadPoolExecutor(max_workers=4) as readers:
            decisions = list(readers.map(
                lambda _: registry.check("agent-a", **request),
                range(8),
            ))
        assert all(item.decision_type is DecisionType.ALLOW for item in decisions)
        release.set()
        execution.result(timeout=2)
        revocation.result(timeout=2)
    assert registry.check("agent-a", **request).decision_type is DecisionType.BLOCK
    registry.revoke("mission-concurrent", 1)


def test_version_registration_waits_for_active_execution_and_does_not_rebind():
    registry = MissionRegistry()
    now = datetime.now(timezone.utc)
    registry.register(Mission(
        "mission-update-race", "agent-a", action="read", resource="db",
        tool="lookup", issued_at=now, expires_at=now + timedelta(minutes=5),
    ))
    registry.bind("agent-a", "mission-update-race", 1)
    entered = threading.Event()
    release = threading.Event()
    updated = threading.Event()

    def execute():
        with registry.execution_guard(
            "agent-a", action="read", resource="db", tool="lookup"
        ) as decision:
            assert decision.decision_type is DecisionType.ALLOW
            entered.set()
            assert release.wait(timeout=2)

    def register_broader_version():
        assert entered.wait(timeout=2)
        registry.register(Mission(
            "mission-update-race",
            "agent-a",
            action="*",
            resource="*",
            tool="*",
            version=2,
            issued_at=now,
            expires_at=now + timedelta(minutes=5),
        ))
        updated.set()

    with ThreadPoolExecutor(max_workers=2) as pool:
        execution = pool.submit(execute)
        registration = pool.submit(register_broader_version)
        assert entered.wait(timeout=2)
        assert not updated.wait(timeout=0.05)
        release.set()
        execution.result(timeout=2)
        registration.result(timeout=2)
    assert registry.binding_for("agent-a").mission_version == 1
    assert registry.check(
        "agent-a", action="write", resource="admin_db", tool="admin",
    ).decision_type is DecisionType.BLOCK
    assert registry.check(
        "agent-a", action="read", resource="db", tool="lookup",
    ).decision_type is DecisionType.ALLOW
