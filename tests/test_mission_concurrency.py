from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

from preventai import Capability, DecisionType, Mission, MissionRegistry


def _mission(**overrides):
    values = {
        "mission_id": "concurrency-mission",
        "agent_id": "agent-a",
        "action": "customer.read",
        "resource": "customer_db",
        "tool": "customer_lookup",
        "allowed_destinations": ("approved-service",),
        "allowed_data_classes": ("internal",),
        "context_constraints": {"tenant": "tenant-a"},
        "expires_at": datetime.now(timezone.utc) + timedelta(minutes=5),
    }
    values.update(overrides)
    return Mission(**values)


def test_delegated_capability_cannot_expand_bound_mission():
    registry = MissionRegistry()
    registry.register(_mission())
    registry.bind("agent-a", "concurrency-mission", 1)

    assert registry.allows_capability(
        "agent-a",
        Capability(
            "customer.read",
            "customer_db",
            "customer_lookup",
            destinations=("approved-service",),
            data_classes=("internal",),
            context_constraints={"tenant": "tenant-a"},
        ),
    )
    assert not registry.allows_capability(
        "agent-a",
        Capability("customer.write", "customer_db", "customer_lookup"),
    )
    assert not registry.allows_capability(
        "agent-a",
        Capability(
            "customer.read",
            "customer_db",
            "customer_lookup",
            destinations=("approved-service",),
            data_classes=("private",),
            context_constraints={"tenant": "tenant-a"},
        ),
    )


def test_revoke_serializes_with_guarded_execution():
    registry = MissionRegistry()
    registry.register(_mission())
    registry.bind("agent-a", "concurrency-mission", 1)
    entered = False

    def execute():
        nonlocal entered
        with registry.execution_guard(
            "agent-a",
            action="customer.read",
            resource="customer_db",
            tool="customer_lookup",
            destination="approved-service",
            data="internal",
            context={"tenant": "tenant-a"},
        ) as decision:
            entered = decision.decision_type is DecisionType.ALLOW
            return decision

    with ThreadPoolExecutor(max_workers=2) as pool:
        result = pool.submit(execute).result()
        registry.revoke("concurrency-mission")

    assert entered is True
    assert result.decision_type is DecisionType.ALLOW
    assert registry.check(
        "agent-a",
        action="customer.read",
        resource="customer_db",
        tool="customer_lookup",
    ).decision_type is DecisionType.BLOCK
