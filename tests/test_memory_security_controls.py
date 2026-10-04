from concurrent.futures import ThreadPoolExecutor

import pytest

from preventai import (
    AuditLogger,
    AuditPolicy,
    DataFlowEngine,
    DataFlowPolicy,
    Decision,
    DecisionReason,
    DecisionType,
)
from preventai.core.dataflow import FlowClassification, FlowTrust
from preventai.memory import (
    MemoryAccessContext,
    MemoryAccessDenied,
    MemoryOperation,
    MemoryPolicy,
    MemoryPolicyRule,
    MemoryStore,
)


def ctx(agent="agent-a", session="session-a", namespace="customer"):
    return MemoryAccessContext(agent, agent, session, namespace)


def policy(*operations):
    return MemoryPolicy([
        MemoryPolicyRule(operation.value, operation, DecisionType.ALLOW)
        for operation in (operations or tuple(MemoryOperation))
    ])


def test_memory_audit_events_are_structural_and_payload_free():
    logger = AuditLogger(secret="synthetic-audit-key")
    memory = MemoryStore(policy=policy(), audit_logger=logger)
    entry = memory.write(ctx(), {"api_key": "synthetic-secret", "email": "synthetic@example.test"})
    memory.read(ctx(), entry.memory_id)
    memory.update(ctx(), entry.memory_id, {"value": 2}, expected_version=1)
    memory.delete(ctx(), entry.memory_id)
    events = logger.events()
    assert [event.event_type.value for event in events] == [
        "memory_write", "memory_read", "memory_update", "memory_delete"
    ]
    assert all(event.correlation_id == entry.memory_id for event in events)
    serialized = "".join(event.to_json() for event in events)
    assert "synthetic-secret" not in serialized
    assert "synthetic@example.test" not in serialized
    assert logger.verify().valid


def test_blocked_operation_emits_memory_blocked_event():
    logger = AuditLogger(secret="synthetic-audit-key")
    memory = MemoryStore(policy=policy(MemoryOperation.WRITE), audit_logger=logger)
    entry = memory.write(ctx(), {"value": 1})
    with pytest.raises(MemoryAccessDenied):
        memory.read(ctx("agent-b", "session-b"), entry.memory_id)
    assert logger.events()[-1].event_type.value == "memory_blocked"


class FailingAudit:
    policy = AuditPolicy(enabled=True, required=True)

    def record(self, *args, **kwargs):
        raise RuntimeError("audit backend failure")


def test_required_audit_failure_blocks_before_commit():
    memory = MemoryStore(policy=policy(), audit_logger=FailingAudit())
    with pytest.raises(MemoryAccessDenied) as error:
        memory.write(ctx(), {"value": 1})
    assert error.value.decision.decision_type is DecisionType.BLOCK
    assert "audit backend failure" not in str(error.value)


class OptionalFailingAudit(FailingAudit):
    policy = AuditPolicy(enabled=True, required=False)


def test_optional_audit_failure_does_not_create_unsafe_allow():
    memory = MemoryStore(policy=policy(), audit_logger=OptionalFailingAudit())
    entry = memory.write(ctx(), {"value": 1})
    assert memory.read(ctx(), entry.memory_id).payload == {"value": 1}


def test_missing_data_flow_policy_blocks_memory_operation():
    memory = MemoryStore(policy=policy(), data_flow_engine=DataFlowEngine())
    with pytest.raises(MemoryAccessDenied) as error:
        memory.write(ctx(), {"value": 1})
    assert error.value.decision.decision_type is DecisionType.BLOCK


def test_data_flow_policy_is_authoritative():
    flow = DataFlowPolicy(
        "memory", DecisionType.ALLOW,
        source="memory.customer", destination="memory.customer",
        action="memory.*", classification=FlowClassification.PRIVATE,
        trust=FlowTrust.TRUSTED,
    )
    memory = MemoryStore(
        policy=policy(),
        data_flow_engine=DataFlowEngine([flow]),
    )
    entry = memory.write(ctx(), {"value": 1})
    assert memory.read(ctx(), entry.memory_id).payload == {"value": 1}


def test_data_flow_evaluator_failure_fails_closed():
    class BrokenFlow(DataFlowEngine):
        def check(self, flow):
            raise RuntimeError("synthetic flow failure")

    memory = MemoryStore(policy=policy(), data_flow_engine=BrokenFlow())
    with pytest.raises(MemoryAccessDenied) as error:
        memory.write(ctx(), {"value": 1})
    assert error.value.decision.decision_type is DecisionType.BLOCK


def test_permission_denial_and_evaluator_failure_fail_closed():
    denied = MemoryStore(
        policy=policy(),
        permission_checker=lambda operation, context, entry: Decision(
            DecisionType.BLOCK, DecisionReason.MISSING_PERMISSION,
        ),
    )
    with pytest.raises(MemoryAccessDenied):
        denied.write(ctx(), {"value": 1})

    broken = MemoryStore(
        policy=policy(),
        permission_checker=lambda operation, context, entry: (_ for _ in ()).throw(
            RuntimeError("synthetic permission failure")
        ),
    )
    with pytest.raises(MemoryAccessDenied):
        broken.write(ctx(), {"value": 1})


def test_concurrent_updates_allow_only_one_writer_per_expected_version():
    memory = MemoryStore(policy=policy())
    entry = memory.write(ctx(), {"value": 0})

    def update(value):
        try:
            return memory.update(ctx(), entry.memory_id, {"value": value}, expected_version=1)
        except MemoryAccessDenied:
            return None

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(update, [1, 2]))
    assert sum(result is not None for result in results) == 1
    current = memory.read(ctx(), entry.memory_id)
    assert current.version == 2
    assert current.payload["value"] in (1, 2)


def test_concurrent_reads_return_consistent_entries():
    memory = MemoryStore(policy=policy())
    entry = memory.write(ctx(), {"value": "stable"})
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _: memory.read(ctx(), entry.memory_id), range(32)))
    assert all(result.version == 1 and result.payload == {"value": "stable"} for result in results)


def test_concurrent_writes_preserve_unique_memory_ids_and_limit():
    memory = MemoryStore(policy=policy(MemoryOperation.WRITE), max_entries_per_namespace=20)

    with ThreadPoolExecutor(max_workers=8) as pool:
        entries = list(pool.map(lambda value: memory.write(ctx(), {"value": value}), range(20)))
    assert len({entry.memory_id for entry in entries}) == 20
    assert all(entry.version == 1 for entry in entries)
    with pytest.raises(MemoryAccessDenied):
        memory.write(ctx(), {"value": "overflow"})
