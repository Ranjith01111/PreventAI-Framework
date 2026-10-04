from concurrent.futures import ThreadPoolExecutor

import pytest

from preventai import (
    AgentLifecycleRegistry,
    RegisteredAgentIdentity,
    AuditEventType,
    AuditLogger,
    LifecycleError,
    LifecycleState,
)
from tests._agent_communication_support import make_boundary, make_message


def active_registry():
    registry = AgentLifecycleRegistry()
    registry.register("agent-a")
    registry.register("agent-b")
    registry.activate("agent-a")
    registry.activate("agent-b")
    return registry


def test_lifecycle_states_and_semantics():
    assert {state.value for state in LifecycleState} == {
        "created", "registered", "active", "suspended",
        "quarantined", "revoked", "terminated",
    }
    registry = AgentLifecycleRegistry()
    registry.register(RegisteredAgentIdentity("agent-a"))
    assert registry.get_state("agent-a") is LifecycleState.REGISTERED
    assert not registry.is_executable("agent-a")
    registry.activate("agent-a")
    assert registry.is_active("agent-a")
    registry.suspend("agent-a")
    assert not registry.is_communicable("agent-a")


def test_invalid_and_terminal_transitions_fail_closed():
    registry = AgentLifecycleRegistry()
    registry.register("agent-a")
    with pytest.raises(LifecycleError):
        registry.terminate("agent-a")
    registry.activate("agent-a")
    registry.revoke("agent-a")
    with pytest.raises(LifecycleError):
        registry.activate("agent-a")
    registry.terminate("agent-a")
    with pytest.raises(LifecycleError):
        registry.suspend("agent-a")
    assert registry.check("missing").decision_type.value == "block"
    assert registry.get_state("malformed\nid") is None


def test_registry_is_instance_local_and_agents_are_isolated():
    first = AgentLifecycleRegistry()
    second = AgentLifecycleRegistry()
    first.register("agent-a")
    first.register("agent-b")
    first.activate("agent-a")
    first.activate("agent-b")
    first.revoke("agent-a")
    assert first.get_state("agent-b") is LifecycleState.ACTIVE
    assert second.get_state("agent-a") is None


def test_lifecycle_snapshot_cannot_be_mutated_without_transition():
    registry = AgentLifecycleRegistry()
    registry.register("agent-a")
    registry.activate("agent-a")
    with pytest.raises(TypeError):
        registry._records["agent-a"] = registry.get_record("agent-a")
    with pytest.raises(AttributeError):
        registry._records = {}
    assert registry.get_state("agent-a") is LifecycleState.ACTIVE


def test_concurrent_transitions_have_one_terminal_outcome():
    registry = AgentLifecycleRegistry()
    registry.register("agent-a")
    registry.activate("agent-a")
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _: _revoke(registry), range(8)))
    assert sum(results) == 1
    assert registry.get_state("agent-a") is LifecycleState.REVOKED


def _revoke(registry):
    try:
        registry.revoke("agent-a")
        return 1
    except LifecycleError:
        return 0


def test_lifecycle_transitions_are_audited():
    audit = AuditLogger()
    registry = AgentLifecycleRegistry(audit_logger=audit)
    registry.register("agent-a")
    registry.activate("agent-a")
    registry.revoke("agent-a")
    assert [event.event_type for event in audit.events()] == [
        AuditEventType.LIFECYCLE_TRANSITION,
        AuditEventType.LIFECYCLE_TRANSITION,
        AuditEventType.LIFECYCLE_TRANSITION,
    ]
    assert audit.verify().valid


def test_inter_agent_auth_revalidates_lifecycle_before_dispatch():
    registry = active_registry()
    boundary, context, calls = make_boundary()
    boundary._lifecycle = registry
    message = make_message()
    authorization = boundary.issue(message, context)
    registry.revoke("agent-a")
    with pytest.raises(Exception):
        boundary.dispatch(message, authorization, context)
    assert calls == []
