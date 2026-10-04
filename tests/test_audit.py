from datetime import datetime, timezone
from dataclasses import replace
import threading

import pytest

from preventai import (
    AuditEvent,
    AuditEventType,
    AuditLogger,
    AuditPolicy,
    DecisionReason,
    DecisionType,
    Agent,
    PreventApp,
)
from preventai.core import (
    PermissionCategory,
    PermissionEngine,
    PermissionRule,
    PolicyEngine,
    PolicyRule,
)
from preventai.core.decision import Decision


def test_event_is_immutable_and_canonical():
    event = AuditEvent(
        "event-1",
        1,
        datetime(2026, 1, 1, tzinfo=timezone.utc),
        AuditEventType.POLICY_EVALUATION,
        metadata={"category": "policy"},
    )
    assert event.to_json() == event.to_json()
    with pytest.raises(Exception):
        event.sequence = 2


def test_logger_chain_trace_and_tamper_detection():
    logger = AuditLogger(secret="audit-secret")
    first = logger.record(
        AuditEventType.AUTHENTICATION,
        correlation_id="trace-1",
        agent_id="agent",
        decision=DecisionType.ALLOW.value,
        reason=DecisionReason.REQUEST_ALLOWED.value,
    )
    logger.record(
        AuditEventType.POLICY_EVALUATION,
        correlation_id="trace-1",
        decision=DecisionType.ALLOW.value,
    )
    assert first is not None
    assert logger.verify().valid
    assert len(logger.trace("trace-1")) == 2
    assert logger.trace("unknown") == ()
    logger._events[0] = replace(logger._events[0], decision="block")
    assert not logger.verify().valid


def test_sensitive_metadata_is_rejected():
    logger = AuditLogger()
    with pytest.raises(ValueError):
        logger.record(AuditEventType.SECRET_DETECTION, metadata={"secret": "x"})


def test_concurrent_append_preserves_chain():
    logger = AuditLogger(secret="audit-secret")

    def append():
        logger.record(AuditEventType.TOOL_EXECUTION_STARTED, agent_id="agent")

    threads = [threading.Thread(target=append) for _ in range(20)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert [event.sequence for event in logger.events()] == list(range(1, 21))
    assert logger.verify().valid


def test_tool_execution_audits_start_and_success():
    logger = AuditLogger(secret="audit-secret")
    agent = Agent("audit-agent")

    @agent.tool(name="lookup")
    def lookup():
        return {"ok": True}

    app = PreventApp(
        policy=PolicyEngine(
            [PolicyRule("allow", DecisionType.ALLOW, action="lookup")]
        ),
        permissions=PermissionEngine(
            [PermissionRule("read", PermissionCategory.READ, action="lookup")]
        ),
        audit_logger=logger,
    )
    app.add_agent(agent)
    app.run()
    assert lookup() == {"ok": True}
    event_types = [event.event_type for event in logger.events()]
    assert AuditEventType.TOOL_EXECUTION_STARTED in event_types
    assert AuditEventType.TOOL_EXECUTION_SUCCEEDED in event_types
    assert logger.verify().valid


def test_required_audit_failure_blocks():
    logger = AuditLogger(policy=AuditPolicy(enabled=True, required=True))
    logger.record = lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("internal"))
    agent = Agent("required-audit-agent")
    calls = []

    @agent.tool(name="lookup")
    def lookup():
        calls.append(True)
        return "ok"

    app = PreventApp(
        policy=PolicyEngine([PolicyRule("allow", DecisionType.ALLOW, action="lookup")]),
        permissions=PermissionEngine(
            [PermissionRule("read", PermissionCategory.READ, action="lookup")]
        ),
        audit_logger=logger,
    )
    with pytest.raises(RuntimeError):
        app.add_agent(agent)
    assert calls == []
