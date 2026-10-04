from datetime import datetime, timezone

from preventai.core import (
    DecisionReason,
    DecisionType,
    EvaluationContext,
    PermissionCategory,
    PermissionEngine,
    PermissionRule,
    SecurityEngine,
)
from preventai.core.policy import PolicyRule, PolicyEngine
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


def context(action="database.read", agent="agent-1", tool="database", resource="db-1"):
    request = SecurityRequest(
        protocol_version="1.0",
        request_id="permission-1",
        timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
        agent=AgentIdentity(agent, "autonomous"),
        session=SessionIdentity("session-1"),
        requested_by=RequestedBy.AGENT,
        action=Action(action, {}),
        resource=Resource(resource, "database"),
        tool=Tool(tool, "1.0"),
        data=DataMetadata(DataClassification.INTERNAL, SourceTrust.TRUSTED),
        context={},
        metadata={},
    )
    return EvaluationContext.from_request(request)


def grant(permission_id="p1", **kwargs):
    return PermissionRule(permission_id, **kwargs)


def test_allowed_action():
    result = PermissionEngine([
        grant(category=PermissionCategory.DATABASE, action="database.read"),
    ]).evaluate(context())
    assert result.decision_type is DecisionType.ALLOW


def test_denied_action():
    result = PermissionEngine([
        grant(category=PermissionCategory.DATABASE, action="database.write"),
    ]).evaluate(context())
    assert result.reason is DecisionReason.MISSING_PERMISSION


def test_missing_permission_denies():
    result = PermissionEngine().evaluate(context())
    assert result.decision_type is DecisionType.BLOCK


def test_wildcard_permission():
    result = PermissionEngine([
        grant(category=PermissionCategory.DATABASE, action="database.*"),
    ]).evaluate(context(action="database.read"))
    assert result.decision_type is DecisionType.ALLOW


def test_resource_specific_permission():
    engine = PermissionEngine([
        grant(category=PermissionCategory.DATABASE, resource="db-safe"),
    ])
    assert engine.evaluate(context(resource="db-safe")).decision_type is DecisionType.ALLOW
    assert engine.evaluate(context(resource="db-other")).decision_type is DecisionType.BLOCK


def test_privilege_escalation_attempt_is_blocked():
    delegator = PermissionEngine([
        grant("authority", category=PermissionCategory.DATABASE, action="database.read"),
    ])
    delegate = PermissionEngine([
        grant("requested", category=PermissionCategory.DATABASE, action="database.delete"),
    ])
    result = delegate.evaluate(context(action="database.delete"), delegating_permissions=delegator)
    assert result.reason is DecisionReason.PRIVILEGE_ESCALATION


def test_permission_engine_integrates_into_core_pipeline():
    policy = PolicyEngine([
        PolicyRule("policy-allow", DecisionType.ALLOW, action="database.read"),
    ])
    permissions = PermissionEngine([
        grant(category=PermissionCategory.DATABASE, action="database.read"),
    ])
    request_context = context()
    result = SecurityEngine(policy, permissions).evaluate(request_context.request)
    assert result.decision_type is DecisionType.ALLOW
