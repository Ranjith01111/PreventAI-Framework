import re

import pytest

from preventai import Agent, PreventApp, SecretsGuard, ToolApprovalRequired, ToolExecutionDenied
from preventai.core import (
    DecisionReason,
    DecisionType,
    PermissionCategory,
    PermissionEngine,
    PermissionRule,
    PolicyEngine,
    PolicyRule,
)


def scan(value, **kwargs):
    return SecretsGuard(**kwargs).scan(value)


@pytest.mark.parametrize(
    ("value", "category"),
    [
        ({"api_key": "abc123"}, "api_key"),
        ({"password": "abc123"}, "password"),
        ({"secret": "abc123"}, "secret"),
        ({"access_token": "abc123"}, "access_token"),
        ("Bearer abcdefgh1234", "bearer_token"),
        ("-----BEGIN PRIVATE KEY-----\nabc", "private_key"),
        ("postgres://user:pass@example.test/db", "connection_string"),
        ("API_KEY=abc123", "environment_secret"),
    ],
)
def test_secret_categories_are_detected_without_leaking_values(value, category):
    decision = scan(value)
    assert decision.decision_type is DecisionType.BLOCK
    assert decision.reason is DecisionReason.SECRETS_DETECTED
    assert decision.metadata["category"] == category
    assert "abc123" not in decision.to_json()
    assert "pass@" not in decision.to_json()


@pytest.mark.parametrize(
    "value",
    [
        {"customer": {"credentials": {"api_key": "secret-value"}}},
        [{"nested": {"password": "secret-value"}}],
        ({"nested": [{"client_secret": "secret-value"}]},),
    ],
)
def test_nested_structures_are_scanned(value):
    assert scan(value).decision_type is DecisionType.BLOCK


def test_multiple_secrets_return_safe_first_finding():
    decision = scan({"password": "one", "api_key": "two"})
    assert decision.decision_type is DecisionType.BLOCK
    assert "one" not in decision.to_json()
    assert "two" not in decision.to_json()


def test_normal_text_and_empty_sensitive_fields_are_allowed():
    assert scan("This is normal documentation.").decision_type is DecisionType.ALLOW
    assert scan({"password": "", "note": "ordinary text"}).decision_type is DecisionType.ALLOW


def test_custom_field_and_pattern():
    assert scan({"internal_secret": "value"}, sensitive_fields={"internal_secret"}).decision_type is DecisionType.BLOCK
    assert scan("ticket=ZX-123", custom_patterns={"ticket": r"ZX-\d+"}).decision_type is DecisionType.BLOCK


def test_enabled_categories_can_reduce_detection():
    assert scan({"password": "value"}, enabled_categories={"api_key"}).decision_type is DecisionType.ALLOW


def test_depth_and_unsupported_values_fail_closed():
    assert scan({"a": {"b": {"c": "value"}}}, max_depth=1).decision_type is DecisionType.BLOCK
    assert scan(object()).decision_type is DecisionType.BLOCK


def test_cycles_fail_closed():
    value = []
    value.append(value)
    assert scan(value).decision_type is DecisionType.BLOCK


def test_tool_secret_does_not_execute_or_leak():
    calls = []
    policy = PolicyEngine([PolicyRule("allow", DecisionType.ALLOW, action="lookup")])
    permissions = PermissionEngine([
        PermissionRule("allow", PermissionCategory.READ, action="lookup")
    ])
    app = PreventApp(policy=policy, permissions=permissions)
    agent = app.add_agent(Agent("secret-agent"))

    @agent.tool()
    def lookup(password):
        calls.append(password)

    app.run()
    with pytest.raises(ToolExecutionDenied) as error:
        lookup("super-secret")
    assert error.value.decision.reason is DecisionReason.SECRETS_DETECTED
    assert "super-secret" not in str(error.value)
    assert "super-secret" not in error.value.decision.to_json()
    assert calls == []


def test_tool_without_secret_executes():
    policy = PolicyEngine([PolicyRule("allow", DecisionType.ALLOW, action="lookup")])
    permissions = PermissionEngine([
        PermissionRule("allow", PermissionCategory.READ, action="lookup")
    ])
    app = PreventApp(policy=policy, permissions=permissions)
    agent = app.add_agent(Agent("safe-agent"))

    @agent.tool()
    def lookup(customer_id):
        return customer_id

    app.run()
    assert lookup("C001") == "C001"


def test_tool_scanner_exception_fails_closed(monkeypatch):
    def broken(self, value):
        raise RuntimeError("internal detail")

    monkeypatch.setattr(SecretsGuard, "scan", broken)
    policy = PolicyEngine([PolicyRule("allow", DecisionType.ALLOW, action="lookup")])
    permissions = PermissionEngine([
        PermissionRule("allow", PermissionCategory.READ, action="lookup")
    ])
    app = PreventApp(policy=policy, permissions=permissions)
    agent = app.add_agent(Agent("scanner-agent"))

    @agent.tool()
    def lookup(value):
        return value

    app.run()
    with pytest.raises(ToolExecutionDenied) as error:
        lookup("ordinary")
    assert error.value.decision.reason is DecisionReason.SECRETS_DETECTED
    assert "internal detail" not in str(error.value)


def test_policy_block_precedes_secrets():
    app = PreventApp(
        policy=PolicyEngine([PolicyRule("block", DecisionType.BLOCK, action="lookup")]),
        permissions=PermissionEngine([
            PermissionRule("allow", PermissionCategory.READ, action="lookup")
        ]),
    )
    agent = app.add_agent(Agent("blocked-agent"))

    @agent.tool()
    def lookup(password):
        return password

    app.run()
    with pytest.raises(ToolExecutionDenied) as error:
        lookup("secret")
    assert error.value.decision.reason is DecisionReason.REQUEST_BLOCKED


def test_permission_block_precedes_secrets():
    app = PreventApp(
        policy=PolicyEngine([PolicyRule("allow", DecisionType.ALLOW, action="lookup")]),
        permissions=PermissionEngine(),
    )
    agent = app.add_agent(Agent("denied-agent"))

    @agent.tool()
    def lookup(password):
        return password

    app.run()
    with pytest.raises(ToolExecutionDenied) as error:
        lookup("secret")
    assert error.value.decision.reason is DecisionReason.MISSING_PERMISSION


def test_schema_block_precedes_secrets():
    app = PreventApp(
        policy=PolicyEngine([PolicyRule("allow", DecisionType.ALLOW, action="lookup")]),
        permissions=PermissionEngine([
            PermissionRule("allow", PermissionCategory.READ, action="lookup")
        ]),
    )
    agent = app.add_agent(Agent("schema-agent"))

    @agent.tool(input_schema={"type": "object", "required": ["customer_id"]})
    def lookup(customer_id):
        return customer_id

    app.run()
    with pytest.raises(ToolExecutionDenied) as error:
        lookup(password="secret")
    assert error.value.decision.reason is DecisionReason.SCHEMA_VALIDATION_FAILED


def test_approval_precedes_secrets():
    app = PreventApp(
        policy=PolicyEngine([
            PolicyRule("approval", DecisionType.APPROVAL_REQUIRED, action="lookup")
        ]),
        permissions=PermissionEngine([
            PermissionRule("allow", PermissionCategory.READ, action="lookup")
        ]),
    )
    agent = app.add_agent(Agent("approval-agent"))

    @agent.tool()
    def lookup(password):
        return password

    app.run()
    with pytest.raises(ToolApprovalRequired):
        lookup("secret")
