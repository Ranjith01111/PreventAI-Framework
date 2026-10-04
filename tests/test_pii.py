import pytest

from preventai import Agent, PIIGuard, PreventApp, ToolApprovalRequired, ToolExecutionDenied
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
    return PIIGuard(**kwargs).scan(value)


@pytest.mark.parametrize(
    ("value", "category"),
    [
        ("alice@example.com", "email"),
        ("+91 98765 43210", "phone"),
        ("192.168.1.10", "ip_address"),
        ("2001:db8::1", "ip_address"),
        ("4111 1111 1111 1111", "credit_card_like"),
        ({"date_of_birth": "1990-01-01"}, "date_of_birth"),
        ({"address": "10 Main Street"}, "physical_address"),
        ({"user_id": "user-12345"}, "user_identifier"),
        ({"national_id": "AB123"}, "government_id_like"),
    ],
)
def test_pii_categories_are_detected(value, category):
    decision = scan(value)
    assert decision.decision_type is DecisionType.BLOCK
    assert decision.reason is DecisionReason.PII_DETECTED
    assert decision.metadata["category"] == category


def test_custom_government_pattern():
    decision = scan("NID-123456", custom_patterns={"government_id_like": r"NID-\d{6}"})
    assert decision.metadata["category"] == "government_id_like"


@pytest.mark.parametrize(
    "value",
    [
        {"customer": {"contact": {"email": "alice@example.com"}}},
        [{"phone": "+1 212-555-0100"}],
        ({"nested": [{"user_id": "user-123"}]},),
    ],
)
def test_nested_values_are_scanned(value):
    assert scan(value).decision_type is DecisionType.BLOCK


def test_multiple_pii_values_do_not_leak():
    decision = scan({"email": "alice@example.com", "phone": "+1 212-555-0100"})
    encoded = decision.to_json()
    assert "alice@example.com" not in encoded
    assert "212-555-0100" not in encoded


def test_false_positive_controls():
    assert scan("ordinary documentation").decision_type is DecisionType.ALLOW
    assert scan({"order_id": "1234567890123456"}).decision_type is DecisionType.ALLOW
    assert scan("2026-09-25").decision_type is DecisionType.ALLOW
    assert scan("1234567890123457").decision_type is DecisionType.ALLOW


def test_sensitive_field_and_category_configuration():
    assert scan({"personal_contact": "alice@example.com"}, sensitive_fields={"personal_contact"}).decision_type is DecisionType.BLOCK
    assert scan("alice@example.com", enabled_categories={"phone"}).decision_type is DecisionType.ALLOW


def test_depth_cycles_and_unsupported_values_fail_closed():
    assert scan({"a": {"b": {"c": {"email": "alice@example.com"}}}}, max_depth=1).decision_type is DecisionType.BLOCK
    cyclic = []
    cyclic.append(cyclic)
    assert scan(cyclic).decision_type is DecisionType.BLOCK
    assert scan(object()).decision_type is DecisionType.BLOCK


def _app(policy_type=DecisionType.ALLOW, permissions=None):
    policy = PolicyEngine([PolicyRule("rule", policy_type, action="*")])
    permissions = permissions or PermissionEngine([
        PermissionRule("permission", PermissionCategory.READ, action="lookup")
    ])
    app = PreventApp(policy=policy, permissions=permissions)
    return app, app.add_agent(Agent("pii-agent"))


def test_pii_tool_block_prevents_side_effect_and_does_not_leak():
    calls = []
    app, agent = _app()

    @agent.tool()
    def lookup(email):
        calls.append(email)
        return email

    app.run()
    with pytest.raises(ToolExecutionDenied) as error:
        lookup("alice@example.com")
    assert error.value.decision.reason is DecisionReason.PII_DETECTED
    assert "alice@example.com" not in str(error.value)
    assert calls == []


def test_clean_tool_executes():
    app, agent = _app()

    @agent.tool()
    def lookup(value):
        return value

    app.run()
    assert lookup("ordinary") == "ordinary"


def test_existing_blockers_precede_pii():
    app, agent = _app(DecisionType.BLOCK)

    @agent.tool()
    def lookup(email):
        return email

    app.run()
    with pytest.raises(ToolExecutionDenied) as error:
        lookup("alice@example.com")
    assert error.value.decision.reason is DecisionReason.REQUEST_BLOCKED

    app, agent = _app(permissions=PermissionEngine())

    @agent.tool()
    def denied(email):
        return email

    app.run()
    with pytest.raises(ToolExecutionDenied) as error:
        denied("alice@example.com")
    assert error.value.decision.reason is DecisionReason.MISSING_PERMISSION


def test_schema_and_secrets_block_before_pii():
    app, agent = _app()

    @agent.tool("lookup", input_schema={"type": "object", "required": ["customer_id"]})
    def schema_tool(customer_id):
        return customer_id

    app.run()
    with pytest.raises(ToolExecutionDenied) as error:
        schema_tool(email="alice@example.com")
    assert error.value.decision.reason is DecisionReason.SCHEMA_VALIDATION_FAILED

    app, agent = _app()

    @agent.tool("lookup")
    def secret_tool(password):
        return password

    app.run()
    with pytest.raises(ToolExecutionDenied) as error:
        secret_tool(password="secret-value")
    assert error.value.decision.reason is DecisionReason.SECRETS_DETECTED


def test_approval_is_not_bypassed():
    app, agent = _app(DecisionType.APPROVAL_REQUIRED)

    @agent.tool()
    def lookup(email):
        return email

    app.run()
    with pytest.raises(ToolApprovalRequired):
        lookup("alice@example.com")


def test_scanner_exception_fails_closed(monkeypatch):
    def broken(self, value):
        raise RuntimeError("internal")

    monkeypatch.setattr(PIIGuard, "scan", broken)
    app, agent = _app()

    @agent.tool()
    def lookup(value):
        return value

    app.run()
    with pytest.raises(ToolExecutionDenied) as error:
        lookup("ordinary")
    assert error.value.decision.reason is DecisionReason.PII_DETECTED
    assert "internal" not in str(error.value)
