import pytest

from preventai import Agent, ContentGuard, PreventApp, ToolApprovalRequired, ToolExecutionDenied
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
    return ContentGuard(**kwargs).scan(value)


def test_blocked_and_allowed_phrases():
    decision = scan("This contains internal-only-command.", blocked_phrases={"internal-only-command"})
    assert decision.decision_type is DecisionType.BLOCK
    assert decision.reason is DecisionReason.CONTENT_POLICY_VIOLATION
    assert decision.metadata["category"] == "prohibited_phrase"
    assert scan("ordinary documentation", blocked_phrases={"internal-only-command"}).decision_type is DecisionType.ALLOW


def test_phrase_matching_is_case_insensitive_and_word_bounded():
    assert scan("INTERNAL   ONLY-COMMAND", blocked_phrases={"internal only-command"}).decision_type is DecisionType.BLOCK
    assert scan("internal-only-commands", blocked_phrases={"internal-only-command"}).decision_type is DecisionType.ALLOW


def test_custom_regex_and_rule_identifier():
    decision = scan("internal override", custom_patterns={"APP-001": r"internal\s+override"})
    assert decision.decision_type is DecisionType.BLOCK
    assert decision.rule_id == "APP-001"
    assert decision.metadata["category"] == "prohibited_pattern"


def test_field_specific_rule_does_not_apply_globally():
    guard = ContentGuard(field_rules={"comment": {"restricted phrase"}})
    assert guard.scan({"comment": "restricted phrase"}).decision_type is DecisionType.BLOCK
    assert guard.scan({"title": "restricted phrase"}).decision_type is DecisionType.ALLOW


def test_category_filtering():
    assert scan("restricted phrase", blocked_phrases={"restricted phrase"}, enabled_categories={"restricted_field"}).decision_type is DecisionType.ALLOW
    assert scan("internal override", custom_patterns={"APP-001": "internal override"}, enabled_categories={"prohibited_phrase"}).decision_type is DecisionType.ALLOW


@pytest.mark.parametrize(
    "value",
    [
        {"message": {"content": "blocked phrase"}},
        [{"content": "blocked phrase"}],
        ({"content": "blocked phrase"},),
    ],
)
def test_nested_values_are_scanned(value):
    assert scan(value, blocked_phrases={"blocked phrase"}).decision_type is DecisionType.BLOCK


def test_normalization_and_safe_diagnostics():
    matched = "restricted phrase"
    decision = scan("  RESTRICTED\u200b   PHRASE  ", blocked_phrases={matched})
    assert decision.decision_type is DecisionType.BLOCK
    encoded = decision.to_json()
    assert matched not in encoded
    assert decision.metadata["location"] == "$"


def test_multiple_matches_return_only_safe_metadata():
    decision = scan({"a": "first blocked", "b": "second blocked"}, blocked_phrases={"first blocked", "second blocked"})
    assert decision.decision_type is DecisionType.BLOCK
    assert "first blocked" not in decision.to_json()
    assert "second blocked" not in decision.to_json()


def test_depth_cycles_and_unsupported_values_fail_closed():
    assert scan({"a": {"b": "blocked phrase"}}, blocked_phrases={"blocked phrase"}, max_depth=0).decision_type is DecisionType.BLOCK
    cyclic = []
    cyclic.append(cyclic)
    assert scan(cyclic).decision_type is DecisionType.BLOCK
    assert scan(object()).decision_type is DecisionType.BLOCK


def test_malformed_configuration_and_runtime_error_fail_closed(monkeypatch):
    with pytest.raises(ValueError):
        ContentGuard(custom_patterns={"bad": "["})
    guard = ContentGuard(blocked_phrases={"blocked phrase"})
    monkeypatch.setattr(guard, "_scan", lambda *args: (_ for _ in ()).throw(RuntimeError("secret detail")))
    decision = guard.scan("ordinary")
    assert decision.reason is DecisionReason.CONTENT_POLICY_VIOLATION
    assert "secret detail" not in decision.to_json()


def _app(policy_type=DecisionType.ALLOW, permissions=None):
    permissions = permissions or PermissionEngine([
        PermissionRule("lookup-permission", PermissionCategory.READ, action="lookup")
    ])
    return PreventApp(
        policy=PolicyEngine([PolicyRule("lookup-policy", policy_type, action="lookup")]),
        permissions=permissions,
    )


def test_blocked_content_prevents_side_effect_and_exception_leak():
    calls = []
    app = _app()
    agent = app.add_agent(Agent("content-agent"))

    @agent.tool(content_guard=ContentGuard(blocked_phrases={"blocked phrase"}))
    def lookup(message):
        calls.append(message)

    app.run()
    with pytest.raises(ToolExecutionDenied) as error:
        lookup("blocked phrase")
    assert error.value.decision.reason is DecisionReason.CONTENT_POLICY_VIOLATION
    assert "blocked phrase" not in str(error.value)
    assert calls == []


def test_allowed_content_executes():
    app = _app()
    agent = app.add_agent(Agent("allowed-content-agent"))

    @agent.tool(content_guard=ContentGuard(blocked_phrases={"blocked phrase"}))
    def lookup(message):
        return message

    app.run()
    assert lookup("ordinary") == "ordinary"


def test_previous_security_decisions_short_circuit_content():
    blocked_app = _app(DecisionType.BLOCK)
    blocked_agent = blocked_app.add_agent(Agent("policy-agent"))

    @blocked_agent.tool(content_guard=ContentGuard(blocked_phrases={"blocked phrase"}))
    def lookup(message):
        return message

    blocked_app.run()
    with pytest.raises(ToolExecutionDenied) as error:
        lookup("blocked phrase")
    assert error.value.decision.reason is DecisionReason.REQUEST_BLOCKED

    denied_app = _app(permissions=PermissionEngine())
    denied_agent = denied_app.add_agent(Agent("permission-agent"))

    @denied_agent.tool("lookup", content_guard=ContentGuard(blocked_phrases={"blocked phrase"}))
    def denied(message):
        return message

    denied_app.run()
    with pytest.raises(ToolExecutionDenied) as error:
        denied("blocked phrase")
    assert error.value.decision.reason is DecisionReason.MISSING_PERMISSION


def test_approval_is_not_bypassed():
    app = _app(DecisionType.APPROVAL_REQUIRED)
    agent = app.add_agent(Agent("approval-content-agent"))

    @agent.tool(content_guard=ContentGuard(blocked_phrases={"blocked phrase"}))
    def lookup(message):
        return message

    app.run()
    with pytest.raises(ToolApprovalRequired):
        lookup("blocked phrase")
