import pytest

from preventai import (
    Agent,
    OutputGuard,
    PreventApp,
    ToolExecutionDenied,
)
from preventai.core import (
    DecisionReason,
    DecisionType,
    PermissionCategory,
    PermissionEngine,
    PermissionRule,
    PolicyEngine,
    PolicyRule,
)


def test_valid_output_types_and_schema():
    assert OutputGuard().validate("ok").decision_type is DecisionType.ALLOW
    assert OutputGuard().validate({"status": "ok"}).decision_type is DecisionType.ALLOW
    assert OutputGuard().validate(["ok"]).decision_type is DecisionType.ALLOW
    guard = OutputGuard(
        output_schema={
            "type": "object",
            "properties": {"status": {"type": "string"}},
            "required": ["status"],
            "additional_properties": False,
        }
    )
    assert guard.validate({"status": "ok"}).decision_type is DecisionType.ALLOW
    assert guard.validate({}).reason is DecisionReason.OUTPUT_SCHEMA_VALIDATION_FAILED
    assert guard.validate({"status": "ok", "extra": True}).decision_type is DecisionType.BLOCK
    assert guard.validate({"status": 1}).decision_type is DecisionType.BLOCK


def test_secrets_and_pii_policies_are_deterministic_and_safe():
    secret = OutputGuard().validate({"api_key": "output-secret"})
    assert secret.decision_type is DecisionType.BLOCK
    assert "output-secret" not in secret.to_json()
    assert OutputGuard(pii_policy="block").validate(
        {"email": "alice@example.com"}
    ).decision_type is DecisionType.BLOCK
    assert OutputGuard(pii_policy="warn").validate(
        {"email": "alice@example.com"}
    ).decision_type is DecisionType.WARN
    assert OutputGuard(pii_policy="allow").validate(
        {"email": "alice@example.com"}
    ).decision_type is DecisionType.ALLOW
    assert "alice@example.com" not in OutputGuard(pii_policy="warn").validate(
        {"email": "alice@example.com"}
    ).to_json()


def test_explicit_output_rules_and_size():
    guard = OutputGuard(
        blocked_phrases={"internal-only"},
        custom_patterns={"OUT-001": r"restricted\s+value"},
        max_output_bytes=10,
    )
    assert guard.validate("internal-only").reason is DecisionReason.OUTPUT_POLICY_VIOLATION
    assert guard.validate("restricted value").decision_type is DecisionType.BLOCK
    assert guard.validate("12345678901").decision_type is DecisionType.BLOCK
    assert OutputGuard(max_output_bytes=10).validate("small").decision_type is DecisionType.ALLOW


def test_unsupported_objects_and_cycles_fail_closed():
    assert OutputGuard().validate(object()).reason is DecisionReason.OUTPUT_POLICY_VIOLATION
    cyclic = []
    cyclic.append(cyclic)
    decision = OutputGuard().validate(cyclic)
    assert decision.decision_type is DecisionType.BLOCK


def test_output_guard_exception_fails_closed(monkeypatch):
    guard = OutputGuard()
    monkeypatch.setattr(guard, "_serialized_size", lambda value: (_ for _ in ()).throw(RuntimeError("secret")))
    decision = guard.validate("ordinary")
    assert decision.reason is DecisionReason.OUTPUT_SECURITY_ERROR
    assert "secret" not in decision.to_json()


def _app():
    return PreventApp(
        policy=PolicyEngine([PolicyRule("allow", DecisionType.ALLOW, action="lookup")]),
        permissions=PermissionEngine([
            PermissionRule("read", PermissionCategory.READ, action="lookup")
        ]),
    )


def test_tool_output_is_validated_and_blocked_result_never_returns():
    calls = []
    app = _app()
    agent = app.add_agent(Agent("output-agent"))

    @agent.tool(output_guard=OutputGuard(blocked_phrases={"blocked output"}))
    def lookup(value):
        calls.append(value)
        return {"message": "blocked output"}

    app.run()
    with pytest.raises(ToolExecutionDenied) as error:
        lookup("ordinary")
    assert error.value.decision.reason is DecisionReason.OUTPUT_POLICY_VIOLATION
    assert "blocked output" not in str(error.value)
    assert calls == ["ordinary"]


def test_tool_without_explicit_guard_uses_secure_default():
    calls = []
    app = _app()
    agent = app.add_agent(Agent("default-output-agent"))

    @agent.tool()
    def lookup(value):
        calls.append(value)
        return {"api_key": "default-secret"}

    app.run()
    with pytest.raises(ToolExecutionDenied) as error:
        lookup("ordinary")
    assert error.value.decision.reason is DecisionReason.OUTPUT_POLICY_VIOLATION
    assert "default-secret" not in str(error.value)
    assert calls == ["ordinary"]


def test_explicit_output_guard_is_selected_once(monkeypatch):
    calls = []
    guard = OutputGuard()
    original_validate = guard.validate

    def counted_validate(result, context=None):
        calls.append(result)
        return original_validate(result, context)

    monkeypatch.setattr(guard, "validate", counted_validate)
    app = _app()
    agent = app.add_agent(Agent("single-output-guard-agent"))

    @agent.tool(output_guard=guard)
    def lookup(value):
        return {"status": value}

    app.run()
    assert lookup("ok") == {"status": "ok"}
    assert calls == [{"status": "ok"}]


def test_explicit_custom_guard_overrides_default_and_preserves_policy():
    app = _app()
    agent = app.add_agent(Agent("custom-output-policy-agent"))

    @agent.tool(output_guard=OutputGuard(pii_policy="block"))
    def lookup(value):
        return {"email": value}

    app.run()
    with pytest.raises(ToolExecutionDenied) as error:
        lookup("alice@example.com")
    assert error.value.decision.reason is DecisionReason.PII_DETECTED


def test_explicit_output_schema_remains_enforced():
    app = _app()
    agent = app.add_agent(Agent("custom-output-schema-agent"))

    @agent.tool(
        output_schema={
            "type": "object",
            "properties": {"status": {"type": "string"}},
            "required": ["status"],
            "additional_properties": False,
        }
    )
    def lookup(value):
        return {"unexpected": value}

    app.run()
    with pytest.raises(ToolExecutionDenied) as error:
        lookup("bad")
    assert error.value.decision.reason is DecisionReason.OUTPUT_SCHEMA_VALIDATION_FAILED


def test_explicit_output_limit_and_unsupported_output_fail_closed():
    app = _app()
    agent = app.add_agent(Agent("output-limit-agent"))

    @agent.tool("lookup", output_guard=OutputGuard(max_output_bytes=5))
    def oversized(value):
        return value

    app.run()
    with pytest.raises(ToolExecutionDenied) as error:
        oversized("too large")
    assert error.value.decision.reason is DecisionReason.OUTPUT_POLICY_VIOLATION

    app = _app()
    agent = app.add_agent(Agent("unsupported-output-agent"))

    @agent.tool("lookup")
    def unsupported(value):
        return object()

    app.run()
    with pytest.raises(ToolExecutionDenied) as error:
        unsupported("ordinary")
    assert error.value.decision.reason is DecisionReason.OUTPUT_POLICY_VIOLATION


def test_output_security_exception_fails_closed(monkeypatch):
    guard = OutputGuard()
    monkeypatch.setattr(
        guard,
        "validate",
        lambda result, context=None: (_ for _ in ()).throw(RuntimeError("internal output")),
    )
    app = _app()
    agent = app.add_agent(Agent("output-error-agent"))

    @agent.tool(output_guard=guard)
    def lookup(value):
        return {"status": value}

    app.run()
    with pytest.raises(ToolExecutionDenied) as error:
        lookup("ordinary")
    assert error.value.decision.reason is DecisionReason.OUTPUT_SECURITY_ERROR
    assert "internal output" not in str(error.value)


def test_tool_returns_valid_output_unchanged():
    app = _app()
    agent = app.add_agent(Agent("valid-output-agent"))

    @agent.tool(
        output_schema={
            "type": "object",
            "properties": {"status": {"type": "string"}},
            "required": ["status"],
            "additional_properties": False,
        }
    )
    def lookup(value):
        return {"status": value}

    app.run()
    assert lookup("ok") == {"status": "ok"}


def test_input_block_precedes_output_guard():
    app = PreventApp(
        policy=PolicyEngine([PolicyRule("block", DecisionType.BLOCK, action="lookup")]),
        permissions=PermissionEngine([
            PermissionRule("read", PermissionCategory.READ, action="lookup")
        ]),
    )
    agent = app.add_agent(Agent("input-block-agent"))

    @agent.tool(output_guard=OutputGuard(blocked_phrases={"blocked output"}))
    def lookup(value):
        return "blocked output"

    app.run()
    with pytest.raises(ToolExecutionDenied) as error:
        lookup("ordinary")
    assert error.value.decision.reason is DecisionReason.REQUEST_BLOCKED
