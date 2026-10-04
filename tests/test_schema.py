from datetime import datetime, timezone

import pytest

from preventai import Agent, PreventApp, SchemaGuard, ToolApprovalRequired, ToolExecutionDenied
from preventai.core import (
    DecisionReason,
    DecisionType,
    PermissionCategory,
    PermissionEngine,
    PermissionRule,
    PolicyEngine,
    PolicyRule,
)


def result(schema, value):
    return SchemaGuard.validate(value, schema)


@pytest.mark.parametrize(
    ("schema", "value"),
    [
        ({"type": "string"}, "x"),
        ({"type": "integer"}, 1),
        ({"type": "number"}, 1.5),
        ({"type": "boolean"}, True),
        ({"type": "null"}, None),
        ({"type": "array", "items": {"type": "integer"}}, [1, 2]),
        ({"type": "object", "properties": {"x": {"type": "string"}}}, {"x": "ok"}),
    ],
)
def test_valid_schema_values(schema, value):
    assert result(schema, value).decision_type is DecisionType.ALLOW


@pytest.mark.parametrize(
    ("schema", "value"),
    [
        ({"type": "string"}, 1),
        ({"type": "integer"}, True),
        ({"type": "number"}, "1"),
        ({"type": "boolean"}, 1),
        ({"type": "string", "min_length": 2}, "x"),
        ({"type": "string", "max_length": 2}, "xxx"),
        ({"type": "string", "pattern": "^C[0-9]+$"}, "bad"),
        ({"type": "integer", "minimum": 2, "maximum": 4}, 5),
        ({"type": "array", "items": {"type": "integer"}}, [1, "bad"]),
        ({"type": "object", "required": ["x"]}, {}),
        ({"type": "object", "properties": {"x": {"type": "string"}}, "additional_properties": False}, {"x": "a", "y": 1}),
        ({"type": "string", "enum": ["read", "update"]}, "destroy"),
        ({"type": "object", "properties": {"x": {"type": "object", "required": ["y"]}}}, {"x": {}}),
    ],
)
def test_invalid_schema_values(schema, value):
    decision = result(schema, value)
    assert decision.decision_type is DecisionType.BLOCK
    assert decision.reason is DecisionReason.SCHEMA_VALIDATION_FAILED


@pytest.mark.parametrize(
    "schema",
    [
        None,
        {},
        {"type": "unknown"},
        {"type": "object", "required": "x"},
        {"type": "string", "pattern": "["},
    ],
)
def test_malformed_schema_fails_closed(schema):
    decision = result(schema, "x")
    assert decision.decision_type is DecisionType.BLOCK
    assert decision.reason is DecisionReason.SCHEMA_VALIDATION_FAILED


def test_validator_exception_fails_closed(monkeypatch):
    def broken(arguments, schema):
        raise RuntimeError("internal detail")

    monkeypatch.setattr(SchemaGuard, "validate", staticmethod(broken))
    app, agent, _ = make_app()

    @agent.tool(input_schema={"type": "object"})
    def lookup():
        return "unsafe"

    app.run()
    with pytest.raises(ToolExecutionDenied) as error:
        lookup()
    assert error.value.decision.decision_type is DecisionType.BLOCK
    assert error.value.decision.reason is DecisionReason.SCHEMA_VALIDATION_FAILED


def make_app(policy_decision=DecisionType.ALLOW, schema=None, permissions=None):
    policy = PolicyEngine([PolicyRule("rule", policy_decision, action="lookup")])
    if permissions is None:
        permissions = PermissionEngine([
            PermissionRule("permission", PermissionCategory.READ, action="lookup")
        ])
    app = PreventApp(policy=policy, permissions=permissions)
    agent = app.add_agent(Agent("schema-agent"))
    return app, agent, schema


def test_tool_schema_failure_prevents_side_effect():
    count = []
    schema = {"type": "object", "required": ["customer_id"]}
    app, agent, _ = make_app(schema=schema)

    @agent.tool(input_schema=schema)
    def lookup(customer_id):
        count.append(customer_id)

    app.run()
    with pytest.raises(ToolExecutionDenied) as error:
        lookup()
    assert error.value.decision.reason is DecisionReason.SCHEMA_VALIDATION_FAILED
    assert count == []


def test_valid_schema_tool_executes():
    app, agent, _ = make_app()

    @agent.tool(input_schema={"type": "object", "properties": {"customer_id": {"type": "string"}}})
    def lookup(customer_id):
        return customer_id

    app.run()
    assert lookup("C001") == "C001"


def test_policy_block_precedes_schema():
    count = []
    app, agent, _ = make_app(DecisionType.BLOCK)

    @agent.tool(input_schema={"type": "object", "required": ["x"]})
    def lookup(x):
        count.append(x)

    app.run()
    with pytest.raises(ToolExecutionDenied) as error:
        lookup()
    assert error.value.decision.reason is DecisionReason.REQUEST_BLOCKED
    assert count == []


def test_permission_block_precedes_schema():
    count = []
    app, agent, _ = make_app(permissions=PermissionEngine())

    @agent.tool(input_schema={"type": "object", "required": ["x"]})
    def lookup(x):
        count.append(x)

    app.run()
    with pytest.raises(ToolExecutionDenied) as error:
        lookup()
    assert error.value.decision.reason is DecisionReason.MISSING_PERMISSION
    assert count == []


def test_approval_precedes_schema():
    app, agent, _ = make_app(DecisionType.APPROVAL_REQUIRED)

    @agent.tool(input_schema={"type": "object", "required": ["x"]})
    def lookup(x):
        return x

    app.run()
    with pytest.raises(ToolApprovalRequired):
        lookup()
