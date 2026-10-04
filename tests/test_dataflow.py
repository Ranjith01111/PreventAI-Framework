import pytest

from preventai import (
    Agent,
    DataFlow,
    DataFlowDestination,
    DataFlowEngine,
    DataFlowPolicy,
    DataSource,
    DecisionReason,
    DecisionType,
    FlowClassification,
    FlowTrust,
    PreventApp,
    ToolExecutionDenied,
    propagate_classification,
    propagate_trust,
)
from preventai.core import (
    PermissionCategory,
    PermissionEngine,
    PermissionRule,
    PolicyEngine,
    PolicyRule,
)


def flow(
    classification=FlowClassification.PUBLIC,
    trust=FlowTrust.TRUSTED,
    source="source",
    destination="internal.logging",
    action="write",
    **kwargs,
):
    return DataFlow(
        DataSource(source, classification, trust),
        DataFlowDestination(destination),
        action,
        **kwargs,
    )


def test_basic_flows_and_explicit_policies():
    engine = DataFlowEngine([
        DataFlowPolicy("public-external", DecisionType.ALLOW, classification=FlowClassification.PUBLIC, destination="external.*"),
        DataFlowPolicy("private-internal", DecisionType.ALLOW, classification=FlowClassification.PRIVATE, destination="internal.*"),
        DataFlowPolicy("private-external", DecisionType.BLOCK, classification=FlowClassification.PRIVATE, destination="external.*"),
        DataFlowPolicy("secret-external", DecisionType.BLOCK, classification=FlowClassification.SECRET, destination="external.*"),
        DataFlowPolicy("secret-internal", DecisionType.ALLOW, classification=FlowClassification.SECRET, destination="internal.*"),
    ])
    assert engine.check(flow()).decision_type is DecisionType.BLOCK
    assert engine.check(flow(destination="external.api")).decision_type is DecisionType.ALLOW
    assert engine.check(flow(FlowClassification.PRIVATE)).decision_type is DecisionType.ALLOW
    assert engine.check(flow(FlowClassification.PRIVATE, destination="external.email")).decision_type is DecisionType.BLOCK
    assert engine.check(flow(FlowClassification.SECRET, destination="external.email")).decision_type is DecisionType.BLOCK
    assert engine.check(flow(FlowClassification.SECRET)).decision_type is DecisionType.ALLOW


def test_matching_and_precedence():
    engine = DataFlowEngine([
        DataFlowPolicy("wildcard", DecisionType.ALLOW, destination="external.*", priority=1),
        DataFlowPolicy("agent-block", DecisionType.BLOCK, destination="external.*", agent="agent-1", priority=2),
        DataFlowPolicy("tool-block", DecisionType.BLOCK, destination="external.*", tool="email", priority=3),
    ])
    assert engine.check(flow(destination="external.api", agent="agent-1")).reason is DecisionReason.DATA_FLOW_VIOLATION
    assert engine.check(flow(destination="external.api", agent="agent-2", tool="email")).rule_id == "tool-block"
    assert engine.check(flow(destination="external.api", agent="agent-2", tool="other")).rule_id == "wildcard"


def test_conflict_and_missing_policy_fail_closed():
    conflict = DataFlowEngine([
        DataFlowPolicy("allow", DecisionType.ALLOW, destination="external.api", priority=5),
        DataFlowPolicy("block", DecisionType.BLOCK, destination="external.api", priority=5),
    ]).check(flow(destination="external.api"))
    assert conflict.decision_type is DecisionType.BLOCK
    assert conflict.metadata["category"] == "conflicting_policies"
    missing = DataFlowEngine().check(flow())
    assert missing.decision_type is DecisionType.BLOCK
    assert missing.reason is DecisionReason.DATA_FLOW_NO_POLICY


@pytest.mark.parametrize(
    "bad",
    [
        lambda: DataFlow(None, DataFlowDestination("internal"), "read"),
        lambda: DataFlow(DataSource("s", FlowClassification.PUBLIC, FlowTrust.TRUSTED), None, "read"),
        lambda: DataSource("", FlowClassification.PUBLIC, FlowTrust.TRUSTED),
        lambda: DataSource("s", "public", FlowTrust.TRUSTED),
    ],
)
def test_malformed_flow_information_fails_closed(bad):
    with pytest.raises((ValueError, TypeError)):
        bad()
    decision = DataFlowEngine().check(object())
    assert decision.decision_type is DecisionType.BLOCK


def test_label_and_trust_propagation_is_conservative():
    assert propagate_classification(FlowClassification.PUBLIC, FlowClassification.PRIVATE) is FlowClassification.PRIVATE
    assert propagate_classification(FlowClassification.PRIVATE, FlowClassification.SECRET) is FlowClassification.SECRET
    assert propagate_trust(FlowTrust.TRUSTED, FlowTrust.UNTRUSTED) is FlowTrust.UNTRUSTED
    assert propagate_trust(FlowTrust.TRUSTED, FlowTrust.TRUSTED) is FlowTrust.TRUSTED


def test_flow_decision_does_not_leak_values():
    decision = DataFlowEngine([
        DataFlowPolicy("block", DecisionType.BLOCK, destination="external.*")
    ]).check(flow(FlowClassification.SECRET, source="private-secret-source", destination="external.api"))
    encoded = decision.to_json()
    assert "private-secret-source" not in encoded
    assert "external.api" not in encoded


def _app(data_flow):
    return PreventApp(
        policy=PolicyEngine([PolicyRule("allow", DecisionType.ALLOW, action="send")]),
        permissions=PermissionEngine([
            PermissionRule("read", PermissionCategory.READ, action="send")
        ]),
        data_flow=data_flow,
    )


def test_framework_flow_boundary_blocks_external_and_allows_internal():
    engine = DataFlowEngine([
        DataFlowPolicy("private-internal", DecisionType.ALLOW, classification=FlowClassification.PRIVATE, destination="internal.*"),
        DataFlowPolicy("private-external", DecisionType.BLOCK, classification=FlowClassification.PRIVATE, destination="external.*"),
    ])
    calls = []
    app = _app(engine)
    agent = app.add_agent(Agent("flow-agent"))

    @agent.tool(
        "send",
        data_source=DataSource("customer-source", FlowClassification.PRIVATE, FlowTrust.TRUSTED),
        data_destination=DataFlowDestination("external.email"),
    )
    def send(value):
        calls.append(value)
        return "sent"

    app.run()
    with pytest.raises(ToolExecutionDenied) as error:
        send("value")
    assert error.value.decision.reason is DecisionReason.DATA_FLOW_VIOLATION
    assert calls == []

    internal_app = _app(engine)
    internal_agent = internal_app.add_agent(Agent("internal-flow-agent"))

    @internal_agent.tool(
        "send",
        data_source=DataSource("customer-source", FlowClassification.PRIVATE, FlowTrust.TRUSTED),
        data_destination=DataFlowDestination("internal.logging"),
    )
    def internal_send(value):
        return value

    internal_app.run()
    assert internal_send("value") == "value"


def test_configured_flow_requires_endpoints_and_prior_controls_precede_it():
    engine = DataFlowEngine([
        DataFlowPolicy("allow", DecisionType.ALLOW, destination="internal.*")
    ])
    app = _app(engine)
    agent = app.add_agent(Agent("missing-endpoint-agent"))

    @agent.tool("send")
    def send(value):
        return value

    app.run()
    with pytest.raises(ToolExecutionDenied) as error:
        send("value")
    assert error.value.decision.reason is DecisionReason.DATA_FLOW_VIOLATION

    blocked_app = PreventApp(
        policy=PolicyEngine([PolicyRule("block", DecisionType.BLOCK, action="send")]),
        permissions=PermissionEngine([
            PermissionRule("read", PermissionCategory.READ, action="send")
        ]),
        data_flow=engine,
    )
    blocked_agent = blocked_app.add_agent(Agent("blocked-before-flow"))

    @blocked_agent.tool(
        "send",
        data_source=DataSource("source", FlowClassification.PRIVATE, FlowTrust.TRUSTED),
        data_destination=DataFlowDestination("external.email"),
    )
    def blocked_send(value):
        return value

    blocked_app.run()
    with pytest.raises(ToolExecutionDenied) as error:
        blocked_send("value")
    assert error.value.decision.reason is DecisionReason.REQUEST_BLOCKED
