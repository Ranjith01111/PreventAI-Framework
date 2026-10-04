from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from preventai import (
    Agent,
    ApprovalManager,
    AuditLogger,
    AuditPolicy,
    AuditEventType,
    DataFlowEngine,
    DataFlowPolicy,
    DecisionType,
    DataReleasePolicy,
    DataReleaseRequest,
    DestinationTrust,
    FlowClassification,
    PermissionCategory,
    PermissionEngine,
    PermissionRule,
    PolicyEngine,
    PolicyRule,
    PreventApp,
    ToolExecutionDenied,
    ToolApprovalRequired,
    Transformation,
    NetworkPolicy,
    NetworkRequest,
    DatabasePolicy,
    DatabaseRequest,
    MCPPolicy,
    MCPBoundary,
    Mission,
    evaluate_data_release,
)
from preventai.core.data_release import DataReleaseAuthorization, _RELEASE_AUTHORITY
from preventai.core.dataflow import DataFlowDestination, DataSource, FlowTrust


def request(**changes):
    values = dict(
        agent_id="agent",
        session_id="session",
        source="database.customer",
        source_classification=FlowClassification.PRIVATE,
        lineage=("database.customer",),
        destination="internal.sink",
        destination_trust=DestinationTrust.INTERNAL,
        operation="export",
        tool_id="tool",
        tool_name="export",
        policy_version=1,
    )
    values.update(changes)
    return DataReleaseRequest(**values)


def test_release_contract_rejects_unknown_classification_and_missing_lineage():
    with pytest.raises(ValueError):
        request(source_classification="private")
    with pytest.raises(ValueError):
        request(lineage=())


def test_release_authorization_rejects_mutated_framework_fields():
    release = request(
        mission_id="mission-1",
        mission_version=1,
        transformation=Transformation.REDACTION,
        target_classification=FlowClassification.PUBLIC,
        field_paths=("customer.name",),
        policy_version=7,
    )
    authorization = DataReleaseAuthorization.issue(
        release,
        application_id="app-1",
        policy_version=7,
        _authority=_RELEASE_AUTHORITY,
    )

    for field, value in (
        ("application_id", "app-2"),
        ("agent_id", "agent-2"),
        ("session_id", "session-2"),
        ("tool_id", "tool-2"),
        ("operation", "delete"),
        ("tool_name", "delete"),
        ("source", "database.customer_backup"),
        ("destination", "internal.other"),
        ("field_paths", ("customer.ssn",)),
        ("transformation", Transformation.MASKING),
        ("target_classification", FlowClassification.INTERNAL),
        ("mission_id", "mission-2"),
        ("mission_version", 2),
        ("policy_version", 8),
    ):
        forged = DataReleaseAuthorization.issue(
            release,
            application_id="app-1",
            policy_version=7,
            _authority=_RELEASE_AUTHORITY,
        )
        object.__setattr__(forged, field, value)
        assert not forged.matches(release, application_id="app-1", policy_version=7)
        assert not forged.matches(release, application_id="app-1")


def test_private_data_requires_explicit_destination_and_declassification():
    policy = DataReleasePolicy(
        allowed_destinations=("internal.sink",),
        allowed_destination_trust=(DestinationTrust.INTERNAL,),
        allowed_classifications=(FlowClassification.PUBLIC,),
        allowed_transformations=(Transformation.REDACTION,),
    )
    assert evaluate_data_release(request(), policy).decision_type.value == "block"
    allowed = evaluate_data_release(
        request(
            transformation=Transformation.REDACTION,
            target_classification=FlowClassification.PUBLIC,
        ),
        policy,
    )
    assert allowed.decision_type.value == "allow"


@pytest.mark.parametrize(
    "operation",
    ["json_serialization", "string_conversion", "base64", "compression", "copy", "message_wrap"],
)
def test_application_transformations_do_not_implicitly_declassify(operation):
    policy = DataReleasePolicy(
        allowed_destinations=("internal.sink",),
        allowed_destination_trust=(DestinationTrust.INTERNAL,),
        allowed_classifications=(FlowClassification.PUBLIC,),
    )
    decision = evaluate_data_release(
        request(operation=operation, source_classification=FlowClassification.PRIVATE),
        policy,
    )
    assert decision.decision_type is DecisionType.BLOCK
    assert decision.metadata["error"] == "declassification_denied"


def test_managed_release_audit_emits_evaluation_and_denial_without_sensitive_data():
    audit = AuditLogger()
    app = PreventApp(
        audit_logger=audit,
        policy=PolicyEngine([PolicyRule("allow", DecisionType.ALLOW, action="*")]),
        permissions=PermissionEngine([
            PermissionRule("read", PermissionCategory.READ, action="*", resource="*"),
        ]),
        data_release_policy=DataReleasePolicy(allowed_destinations=("other.sink",)),
    )
    agent = app.add_agent(Agent("release-audit-agent"))

    @agent.tool(
        "export",
        data_source=DataSource("private-db", FlowClassification.PRIVATE, FlowTrust.TRUSTED),
        data_destination=DataFlowDestination("internal.sink"),
    )
    def export():
        return "secret-payload"

    app.run()
    with pytest.raises(ToolExecutionDenied):
        export()
    event_types = [event.event_type for event in audit.events()]
    assert AuditEventType.DATA_RELEASE_EVALUATED in event_types
    assert AuditEventType.DATA_DESTINATION_DENIED in event_types
    serialized = str([event.to_dict() for event in audit.events()])
    assert "secret-payload" not in serialized


def test_required_release_audit_failure_blocks_before_callable():
    audit = AuditLogger(policy=AuditPolicy(required=True))

    def fail_record(*args, **kwargs):
        raise OSError("audit unavailable")

    app = PreventApp(
        audit_logger=audit,
        policy=PolicyEngine([PolicyRule("allow", DecisionType.ALLOW, action="*")]),
        permissions=PermissionEngine([
            PermissionRule("read", PermissionCategory.READ, action="*", resource="*"),
        ]),
        data_release_policy=DataReleasePolicy(allowed_destinations=("other.sink",)),
    )
    agent = app.add_agent(Agent("release-audit-failure-agent"))
    calls = []

    @agent.tool(
        "export",
        data_source=DataSource("private-db", FlowClassification.PRIVATE, FlowTrust.TRUSTED),
        data_destination=DataFlowDestination("internal.sink"),
    )
    def export():
        calls.append(True)
        return "secret"

    app.run()
    audit.record = fail_record
    with pytest.raises(ToolExecutionDenied):
        export()
    assert calls == []


def test_release_approval_required_stops_before_callable():
    policy = DataReleasePolicy(
        allowed_destinations=("internal.sink",),
        allowed_classifications=(FlowClassification.PRIVATE,),
        require_approval_for=(FlowClassification.PRIVATE,),
    )
    app = PreventApp(
        policy=PolicyEngine([PolicyRule("allow", DecisionType.ALLOW, action="*")]),
        permissions=PermissionEngine([
            PermissionRule("read", PermissionCategory.READ, action="*", resource="*"),
        ]),
        data_release_policy=policy,
    )
    agent = app.add_agent(Agent("release-approval-agent"))
    calls = []

    @agent.tool(
        "export",
        data_source=DataSource("private-db", FlowClassification.PRIVATE, FlowTrust.TRUSTED),
        data_destination=DataFlowDestination("internal.sink"),
    )
    def export():
        calls.append(True)
        return "secret"

    app.run()
    with pytest.raises(ToolApprovalRequired):
        export()
    assert calls == []


def test_release_approval_is_exact_and_revalidated_at_final_dispatch():
    manager = ApprovalManager("release-test-secret")
    policy = DataReleasePolicy(
        allowed_destinations=("internal.sink",),
        allowed_classifications=(FlowClassification.PRIVATE,),
        require_approval_for=(FlowClassification.PRIVATE,),
    )
    app = PreventApp(
        approval_manager=manager,
        policy=PolicyEngine([PolicyRule("allow", DecisionType.ALLOW, action="*")]),
        permissions=PermissionEngine([
            PermissionRule("read", PermissionCategory.READ, action="*", resource="*"),
        ]),
        data_release_policy=policy,
    )
    agent = app.add_agent(Agent("release-approval-revalidated-agent"))

    @agent.tool(
        "export",
        data_source=DataSource("private-db", FlowClassification.PRIVATE, FlowTrust.TRUSTED),
        data_destination=DataFlowDestination("internal.sink"),
    )
    def export():
        return "approved"

    app.run()
    with pytest.raises(ToolApprovalRequired):
        export()
    pending = export.pending_release_approval_request
    assert pending is not None
    approval = manager.approve(pending, approved_by="reviewer")
    export.use_release_approval(approval)
    assert export() == "approved"

    changed = replace(
        request(
            destination="other.sink",
            source_classification=FlowClassification.PRIVATE,
        ),
        agent_id=agent.agent_id,
        session_id=agent.session_id,
        tool_id=export.tool_id,
        tool_name=export.name,
    )
    assert manager.verify_release(
        approval,
        changed,
        authenticated_identity=agent.agent_id,
        consume=False,
    ).decision_type is DecisionType.BLOCK


def test_release_approval_is_blocked_when_policy_changes_before_execution():
    manager = ApprovalManager("release-policy-change-secret")
    app = PreventApp(
        approval_manager=manager,
        policy=PolicyEngine([PolicyRule("allow", DecisionType.ALLOW, action="*")]),
        permissions=PermissionEngine([
            PermissionRule("read", PermissionCategory.READ, action="*", resource="*"),
        ]),
        data_release_policy=DataReleasePolicy(
            allowed_destinations=("internal.sink",),
            allowed_classifications=(FlowClassification.PRIVATE,),
            require_approval_for=(FlowClassification.PRIVATE,),
        ),
    )
    agent = app.add_agent(Agent("release-policy-change-agent"))
    calls = []

    @agent.tool(
        "export",
        data_source=DataSource("private-db", FlowClassification.PRIVATE, FlowTrust.TRUSTED),
        data_destination=DataFlowDestination("internal.sink"),
    )
    def export():
        calls.append(True)
        return "approved"

    app.run()
    with pytest.raises(ToolApprovalRequired):
        export()
    approval = manager.approve(
        export.pending_release_approval_request,
        approved_by="reviewer",
    )
    export.use_release_approval(approval)
    export._data_release_policy = DataReleasePolicy(
        version=2,
        allowed_destinations=("internal.sink",),
        allowed_classifications=(FlowClassification.PRIVATE,),
        require_approval_for=(FlowClassification.PRIVATE,),
    )
    with pytest.raises(ToolExecutionDenied):
        export()
    assert calls == []


def test_app_issues_release_authorization_for_mcp_sink():
    calls = []
    app = PreventApp(
        data_release_policy=DataReleasePolicy(
            allowed_destinations=("preventai-server",),
            allowed_classifications=(FlowClassification.PUBLIC,),
        )
    )
    agent = app.add_agent(Agent("mcp-release-agent"))
    mission = Mission(
        mission_id="mission",
        agent_id=agent.agent_id,
        action="tools/call",
        resource="echo",
        tool="echo",
        expires_at=datetime.now(timezone.utc) + timedelta(minutes=5),
    )
    app.missions.register(mission)
    app.bind_mission(agent, "mission", 1)
    boundary = MCPBoundary(
        MCPPolicy(allowed_tools=("echo",)),
        lambda request: (calls.append(request), {"ok": True})[1],
        policy_check=lambda request: True,
        permission_check=lambda request: True,
        delegation_check=lambda request: True,
        dataflow_check=lambda request: True,
        runtime_check=lambda request: True,
        approval_check=lambda request: True,
    )
    app.bind_mcp(boundary)
    request = boundary.parse(
        {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
         "params": {"name": "echo", "arguments": {}}},
        protocol_version="2025-06-18",
        client_id="client",
        session_id=agent.session_id,
    )
    release_request = DataReleaseRequest(
        agent_id=agent.agent_id,
        session_id=agent.session_id,
        source="public-source",
        source_classification=FlowClassification.PUBLIC,
        lineage=("public-source",),
        destination="preventai-server",
        destination_trust=DestinationTrust.INTERNAL,
        operation="tools/call",
        tool_id="mcp-tool",
        tool_name="echo",
    )
    result = boundary.dispatch(
        request,
        boundary.issue(request),
        release_request=release_request,
    )
    assert result.result == {"ok": True}
    assert len(calls) == 1
    release_authorization = app.authorize_data_release(release_request)
    app._data_release_policy = DataReleasePolicy(
        version=2,
        allowed_destinations=("preventai-server",),
        allowed_classifications=(FlowClassification.PUBLIC,),
    )
    with pytest.raises(Exception):
        boundary.dispatch(
            request,
            boundary.issue(request),
            release_authorization=release_authorization,
            release_request=release_request,
        )
    assert len(calls) == 1


def test_public_destination_is_not_authorized_by_serialization():
    policy = DataReleasePolicy(
        allowed_destination_trust=(DestinationTrust.PUBLIC,),
        allowed_classifications=(FlowClassification.PUBLIC,),
    )
    assert evaluate_data_release(
        request(destination="public", destination_trust=DestinationTrust.PUBLIC),
        policy,
    ).decision_type.value == "block"


def test_managed_tool_denies_release_before_callable():
    policy = PolicyEngine([PolicyRule("allow", DecisionType.ALLOW, action="*")])
    permissions = PermissionEngine([
        PermissionRule("read", PermissionCategory.READ, action="*", resource="*"),
    ])
    app = PreventApp(
        policy=policy,
        permissions=permissions,
        data_release_policy=DataReleasePolicy(
            allowed_destinations=("other.sink",),
        ),
    )
    agent = app.add_agent(Agent("release-agent"))
    calls = []

    @agent.tool(
        "export",
        data_source=DataSource("db", FlowClassification.PRIVATE, FlowTrust.TRUSTED),
        data_destination=DataFlowDestination("internal.sink"),
    )
    def export():
        calls.append(True)
        return "secret"

    app.run()
    with pytest.raises(ToolExecutionDenied):
        export()
    assert calls == []


def test_release_evaluation_is_thread_safe():
    policy = DataReleasePolicy()
    with ThreadPoolExecutor(max_workers=4) as pool:
        decisions = list(pool.map(lambda _: evaluate_data_release(request(
            source_classification=FlowClassification.PUBLIC,
        ), policy), range(12)))
    assert all(decision.decision_type.value == "allow" for decision in decisions)


def test_field_restriction_is_policy_controlled():
    policy = DataReleasePolicy(
        allowed_destinations=("internal.sink",),
        allowed_field_paths=("id",),
    )
    assert evaluate_data_release(
        request(source_classification=FlowClassification.PUBLIC, field_paths=("email",)),
        policy,
    ).metadata["error"] == "field_restriction_denied"
    assert evaluate_data_release(
        request(source_classification=FlowClassification.PUBLIC, field_paths=("id",)),
        policy,
    ).decision_type is DecisionType.ALLOW


def test_managed_network_release_denial_prevents_connection(monkeypatch):
    app = PreventApp(
        policy=PolicyEngine([PolicyRule("allow", DecisionType.ALLOW, action="*")]),
        permissions=PermissionEngine([
            PermissionRule("network", PermissionCategory.NETWORK, action="*", resource="*"),
        ]),
        data_flow=DataFlowEngine([DataFlowPolicy("allow", DecisionType.ALLOW)]),
        data_release_policy=DataReleasePolicy(allowed_destinations=("internal.sink",)),
    )
    agent = app.add_agent(Agent("network-release-agent"))
    calls = []

    @agent.tool(
        "fetch",
        network_policy=NetworkPolicy(allowed_hosts=("example.com",)),
        data_source=DataSource("private-db", FlowClassification.PRIVATE, FlowTrust.TRUSTED),
        data_destination=DataFlowDestination("internal.sink"),
    )
    def fetch():
        return NetworkRequest("https://example.com/data")

    app.run()
    network = agent._tools["fetch"]._network
    monkeypatch.setattr(
        network,
        "_request_once",
        lambda *args, **kwargs: calls.append(True),
    )
    with pytest.raises(ToolExecutionDenied):
        fetch()
    assert calls == []
