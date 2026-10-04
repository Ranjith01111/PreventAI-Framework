from datetime import datetime, timedelta, timezone
import time

import pytest

from _filesystem_test_support import instrument_filesystem, make_filesystem_tool
from preventai import (
    AuditLogger,
    Authenticator,
    ApprovalPolicy,
    ApprovalRule,
    CircuitBreaker,
    Capability,
    DataFlowEngine,
    DataFlowPolicy,
    DecisionType,
    DataReleasePolicy,
    DestinationTrust,
    FlowClassification,
    FlowTrust,
    RuntimePolicy,
    RateLimit,
    LoopPolicy,
    RuntimeBudget,
    Timeout,
    ToolApprovalRequired,
    ToolExecutionDenied,
)
from preventai.core import PermissionCategory, PermissionEngine, PermissionRule


def test_approval_required_without_approval_stops_before_operation(tmp_path, monkeypatch):
    app, _, tool = make_filesystem_tool(
        tmp_path,
        approval_policy=ApprovalPolicy([ApprovalRule("approve", action="filesystem.*")]),
    )
    operations = instrument_filesystem(tool, monkeypatch)
    with pytest.raises(ToolApprovalRequired):
        tool(str(tmp_path / "item"), "create", data=b"x")
    assert operations == []


def test_authorized_data_release_reaches_filesystem_backend(tmp_path, monkeypatch):
    app, _, tool = make_filesystem_tool(
        tmp_path,
        source_classification=FlowClassification.INTERNAL,
        data_release_policy=DataReleasePolicy(
            allowed_destinations=("sink", str(tmp_path / "item")),
            allowed_destination_trust=(DestinationTrust.INTERNAL,),
            allowed_classifications=(FlowClassification.INTERNAL,),
        ),
    )
    operations = instrument_filesystem(tool, monkeypatch)
    tool(str(tmp_path / "item"), "create", data=b"authorized")
    assert operations == ["write"]
    assert (tmp_path / "item").read_bytes() == b"authorized"


def test_valid_approval_does_not_bypass_file_permission(tmp_path, monkeypatch):
    from preventai import ApprovalManager

    manager = ApprovalManager("test-secret")
    _, _, tool = make_filesystem_tool(
        tmp_path,
        approval_policy=ApprovalPolicy([ApprovalRule("approve", action="filesystem.*")]),
        approval_manager=manager,
    )
    operations = instrument_filesystem(tool, monkeypatch)
    with pytest.raises(ToolApprovalRequired):
        tool(str(tmp_path / "item"), "create", data=b"x")
    pending = tool.pending_approval_request
    approval = manager.approve(pending, approved_by="human")
    tool.use_approval(approval)
    tool._engine._permissions = PermissionEngine([
        PermissionRule("read-tool", PermissionCategory.READ, action="operate"),
    ])
    operations.clear()
    with pytest.raises(ToolExecutionDenied):
        tool(str(tmp_path / "item"), "create", data=b"x")
    assert operations == []


def test_approved_request_still_obeys_dataflow_denial(tmp_path, monkeypatch):
    from preventai import ApprovalManager

    manager = ApprovalManager("test-secret")
    _, _, tool = make_filesystem_tool(
        tmp_path,
        approval_policy=ApprovalPolicy([ApprovalRule("approve", action="filesystem.*")]),
        approval_manager=manager,
    )
    operations = instrument_filesystem(tool, monkeypatch)
    with pytest.raises(ToolApprovalRequired):
        tool(str(tmp_path / "item"), "create", data=b"x")
    pending = tool.pending_approval_request
    tool.use_approval(manager.approve(pending, approved_by="human"))
    tool._data_flow_engine = DataFlowEngine([
        DataFlowPolicy("block-filesystem", DecisionType.BLOCK, destination="filesystem:*")
    ])
    operations.clear()
    with pytest.raises(ToolExecutionDenied):
        tool(str(tmp_path / "item"), "create", data=b"x")
    assert operations == []


def test_filesystem_approval_is_bound_to_exact_path_and_request(tmp_path, monkeypatch):
    from preventai import ApprovalManager

    manager = ApprovalManager("test-secret")
    _, _, tool = make_filesystem_tool(
        tmp_path,
        approval_policy=ApprovalPolicy([ApprovalRule("approve", action="filesystem.*")]),
        approval_manager=manager,
    )
    with pytest.raises(ToolApprovalRequired):
        tool(str(tmp_path / "first"), "create", data=b"first")
    approval = manager.approve(tool.pending_approval_request, approved_by="human")
    tool.use_approval(approval)
    operations = instrument_filesystem(tool, monkeypatch)
    with pytest.raises(ToolExecutionDenied):
        tool(str(tmp_path / "second"), "create", data=b"second")
    assert operations == []
    assert not (tmp_path / "second").exists()


def test_valid_approval_does_not_bypass_runtime_limit(tmp_path, monkeypatch):
    from preventai import ApprovalManager

    manager = ApprovalManager("test-secret")
    _, _, tool = make_filesystem_tool(
        tmp_path,
        approval_policy=ApprovalPolicy([ApprovalRule("approve", action="filesystem.*")]),
        approval_manager=manager,
        runtime_policy=RuntimePolicy(max_tool_calls=1),
    )
    operations = instrument_filesystem(tool, monkeypatch)
    with pytest.raises(ToolApprovalRequired):
        tool(str(tmp_path / "item"), "create", data=b"x")
    tool.use_approval(
        manager.approve(tool.pending_approval_request, approved_by="human")
    )
    operations.clear()
    with pytest.raises(ToolExecutionDenied):
        tool(str(tmp_path / "item"), "create", data=b"x")
    assert operations == []
    assert not (tmp_path / "item").exists()


def test_filesystem_read_output_guard_blocks_secret_content(tmp_path, monkeypatch):
    path = tmp_path / "credential.txt"
    path.write_text("AKIAABCDEFGHIJKLMNOP", encoding="utf-8")
    _, _, tool = make_filesystem_tool(tmp_path)
    operations = instrument_filesystem(tool, monkeypatch)
    with pytest.raises(ToolExecutionDenied):
        tool(str(path), "read")
    assert operations == ["read"]


@pytest.mark.parametrize(
    "guard",
    [
        pytest.param("pii", id="pii"),
        pytest.param("blocked-content", id="blocked-content"),
        pytest.param("oversized-output", id="oversized-output"),
        pytest.param("exception", id="guard-exception"),
    ],
)
def test_output_guard_denials_follow_read_operation(tmp_path, monkeypatch, guard):
    from preventai import OutputGuard

    path = tmp_path / "content"
    path.write_text(
        "4111 1111 1111 1111" if guard == "pii" else "blocked sample",
        encoding="utf-8",
    )
    _, _, tool = make_filesystem_tool(tmp_path)
    if guard == "pii":
        tool._output_guard = OutputGuard(pii_policy="block")
    elif guard == "blocked-content":
        tool._output_guard = OutputGuard(blocked_phrases=("blocked sample",))
    elif guard == "oversized-output":
        tool._output_guard = OutputGuard(max_output_bytes=1)
    else:
        tool._output_guard.validate = lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("guard failure")
        )
    operations = instrument_filesystem(tool, monkeypatch)
    with pytest.raises(ToolExecutionDenied):
        tool(str(path), "read")
    assert operations == ["read"]


def test_revoked_human_approval_blocks_before_filesystem_operation(tmp_path, monkeypatch):
    from preventai import ApprovalManager

    manager = ApprovalManager("test-secret")
    _, _, tool = make_filesystem_tool(
        tmp_path,
        approval_policy=ApprovalPolicy([ApprovalRule("approve", action="filesystem.*")]),
        approval_manager=manager,
    )
    operations = instrument_filesystem(tool, monkeypatch)
    with pytest.raises(ToolApprovalRequired):
        tool(str(tmp_path / "item"), "create", data=b"x")
    approval = manager.approve(tool.pending_approval_request, approved_by="human")
    manager.revoke(approval.approval_id)
    tool.use_approval(approval)
    operations.clear()
    with pytest.raises(ToolExecutionDenied):
        tool(str(tmp_path / "item"), "create", data=b"x")
    assert operations == []
    assert not (tmp_path / "item").exists()


def test_valid_approval_authorizes_exact_filesystem_request(tmp_path, monkeypatch):
    from preventai import ApprovalManager

    manager = ApprovalManager("test-secret")
    _, _, tool = make_filesystem_tool(
        tmp_path,
        approval_policy=ApprovalPolicy([ApprovalRule("approve", action="filesystem.*")]),
        approval_manager=manager,
    )
    operations = instrument_filesystem(tool, monkeypatch)
    with pytest.raises(ToolApprovalRequired):
        tool(str(tmp_path / "item"), "create", data=b"approved")
    approval = manager.approve(tool.pending_approval_request, approved_by="human")
    tool.use_approval(approval)
    operations.clear()
    result = tool(str(tmp_path / "item"), "create", data=b"approved")
    assert result["operation"] == "create"
    assert operations == ["write"]
    assert (tmp_path / "item").read_bytes() == b"approved"
    with pytest.raises(ToolExecutionDenied):
        tool(str(tmp_path / "item"), "create", data=b"approved")
    assert operations == ["write"]


def test_expired_human_approval_blocks_before_filesystem_operation(tmp_path, monkeypatch):
    from preventai import ApprovalManager

    manager = ApprovalManager("test-secret")
    audit = AuditLogger()
    _, _, tool = make_filesystem_tool(
        tmp_path,
        approval_policy=ApprovalPolicy([ApprovalRule("approve", action="filesystem.*")]),
        approval_manager=manager,
        audit_logger=audit,
    )
    operations = instrument_filesystem(tool, monkeypatch)
    with pytest.raises(ToolApprovalRequired):
        tool(str(tmp_path / "item"), "create", data=b"x")
    approval = manager.approve(
        tool.pending_approval_request,
        approved_by="human",
        expires_at=datetime.now(timezone.utc) + timedelta(milliseconds=2),
    )
    tool.use_approval(approval)
    time.sleep(0.01)
    operations.clear()
    with pytest.raises(ToolExecutionDenied):
        tool(str(tmp_path / "item"), "create", data=b"x")
    assert operations == []
    assert not (tmp_path / "item").exists()
    from preventai.audit import AuditEventType

    assert any(
        event.event_type is AuditEventType.APPROVAL_DECISION
        and event.decision == DecisionType.BLOCK.value
        for event in audit.events()
    )


@pytest.mark.parametrize(
    ("approved_operation", "approved_path", "approved_destination", "attempt_operation", "attempt_path", "attempt_destination"),
    [
        ("create", "first", None, "create", "second", None),
        ("create", "first", None, "write", "first", None),
        ("rename", "source", "destination-one", "rename", "source", "destination-two"),
        ("create", "first", None, "create", "first", None),
    ],
)
def test_approval_scope_mismatches_block_without_filesystem_operation(
    tmp_path,
    monkeypatch,
    approved_operation,
    approved_path,
    approved_destination,
    attempt_operation,
    attempt_path,
    attempt_destination,
):
    from preventai import ApprovalManager

    manager = ApprovalManager("test-secret")
    _, _, tool = make_filesystem_tool(
        tmp_path,
        approval_policy=ApprovalPolicy([ApprovalRule("approve", action="filesystem.*")]),
        approval_manager=manager,
    )
    if approved_operation == "rename":
        (tmp_path / approved_path).write_text("source", encoding="utf-8")
    with pytest.raises(ToolApprovalRequired):
        tool(
            str(tmp_path / approved_path),
            approved_operation,
            data=b"x" if approved_operation == "create" else b"",
            destination_path=(
                str(tmp_path / approved_destination) if approved_destination else None
            ),
        )
    tool.use_approval(
        manager.approve(tool.pending_approval_request, approved_by="human")
    )
    operations = instrument_filesystem(tool, monkeypatch)
    if attempt_operation == "write":
        (tmp_path / attempt_path).write_text("existing", encoding="utf-8")
    with pytest.raises(ToolExecutionDenied):
        tool(
            str(tmp_path / attempt_path),
            attempt_operation,
            data=b"changed" if attempt_operation in {"create", "write"} else b"",
            overwrite=attempt_operation == "write",
            destination_path=(
                str(tmp_path / attempt_destination) if attempt_destination else None
            ),
        )
    assert operations == []


def test_valid_approval_does_not_replace_filesystem_authorization(tmp_path, monkeypatch):
    from preventai import ApprovalManager
    from preventai.tools import FileSystemAuthorization

    manager = ApprovalManager("test-secret")
    _, _, tool = make_filesystem_tool(
        tmp_path,
        approval_policy=ApprovalPolicy([ApprovalRule("approve", action="filesystem.*")]),
        approval_manager=manager,
    )
    with pytest.raises(ToolApprovalRequired):
        tool(str(tmp_path / "item"), "create", data=b"x")
    tool.use_approval(
        manager.approve(tool.pending_approval_request, approved_by="human")
    )
    operations = instrument_filesystem(tool, monkeypatch)
    monkeypatch.setattr(
        tool,
        "_filesystem_authorizer",
        lambda request, plan: FileSystemAuthorization(
            request.digest(), str(plan.path), None, time.time() + 30, "forged", "bad"
        ),
    )
    with pytest.raises(ToolExecutionDenied):
        tool(str(tmp_path / "item"), "create", data=b"x")
    assert operations == []


@pytest.mark.parametrize(
    "result",
    ["string result", b"bytes", {"path": "file"}, None, object()],
)
def test_filesystem_tool_never_falls_back_to_normal_callable_result(
    tmp_path, monkeypatch, result
):
    _, _, tool = make_filesystem_tool(tmp_path)
    operations = instrument_filesystem(tool, monkeypatch)
    tool._function = lambda *args, **kwargs: result
    with pytest.raises(ToolExecutionDenied):
        tool(str(tmp_path / "item"), "create")
    assert operations == []


def test_dataflow_denial_covers_directory_listing_and_rename(tmp_path, monkeypatch):
    (tmp_path / "source").write_text("source", encoding="utf-8")
    _, _, tool = make_filesystem_tool(tmp_path)
    tool._data_flow_engine = DataFlowEngine([
        DataFlowPolicy(
            "block-filesystem-output",
            DecisionType.BLOCK,
            destination="sink",
        )
    ])
    operations = instrument_filesystem(tool, monkeypatch)
    with pytest.raises(ToolExecutionDenied):
        tool(str(tmp_path), "list")
    assert operations == []
    with pytest.raises(ToolExecutionDenied):
        tool(
            str(tmp_path / "source"),
            "rename",
            destination_path=str(tmp_path / "destination"),
        )
    assert operations == []


@pytest.mark.parametrize(
    ("operation", "policy_options", "data"),
    [
        ("read", {"max_read_bytes": 3}, b""),
        ("create", {"max_write_bytes": 3}, b"four"),
    ],
)
def test_filesystem_size_limits_deny_before_backend_operation(
    tmp_path, monkeypatch, operation, policy_options, data
):
    path = tmp_path / "item"
    path.write_bytes(b"four")
    _, _, tool = make_filesystem_tool(
        tmp_path, filesystem_policy_options=policy_options
    )
    operations = instrument_filesystem(tool, monkeypatch)
    with pytest.raises(ToolExecutionDenied):
        tool(str(path), operation, data=data)
    assert operations == []
    assert path.read_bytes() == b"four"


@pytest.mark.parametrize(
    "classification",
    [FlowClassification.PRIVATE, FlowClassification.SENSITIVE, FlowClassification.SECRET],
)
def test_filesystem_dataflow_preserves_classification_and_untrusted_source(
    tmp_path, monkeypatch, classification
):
    _, _, tool = make_filesystem_tool(tmp_path)
    flows = []
    engine = tool._data_flow_engine
    original_check = engine.check

    def record(flow):
        flows.append(flow)
        return original_check(flow)

    monkeypatch.setattr(engine, "check", record)
    result_path = tmp_path / classification.value
    tool(str(result_path), "create", data=b"payload", classification=classification)
    filesystem_flow = next(
        flow for flow in flows if flow.destination.name.startswith("filesystem:")
    )
    assert filesystem_flow.classification is classification
    assert filesystem_flow.trust is FlowTrust.UNTRUSTED
    assert filesystem_flow.destination.name == f"filesystem:{result_path.resolve()}"


def test_filesystem_dataflow_preserves_trusted_source_for_write(tmp_path, monkeypatch):
    _, _, tool = make_filesystem_tool(tmp_path)
    flows = []
    engine = tool._data_flow_engine
    original_check = engine.check
    monkeypatch.setattr(
        engine,
        "check",
        lambda flow: (flows.append(flow), original_check(flow))[1],
    )
    tool(
        str(tmp_path / "trusted"),
        "create",
        data=b"trusted",
        request_trust=FlowTrust.TRUSTED,
    )
    filesystem_flow = next(
        flow for flow in flows if flow.destination.name.startswith("filesystem:")
    )
    assert filesystem_flow.trust is FlowTrust.TRUSTED


def test_filesystem_read_dataflow_uses_strongest_classification(tmp_path, monkeypatch):
    path = tmp_path / "classified"
    path.write_text("classified data", encoding="utf-8")
    _, _, tool = make_filesystem_tool(
        tmp_path, source_classification=FlowClassification.SENSITIVE
    )
    flows = []
    engine = tool._data_flow_engine
    original_check = engine.check
    monkeypatch.setattr(
        engine,
        "check",
        lambda flow: (flows.append(flow), original_check(flow))[1],
    )
    tool(str(path), "read", classification=FlowClassification.PRIVATE)
    read_flow = next(
        flow for flow in flows if flow.source.name.startswith("filesystem:")
    )
    assert read_flow.classification is FlowClassification.SENSITIVE
    assert read_flow.trust is FlowTrust.UNTRUSTED


def test_rename_dataflow_binds_source_and_destination(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.write_text("move", encoding="utf-8")
    _, _, tool = make_filesystem_tool(tmp_path)
    flows = []
    engine = tool._data_flow_engine
    original_check = engine.check
    monkeypatch.setattr(
        engine,
        "check",
        lambda flow: (flows.append(flow), original_check(flow))[1],
    )
    result = tool(
        str(source),
        "rename",
        destination_path=str(tmp_path / "destination"),
    )
    assert result["operation"] == "rename"
    rename_flow = next(
        flow for flow in flows if flow.source.name.startswith("filesystem:")
    )
    assert rename_flow.source.name == f"filesystem:{source.resolve()}"
    assert rename_flow.destination.name == f"filesystem:{(tmp_path / 'destination').resolve()}"


def test_authentication_denial_starts_no_filesystem_operation(tmp_path, monkeypatch):
    from preventai import Credential

    identity = "filesystem-auth-agent"
    credential = Credential(identity, "auth-secret")
    authenticator = Authenticator()
    _, _, tool = make_filesystem_tool(
        tmp_path,
        authenticator=authenticator,
        agent_id=identity,
        credential=credential,
    )
    operations = instrument_filesystem(tool, monkeypatch)
    authenticator.revoke(identity)
    with pytest.raises(ToolExecutionDenied):
        tool(str(tmp_path / "item"), "create", data=b"x")
    assert operations == []


def test_delegation_scope_cannot_be_expanded_by_filesystem_policy(tmp_path, monkeypatch):
    _, agent, tool = make_filesystem_tool(tmp_path)
    allowed = tmp_path / "data" / "item"
    allowed.parent.mkdir()
    allowed.write_text("allowed", encoding="utf-8")
    registry = tool._delegation_registry
    grant = registry.create(
        "issuer-agent",
        agent.agent_id,
        [
            Capability(
                "operate", tool.tool_id, "operate", PermissionCategory.READ
            ),
            Capability(
                "filesystem.*",
                str(tmp_path / "data") + "\\*",
                "operate",
                PermissionCategory.FILE,
            ),
        ],
        expires_at=datetime.now(timezone.utc) + timedelta(minutes=1),
        session_id=agent.session_id,
        issuer_permissions=tool._engine.permissions,
    )
    agent.use_delegation(grant)
    operations = instrument_filesystem(tool, monkeypatch)
    assert tool(str(allowed), "read")["content"] == "allowed"
    assert operations == ["read"]
    operations.clear()
    secret_dir = tmp_path / "secrets"
    secret_dir.mkdir()
    secret = secret_dir / "item"
    secret.write_text("denied", encoding="utf-8")
    with pytest.raises(ToolExecutionDenied):
        tool(str(secret), "read")
    assert operations == []


def test_filesystem_rate_limit_denies_before_second_operation(tmp_path, monkeypatch):
    _, _, tool = make_filesystem_tool(
        tmp_path, runtime_policy=RuntimePolicy(rate_limit=RateLimit(1, 60))
    )
    operations = instrument_filesystem(tool, monkeypatch)
    tool(str(tmp_path / "first"), "create", data=b"first")
    assert operations == ["write"]
    operations.clear()
    with pytest.raises(ToolExecutionDenied):
        tool(str(tmp_path / "second"), "create", data=b"second")
    assert operations == []
    assert not (tmp_path / "second").exists()


def test_filesystem_runtime_budget_denies_before_second_operation(
    tmp_path, monkeypatch
):
    _, _, tool = make_filesystem_tool(
        tmp_path,
        runtime_policy=RuntimePolicy(budget=RuntimeBudget(max_tool_calls=1)),
    )
    operations = instrument_filesystem(tool, monkeypatch)
    tool(str(tmp_path / "first"), "create", data=b"first")
    assert operations == ["write"]
    operations.clear()
    with pytest.raises(ToolExecutionDenied):
        tool(str(tmp_path / "second"), "create", data=b"second")
    assert operations == []
    assert not (tmp_path / "second").exists()


def test_filesystem_loop_detection_denies_before_repeated_operation(
    tmp_path, monkeypatch
):
    _, _, tool = make_filesystem_tool(
        tmp_path,
        runtime_policy=RuntimePolicy(
            loop_policy=LoopPolicy(max_repeated_calls=2, cycle_length=2, max_history=10)
        ),
    )
    operations = instrument_filesystem(tool, monkeypatch)
    tool(str(tmp_path / "first"), "create", data=b"first")
    assert operations == ["write"]
    operations.clear()
    with pytest.raises(ToolExecutionDenied):
        tool(str(tmp_path / "second"), "create", data=b"second")
    assert operations == []
    assert not (tmp_path / "second").exists()


def test_filesystem_circuit_breaker_denies_before_backend_operation(
    tmp_path, monkeypatch
):
    app, _, tool = make_filesystem_tool(
        tmp_path,
        runtime_policy=RuntimePolicy(circuit_breaker=CircuitBreaker(failure_threshold=1)),
        callable_failure=True,
    )
    operations = instrument_filesystem(tool, monkeypatch)
    with pytest.raises(RuntimeError):
        tool(str(tmp_path / "first"), "create", data=b"first")
    with pytest.raises(ToolExecutionDenied):
        tool(str(tmp_path / "second"), "create", data=b"second")
    assert operations == []
    assert not (tmp_path / "second").exists()


def test_runtime_concurrency_denies_second_filesystem_backend_entry(
    tmp_path, monkeypatch
):
    import threading

    _, _, tool = make_filesystem_tool(
        tmp_path,
        runtime_policy=RuntimePolicy(max_concurrent_tools=1),
        filesystem_policy_options={"max_concurrent_operations": 2},
    )
    operations = instrument_filesystem(tool, monkeypatch)
    entered = threading.Event()
    release = threading.Event()
    original_write = tool._filesystem._backend.write

    def slow_write(*args, **kwargs):
        entered.set()
        if not release.wait(2):
            raise TimeoutError("test did not release the filesystem operation")
        return original_write(*args, **kwargs)

    monkeypatch.setattr(tool._filesystem._backend, "write", slow_write)
    failures = []

    def first_call():
        try:
            tool(str(tmp_path / "first"), "create", data=b"first")
        except Exception as exc:
            failures.append(exc)

    worker = threading.Thread(target=first_call)
    worker.start()
    assert entered.wait(1)
    with pytest.raises(ToolExecutionDenied):
        tool(str(tmp_path / "second"), "create", data=b"second")
    assert operations == []
    release.set()
    worker.join(2)
    assert not worker.is_alive()
    assert failures == []
    assert operations == ["write"]
    assert (tmp_path / "first").read_bytes() == b"first"
    assert not (tmp_path / "second").exists()


def test_timeout_blocks_filesystem_result_after_non_cancellable_write(
    tmp_path, monkeypatch
):
    import threading

    from preventai import AuditLogger
    from preventai.audit import AuditEventType

    audit = AuditLogger()
    _, _, tool = make_filesystem_tool(
        tmp_path,
        runtime_policy=RuntimePolicy(timeout=Timeout(0.01)),
        audit_logger=audit,
    )
    operations = instrument_filesystem(tool, monkeypatch)
    entered = threading.Event()
    release = threading.Event()
    original_write = tool._filesystem._backend.write

    def slow_write(*args, **kwargs):
        entered.set()
        if not release.wait(2):
            raise TimeoutError("test did not release the filesystem operation")
        return original_write(*args, **kwargs)

    monkeypatch.setattr(tool._filesystem._backend, "write", slow_write)
    worker_errors = []

    def call_tool():
        try:
            tool(str(tmp_path / "item"), "create", data=b"written")
        except ToolExecutionDenied as exc:
            worker_errors.append(exc)

    worker = threading.Thread(target=call_tool)
    worker.start()
    assert entered.wait(1)
    time.sleep(0.02)
    release.set()
    worker.join(2)
    assert not worker.is_alive()
    assert len(worker_errors) == 1
    assert operations == ["write"]
    assert (tmp_path / "item").read_bytes() == b"written"
    assert any(
        event.event_type is AuditEventType.TIMEOUT
        and event.decision == DecisionType.BLOCK.value
        for event in audit.events()
    )
