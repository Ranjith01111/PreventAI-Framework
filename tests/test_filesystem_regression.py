from pathlib import Path

import pytest

from _filesystem_test_support import instrument_filesystem, make_filesystem_tool
from preventai import AuditLogger, Decision, DecisionReason, DecisionType, ToolExecutionDenied
from preventai.audit import AuditEventType
from preventai.core import DataFlowPolicy
from preventai.tools import FileSystemRequest


def test_filesystem_audit_records_operation_without_content(tmp_path):
    audit = AuditLogger()
    path = tmp_path / "item"
    app, _, tool = make_filesystem_tool(tmp_path)
    tool._audit_logger = audit
    tool._filesystem._audit_logger = audit
    tool(str(path), "create", data=b"highly confidential payload")
    events = audit.events()
    assert any(event.event_type is AuditEventType.FILESYSTEM_OPERATION_STARTED for event in events)
    assert any(event.event_type is AuditEventType.FILESYSTEM_OPERATION_COMPLETED for event in events)
    assert all("highly confidential payload" not in str(event.to_dict()) for event in events)


def test_instrumented_backend_confirms_successful_single_operation(tmp_path, monkeypatch):
    _, _, tool = make_filesystem_tool(tmp_path)
    operations = instrument_filesystem(tool, monkeypatch)
    tool(str(tmp_path / "item"), "create", data=b"x")
    assert operations == ["write"]


def test_denied_operation_never_reaches_backend(tmp_path, monkeypatch):
    _, _, tool = make_filesystem_tool(tmp_path)
    operations = instrument_filesystem(tool, monkeypatch)
    with pytest.raises(ToolExecutionDenied):
        tool(str(tmp_path.parent / "outside"), "create", data=b"x")
    assert operations == []


def test_filesystem_denial_audit_redacts_path_and_payload(tmp_path):
    audit = AuditLogger()
    app, _, tool = make_filesystem_tool(
        tmp_path, audit_logger=audit,
        request_path=tmp_path / "id_rsa",
        request_data=b"******",
    )
    tool._audit_logger = audit
    tool._filesystem._audit_logger = audit
    sensitive_payload = b"password=private-test-value"
    with pytest.raises(ToolExecutionDenied):
        tool(str(tmp_path / "ignored"), "create")
    serialized = str([event.to_dict() for event in audit.events()])
    assert str(sensitive_payload) not in serialized
    assert "id_rsa" not in serialized
    assert any(
        event.event_type is AuditEventType.FILESYSTEM_BLOCKED
        for event in audit.events()
    )


def test_filesystem_dataflow_denial_is_audited_without_content(tmp_path):
    audit = AuditLogger()
    _, _, tool = make_filesystem_tool(
        tmp_path,
        audit_logger=audit,
        flow_rules=[
            DataFlowPolicy(
                "block-filesystem",
                DecisionType.BLOCK,
                destination="filesystem:*",
            )
        ],
    )
    tool._audit_logger = audit
    tool._filesystem._audit_logger = audit
    with pytest.raises(ToolExecutionDenied):
        tool(str(tmp_path / "item"), "create", data=b"payload-private-marker")
    serialized = str([event.to_dict() for event in audit.events()])
    assert "payload-private-marker" not in serialized
    assert any(
        event.event_type is AuditEventType.DATA_FLOW
        and event.decision == DecisionType.BLOCK.value
        for event in audit.events()
    )


@pytest.mark.parametrize("operation", ["write", "delete", "rename"])
def test_denied_mutation_is_audited_without_path_or_content(
    tmp_path, monkeypatch, operation
):
    from _filesystem_test_support import instrument_filesystem

    from preventai import ToolExecutionDenied

    audit = AuditLogger()
    _, _, tool = make_filesystem_tool(
        tmp_path,
        audit_logger=audit,
        allowed_operations=("read",),
    )
    operations = instrument_filesystem(tool, monkeypatch)
    path = tmp_path / "private-name"
    path.write_text("preserve", encoding="utf-8")
    kwargs = {"data": b"private-payload"} if operation == "write" else {}
    if operation == "rename":
        kwargs["destination_path"] = str(tmp_path / "destination")
    with pytest.raises(ToolExecutionDenied):
        tool(str(path), operation, **kwargs)
    serialized = str([event.to_dict() for event in audit.events()])
    assert operations == []
    assert "private-name" not in serialized
    assert "private-payload" not in serialized
    assert any(
        event.event_type is AuditEventType.FILESYSTEM_BLOCKED
        and event.decision == DecisionType.BLOCK.value
        for event in audit.events()
    )


def test_traversal_denial_is_audited_without_path(tmp_path):
    audit = AuditLogger()
    root = tmp_path / "root"
    root.mkdir()
    _, _, tool = make_filesystem_tool(root, audit_logger=audit)
    with pytest.raises(ToolExecutionDenied):
        tool(str(root / ".." / "private-name"), "create", data=b"payload")
    serialized = str([event.to_dict() for event in audit.events()])
    assert "private-name" not in serialized
    assert "payload" not in serialized
    assert any(
        event.event_type is AuditEventType.FILESYSTEM_BLOCKED
        for event in audit.events()
    )


def test_denied_read_is_audited_without_requested_path(tmp_path, monkeypatch):
    audit = AuditLogger()
    allowed = tmp_path / "allowed"
    outside = tmp_path / "private-name"
    allowed.mkdir()
    outside.write_text("private-content", encoding="utf-8")
    _, _, tool = make_filesystem_tool(allowed, audit_logger=audit)
    operations = instrument_filesystem(tool, monkeypatch)
    with pytest.raises(ToolExecutionDenied):
        tool(str(outside), "read")
    serialized = str([event.to_dict() for event in audit.events()])
    assert operations == []
    assert "private-name" not in serialized
    assert "private-content" not in serialized
    assert any(
        event.event_type is AuditEventType.FILESYSTEM_BLOCKED
        for event in audit.events()
    )


def test_fake_symlink_denial_is_audited_without_link_path(tmp_path, monkeypatch):
    audit = AuditLogger()
    link = tmp_path / "link"
    _, _, tool = make_filesystem_tool(tmp_path, audit_logger=audit)
    operations = instrument_filesystem(tool, monkeypatch)
    original_is_symlink = Path.is_symlink
    monkeypatch.setattr(
        Path,
        "is_symlink",
        lambda path: True if path == link else original_is_symlink(path),
    )
    with pytest.raises(ToolExecutionDenied):
        tool(str(link), "read")
    serialized = str([event.to_dict() for event in audit.events()])
    assert operations == []
    assert "link" not in serialized
    assert any(
        event.event_type is AuditEventType.FILESYSTEM_BLOCKED
        for event in audit.events()
    )


def test_security_evaluation_failure_is_audited_before_filesystem_operation(
    tmp_path, monkeypatch
):
    audit = AuditLogger()
    _, _, tool = make_filesystem_tool(tmp_path, audit_logger=audit)
    operations = instrument_filesystem(tool, monkeypatch)
    monkeypatch.setattr(
        tool._engine._policy,
        "evaluate",
        lambda context: Decision(
            DecisionType.BLOCK, DecisionReason.SECURITY_EVALUATION_ERROR
        ),
    )
    with pytest.raises(ToolExecutionDenied):
        tool(str(tmp_path / "item"), "create", data=b"payload")
    assert operations == []
    assert any(
        event.event_type is AuditEventType.POLICY_EVALUATION
        and event.decision == DecisionType.BLOCK.value
        for event in audit.events()
    )
