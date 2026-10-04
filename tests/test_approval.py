from datetime import datetime, timedelta, timezone
import threading

from preventai import (
    Action,
    AgentIdentity,
    ApprovalManager,
    DataClassification,
    DataMetadata,
    RequestedBy,
    Resource,
    SecurityRequest,
    SessionIdentity,
    SourceTrust,
    ProtocolTool,
)


def request():
    return SecurityRequest(
        "1.0",
        "approval-request",
        datetime(2026, 1, 1, tzinfo=timezone.utc),
        AgentIdentity("agent-1", "test"),
        SessionIdentity("session-1"),
        RequestedBy.AGENT,
        Action("database.delete", {"record": "opaque-id"}),
        Resource("customer-db", "database"),
        ProtocolTool("database", "1.0"),
        DataMetadata(DataClassification.INTERNAL, SourceTrust.TRUSTED),
        {"purpose": "maintenance"},
        {},
    )


def test_exact_request_approval_is_signed_and_one_time():
    manager = ApprovalManager("approval-secret")
    pending = manager.request(
        request(),
        identity="agent-1",
        reason="destructive action",
        expires_at=datetime.now(timezone.utc) + timedelta(minutes=1),
    )
    approval = manager.approve(pending, approved_by="operator")
    assert "approval-secret" not in approval.to_json()
    assert manager.verify(
        approval, request(), authenticated_identity="agent-1"
    ).reason.value == "request_allowed"
    assert manager.verify(
        approval, request(), authenticated_identity="agent-1"
    ).reason.value == "approval_already_consumed"


def test_mismatch_tamper_and_revocation_fail_closed():
    manager = ApprovalManager("approval-secret")
    pending = manager.request(
        request(),
        identity="agent-1",
        reason="destructive action",
        expires_at=datetime.now(timezone.utc) + timedelta(minutes=1),
    )
    approval = manager.approve(pending, approved_by="operator", one_time=False)
    assert manager.verify(
        approval, request(), authenticated_identity="agent-2"
    ).reason.value == "approval_subject_mismatch"
    manager.revoke(approval.approval_id)
    assert manager.verify(
        approval, request(), authenticated_identity="agent-1"
    ).reason.value == "approval_revoked"


def test_concurrent_one_time_consumption_is_atomic():
    manager = ApprovalManager("approval-secret")
    pending = manager.request(
        request(),
        identity="agent-1",
        reason="destructive action",
        expires_at=datetime.now(timezone.utc) + timedelta(minutes=1),
    )
    approval = manager.approve(pending, approved_by="operator")
    results = []

    def consume():
        results.append(manager.verify(approval, request(), authenticated_identity="agent-1"))

    threads = [threading.Thread(target=consume) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert sum(result.reason.value == "request_allowed" for result in results) == 1
