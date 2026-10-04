from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor

import pytest

from preventai import (
    Agent,
    DecisionType,
    PermissionCategory,
    PermissionEngine,
    PermissionRule,
    PolicyEngine,
    PolicyIntegrityRegistry,
    PolicyRule,
    PreventApp,
    ToolExecutionDenied,
    configuration_digest,
)
from preventai.audit import AuditEventType, AuditLogger
from preventai.tools import (
    DatabasePolicy,
    FileSystemPolicy,
    NetworkPolicy,
    SandboxPolicy,
)


def _config():
    return (
        PolicyEngine([PolicyRule("allow", DecisionType.ALLOW, action="*")]),
        PermissionEngine([
            PermissionRule("execute", PermissionCategory.EXECUTE, action="*", resource="*"),
            PermissionRule("read", PermissionCategory.READ, action="*", resource="*"),
        ]),
    )


def test_registry_is_app_owned_and_digest_is_deterministic():
    policy_a, permissions_a = _config()
    policy_b, permissions_b = _config()
    assert configuration_digest(policy_a, permissions_a, None) == configuration_digest(
        policy_b, permissions_b, None
    )
    with pytest.raises(ValueError):
        PolicyIntegrityRegistry(audit_logger=AuditLogger())
    app = PreventApp(policy=policy_a, permissions=permissions_a)
    assert app.policy_integrity.active().version == 1


def test_invalid_and_stale_versions_fail_closed():
    policy, permissions = _config()
    app = PreventApp(policy=policy, permissions=permissions)
    with pytest.raises(ValueError):
        app.update_security_policy(
            policy=policy, permissions=permissions, version=1
        )
    with pytest.raises(ValueError):
        app.update_security_policy(
            policy=policy, permissions=permissions, version=0
        )


def test_tampered_active_snapshot_blocks_real_tool_execution():
    policy, permissions = _config()
    app = PreventApp(policy=policy, permissions=permissions)
    agent = app.add_agent(Agent("integrity-agent", agent_id="integrity-agent"))
    calls = []

    @agent.tool("lookup")
    def lookup():
        calls.append(True)
        return "ok"

    app.run()
    snapshot = app.policy_integrity.active()
    object.__setattr__(snapshot.policy, "_rules", ())
    with pytest.raises(ToolExecutionDenied):
        lookup()
    assert calls == []


def test_explicit_versioned_replacement_is_atomic_and_audited():
    audit = AuditLogger()
    policy, permissions = _config()
    app = PreventApp(policy=policy, permissions=permissions, audit_logger=audit)
    replacement = PolicyEngine([
        PolicyRule("allow-v2", DecisionType.ALLOW, action="*"),
    ])
    app.update_security_policy(
        policy=replacement, permissions=permissions, version=2, activate=True
    )
    assert app.policy_integrity.active().version == 2
    assert any(
        event.event_type is AuditEventType.POLICY_ACTIVATED
        for event in audit.events()
    )


def test_revoked_configuration_blocks_evaluation():
    policy, permissions = _config()
    app = PreventApp(policy=policy, permissions=permissions)
    app.policy_integrity.revoke("application-security", 1)
    with pytest.raises(RuntimeError):
        app.policy_integrity.active()


def test_expired_configuration_fails_closed():
    policy, permissions = _config()
    app = PreventApp(policy=policy, permissions=permissions)
    app.update_security_policy(
        policy=policy,
        permissions=permissions,
        version=2,
        expires_at=datetime.now(timezone.utc) - timedelta(seconds=1),
    )
    with pytest.raises(RuntimeError):
        app.policy_integrity.active()


def test_cross_app_configuration_cannot_be_substituted():
    policy, permissions = _config()
    first = PreventApp(policy=policy, permissions=permissions)
    second = PreventApp(policy=policy, permissions=permissions)
    assert first.policy_integrity is not second.policy_integrity
    with pytest.raises(ValueError):
        first.policy_integrity.activate("other-app", 1)


def test_audit_failure_does_not_leave_new_configuration_active():
    logger = AuditLogger()
    policy, permissions = _config()
    app = PreventApp(policy=policy, permissions=permissions, audit_logger=logger)
    logger.record = lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("audit"))
    with pytest.raises(RuntimeError):
        app.update_security_policy(
            policy=policy, permissions=permissions, version=2
        )
    logger.record = AuditLogger.record.__get__(logger, AuditLogger)
    assert app.policy_integrity.active().version == 1


def test_concurrent_versions_leave_one_complete_active_snapshot():
    policy, permissions = _config()
    app = PreventApp(policy=policy, permissions=permissions)

    def update(version):
        try:
            app.update_security_policy(
                policy=policy, permissions=permissions, version=version
            )
            return "ok"
        except ValueError:
            return "rejected"

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(update, (2, 3)))
    assert "ok" in results
    assert app.policy_integrity.active().version in {2, 3}


@pytest.mark.parametrize(
    "boundary_policy",
    [
        SandboxPolicy(allowed_commands=("echo",)),
        NetworkPolicy(allowed_hosts=("example.com",)),
        FileSystemPolicy(allowed_roots=(".",), allowed_operations=("read",)),
        DatabasePolicy(allowed_databases=("db",), allowed_tables=("table",)),
    ],
)
def test_tampered_policy_blocks_before_restricted_boundary(boundary_policy):
    policy, permissions = _config()
    app = PreventApp(policy=policy, permissions=permissions)
    agent = app.add_agent(Agent("boundary-integrity-agent"))

    @agent.tool("restricted", **{
        {
            SandboxPolicy: "sandbox_policy",
            NetworkPolicy: "network_policy",
            FileSystemPolicy: "filesystem_policy",
            DatabasePolicy: "database_policy",
        }[type(boundary_policy)]: boundary_policy
    })
    def restricted():
        raise AssertionError("callable must not run")

    app.run()
    app._engine._policy = PolicyEngine([
        PolicyRule("deny", DecisionType.BLOCK, action="*"),
    ])
    with pytest.raises(ToolExecutionDenied):
        restricted()
