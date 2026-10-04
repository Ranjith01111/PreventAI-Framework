import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from preventai import (
    Agent,
    Decision,
    DecisionReason,
    DecisionType,
    PreventApp,
    ToolApprovalRequired,
    ToolExecutionDenied,
)
from preventai.core import PermissionCategory, PermissionEngine, PermissionRule, PolicyEngine, PolicyRule
from preventai.tools import Sandbox as _Sandbox, SandboxCommand, SandboxPolicy
from preventai.tools.sandbox import _FRAMEWORK_ISSUER
from _boundary_test_support import bind_test_boundary


def Sandbox(policy, **kwargs):
    return bind_test_boundary(_Sandbox(policy, **kwargs))


def make_policy(tmp_path, **overrides):
    values = {
        "allowed_commands": (sys.executable,),
        "working_directory": str(tmp_path),
        "allowed_paths": (str(tmp_path),),
    }
    values.update(overrides)
    return SandboxPolicy(**values)


def make_command(code="print('ok')", **kwargs):
    return SandboxCommand(sys.executable, ("-c", code), **kwargs)


def authorize(sandbox, command, **scope):
    values = {
        "agent_id": "agent",
        "session_id": "session",
        "tool_id": "tool",
        "action": "run",
        "resource": "resource",
        "command": command,
        "_issuer": _FRAMEWORK_ISSUER,
    }
    values.update(scope)
    return sandbox._issue_authorization(**values)


def execute(sandbox, command, **scope):
    values = {
        "agent_id": "agent",
        "session_id": "session",
        "tool_id": "tool",
        "action": "run",
        "resource": "resource",
    }
    values.update(scope)
    return sandbox.execute(command, authorization=authorize(sandbox, command, **values), **values)


def assert_no_process(monkeypatch):
    started = []

    def forbidden(*args, **kwargs):
        started.append((args, kwargs))
        raise AssertionError("subprocess started")

    monkeypatch.setattr(subprocess, "Popen", forbidden)
    return started


def test_issuer_capability_is_not_stored_on_sandbox(tmp_path):
    sandbox = Sandbox(make_policy(tmp_path))
    assert not hasattr(sandbox, "_issuer_token")
    with pytest.raises(PermissionError):
        sandbox._issue_authorization(
            agent_id="agent", session_id="session", tool_id="tool",
            action="run", resource="resource", command=make_command(),
        )


def test_action_and_resource_scope_are_checked(tmp_path):
    sandbox = Sandbox(make_policy(tmp_path))
    command = make_command()
    authorization = authorize(sandbox, command)
    result = sandbox.execute(
        command, authorization=authorization, agent_id="agent",
        session_id="session", tool_id="tool", action="other", resource="resource",
    )
    assert result.decision.decision_type is DecisionType.BLOCK


def test_authorization_cannot_be_reused_for_another_session(tmp_path):
    sandbox = Sandbox(make_policy(tmp_path))
    command = make_command()
    authorization = authorize(sandbox, command)
    result = sandbox.execute(
        command, authorization=authorization, agent_id="agent",
        session_id="other", tool_id="tool", action="run", resource="resource",
    )
    assert result.decision.decision_type is DecisionType.BLOCK


def test_digest_mismatch_starts_no_process(tmp_path, monkeypatch):
    started = assert_no_process(monkeypatch)
    sandbox = Sandbox(make_policy(tmp_path))
    authorization = authorize(sandbox, make_command("print('one')"))
    result = sandbox.execute(
        make_command("print('two')"), authorization=authorization,
        agent_id="agent", session_id="session", tool_id="tool",
        action="run", resource="resource",
    )
    assert result.decision.decision_type is DecisionType.BLOCK
    assert started == []


def test_invalid_executable_starts_no_process(tmp_path, monkeypatch):
    started = assert_no_process(monkeypatch)
    sandbox = Sandbox(make_policy(tmp_path))
    command = SandboxCommand("not-allowlisted", (), cwd=str(tmp_path))
    result = execute(sandbox, command)
    assert result.decision.decision_type is DecisionType.BLOCK
    assert started == []


def test_invalid_cwd_starts_no_process(tmp_path, monkeypatch):
    started = assert_no_process(monkeypatch)
    sandbox = Sandbox(make_policy(tmp_path))
    command = make_command(cwd=str(Path(tmp_path).parent))
    result = execute(sandbox, command)
    assert result.decision.decision_type is DecisionType.BLOCK
    assert started == []


def test_invalid_environment_starts_no_process(tmp_path, monkeypatch):
    started = assert_no_process(monkeypatch)
    sandbox = Sandbox(make_policy(tmp_path))
    command = make_command(environment={"NOT_ALLOWED": "value"})
    result = execute(sandbox, command)
    assert result.decision.decision_type is DecisionType.BLOCK
    assert started == []


def test_malformed_null_argument_is_rejected_before_execution(tmp_path, monkeypatch):
    started = assert_no_process(monkeypatch)
    with pytest.raises(ValueError):
        SandboxCommand(sys.executable, ("\x00",))
    assert started == []


def test_malformed_non_string_argument_is_rejected(tmp_path, monkeypatch):
    started = assert_no_process(monkeypatch)
    with pytest.raises(ValueError):
        SandboxCommand(sys.executable, (123,))
    assert started == []


def test_denylist_overrides_allowlist(tmp_path, monkeypatch):
    started = assert_no_process(monkeypatch)
    sandbox = Sandbox(make_policy(tmp_path, allowed_commands=("python",), denied_commands=("cmd.exe",)))
    command = SandboxCommand("cmd.exe", (), cwd=str(tmp_path))
    result = execute(sandbox, command)
    assert result.decision.reason is DecisionReason.SANDBOX_POLICY_VIOLATION
    assert started == []


def test_shell_flag_is_never_forwarded_as_enabled(tmp_path, monkeypatch):
    started = assert_no_process(monkeypatch)
    sandbox = Sandbox(make_policy(tmp_path))
    result = execute(sandbox, SandboxCommand(sys.executable, ("-c", "print(1)"), cwd=str(tmp_path), shell=True))
    assert result.decision.decision_type is DecisionType.BLOCK
    assert started == []


def test_input_limit_starts_no_process(tmp_path, monkeypatch):
    started = assert_no_process(monkeypatch)
    sandbox = Sandbox(make_policy(tmp_path, max_input_bytes=4))
    result = execute(sandbox, make_command("print('large')"))
    assert result.decision.decision_type is DecisionType.BLOCK
    assert started == []


def test_environment_allowlist_does_not_leak_unlisted_values(tmp_path):
    sandbox = Sandbox(make_policy(tmp_path, environment_allowlist=("PATH",)))
    result = execute(sandbox, make_command("import os; print(os.getenv('PREVENTAI_TEST_SECRET', 'missing'))"))
    assert result.decision.decision_type is DecisionType.ALLOW
    assert "missing" in result.stdout


def test_output_guard_exception_fails_closed(tmp_path):
    sandbox = Sandbox(
        make_policy(tmp_path),
        output_guard=lambda output: (_ for _ in ()).throw(RuntimeError("guard failure")),
    )
    result = execute(sandbox, make_command("print('output')"))
    assert result.decision.decision_type is DecisionType.BLOCK
    assert result.stdout == ""


def test_output_guard_block_removes_raw_output(tmp_path):
    sandbox = Sandbox(
        make_policy(tmp_path),
        output_guard=lambda output: Decision(DecisionType.BLOCK, DecisionReason.OUTPUT_POLICY_VIOLATION),
    )
    result = execute(sandbox, make_command("print('secret-output')"))
    assert result.decision.decision_type is DecisionType.BLOCK
    assert result.stdout == ""
    assert result.stderr == ""


def test_timeout_kills_direct_process(tmp_path):
    sandbox = Sandbox(make_policy(tmp_path, max_execution_seconds=0.02))
    result = execute(sandbox, make_command("import time; time.sleep(1)"))
    assert result.timed_out
    assert result.decision.reason is DecisionReason.SANDBOX_TIMEOUT


def test_audit_records_blocked_execution(tmp_path):
    from preventai.audit import AuditLogger

    logger = AuditLogger(secret="sandbox-audit")
    sandbox = Sandbox(make_policy(tmp_path), audit_logger=logger)
    result = sandbox.execute(make_command(), action="run", resource="resource")
    assert result.decision.decision_type is DecisionType.BLOCK
    assert any(event.event_type.value == "sandbox_blocked" for event in logger.events())


def test_concurrent_executions_preserve_result_integrity(tmp_path):
    sandbox = Sandbox(make_policy(tmp_path))

    def run_one(index):
        result = execute(sandbox, make_command(f"print({index})"))
        return result.decision.decision_type, result.stdout.strip()

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(run_one, range(4)))
    assert all(decision is DecisionType.ALLOW for decision, _ in results)
    assert {output for _, output in results} == {"0", "1", "2", "3"}


def framework_tool(tmp_path, *, policy, runtime_policy=None, approval_policy=None, authenticator=None):
    permissions = PermissionEngine([
        PermissionRule("execute", PermissionCategory.EXECUTE, action="run")
    ])
    app = PreventApp(
        policy=policy, permissions=permissions, runtime_policy=runtime_policy,
        approval_policy=approval_policy, authenticator=authenticator,
    )
    agent = Agent("sandbox-regression")
    if authenticator is not None:
        from preventai import Credential
        agent._credential = Credential(agent.agent_id, "wrong-secret")
    calls = []

    @agent.tool(
        sandbox_policy=make_policy(tmp_path),
        runtime_policy=runtime_policy,
    )
    def run():
        calls.append(True)
        return make_command("print('framework')")

    app.add_agent(agent)
    app.run()
    return run, calls


def test_authentication_denial_starts_no_process(tmp_path, monkeypatch):
    from preventai import Credential, RegisteredAgentIdentity
    from preventai.identity import Authenticator

    started = assert_no_process(monkeypatch)
    authenticator = Authenticator()
    agent_id = Agent("sandbox-regression").agent_id
    authenticator.register(
        RegisteredAgentIdentity(agent_id, "test"),
        Credential(agent_id, "correct-secret"),
    )
    run, calls = framework_tool(
        tmp_path,
        policy=PolicyEngine([PolicyRule("allow", DecisionType.ALLOW, action="run")]),
        authenticator=authenticator,
    )
    with pytest.raises(ToolExecutionDenied):
        run()
    assert calls == []
    assert started == []


def test_policy_denial_starts_no_process(tmp_path, monkeypatch):
    started = assert_no_process(monkeypatch)
    run, calls = framework_tool(
        tmp_path,
        policy=PolicyEngine([PolicyRule("deny", DecisionType.BLOCK, action="run")]),
    )
    with pytest.raises(ToolExecutionDenied):
        run()
    assert calls == []
    assert started == []


def test_runtime_max_call_denial_starts_no_process(tmp_path, monkeypatch):
    from preventai.runtime import RuntimePolicy

    runtime = RuntimePolicy(max_tool_calls=1)
    run, calls = framework_tool(
        tmp_path,
        policy=PolicyEngine([PolicyRule("allow", DecisionType.ALLOW, action="run")]),
        runtime_policy=runtime,
    )
    run()
    started = assert_no_process(monkeypatch)
    with pytest.raises(ToolExecutionDenied):
        run()
    assert len(calls) == 1
    assert started == []


def test_rate_limit_denial_starts_no_process(tmp_path, monkeypatch):
    from preventai.runtime import RateLimit, RuntimePolicy

    runtime = RuntimePolicy(rate_limit=RateLimit(1, 60))
    run, calls = framework_tool(
        tmp_path,
        policy=PolicyEngine([PolicyRule("allow", DecisionType.ALLOW, action="run")]),
        runtime_policy=runtime,
    )
    run()
    started = assert_no_process(monkeypatch)
    with pytest.raises(ToolExecutionDenied):
        run()
    assert len(calls) == 1
    assert started == []


def test_budget_denial_starts_no_process(tmp_path, monkeypatch):
    from preventai.runtime import RuntimeBudget, RuntimePolicy

    runtime = RuntimePolicy(budget=RuntimeBudget(max_tool_calls=1))
    run, calls = framework_tool(
        tmp_path,
        policy=PolicyEngine([PolicyRule("allow", DecisionType.ALLOW, action="run")]),
        runtime_policy=runtime,
    )
    run()
    started = assert_no_process(monkeypatch)
    with pytest.raises(ToolExecutionDenied):
        run()
    assert len(calls) == 1
    assert started == []


def test_concurrency_admission_denial_starts_no_process(tmp_path, monkeypatch):
    import threading
    from preventai.runtime import RuntimePolicy

    entered = threading.Event()
    release = threading.Event()
    permissions = PermissionEngine([
        PermissionRule("execute", PermissionCategory.EXECUTE, action="run")
    ])
    app = PreventApp(
        policy=PolicyEngine([PolicyRule("allow", DecisionType.ALLOW, action="run")]),
        permissions=permissions,
        runtime_policy=RuntimePolicy(max_concurrent_tools=1),
    )
    agent = Agent("sandbox-concurrency")
    calls = []

    @agent.tool(sandbox_policy=make_policy(tmp_path))
    def run():
        calls.append(True)
        entered.set()
        release.wait(2)
        return make_command()

    app.add_agent(agent)
    app.run()
    errors = []
    worker = threading.Thread(target=lambda: run(),)
    worker.start()
    assert entered.wait(2)
    started = assert_no_process(monkeypatch)
    with pytest.raises(ToolExecutionDenied):
        run()
    monkeypatch.undo()
    release.set()
    worker.join(3)
    assert not errors
    assert started == []


def test_delegation_denial_starts_no_process(tmp_path, monkeypatch):
    started = assert_no_process(monkeypatch)
    run, calls = framework_tool(
        tmp_path,
        policy=PolicyEngine([PolicyRule("allow", DecisionType.ALLOW, action="run")]),
    )
    agent = run._agent
    agent._delegation = object()
    agent._delegation_registry.evaluate = lambda *args, **kwargs: Decision(
        DecisionType.BLOCK, DecisionReason.DELEGATION_INVALID
    )
    with pytest.raises(ToolExecutionDenied):
        run()
    assert calls == []
    assert started == []


def test_approval_denial_starts_no_process(tmp_path, monkeypatch):
    started = assert_no_process(monkeypatch)
    from preventai.approval import ApprovalPolicy, ApprovalRule

    run, calls = framework_tool(
        tmp_path,
        policy=PolicyEngine([PolicyRule("allow", DecisionType.ALLOW, action="run")]),
        approval_policy=ApprovalPolicy([ApprovalRule("human", action="run")]),
    )
    with pytest.raises(ToolApprovalRequired):
        run()
    assert calls == []
    assert started == []


def test_sandbox_configured_tool_has_no_normal_value_fallback(tmp_path, monkeypatch):
    started = assert_no_process(monkeypatch)
    permissions = PermissionEngine([
        PermissionRule("execute", PermissionCategory.EXECUTE, action="run")
    ])
    app = PreventApp(
        policy=PolicyEngine([PolicyRule("allow", DecisionType.ALLOW, action="run")]),
        permissions=permissions,
    )
    agent = app.add_agent(Agent("fallback-regression"))

    @agent.tool(sandbox_policy=make_policy(tmp_path))
    def run():
        return {"safe": "ordinary result"}

    app.run()
    with pytest.raises(ToolExecutionDenied):
        run()
    assert started == []
