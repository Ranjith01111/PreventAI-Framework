import sys
import time

from preventai import DecisionReason, DecisionType
from preventai import Decision
from preventai.audit import AuditLogger
from preventai.tools import Sandbox as _Sandbox, SandboxCommand, SandboxPolicy
from preventai.tools.sandbox import _FRAMEWORK_ISSUER
from _boundary_test_support import bind_test_boundary


def Sandbox(policy, **kwargs):
    return bind_test_boundary(_Sandbox(policy, **kwargs))


def policy(tmp_path, **kwargs):
    values = dict(
        allowed_commands=("python", "python.exe"),
        working_directory=str(tmp_path),
        allowed_paths=(str(tmp_path),),
        environment_allowlist=(),
    )
    values.update(kwargs)
    return SandboxPolicy(**values)


def command(code):
    return SandboxCommand(sys.executable, ("-c", code))


def run(sandbox, command):
    return sandbox.execute(
        command,
        authorization=sandbox._issue_authorization(
            agent_id="agent", session_id="session", tool_id="tool",
            action="run", resource="resource",
            command=command,
            _issuer=_FRAMEWORK_ISSUER,
        ),
        agent_id="agent", session_id="session", tool_id="tool",
        action="run", resource="resource",
    )


def test_safe_execution_and_bounded_output(tmp_path):
    sandbox = Sandbox(policy(tmp_path, allowed_commands=(sys.executable,)))
    result = run(sandbox, command("print('safe')"))
    assert result.decision.decision_type is DecisionType.ALLOW
    assert result.stdout.strip() == "safe"

    limited = Sandbox(policy(tmp_path, allowed_commands=(sys.executable,), max_output_bytes=4))
    result = run(limited, command("print('0123456789')"))
    assert result.output_truncated
    assert len(result.stdout.encode()) <= 4


def test_blocked_command_never_starts_and_timeout_blocks(tmp_path):
    calls = []
    sandbox = Sandbox(
        policy(tmp_path, allowed_commands=(sys.executable,)),
        permission_checker=lambda command: (
            calls.append(command) or __import__("preventai").Decision(
                DecisionType.BLOCK, DecisionReason.MISSING_PERMISSION
            )
        ),
    )
    result = run(sandbox, command("print('must-not-run')"))
    assert result.decision.decision_type is DecisionType.BLOCK
    assert calls

    timed = Sandbox(policy(tmp_path, allowed_commands=(sys.executable,), max_execution_seconds=0.05))
    result = run(timed, command("import time; time.sleep(1)"))
    assert result.timed_out
    assert result.decision.reason is DecisionReason.SANDBOX_TIMEOUT


def test_audit_excludes_output_and_environment(tmp_path):
    logger = AuditLogger(secret="audit-secret")
    sandbox = Sandbox(policy(tmp_path, allowed_commands=(sys.executable,)), audit_logger=logger)
    result = run(sandbox, command("print('synthetic-output')"))
    assert result.decision.decision_type is DecisionType.ALLOW
    serialized = "".join(event.to_json() for event in logger.events())
    assert "synthetic-output" not in serialized
    assert "audit-secret" not in serialized
    assert logger.verify().valid


def test_data_flow_and_output_guard_block(tmp_path):
    from preventai import Decision
    sandbox = Sandbox(
        policy(tmp_path, allowed_commands=(sys.executable,)),
        data_flow_checker=lambda command: Decision(DecisionType.BLOCK, DecisionReason.DATA_FLOW_VIOLATION),
    )
    assert sandbox.execute(command("print('blocked')")).decision.decision_type is DecisionType.BLOCK

    guarded = Sandbox(
        policy(tmp_path, allowed_commands=(sys.executable,)),
        output_guard=lambda output: Decision(DecisionType.BLOCK, DecisionReason.OUTPUT_POLICY_VIOLATION),
    )
    assert run(guarded, command("print('blocked')")).stdout == ""


def test_missing_authorization_context_fails_closed(tmp_path):
    sandbox = Sandbox(policy(tmp_path, allowed_commands=(sys.executable,)))
    result = sandbox.execute(command("print('must-not-run')"))
    assert result.decision.decision_type is DecisionType.BLOCK
    assert result.decision.reason is DecisionReason.MISSION_INVALID
