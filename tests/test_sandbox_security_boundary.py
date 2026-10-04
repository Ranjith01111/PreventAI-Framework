import sys
from dataclasses import replace

from preventai import DecisionType
from preventai.tools import Sandbox, SandboxCommand, SandboxPolicy
from preventai.tools.sandbox import _FRAMEWORK_ISSUER


def make(tmp_path):
    return Sandbox(SandboxPolicy(
        allowed_commands=(sys.executable,),
        working_directory=str(tmp_path),
        allowed_paths=(str(tmp_path),),
    ))


def command():
    return SandboxCommand(sys.executable, ("-c", "print('boundary')"))


def auth(sandbox, command_value=None):
    return sandbox._issue_authorization(
        agent_id="agent", session_id="session", tool_id="tool",
        action="execute", resource="resource", command=command_value or command(),
        ttl=30,
        _issuer=_FRAMEWORK_ISSUER,
    )


def test_missing_fake_and_malformed_contexts_block(tmp_path):
    sandbox = make(tmp_path)
    value = command()
    try:
        sandbox._issue_authorization(
            agent_id="agent", session_id="session", tool_id="tool",
            action="execute", resource="resource", command=value,
        )
    except PermissionError:
        pass
    else:
        raise AssertionError("direct authorization issuance must be rejected")
    assert sandbox.execute(value).decision.decision_type is DecisionType.BLOCK
    assert sandbox.execute(value, authorization=True, agent_id="agent", session_id="session", tool_id="tool").decision.decision_type is DecisionType.BLOCK
    assert sandbox.execute(value, authorization=object(), agent_id="agent", session_id="session", tool_id="tool").decision.decision_type is DecisionType.BLOCK


def test_context_scope_and_command_binding_block_reuse(tmp_path):
    sandbox = make(tmp_path)
    authorization = auth(sandbox, command())
    assert sandbox.execute(command(), authorization=authorization, agent_id="other", session_id="session", tool_id="tool").decision.decision_type is DecisionType.BLOCK
    changed = SandboxCommand(sys.executable, ("-c", "print('different')"))
    assert sandbox.execute(changed, authorization=authorization, agent_id="agent", session_id="session", tool_id="tool").decision.decision_type is DecisionType.BLOCK


def test_revoked_and_expired_contexts_block(tmp_path):
    sandbox = make(tmp_path)
    authorization = auth(sandbox)
    sandbox.revoke_authorization(authorization)
    assert sandbox.execute(command(), authorization=authorization, agent_id="agent", session_id="session", tool_id="tool").decision.decision_type is DecisionType.BLOCK
    expired = sandbox._issue_authorization(
        agent_id="agent", session_id="session", tool_id="tool",
        action="execute", resource="resource", command=command(), ttl=-1,
        _issuer=_FRAMEWORK_ISSUER,
    )
    assert sandbox.execute(command(), authorization=expired, agent_id="agent", session_id="session", tool_id="tool").decision.decision_type is DecisionType.BLOCK


def test_boolean_and_string_authorization_cannot_execute(tmp_path):
    sandbox = make(tmp_path)
    for fake in (True, False, "approved", {"authorized": True}):
        result = sandbox.execute(command(), authorization=fake, agent_id="agent", session_id="session", tool_id="tool")
        assert result.decision.decision_type is DecisionType.BLOCK
