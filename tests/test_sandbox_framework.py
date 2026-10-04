import sys

import pytest

from preventai import Agent, DecisionType, PreventApp, ToolExecutionDenied
from preventai.core import PermissionCategory, PermissionEngine, PermissionRule, PolicyEngine, PolicyRule
from preventai.tools import SandboxCommand, SandboxPolicy


def test_framework_managed_sandbox_runs_after_security_pipeline(tmp_path):
    policy = PolicyEngine([PolicyRule("allow", DecisionType.ALLOW, action="run")])
    permissions = PermissionEngine([
        PermissionRule("execute", PermissionCategory.EXECUTE, action="run")
    ])
    app = PreventApp(policy=policy, permissions=permissions)
    agent = app.add_agent(Agent("sandbox-agent"))

    @agent.tool(
        sandbox_policy=SandboxPolicy(
            allowed_commands=(sys.executable,),
            working_directory=str(tmp_path),
            allowed_paths=(str(tmp_path),),
        )
    )
    def run():
        return SandboxCommand(sys.executable, ("-c", "print('framework-safe')"))

    app.run()
    result = run()
    assert result["stdout"].strip() == "framework-safe"


def test_framework_sandbox_requires_a_command_result(tmp_path):
    policy = PolicyEngine([PolicyRule("allow", DecisionType.ALLOW, action="run")])
    permissions = PermissionEngine([
        PermissionRule("execute", PermissionCategory.EXECUTE, action="run")
    ])
    app = PreventApp(policy=policy, permissions=permissions)
    agent = app.add_agent(Agent("sandbox-agent"))

    @agent.tool(
        sandbox_policy=SandboxPolicy(
            allowed_commands=(sys.executable,),
            working_directory=str(tmp_path),
            allowed_paths=(str(tmp_path),),
        )
    )
    def run():
        return "not a command"

    app.run()
    with pytest.raises(ToolExecutionDenied) as error:
        run()
    assert error.value.decision.decision_type is DecisionType.BLOCK


def test_pre_callable_policy_denial_starts_no_subprocess(tmp_path, monkeypatch):
    policy = PolicyEngine([PolicyRule("deny", DecisionType.BLOCK, action="run")])
    permissions = PermissionEngine([
        PermissionRule("execute", PermissionCategory.EXECUTE, action="run")
    ])
    app = PreventApp(policy=policy, permissions=permissions)
    agent = app.add_agent(Agent("sandbox-agent"))
    started = []

    def unexpected_process(*args, **kwargs):
        started.append(True)
        raise AssertionError("subprocess must not start")

    monkeypatch.setattr("preventai.tools.sandbox.subprocess.Popen", unexpected_process)

    @agent.tool(
        sandbox_policy=SandboxPolicy(
            allowed_commands=(sys.executable,),
            working_directory=str(tmp_path),
            allowed_paths=(str(tmp_path),),
        )
    )
    def run():
        return SandboxCommand(sys.executable, ("-c", "print('must-not-run')"))

    app.run()
    with pytest.raises(ToolExecutionDenied):
        run()
    assert started == []
