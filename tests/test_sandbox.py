from pathlib import Path

import pytest

from preventai import DecisionType
from preventai.tools import SandboxCommand, SandboxPolicy, SandboxValidator


def policy(tmp_path, **kwargs):
    return SandboxPolicy(
        allowed_commands=("python",),
        working_directory=str(tmp_path),
        allowed_paths=(str(tmp_path),),
        **kwargs,
    )


def test_secure_policy_requires_explicit_command_and_path(tmp_path):
    validator = SandboxValidator(policy(tmp_path))
    assert validator.validate(SandboxCommand("python", ("-c", "print(1)"))).decision_type is DecisionType.ALLOW
    assert validator.validate(SandboxCommand("cmd.exe", ())).decision_type is DecisionType.BLOCK
    assert validator.validate(SandboxCommand("python", (), cwd=str(Path(tmp_path).parent))).decision_type is DecisionType.BLOCK


def test_shell_and_environment_are_denied_by_default(tmp_path):
    validator = SandboxValidator(policy(tmp_path))
    assert validator.validate(SandboxCommand("python", (), shell=True)).decision_type is DecisionType.BLOCK
    assert validator.validate(SandboxCommand("python", (), environment={"SECRET": "value"})).decision_type is DecisionType.BLOCK


def test_argument_and_policy_validation_is_strict(tmp_path):
    with pytest.raises(ValueError):
        SandboxCommand("", ())
    with pytest.raises(ValueError):
        SandboxCommand("python", ("\x00",))
    with pytest.raises(ValueError):
        SandboxPolicy(max_output_bytes=0)
    assert SandboxValidator(policy(tmp_path, max_input_bytes=2)).validate(
        SandboxCommand("python", ("123",))
    ).decision_type is DecisionType.BLOCK


def test_path_traversal_is_blocked(tmp_path):
    validator = SandboxValidator(policy(tmp_path))
    command = SandboxCommand("python", (), cwd=str(Path(tmp_path).parent))
    assert validator.validate(command).decision_type is DecisionType.BLOCK
