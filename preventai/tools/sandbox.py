"""Policy-enforced, non-shell subprocess execution.

This is a process-execution control, not an OS or kernel security boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess
import threading
import time
import hmac
import hashlib
import secrets
from typing import Any, Callable, Mapping

from ..audit.event import AuditEventType
from ..audit.logger import AuditLogger
from ..core.decision import Decision, DecisionReason, DecisionType, Severity

_FRAMEWORK_ISSUER = object()


def _text(value: Any, field: str, limit: int = 256) -> str:
    if not isinstance(value, str) or not value or len(value) > limit or "\x00" in value:
        raise ValueError(f"{field} is invalid")
    return value


def _path(value: str, field: str) -> Path:
    return Path(_text(value, field)).resolve(strict=False)


@dataclass(frozen=True, slots=True)
class SandboxPolicy:
    enabled: bool = True
    allowed_commands: tuple[str, ...] = ()
    denied_commands: tuple[str, ...] = ()
    working_directory: str | None = None
    allowed_paths: tuple[str, ...] = ()
    denied_paths: tuple[str, ...] = ()
    environment_allowlist: tuple[str, ...] = ()
    allow_shell: bool = False
    max_processes: int = 1
    max_execution_seconds: float = 30.0
    max_output_bytes: int = 1_000_000
    max_input_bytes: int = 64_000

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool) or not isinstance(self.allow_shell, bool):
            raise ValueError("boolean policy fields are invalid")
        for field in (
            "allowed_commands", "denied_commands", "allowed_paths",
            "denied_paths", "environment_allowlist",
        ):
            values = getattr(self, field)
            if not isinstance(values, tuple) or any(
                not isinstance(value, str) or not value or "\x00" in value for value in values
            ):
                raise ValueError(f"{field} is invalid")
        if self.working_directory is not None:
            _path(self.working_directory, "working_directory")
        if self.max_processes != 1 or not isinstance(self.max_processes, int):
            raise ValueError("V1 supports exactly one subprocess")
        if not isinstance(self.max_execution_seconds, (int, float)) or self.max_execution_seconds <= 0:
            raise ValueError("max_execution_seconds must be positive")
        for field in ("max_output_bytes", "max_input_bytes"):
            value = getattr(self, field)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{field} must be positive")
        if set(self.allowed_commands) & set(self.denied_commands):
            raise ValueError("command allow and deny lists conflict")
        if not self.enabled and self.allow_shell:
            raise ValueError("disabled policy cannot enable shell execution")


@dataclass(frozen=True, slots=True)
class SandboxCommand:
    executable: str
    args: tuple[str, ...] = ()
    cwd: str | None = None
    environment: Mapping[str, str] | None = None
    shell: bool = False

    def __post_init__(self) -> None:
        _text(self.executable, "executable")
        if not isinstance(self.args, tuple) or any(
            not isinstance(arg, str) or "\x00" in arg or len(arg) > 4096 for arg in self.args
        ):
            raise ValueError("args are invalid")
        if self.cwd is not None:
            _path(self.cwd, "cwd")
        if not isinstance(self.shell, bool):
            raise ValueError("shell must be boolean")
        if self.environment is not None:
            if not isinstance(self.environment, Mapping):
                raise ValueError("environment must be an object")
            for key, value in self.environment.items():
                _text(key, "environment key", 128)
                _text(value, "environment value", 4096)


@dataclass(frozen=True, slots=True)
class SandboxResult:
    decision: Decision
    returncode: int | None = None
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False
    output_truncated: bool = False


@dataclass(frozen=True, slots=True)
class SandboxAuthorization:
    agent_id: str
    session_id: str
    tool_id: str
    action: str
    resource: str
    command_digest: str
    expires_at: float
    _proof: str


class SandboxValidator:
    def __init__(self, policy: SandboxPolicy) -> None:
        if not isinstance(policy, SandboxPolicy):
            raise ValueError("policy must be a SandboxPolicy")
        self.policy = policy

    def validate(self, command: SandboxCommand) -> Decision:
        try:
            if not self.policy.enabled:
                return self._block("disabled")
            executable = Path(command.executable).name
            if command.shell and not self.policy.allow_shell:
                return self._block("shell_disabled")
            if command.shell:
                return self._block("shell_not_supported")
            if not self.policy.allowed_commands or executable not in self.policy.allowed_commands and command.executable not in self.policy.allowed_commands:
                return self._block("executable_not_allowed")
            if executable in self.policy.denied_commands or command.executable in self.policy.denied_commands:
                return self._block("executable_denied")
            input_size = sum(len(arg.encode()) for arg in command.args) + len(command.executable.encode())
            if input_size > self.policy.max_input_bytes:
                return self._block("input_limit")
            cwd = self._cwd(command)
            if cwd is None or not cwd.is_dir() or not self._allowed(cwd):
                return self._block("working_directory")
            for variable in (command.environment or {}):
                if variable not in self.policy.environment_allowlist:
                    return self._block("environment_not_allowed")
            return Decision(DecisionType.ALLOW, DecisionReason.REQUEST_ALLOWED)
        except Exception:
            return self._block("validation_error")

    def cwd(self, command: SandboxCommand) -> str | None:
        value = self._cwd(command)
        return str(value) if value is not None and self._allowed(value) else None

    def _cwd(self, command: SandboxCommand) -> Path | None:
        if command.cwd is None and self.policy.working_directory is None:
            return None
        return _path(command.cwd or self.policy.working_directory or "", "cwd")

    def _allowed(self, value: Path) -> bool:
        allowed = tuple(_path(item, "allowed_path") for item in self.policy.allowed_paths)
        denied = tuple(_path(item, "denied_path") for item in self.policy.denied_paths)
        return bool(allowed) and any(value == root or root in value.parents for root in allowed) and not any(
            value == root or root in value.parents for root in denied
        )

    @staticmethod
    def _block(category: str) -> Decision:
        return Decision(
            DecisionType.BLOCK,
            DecisionReason.SANDBOX_POLICY_VIOLATION,
            severity=Severity.HIGH,
            metadata={"category": category},
        )


class Sandbox:
    def __init__(
        self,
        policy: SandboxPolicy,
        *,
        audit_logger: AuditLogger | None = None,
        permission_checker: Callable[[SandboxCommand], Decision] | None = None,
        data_flow_checker: Callable[[SandboxCommand], Decision] | None = None,
        output_guard: Callable[[Mapping[str, Any]], Any] | None = None,
    ) -> None:
        self.policy = policy
        self.validator = SandboxValidator(policy)
        self.audit_logger = audit_logger
        self.permission_checker = permission_checker
        self.data_flow_checker = data_flow_checker
        self.output_guard = output_guard
        self._lock = threading.RLock()
        self._authorization_secret = secrets.token_bytes(32)
        self._revoked_authorizations: set[str] = set()
        self._mission_registry = None
        self._mission_agent_id: str | None = None
        self._mission_tool: str | None = None

    def bind_mission_context(
        self, mission_registry: object, agent_id: str, *, tool: str | None = None
    ) -> None:
        from ..identity.mission import MissionRegistry
        if (
            not isinstance(mission_registry, MissionRegistry)
            or not mission_registry.application_owned
            or not isinstance(agent_id, str)
        ):
            raise TypeError("authoritative mission context is required")
        self._mission_registry = mission_registry
        self._mission_agent_id = agent_id
        self._mission_tool = tool

    def revoke_authorization(self, authorization: SandboxAuthorization) -> None:
        if not isinstance(authorization, SandboxAuthorization):
            raise ValueError("authorization is invalid")
        self._revoked_authorizations.add(authorization._proof)

    def _issue_authorization(
        self, *, agent_id: str, session_id: str, tool_id: str,
        action: str, resource: str, command: SandboxCommand | None = None,
        ttl: float = 30.0, _issuer: object | None = None,
    ) -> SandboxAuthorization:
        if _issuer is not _FRAMEWORK_ISSUER:
            raise PermissionError("sandbox authorization issuance is framework-only")
        digest = self._command_digest(command) if command is not None else ""
        values = (agent_id, session_id, tool_id, action, resource, digest, str(time.monotonic() + ttl))
        proof = self._sign(values)
        return SandboxAuthorization(*values[:6], float(values[6]), proof)

    def _sign(self, values: tuple[str, ...]) -> str:
        return hmac.new(
            self._authorization_secret,
            "\x1f".join(values).encode(),
            hashlib.sha256,
        ).hexdigest()

    def _valid_authorization(
        self, authorization: SandboxAuthorization, *,
        agent_id: str | None, session_id: str | None, tool_id: str | None,
        action: str | None, resource: str | None,
        command: SandboxCommand,
    ) -> bool:
        if not isinstance(authorization, SandboxAuthorization):
            return False
        if authorization._proof in self._revoked_authorizations:
            return False
        if time.monotonic() >= authorization.expires_at:
            return False
        if (authorization.agent_id, authorization.session_id, authorization.tool_id) != (
            agent_id, session_id, tool_id
        ):
            return False
        if (authorization.action, authorization.resource) != (action, resource):
            return False
        values = (
            authorization.agent_id, authorization.session_id, authorization.tool_id,
            authorization.action, authorization.resource, authorization.command_digest,
            str(authorization.expires_at),
        )
        return (
            hmac.compare_digest(authorization._proof, self._sign(values))
            and (
                not authorization.command_digest
                or authorization.command_digest == self._command_digest(command)
            )
        )

    @staticmethod
    def _command_digest(command: SandboxCommand | None) -> str:
        if command is None:
            return ""
        value = "\x1f".join((command.executable, *command.args, command.cwd or ""))
        return hashlib.sha256(value.encode()).hexdigest()

    def execute(self, command: SandboxCommand, *, authorization: SandboxAuthorization | None = None,
                agent_id: str | None = None, session_id: str | None = None,
                tool_id: str | None = None, action: str | None = None,
                resource: str | None = None) -> SandboxResult:
        correlation_id = f"sandbox-{time.monotonic_ns()}"
        try:
            if self._mission_registry is None:
                decision = self._error("mission_context_missing", DecisionReason.MISSION_INVALID)
                self._audit(AuditEventType.SANDBOX_BLOCKED, correlation_id, decision, agent_id, session_id, tool_id)
                return SandboxResult(decision)
            decision = self._mission_registry.check(
                self._mission_agent_id,
                action=action or "",
                resource=resource or "",
                tool=self._mission_tool or tool_id or "",
                data="internal",
                operation="sandbox_execution",
            )
            if decision.decision_type is not DecisionType.ALLOW:
                return SandboxResult(decision)
            decision = self.validator.validate(command)
            if decision.decision_type is not DecisionType.ALLOW:
                self._audit(AuditEventType.SANDBOX_BLOCKED, correlation_id, decision, agent_id, session_id, tool_id)
                return SandboxResult(decision)
            if not self._valid_authorization(
                authorization, agent_id=agent_id, session_id=session_id, tool_id=tool_id,
                action=action, resource=resource, command=command,
            ):
                decision = self._error("missing_security_context")
                self._audit(AuditEventType.SANDBOX_BLOCKED, correlation_id, decision, agent_id, session_id, tool_id)
                return SandboxResult(decision)
            for checker in (self.permission_checker, self.data_flow_checker):
                if checker is not None:
                    checked = checker(command)
                    if not isinstance(checked, Decision) or checked.decision_type is not DecisionType.ALLOW:
                        result = checked if isinstance(checked, Decision) else self._error("security_evaluator")
                        self._audit(AuditEventType.SANDBOX_BLOCKED, correlation_id, result, agent_id, session_id, tool_id)
                        return SandboxResult(result)
            cwd = self.validator.cwd(command)
            environment = {
                key: os.environ[key] for key in self.policy.environment_allowlist if key in os.environ
            }
            environment.update(command.environment or {})
            self._audit(AuditEventType.SANDBOX_EXECUTION_STARTED, correlation_id, decision, agent_id, session_id, tool_id)
            with self._lock:
                process = subprocess.Popen(
                    [command.executable, *command.args],
                    cwd=cwd,
                    env=environment,
                    shell=False,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=False,
                )
                stdout, stderr, exceeded, timed_out = self._communicate_bounded(process)
                if timed_out:
                    return SandboxResult(
                        self._error("timeout", DecisionReason.SANDBOX_TIMEOUT),
                        process.returncode, self._bounded(stdout), self._bounded(stderr),
                        timed_out=True, output_truncated=exceeded,
                    )
            truncated = exceeded or len(stdout) > self.policy.max_output_bytes or len(stderr) > self.policy.max_output_bytes
            result = SandboxResult(
                Decision(DecisionType.ALLOW, DecisionReason.REQUEST_ALLOWED),
                process.returncode, self._bounded(stdout), self._bounded(stderr),
                output_truncated=truncated,
            )
            if self.output_guard is not None:
                guarded = self.output_guard({"stdout": result.stdout, "stderr": result.stderr, "returncode": result.returncode})
                if isinstance(guarded, Decision) and guarded.decision_type is not DecisionType.ALLOW:
                    return SandboxResult(guarded, result.returncode, "", "")
            self._audit(AuditEventType.SANDBOX_EXECUTION_SUCCEEDED, correlation_id, result.decision, agent_id, session_id, tool_id)
            return result
        except Exception:
            result = SandboxResult(self._error("execution_error"))
            self._audit(AuditEventType.SANDBOX_EXECUTION_FAILED, correlation_id, result.decision, agent_id, session_id, tool_id)
            return result

    def _bounded(self, value: bytes) -> str:
        return value[: self.policy.max_output_bytes].decode("utf-8", errors="replace")

    def _communicate_bounded(self, process: subprocess.Popen[bytes]) -> tuple[bytes, bytes, bool, bool]:
        outputs: dict[str, bytearray] = {"stdout": bytearray(), "stderr": bytearray()}
        exceeded = threading.Event()

        def read_stream(name: str, stream) -> None:
            try:
                while True:
                    chunk = stream.read(4096)
                    if not chunk:
                        return
                    if len(outputs[name]) < self.policy.max_output_bytes + 1:
                        remaining = self.policy.max_output_bytes + 1 - len(outputs[name])
                        outputs[name].extend(chunk[:remaining])
                    if len(outputs[name]) > self.policy.max_output_bytes:
                        exceeded.set()
                        return
            finally:
                stream.close()

        threads = [
            threading.Thread(target=read_stream, args=("stdout", process.stdout), daemon=True),
            threading.Thread(target=read_stream, args=("stderr", process.stderr), daemon=True),
        ]
        for thread in threads:
            thread.start()
        if process.stdin is not None:
            process.stdin.close()
        deadline = time.monotonic() + self.policy.max_execution_seconds
        timed_out = False
        while process.poll() is None:
            if exceeded.is_set():
                process.kill()
                break
            if time.monotonic() >= deadline:
                process.kill()
                timed_out = True
                break
            time.sleep(0.005)
        process.wait()
        for thread in threads:
            thread.join(timeout=1)
        return bytes(outputs["stdout"]), bytes(outputs["stderr"]), exceeded.is_set(), timed_out

    def _audit(self, event_type, correlation_id, decision, agent_id, session_id, tool_id):
        if self.audit_logger is None:
            return
        try:
            self.audit_logger.record(
                event_type, correlation_id=correlation_id, agent_id=agent_id,
                session_id=session_id, tool_id=tool_id,
                decision=decision.decision_type.value, reason=decision.reason.value,
                metadata={"category": "sandbox"},
            )
        except Exception:
            if self.audit_logger.policy.required:
                raise

    @staticmethod
    def _error(category: str, reason: DecisionReason = DecisionReason.SANDBOX_SECURITY_ERROR) -> Decision:
        return Decision(DecisionType.BLOCK, reason, severity=Severity.HIGH, metadata={"category": category})
