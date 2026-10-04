"""Runtime policy and atomic execution admission."""

from __future__ import annotations

from dataclasses import dataclass
import threading
from typing import Any

from ..core.decision import Decision, DecisionReason, DecisionType, Severity
from .budget import RuntimeBudget
from .rate_limit import RateLimit
from .timeout import Timeout
from .loop_detection import ExecutionSignature, LoopDetector, LoopPolicy
from .circuit_breaker import CircuitBreaker


@dataclass(frozen=True, slots=True)
class RuntimePolicy:
    rate_limit: RateLimit | None = None
    timeout: Timeout | None = None
    budget: RuntimeBudget | None = None
    max_tool_calls: int | None = None
    max_concurrent_tools: int | None = None
    loop_policy: LoopPolicy | None = None
    circuit_breaker: CircuitBreaker | None = None

    def __post_init__(self) -> None:
        if self.rate_limit is not None and not isinstance(self.rate_limit, RateLimit):
            raise ValueError("rate_limit must be a RateLimit")
        if self.timeout is not None and not isinstance(self.timeout, Timeout):
            raise ValueError("timeout must be a Timeout")
        if self.budget is not None and not isinstance(self.budget, RuntimeBudget):
            raise ValueError("budget must be a RuntimeBudget")
        if self.max_tool_calls is not None and (
            not isinstance(self.max_tool_calls, int)
            or isinstance(self.max_tool_calls, bool)
            or self.max_tool_calls <= 0
        ):
            raise ValueError("max_tool_calls must be a positive integer")
        if self.max_concurrent_tools is not None and (
            not isinstance(self.max_concurrent_tools, int)
            or isinstance(self.max_concurrent_tools, bool)
            or self.max_concurrent_tools <= 0
        ):
            raise ValueError("max_concurrent_tools must be a positive integer")
        if self.loop_policy is not None and not isinstance(self.loop_policy, LoopPolicy):
            raise ValueError("loop_policy must be a LoopPolicy")
        if self.circuit_breaker is not None and not isinstance(self.circuit_breaker, CircuitBreaker):
            raise ValueError("circuit_breaker must be a CircuitBreaker")


class RuntimeController:
    def __init__(self, policy: RuntimePolicy, *, application_id: str) -> None:
        self.policy = policy
        self.application_id = application_id
        self._calls: dict[str, int] = {}
        self._active = 0
        self._lock = threading.Lock()
        self._loop = LoopDetector(policy.loop_policy) if policy.loop_policy else None
        self._breakers: dict[tuple[str, str], CircuitBreaker] = {}

    def _breaker(self, agent_id: str, tool_id: str) -> CircuitBreaker | None:
        if self.policy.circuit_breaker is None:
            return None
        key = (agent_id, tool_id)
        with self._lock:
            return self._breakers.setdefault(
                key,
                CircuitBreaker(
                    self.policy.circuit_breaker.failure_threshold,
                    self.policy.circuit_breaker.recovery_timeout,
                    self.policy.circuit_breaker.half_open_max_calls,
                ),
            )

    def acquire(self, agent_id: str, tool_id: str, session_id: str) -> Decision | None:
        try:
            if self.policy.rate_limit and not self.policy.rate_limit.allow(
                application_id=self.application_id,
                agent_id=agent_id,
                tool_id=tool_id,
                session_id=session_id,
            ):
                return self._blocked(DecisionReason.RATE_LIMIT_EXCEEDED)
            with self._lock:
                key = session_id
                limit = self.policy.max_tool_calls
                if limit is not None and self._calls.get(key, 0) >= limit:
                    return self._blocked(DecisionReason.TOOL_CALL_LIMIT_EXCEEDED)
                if (
                    self.policy.max_concurrent_tools is not None
                    and self._active >= self.policy.max_concurrent_tools
                ):
                    return self._blocked(DecisionReason.CONCURRENCY_LIMIT_EXCEEDED)
                if self.policy.budget and not self.policy.budget.admit():
                    return self._blocked(DecisionReason.BUDGET_EXCEEDED)
                self._calls[key] = self._calls.get(key, 0) + 1
                self._active += 1
                return None
        except Exception:
            return self._blocked(DecisionReason.RUNTIME_SECURITY_ERROR)

    def release(self) -> None:
        with self._lock:
            self._active = max(0, self._active - 1)

    def check_loop(self, signature: ExecutionSignature) -> Decision | None:
        if self._loop is None:
            return None
        try:
            return self._loop.observe(signature)
        except Exception:
            return self._blocked(DecisionReason.RUNTIME_SECURITY_ERROR)

    def check_circuit(self, agent_id: str, tool_id: str) -> Decision | None:
        try:
            breaker = self._breaker(agent_id, tool_id)
            return breaker.before_execution() if breaker is not None else None
        except Exception:
            return self._blocked(DecisionReason.RUNTIME_SECURITY_ERROR)

    def record_success(self, agent_id: str, tool_id: str) -> Decision | None:
        try:
            breaker = self._breaker(agent_id, tool_id)
            if breaker is not None:
                breaker.record_success()
            return None
        except Exception:
            return self._blocked(DecisionReason.RUNTIME_SECURITY_ERROR)

    def record_failure(self, agent_id: str, tool_id: str) -> Decision | None:
        try:
            breaker = self._breaker(agent_id, tool_id)
            if breaker is not None:
                breaker.record_failure()
            return None
        except Exception:
            return self._blocked(DecisionReason.RUNTIME_SECURITY_ERROR)

    def record(self, elapsed_seconds: float) -> Decision | None:
        try:
            if self.policy.budget and not self.policy.budget.consume(elapsed_seconds, admitted=True):
                return self._blocked(DecisionReason.BUDGET_EXCEEDED)
            if self.policy.timeout and self.policy.timeout.exceeded_elapsed(elapsed_seconds):
                return self._blocked(DecisionReason.TIMEOUT)
            return None
        except Exception:
            return self._blocked(DecisionReason.RUNTIME_SECURITY_ERROR)

    @staticmethod
    def _blocked(reason: DecisionReason) -> Decision:
        return Decision(
            decision_type=DecisionType.BLOCK,
            reason=reason,
            severity=Severity.HIGH,
            metadata={"stage": "runtime"},
        )
