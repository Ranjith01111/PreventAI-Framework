"""Thread-safe deterministic circuit breaker."""

from __future__ import annotations

from enum import Enum
import threading
import time

from ..core.decision import Decision, DecisionReason, DecisionType, Severity


class CircuitState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 30,
        half_open_max_calls: int = 1,
    ) -> None:
        if not isinstance(failure_threshold, int) or isinstance(failure_threshold, bool) or failure_threshold <= 0:
            raise ValueError("failure_threshold must be positive")
        if not isinstance(recovery_timeout, (int, float)) or isinstance(recovery_timeout, bool) or recovery_timeout <= 0:
            raise ValueError("recovery_timeout must be positive")
        if not isinstance(half_open_max_calls, int) or isinstance(half_open_max_calls, bool) or half_open_max_calls <= 0:
            raise ValueError("half_open_max_calls must be positive")
        self.failure_threshold = failure_threshold
        self.recovery_timeout = float(recovery_timeout)
        self.half_open_max_calls = half_open_max_calls
        self._state = CircuitState.CLOSED
        self._failures = 0
        self._opened_at: float | None = None
        self._half_open_calls = 0
        self._lock = threading.Lock()

    @property
    def state(self) -> CircuitState:
        with self._lock:
            return self._state

    def before_execution(self) -> Decision | None:
        try:
            with self._lock:
                if self._state is CircuitState.OPEN:
                    if self._opened_at is None or time.monotonic() - self._opened_at < self.recovery_timeout:
                        return self._blocked()
                    self._state = CircuitState.HALF_OPEN
                    self._half_open_calls = 0
                if self._state is CircuitState.HALF_OPEN:
                    if self._half_open_calls >= self.half_open_max_calls:
                        return self._blocked()
                    self._half_open_calls += 1
                return None
        except Exception:
            return self._blocked()

    def record_success(self) -> None:
        with self._lock:
            self._state = CircuitState.CLOSED
            self._failures = 0
            self._opened_at = None
            self._half_open_calls = 0

    def record_failure(self) -> None:
        with self._lock:
            self._failures += 1
            if self._state is CircuitState.HALF_OPEN or self._failures >= self.failure_threshold:
                self._state = CircuitState.OPEN
                self._opened_at = time.monotonic()
                self._half_open_calls = 0

    @staticmethod
    def _blocked() -> Decision:
        return Decision(
            decision_type=DecisionType.BLOCK,
            reason=DecisionReason.CIRCUIT_OPEN,
            severity=Severity.HIGH,
            metadata={"state": "open"},
        )
