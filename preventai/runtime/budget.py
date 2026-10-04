"""Thread-safe execution budgets."""

from __future__ import annotations

import threading
from typing import Callable


class RuntimeBudget:
    def __init__(
        self,
        *,
        max_tool_calls: int | None = None,
        max_execution_seconds: float | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        if max_tool_calls is not None and (
            not isinstance(max_tool_calls, int)
            or isinstance(max_tool_calls, bool)
            or max_tool_calls <= 0
        ):
            raise ValueError("max_tool_calls must be a positive integer")
        if max_execution_seconds is not None and (
            not isinstance(max_execution_seconds, (int, float))
            or isinstance(max_execution_seconds, bool)
            or max_execution_seconds <= 0
        ):
            raise ValueError("max_execution_seconds must be positive")
        self.max_tool_calls = max_tool_calls
        self.max_execution_seconds = (
            float(max_execution_seconds) if max_execution_seconds is not None else None
        )
        self._clock = clock
        self._calls = 0
        self._elapsed = 0.0
        self._lock = threading.Lock()

    def admit(self) -> bool:
        with self._lock:
            if self.max_tool_calls is not None and self._calls >= self.max_tool_calls:
                return False
            self._calls += 1
            return True

    def consume(self, elapsed_seconds: float = 0.0, *, admitted: bool = False) -> bool:
        if not isinstance(elapsed_seconds, (int, float)) or elapsed_seconds < 0:
            return False
        with self._lock:
            if not admitted and self.max_tool_calls is not None and self._calls >= self.max_tool_calls:
                return False
            if (
                self.max_execution_seconds is not None
                and self._elapsed + elapsed_seconds > self.max_execution_seconds
            ):
                return False
            if not admitted:
                self._calls += 1
            self._elapsed += float(elapsed_seconds)
            return True

    @property
    def calls(self) -> int:
        with self._lock:
            return self._calls

    @property
    def elapsed_seconds(self) -> float:
        with self._lock:
            return self._elapsed
