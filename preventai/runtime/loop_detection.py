"""Bounded deterministic execution-pattern loop detection."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import threading
from typing import Any

from ..core.decision import Decision, DecisionReason, DecisionType, Severity


@dataclass(frozen=True, slots=True)
class ExecutionSignature:
    agent_id: str
    session_id: str
    tool_id: str
    action: str
    resource: str

    def __post_init__(self) -> None:
        for field_name in ("agent_id", "session_id", "tool_id", "action", "resource"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip() or len(value) > 256:
                raise ValueError(f"{field_name} must be a non-empty string")
            if any(ord(char) < 32 for char in value):
                raise ValueError(f"{field_name} contains control characters")


@dataclass(frozen=True, slots=True)
class LoopPolicy:
    max_repeated_calls: int = 5
    cycle_length: int = 2
    max_history: int = 50

    def __post_init__(self) -> None:
        if not isinstance(self.max_repeated_calls, int) or isinstance(self.max_repeated_calls, bool) or self.max_repeated_calls < 2:
            raise ValueError("max_repeated_calls must be at least 2")
        if not isinstance(self.cycle_length, int) or isinstance(self.cycle_length, bool) or self.cycle_length < 2:
            raise ValueError("cycle_length must be at least 2")
        if not isinstance(self.max_history, int) or isinstance(self.max_history, bool) or self.max_history < self.cycle_length * 2:
            raise ValueError("max_history must contain at least two cycles")


class LoopDetector:
    """Detect consecutive repeats and fixed repeated cycles per scope."""

    def __init__(self, policy: LoopPolicy | None = None) -> None:
        self.policy = policy or LoopPolicy()
        self._history: dict[tuple[str, str], deque[ExecutionSignature]] = {}
        self._lock = threading.Lock()

    def observe(self, signature: ExecutionSignature) -> Decision | None:
        try:
            if not isinstance(signature, ExecutionSignature):
                return self._blocked("invalid_signature", 0)
            with self._lock:
                scope = (signature.agent_id, signature.session_id)
                history = self._history.setdefault(scope, deque(maxlen=self.policy.max_history))
                history.append(signature)
                pattern, length = self._detect(history)
            if pattern is None:
                return None
            return self._blocked(pattern, length, signature)
        except Exception:
            return self._blocked("detector_error", 0)

    def reset(self, session_id: str | None = None, agent_id: str | None = None) -> None:
        with self._lock:
            if session_id is None and agent_id is None:
                self._history.clear()
            elif session_id is not None and agent_id is not None:
                self._history.pop((agent_id, session_id), None)
            elif session_id is not None:
                for scope in tuple(self._history):
                    if scope[1] == session_id:
                        self._history.pop(scope, None)
            else:
                for scope in tuple(self._history):
                    if scope[0] == agent_id:
                        self._history.pop(scope, None)

    def history_size(self, session_id: str, agent_id: str | None = None) -> int:
        with self._lock:
            if agent_id is not None:
                return len(self._history.get((agent_id, session_id), ()))
            return sum(
                len(history)
                for (stored_agent, stored_session), history in self._history.items()
                if stored_session == session_id
            )

    def _detect(self, history: deque[ExecutionSignature]) -> tuple[str | None, int]:
        if len(history) >= self.policy.max_repeated_calls:
            tail = list(history)[-self.policy.max_repeated_calls:]
            if len({item for item in tail}) == 1:
                return "consecutive", 1
        for length in range(2, self.policy.cycle_length + 1):
            required = length * 2
            if len(history) < required:
                continue
            tail = list(history)[-required:]
            if tail[:length] == tail[length:] and len(set(tail)) > 1:
                return "cycle", length
        return None, 0

    @staticmethod
    def _blocked(pattern: str, cycle_length: int, signature: ExecutionSignature | None = None) -> Decision:
        metadata: dict[str, Any] = {"pattern": pattern}
        if cycle_length:
            metadata["cycle_length"] = cycle_length
        if signature is not None:
            metadata["agent_id"] = signature.agent_id
            metadata["tool_id"] = signature.tool_id
        return Decision(
            decision_type=DecisionType.BLOCK,
            reason=DecisionReason.LOOP_DETECTED,
            severity=Severity.HIGH,
            metadata=metadata,
        )
