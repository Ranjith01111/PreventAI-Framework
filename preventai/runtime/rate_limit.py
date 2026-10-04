"""Thread-safe in-process rate limiting."""

from __future__ import annotations

from collections import deque
import threading
import time
from typing import Callable


class RateLimit:
    def __init__(
        self,
        max_calls: int,
        window_seconds: float,
        *,
        scope: str = "session",
        clock: Callable[[], float] | None = None,
    ) -> None:
        if not isinstance(max_calls, int) or isinstance(max_calls, bool) or max_calls <= 0:
            raise ValueError("max_calls must be a positive integer")
        if not isinstance(window_seconds, (int, float)) or isinstance(window_seconds, bool) or window_seconds <= 0:
            raise ValueError("window_seconds must be positive")
        if scope not in {"application", "agent", "tool", "session"}:
            raise ValueError("scope must be application, agent, tool, or session")
        self.max_calls = max_calls
        self.window_seconds = float(window_seconds)
        self.scope = scope
        self._clock = clock or time.monotonic
        self._events: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def allow(self, *, application_id: str, agent_id: str, tool_id: str, session_id: str) -> bool:
        key = {
            "application": application_id,
            "agent": agent_id,
            "tool": tool_id,
            "session": session_id,
        }[self.scope]
        now = self._clock()
        with self._lock:
            events = self._events.setdefault(key, deque())
            cutoff = now - self.window_seconds
            while events and events[0] <= cutoff:
                events.popleft()
            if len(events) >= self.max_calls:
                return False
            events.append(now)
            return True
