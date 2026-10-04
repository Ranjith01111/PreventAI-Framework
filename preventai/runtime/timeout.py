"""Synchronous elapsed-time timeout configuration."""

from __future__ import annotations

import time


class Timeout:
    def __init__(self, seconds: float) -> None:
        if not isinstance(seconds, (int, float)) or isinstance(seconds, bool) or seconds <= 0:
            raise ValueError("seconds must be positive")
        self.seconds = float(seconds)

    @staticmethod
    def start() -> float:
        return time.monotonic()

    def exceeded(self, started_at: float) -> bool:
        return time.monotonic() - started_at > self.seconds

    def exceeded_elapsed(self, elapsed_seconds: float) -> bool:
        return elapsed_seconds > self.seconds
