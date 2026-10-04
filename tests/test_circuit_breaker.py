import time
import threading

import pytest

from preventai import CircuitBreaker, CircuitState, DecisionReason, DecisionType


def test_failure_threshold_opens_and_open_blocks():
    breaker = CircuitBreaker(failure_threshold=2, recovery_timeout=0.02)
    assert breaker.before_execution() is None
    breaker.record_failure()
    assert breaker.state is CircuitState.CLOSED
    breaker.record_failure()
    assert breaker.state is CircuitState.OPEN
    decision = breaker.before_execution()
    assert decision is not None
    assert decision.reason is DecisionReason.CIRCUIT_OPEN
    assert decision.metadata == {"state": "open"}


def test_half_open_success_closes_and_failure_reopens():
    breaker = CircuitBreaker(failure_threshold=1, recovery_timeout=0.01, half_open_max_calls=1)
    breaker.record_failure()
    time.sleep(0.02)
    assert breaker.before_execution() is None
    assert breaker.state is CircuitState.HALF_OPEN
    breaker.record_success()
    assert breaker.state is CircuitState.CLOSED
    breaker.record_failure()
    time.sleep(0.02)
    assert breaker.before_execution() is None
    breaker.record_failure()
    assert breaker.state is CircuitState.OPEN


def test_half_open_allows_only_configured_calls_concurrently():
    breaker = CircuitBreaker(failure_threshold=1, recovery_timeout=0.01, half_open_max_calls=1)
    breaker.record_failure()
    time.sleep(0.02)
    results = []

    def admit():
        results.append(breaker.before_execution())

    threads = [threading.Thread(target=admit) for _ in range(3)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert sum(result is None for result in results) == 1
    assert sum(result is not None for result in results) == 2


def test_configuration_validation():
    with pytest.raises(ValueError):
        CircuitBreaker(failure_threshold=0)
    with pytest.raises(ValueError):
        CircuitBreaker(recovery_timeout=0)
    with pytest.raises(ValueError):
        CircuitBreaker(half_open_max_calls=0)


def test_circuit_failure_does_not_expose_details():
    breaker = CircuitBreaker(failure_threshold=1)
    breaker.record_failure()
    decision = breaker.before_execution()
    assert decision.decision_type is DecisionType.BLOCK
    assert "failure" not in decision.to_json()
