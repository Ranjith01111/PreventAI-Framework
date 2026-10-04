import threading

import pytest

from preventai import (
    ExecutionSignature,
    LoopDetector,
    LoopPolicy,
    DecisionReason,
    DecisionType,
)


def signature(tool):
    return ExecutionSignature("agent", "session", tool, tool, tool)


def test_consecutive_and_single_tool_repeats():
    detector = LoopDetector(LoopPolicy(max_repeated_calls=3, cycle_length=2, max_history=10))
    assert detector.observe(signature("a")) is None
    assert detector.observe(signature("a")) is None
    decision = detector.observe(signature("a"))
    assert decision is not None
    assert decision.reason is DecisionReason.LOOP_DETECTED
    assert decision.metadata["pattern"] == "consecutive"


def test_alternating_and_fixed_length_cycles():
    detector = LoopDetector(LoopPolicy(max_repeated_calls=5, cycle_length=3, max_history=20))
    for tool in ("a", "b", "a"):
        assert detector.observe(signature(tool)) is None
    decision = detector.observe(signature("b"))
    assert decision is not None
    assert decision.metadata["pattern"] == "cycle"
    detector.reset()
    for tool in ("a", "b", "c", "a", "b", "c"):
        decision = detector.observe(signature(tool))
    assert decision is not None
    assert decision.metadata["cycle_length"] == 3


def test_non_loop_history_bound_reset_and_isolation():
    detector = LoopDetector(LoopPolicy(max_repeated_calls=4, cycle_length=2, max_history=4))
    assert all(detector.observe(signature(tool)) is None for tool in ("a", "b", "c", "d"))
    assert detector.history_size("session") == 4
    detector.reset("session")
    assert detector.history_size("session") == 0
    detector.observe(ExecutionSignature("agent-a", "session-a", "a", "a", "a"))
    detector.observe(ExecutionSignature("agent-b", "session-b", "a", "a", "a"))
    assert detector.history_size("session-a") == 1
    assert detector.history_size("session-b") == 1


def test_agents_with_same_session_id_do_not_share_loop_state():
    detector = LoopDetector(LoopPolicy(max_repeated_calls=3, cycle_length=2, max_history=10))
    for agent in ("agent-a", "agent-b", "agent-a", "agent-b"):
        assert detector.observe(
            ExecutionSignature(agent, "shared-session", "tool", "action", "resource")
        ) is None
    assert detector.history_size("shared-session", "agent-a") == 2
    assert detector.history_size("shared-session", "agent-b") == 2


def test_invalid_signature_is_blocked_without_recording_payload():
    detector = LoopDetector()
    decision = detector.observe("not-a-signature")
    assert decision is not None
    assert decision.reason is DecisionReason.LOOP_DETECTED
    assert detector.history_size("session") == 0


def test_thread_safe_observation_and_configuration_validation():
    detector = LoopDetector(LoopPolicy(max_repeated_calls=10, cycle_length=2, max_history=20))
    threads = [
        threading.Thread(target=lambda: [detector.observe(signature("a")) for _ in range(5)])
        for _ in range(4)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert detector.history_size("session") == 20
    with pytest.raises(ValueError):
        LoopPolicy(max_repeated_calls=1)
    with pytest.raises(ValueError):
        LoopPolicy(cycle_length=1)
    with pytest.raises(ValueError):
        LoopPolicy(max_history=2)


def test_detector_error_fails_closed():
    detector = LoopDetector()
    detector._history = None
    decision = detector.observe(signature("a"))
    assert decision.decision_type is DecisionType.BLOCK
    assert decision.reason is DecisionReason.LOOP_DETECTED
