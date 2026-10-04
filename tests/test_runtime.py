import threading
import time

import pytest

from preventai import (
    Agent,
    DecisionReason,
    DecisionType,
    PreventApp,
    RateLimit,
    RuntimeBudget,
    RuntimePolicy,
    Timeout,
    ToolExecutionDenied,
    CircuitBreaker,
    LoopPolicy,
)
from preventai.core import (
    PermissionCategory,
    PermissionEngine,
    PermissionRule,
    PolicyEngine,
    PolicyRule,
)


def app(runtime_policy, *, policy_type=DecisionType.ALLOW):
    return PreventApp(
        policy=PolicyEngine([PolicyRule("policy", policy_type, action="lookup")]),
        permissions=PermissionEngine([
            PermissionRule("permission", PermissionCategory.READ, action="lookup")
        ]),
        runtime_policy=runtime_policy,
    )


def test_max_calls_and_budget_block_before_callable():
    calls = []
    application = app(RuntimePolicy(max_tool_calls=2, budget=RuntimeBudget(max_tool_calls=2)))
    agent = application.add_agent(Agent("runtime-agent"))

    @agent.tool()
    def lookup(value):
        calls.append(value)
        return value

    application.run()
    assert lookup("one") == "one"
    assert lookup("two") == "two"
    with pytest.raises(ToolExecutionDenied) as error:
        lookup("three")
    assert error.value.decision.reason is DecisionReason.TOOL_CALL_LIMIT_EXCEEDED
    assert calls == ["one", "two"]


def test_rate_limit_window_and_session_scope():
    limit = RateLimit(1, 0.02)
    application = app(RuntimePolicy(rate_limit=limit))
    agent = application.add_agent(Agent("rate-agent"))

    @agent.tool()
    def lookup(value):
        return value

    application.run()
    assert lookup("one") == "one"
    with pytest.raises(ToolExecutionDenied) as error:
        lookup("two")
    assert error.value.decision.reason is DecisionReason.RATE_LIMIT_EXCEEDED
    time.sleep(0.03)
    assert lookup("three") == "three"


def test_concurrency_limit_releases_after_completion():
    started = threading.Event()
    release = threading.Event()
    application = app(RuntimePolicy(max_concurrent_tools=1))
    agent = application.add_agent(Agent("concurrency-agent"))

    @agent.tool()
    def lookup(value):
        started.set()
        release.wait(1)
        return value

    application.run()
    first = threading.Thread(target=lambda: lookup("one"))
    first.start()
    assert started.wait(1)
    with pytest.raises(ToolExecutionDenied) as error:
        lookup("two")
    assert error.value.decision.reason is DecisionReason.CONCURRENCY_LIMIT_EXCEEDED
    release.set()
    first.join(1)
    assert lookup("three") == "three"


def test_timeout_blocks_result_but_does_not_claim_cancellation():
    application = app(RuntimePolicy(timeout=Timeout(0.01)))
    agent = application.add_agent(Agent("timeout-agent"))

    @agent.tool()
    def lookup(value):
        time.sleep(0.03)
        return value

    application.run()
    with pytest.raises(ToolExecutionDenied) as error:
        lookup("slow")
    assert error.value.decision.reason is DecisionReason.TIMEOUT
    assert "slow" not in str(error.value)


def test_runtime_configuration_validation():
    with pytest.raises(ValueError):
        RateLimit(0, 1)
    with pytest.raises(ValueError):
        Timeout(0)
    with pytest.raises(ValueError):
        RuntimeBudget(max_tool_calls=0)
    with pytest.raises(ValueError):
        RuntimePolicy(max_concurrent_tools=0)


def test_policy_and_permission_precede_runtime_controls():
    blocked = app(RuntimePolicy(max_tool_calls=1), policy_type=DecisionType.BLOCK)
    agent = blocked.add_agent(Agent("blocked-runtime-agent"))

    @agent.tool()
    def lookup(value):
        return value

    blocked.run()
    with pytest.raises(ToolExecutionDenied) as error:
        lookup("value")
    assert error.value.decision.reason is DecisionReason.REQUEST_BLOCKED

    denied = PreventApp(
        policy=PolicyEngine([PolicyRule("policy", DecisionType.ALLOW, action="lookup")]),
        permissions=PermissionEngine(),
        runtime_policy=RuntimePolicy(max_tool_calls=1),
    )
    agent = denied.add_agent(Agent("denied-runtime-agent"))

    @agent.tool("lookup")
    def denied_lookup(value):
        return value

    denied.run()
    with pytest.raises(ToolExecutionDenied) as error:
        denied_lookup("value")
    assert error.value.decision.reason is DecisionReason.MISSING_PERMISSION


def test_runtime_state_exception_fails_closed(monkeypatch):
    policy = RuntimePolicy(max_tool_calls=1)
    application = app(policy)
    agent = application.add_agent(Agent("runtime-error-agent"))

    @agent.tool()
    def lookup(value):
        return value

    application.run()
    monkeypatch.setattr(
        agent.tools["lookup"]._runtime_controller,
        "acquire",
        lambda *args: (_ for _ in ()).throw(RuntimeError("secret runtime detail")),
    )
    with pytest.raises(ToolExecutionDenied) as error:
        lookup("value")
    assert error.value.decision.reason is DecisionReason.RUNTIME_SECURITY_ERROR
    assert "secret runtime detail" not in str(error.value)


def test_runtime_policy_can_be_configured_per_tool():
    application = app(None)
    agent = application.add_agent(Agent("tool-runtime-agent"))

    @agent.tool(runtime_policy=RuntimePolicy(max_tool_calls=1))
    def lookup(value):
        return value

    application.run()
    assert lookup("one") == "one"
    with pytest.raises(ToolExecutionDenied) as error:
        lookup("two")
    assert error.value.decision.reason is DecisionReason.TOOL_CALL_LIMIT_EXCEEDED


def test_loop_detection_blocks_before_callable():
    calls = []
    application = app(RuntimePolicy(loop_policy=LoopPolicy(max_repeated_calls=2, cycle_length=2, max_history=10)))
    agent = application.add_agent(Agent("loop-runtime-agent"))

    @agent.tool()
    def lookup(value):
        calls.append(value)
        return value

    application.run()
    assert lookup("one") == "one"
    with pytest.raises(ToolExecutionDenied) as error:
        lookup("two")
    assert error.value.decision.reason is DecisionReason.LOOP_DETECTED
    assert calls == ["one"]


def test_circuit_breaker_counts_callable_failures_only():
    application = app(RuntimePolicy(circuit_breaker=CircuitBreaker(failure_threshold=2)))
    agent = application.add_agent(Agent("circuit-runtime-agent"))

    @agent.tool()
    def lookup(value):
        raise RuntimeError("tool failure")

    application.run()
    with pytest.raises(RuntimeError):
        lookup("one")
    with pytest.raises(RuntimeError):
        lookup("two")
    with pytest.raises(ToolExecutionDenied) as error:
        lookup("three")
    assert error.value.decision.reason is DecisionReason.CIRCUIT_OPEN
