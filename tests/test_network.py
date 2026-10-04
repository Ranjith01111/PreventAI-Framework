from dataclasses import replace
from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor
import ssl

import pytest

import preventai.tools.network as network_module
from preventai import (
    Agent,
    ApprovalManager,
    ApprovalPolicy,
    ApprovalRule,
    Authenticator,
    Capability,
    ContentGuard,
    Credential,
    AuditLogger,
    DataFlowEngine,
    DataFlowPolicy,
    DataSource,
    DecisionType,
    DataFlowDestination,
    FlowClassification,
    FlowTrust,
    OutputGuard,
    PreventApp,
    ToolExecutionDenied,
    ToolApprovalRequired,
)
from preventai.core.decision import DecisionReason
from preventai.core import (
    PermissionCategory,
    PermissionEngine,
    PermissionRule,
    PolicyEngine,
    PolicyRule,
)
from preventai.approval import human_approval as approval_module
from preventai.runtime import RuntimeController, RuntimePolicy
from preventai.tools import (
    Network,
    NetworkAuthorization,
    NetworkExecutionDenied,
    NetworkPolicy,
    NetworkRequest,
    NetworkResponse,
)
from preventai.tools.network import _FRAMEWORK_ISSUER, _PinnedHTTPSConnection
from _boundary_test_support import bind_test_boundary


def make_request(url="https://example.com/data", **kwargs):
    return NetworkRequest(url=url, **kwargs)


def make_network(**kwargs):
    kwargs.setdefault("allowed_hosts", ("example.com",))
    return bind_test_boundary(Network(NetworkPolicy(**kwargs)))


def authorized(network, request=None, *, addresses=("93.184.216.34",), ttl=30):
    request = request or make_request()
    request = replace(request, agent_id="agent", session_id="session", tool_id="tool")
    plan = network.prepare(request)
    assert plan.addresses == addresses
    token = network._issue_authorization(
        request,
        resolved_addresses=plan.addresses,
        ttl=ttl,
        _issuer=_FRAMEWORK_ISSUER,
    )
    return request, token


def fake_io(monkeypatch, network, *, responses=None, addresses=("93.184.216.34",)):
    calls = []
    queue = list(responses or [(200, {"content-type": "text/plain"}, b"ok")])
    monkeypatch.setattr(network, "_resolve", lambda host, port: addresses)

    def request_once(request, resolved_addresses, timeout):
        calls.append((request, resolved_addresses, timeout))
        if len(queue) > 1:
            return queue.pop(0)
        return queue[0]

    monkeypatch.setattr(network, "_request_once", request_once)
    return calls


def test_policy_denies_all_hosts_by_default():
    network = bind_test_boundary(Network(NetworkPolicy()))
    assert network._policy_decision(make_request()) is not None


def test_default_scheme_and_https_requirement_are_restrictive():
    assert make_network()._policy_decision(make_request()) is None
    assert make_network()._policy_decision(make_request("http://example.com/")) is not None


def test_explicit_http_requires_both_scheme_and_https_override():
    network = make_network(allowed_schemes=("http",), require_https=False)
    assert network._policy_decision(make_request("http://example.com/")) is None


@pytest.mark.parametrize(
    "url",
    [
        "ftp://example.com/",
        "file:///etc/passwd",
        "https:///missing-host",
        "https://user:password@example.com/",
        "https://example.com:bad/",
        "https://example.com:0/",
        "https://[xyz]/",
        "https://example.com\\@evil.test/",
        "https://example.com/%GG",
    ],
)
def test_malformed_or_unsupported_urls_are_rejected(url):
    with pytest.raises(ValueError):
        make_request(url)


def test_url_canonicalization_normalizes_scheme_host_default_port_and_fragment():
    request = make_request("HTTPS://EXAMPLE.COM:443/a%2fb?q=x#ignored")
    assert request.url == "https://example.com/a%2Fb?q=x"
    assert request.scheme == "https"
    assert request.host == "example.com"
    assert request.port == 443
    assert request.path == "/a%2Fb?q=x"


def test_explicit_non_default_port_requires_policy_allowlist():
    network = make_network()
    assert network._policy_decision(make_request("https://example.com:8443/")) is not None
    network = make_network(allowed_ports=(8443,))
    assert network._policy_decision(make_request("https://example.com:8443/")) is None


def test_denied_port_takes_precedence_over_allowed_ports():
    network = make_network(allowed_ports=(443, 8443), denied_ports=(8443,))
    assert network._policy_decision(make_request("https://example.com:8443/")) is not None


def test_denied_http_method_blocks():
    network = make_network(allowed_methods=("GET",))
    assert network._policy_decision(make_request(method="POST")) is not None


def test_wildcard_host_is_suffix_scoped():
    network = bind_test_boundary(Network(NetworkPolicy(allowed_hosts=("*.example.com",))))
    assert network._policy_decision(make_request("https://api.example.com/")) is None
    assert network._policy_decision(make_request("https://example.com/")) is not None
    assert network._policy_decision(make_request("https://example.com.evil.test/")) is not None


def test_denied_host_overrides_allowlist():
    network = bind_test_boundary(Network(NetworkPolicy(
        allowed_hosts=("*.example.com",),
        denied_hosts=("private.example.com",),
    )))
    assert network._policy_decision(make_request("https://private.example.com/")) is not None


def test_request_body_limit_is_enforced_before_connection():
    network = make_network(max_request_bytes=2)
    request = make_request(method="POST", body=b"too large")
    assert network._policy_decision(request) is not None


def test_request_size_limit_includes_headers():
    network = make_network(max_request_bytes=32)
    request = make_request(headers={"x-trace": "v" * 64})
    assert network._policy_decision(request) is not None


def test_request_digest_changes_for_body_method_header_and_context():
    original = make_request()
    variants = (
        make_request(url=original.url, method="POST"),
        make_request(url=original.url, headers={"accept": "application/json"}),
        make_request(url=original.url, body=b"payload"),
        make_request(url=original.url, context={"version": 1}),
    )
    assert all(item.digest != original.digest for item in variants)
    assert make_request("https://example.com/data?x=1").digest != make_request(
        "https://example.com/data?x=2"
    ).digest


@pytest.mark.parametrize(
    "header",
    ["Host", "Forwarded", "X-Forwarded-Host", "X-Forwarded-For", "Content-Length", "Proxy-Authorization"],
)
def test_destination_and_framing_headers_are_framework_controlled(header):
    with pytest.raises(ValueError):
        make_request(headers={header: "attacker-controlled"})


def test_allowed_private_cidr_is_explicit_internal_authority(monkeypatch):
    network = make_network(allowed_ip_ranges=("10.0.0.0/8",))
    monkeypatch.setattr(network, "_resolve", lambda host, port: ("10.2.3.4",))
    plan = network.prepare(make_request())
    assert plan.internal


def test_configured_ip_allowlist_also_restricts_public_dns_answers(monkeypatch):
    network = make_network(allowed_ip_ranges=("93.184.216.0/24",))
    monkeypatch.setattr(network, "_resolve", lambda host, port: ("8.8.8.8",))
    with pytest.raises(NetworkExecutionDenied):
        network.prepare(make_request())


@pytest.mark.parametrize(
    "address",
    ["127.0.0.1", "::1", "10.2.3.4", "169.254.1.1", "0.0.0.0", "224.0.0.1", "100.64.0.1"],
)
def test_non_public_destinations_block_by_default(monkeypatch, address):
    network = make_network()
    monkeypatch.setattr(network, "_resolve", lambda host, port: (address,))
    with pytest.raises(NetworkExecutionDenied) as error:
        network.prepare(make_request())
    assert error.value.decision.reason is DecisionReason.NETWORK_SSRF_BLOCKED


def test_cloud_metadata_destination_is_blocked(monkeypatch):
    network = make_network()
    monkeypatch.setattr(network, "_resolve", lambda host, port: ("169.254.169.254",))
    with pytest.raises(NetworkExecutionDenied):
        network.prepare(make_request())


def test_denied_cidr_overrides_explicitly_allowed_cidr(monkeypatch):
    network = make_network(
        allowed_ip_ranges=("10.0.0.0/8",),
        denied_ip_ranges=("10.1.0.0/16",),
    )
    monkeypatch.setattr(network, "_resolve", lambda host, port: ("10.1.2.3",))
    with pytest.raises(NetworkExecutionDenied):
        network.prepare(make_request())


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("2001:0db8:0:0:0:0:0:1", "2001:db8::1"),
        ("::ffff:93.184.216.34", "93.184.216.34"),
    ],
)
def test_ip_literals_are_normalized(value, expected):
    from preventai.tools.network import _host

    assert _host(value, "host") == expected


def test_dns_mixed_public_and_private_answers_block(monkeypatch):
    network = make_network()
    monkeypatch.setattr(network, "_resolve", lambda host, port: ("93.184.216.34", "127.0.0.1"))
    with pytest.raises(NetworkExecutionDenied):
        network.prepare(make_request())


def test_dns_resolution_failure_blocks_without_request_connection(monkeypatch):
    network = make_network()
    connected = []
    monkeypatch.setattr(network, "_resolve", lambda host, port: (_ for _ in ()).throw(OSError()))
    monkeypatch.setattr(network, "_request_once", lambda *args: connected.append(True))
    with pytest.raises(NetworkExecutionDenied) as error:
        network.prepare(make_request())
    assert error.value.decision.reason is DecisionReason.NETWORK_DNS_BLOCKED
    assert connected == []


def test_validated_addresses_are_used_by_authoritative_connector(monkeypatch):
    network = make_network()
    calls = fake_io(monkeypatch, network)
    request, token = authorized(network)
    result = network.execute(request, authorization=token)
    assert result.body == b"ok"
    assert calls[0][1] == ("93.184.216.34",)


def test_missing_fake_and_malformed_authorizations_block_without_connection(monkeypatch):
    network = make_network()
    calls = fake_io(monkeypatch, network)
    request = replace(make_request(), agent_id="agent", session_id="session", tool_id="tool")
    for value in (None, True, "authorized", {"allow": True}, object()):
        with pytest.raises(NetworkExecutionDenied):
            network.execute(request, authorization=value)
    assert calls == []


def test_caller_cannot_mint_framework_authorization():
    network = make_network()
    with pytest.raises(PermissionError):
        network._issue_authorization(
            replace(make_request(), agent_id="agent", session_id="session", tool_id="tool"),
            resolved_addresses=("93.184.216.34",),
        )


def test_expired_authorization_blocks(monkeypatch):
    network = make_network()
    calls = fake_io(monkeypatch, network)
    current = [100.0]
    monkeypatch.setattr(network_module.time, "monotonic", lambda: current[0])
    request, token = authorized(network, ttl=1)
    current[0] = 102.0
    with pytest.raises(NetworkExecutionDenied):
        network.execute(request, authorization=token)
    assert calls == []


def test_revoked_authorization_blocks(monkeypatch):
    network = make_network()
    calls = fake_io(monkeypatch, network)
    request, token = authorized(network)
    network.revoke_authorization(token)
    with pytest.raises(NetworkExecutionDenied):
        network.execute(request, authorization=token)
    assert calls == []


@pytest.mark.parametrize(
    "change",
    [
        {"agent_id": "other"},
        {"session_id": "other"},
        {"tool_id": "other"},
        {"method": "POST"},
        {"url": "http://example.com/data"},
        {"url": "https://other.example.com/data"},
        {"url": "https://example.com:8443/data"},
        {"url": "https://example.com/changed"},
        {"body": b"changed"},
        {"headers": {"accept": "text/plain"}},
    ],
)
def test_authorization_is_exact_context_and_request_bound(monkeypatch, change):
    network = make_network(
        allowed_hosts=("example.com", "other.example.com"),
        allowed_ports=(443, 8443),
    )
    calls = fake_io(monkeypatch, network)
    request, token = authorized(network)
    changed = replace(request, **change)
    with pytest.raises(NetworkExecutionDenied):
        network.execute(changed, authorization=token)
    assert calls == []


def test_authorization_is_bound_to_validated_addresses(monkeypatch):
    network = make_network()
    calls = fake_io(monkeypatch, network, addresses=("93.184.216.34",))
    request, token = authorized(network)
    tampered = replace(token, resolved_addresses=("127.0.0.1",))
    with pytest.raises(NetworkExecutionDenied):
        network.execute(request, authorization=tampered)
    assert calls == []


def test_authorization_is_single_use(monkeypatch):
    network = make_network()
    calls = fake_io(monkeypatch, network)
    request, token = authorized(network)
    assert network.execute(request, authorization=token).body == b"ok"
    with pytest.raises(NetworkExecutionDenied):
        network.execute(request, authorization=token)
    assert len(calls) == 1


def test_concurrent_replay_uses_authorization_once(monkeypatch):
    network = make_network(max_concurrent_requests=2)
    calls = fake_io(monkeypatch, network)
    request, token = authorized(network)

    def execute():
        try:
            network.execute(request, authorization=token)
            return "allowed"
        except NetworkExecutionDenied:
            return "blocked"

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(lambda _: execute(), range(2)))
    assert sorted(outcomes) == ["allowed", "blocked"]
    assert len(calls) == 1


def test_malformed_authorization_fields_fail_closed(monkeypatch):
    network = make_network()
    calls = fake_io(monkeypatch, network)
    request, token = authorized(network)
    malformed = replace(token, expires_at="not-a-timestamp")
    with pytest.raises(NetworkExecutionDenied):
        network.execute(request, authorization=malformed)
    assert calls == []


def test_same_origin_redirect_is_policy_checked_and_followed(monkeypatch):
    network = make_network(max_redirects=1)
    calls = fake_io(
        monkeypatch,
        network,
        responses=[
            (302, {"Location": "/next"}, b""),
            (200, {"content-type": "text/plain"}, b"done"),
        ],
    )
    request, token = authorized(network)
    response = network.execute(request, authorization=token)
    assert response.redirects == 1
    assert response.url.endswith("/next")
    assert len(calls) == 2


@pytest.mark.parametrize(
    "location",
    [
        "https://other.example.com/next",
        "https://127.0.0.1/admin",
        "https://169.254.169.254/latest/meta-data",
    ],
)
def test_cross_origin_private_and_metadata_redirects_block(location, monkeypatch):
    network = make_network(allowed_hosts=("example.com", "*.example.com"), max_redirects=2)
    calls = fake_io(monkeypatch, network, responses=[(302, {"Location": location}, b"")])
    request, token = authorized(network)
    with pytest.raises(NetworkExecutionDenied):
        network.execute(request, authorization=token)
    assert len(calls) == 1


def test_redirect_limit_and_loop_block(monkeypatch):
    network = make_network(max_redirects=1)
    calls = fake_io(
        monkeypatch,
        network,
        responses=[(302, {"Location": "/loop"}, b""), (302, {"Location": "/loop"}, b"")],
    )
    request, token = authorized(network)
    with pytest.raises(NetworkExecutionDenied) as error:
        network.execute(request, authorization=token)
    assert error.value.decision.reason is DecisionReason.NETWORK_REDIRECT_BLOCKED
    assert len(calls) == 2


def test_same_origin_redirect_with_rebound_private_dns_is_blocked(monkeypatch):
    network = make_network(max_redirects=1)
    calls = fake_io(
        monkeypatch,
        network,
        responses=[(302, {"Location": "/private"}, b""), (200, {}, b"unexpected")],
    )
    resolved = [("93.184.216.34",), ("127.0.0.1",)]
    monkeypatch.setattr(network, "_resolve", lambda host, port: resolved.pop(0))
    request, token = authorized(network)
    with pytest.raises(NetworkExecutionDenied):
        network.execute(request, authorization=token)
    assert len(calls) == 1


def test_connection_timeout_fails_closed():
    network = make_network()
    request = replace(make_request(), agent_id="agent", session_id="session", tool_id="tool")
    token = network._issue_authorization(
        request,
        resolved_addresses=("93.184.216.34",),
        _issuer=_FRAMEWORK_ISSUER,
    )
    network._request_once = lambda *args: (_ for _ in ()).throw(TimeoutError())
    with pytest.raises(NetworkExecutionDenied) as error:
        network.execute(request, authorization=token)
    assert error.value.decision.reason is DecisionReason.TIMEOUT


def test_proxy_environment_variables_do_not_change_destination(monkeypatch):
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        monkeypatch.setenv(name, "http://127.0.0.1:9999")
    network = make_network()
    calls = fake_io(monkeypatch, network)
    request, token = authorized(network)
    network.execute(request, authorization=token)
    assert calls[0][1] == ("93.184.216.34",)


def test_https_connection_requires_verified_tls():
    connection = _PinnedHTTPSConnection(
        "example.com", 443, addresses=("93.184.216.34",), timeout=1
    )
    try:
        assert connection._context.verify_mode == ssl.CERT_REQUIRED
        assert connection._context.check_hostname
    finally:
        connection.close()


def test_response_size_limit_blocks(monkeypatch):
    network = make_network(max_response_bytes=2)
    fake_io(monkeypatch, network, responses=[(200, {}, b"large")])
    request, token = authorized(network)
    with pytest.raises(NetworkExecutionDenied) as error:
        network.execute(request, authorization=token)
    assert error.value.decision.reason is DecisionReason.NETWORK_RESPONSE_BLOCKED


def test_response_size_limit_includes_headers(monkeypatch):
    network = make_network(max_response_bytes=16)
    fake_io(monkeypatch, network, responses=[(200, {"x-response": "v" * 32}, b"")])
    request, token = authorized(network)
    with pytest.raises(NetworkExecutionDenied):
        network.execute(request, authorization=token)


def test_concurrency_limit_blocks_without_second_connection(monkeypatch):
    network = make_network()
    calls = fake_io(monkeypatch, network)
    network._concurrency.acquire()
    try:
        request, token = authorized(network)
        with pytest.raises(NetworkExecutionDenied):
            network.execute(request, authorization=token)
    finally:
        network._concurrency.release()
    assert calls == []


def test_response_conversion_filters_sensitive_headers_and_query():
    response = NetworkResponse(
        200,
        {"Set-Cookie": "session=secret", "Content-Type": "text/plain"},
        b"ok",
        "https://example.com/path?api_key=secret",
    )
    result = response.output_value()
    assert "set-cookie" not in result["headers"]
    assert "api_key" not in result["url"]


def test_malformed_utf8_response_is_rejected():
    response = NetworkResponse(200, {}, b"\xff", "https://example.com/")
    with pytest.raises(UnicodeDecodeError):
        response.output_value()


def test_audit_does_not_contain_headers_body_or_query_secrets(monkeypatch):
    logger = AuditLogger()
    network = bind_test_boundary(Network(NetworkPolicy(allowed_hosts=("example.com",)), audit_logger=logger))
    fake_io(monkeypatch, network)
    request = make_request(
        "https://example.com/path?token=secret-query-value",
        headers={"authorization": "Bearer secret-header-value"},
        body=b"secret-body-value",
    )
    request, token = authorized(network, request)
    network.execute(request, authorization=token)
    records = repr([event.to_dict() for event in logger.events()])
    assert "secret-query-value" not in records
    assert "secret-header-value" not in records
    assert "secret-body-value" not in records


def framework_app(*, data_flow=True, output_guard=None, network_policy=None):
    permissions = PermissionEngine([
        PermissionRule("call-tool", PermissionCategory.READ, action="fetch"),
        PermissionRule(
            "network-read",
            PermissionCategory.NETWORK,
            action="http.get",
            resource="https://example.com:443",
        ),
    ])
    flow = (
        DataFlowEngine([
            DataFlowPolicy("allow-external", DecisionType.ALLOW, destination="external.*")
        ])
        if data_flow
        else None
    )
    app = PreventApp(
        policy=PolicyEngine([PolicyRule("allow", DecisionType.ALLOW)]),
        permissions=permissions,
        data_flow=flow,
    )
    agent = app.add_agent(Agent("network-agent"))
    kwargs = {
        "data_source": DataSource("public-source", FlowClassification.PUBLIC, FlowTrust.TRUSTED),
        "data_destination": DataFlowDestination("external.api"),
        "network_policy": network_policy or NetworkPolicy(allowed_hosts=("example.com",)),
    }
    if output_guard is not None:
        kwargs["output_guard"] = output_guard
    return app, agent, kwargs


def approval_harness(monkeypatch, *, url="https://example.com/data", responses=None):
    manager = ApprovalManager()
    app = PreventApp(
        policy=PolicyEngine([PolicyRule("allow-all", DecisionType.ALLOW)]),
        permissions=PermissionEngine([
            PermissionRule("read-tools", PermissionCategory.READ, action="*"),
            PermissionRule(
                "network",
                PermissionCategory.NETWORK,
                action="http.*",
                tool="*",
                resource="*",
            ),
        ]),
        data_flow=DataFlowEngine([
            DataFlowPolicy("external", DecisionType.ALLOW, destination="external.*")
        ]),
        approval_manager=manager,
        approval_policy=ApprovalPolicy([
            ApprovalRule("approve-http", action="http.*", tool="fetch*")
        ]),
    )
    agent = app.add_agent(Agent("approval-agent", agent_id="approval-agent"))
    request_state = {"value": make_request(url)}
    request_states = {}
    tools = {}

    def register_tool(target_agent, name):
        state = {"value": make_request(url)}
        request_states[(target_agent.agent_id, name)] = state

        @target_agent.tool(
            name,
            network_policy=NetworkPolicy(
                allowed_schemes=("https", "http"),
                allowed_hosts=("example.com", "other.example.com"),
                allowed_ports=(80, 443, 8443),
                allowed_methods=("GET", "POST"),
                require_https=False,
                max_redirects=2,
            ),
            data_source=DataSource(
                "public", FlowClassification.PUBLIC, FlowTrust.TRUSTED
            ),
            data_destination=DataFlowDestination("external.api"),
        )
        def fetch():
            return state["value"]

        tools[(target_agent.agent_id, name)] = fetch
        return fetch, state

    primary, primary_state = register_tool(agent, "fetch")
    other_tool, _ = register_tool(agent, "fetch_other")
    other_agent = app.add_agent(Agent("other-approval-agent", agent_id="other-agent"))
    other_agent_tool, _ = register_tool(other_agent, "fetch")
    app.run()

    calls = []
    response_queue = list(responses or [(200, {"content-type": "text/plain"}, b"approved")])

    def resolve(self, host, port):
        return ("93.184.216.34",)

    def request_once(self, request, addresses, timeout):
        calls.append((request.host, request.port, request.scheme, request.path))
        if len(response_queue) > 1:
            return response_queue.pop(0)
        return response_queue[0]

    monkeypatch.setattr(Network, "_resolve", resolve)
    monkeypatch.setattr(Network, "_request_once", request_once)
    with pytest.raises(ToolApprovalRequired):
        primary()
    assert calls == []
    return {
        "app": app,
        "agent": agent,
        "manager": manager,
        "primary": primary,
        "tool": agent.tools["fetch"],
        "other_tool": agent.tools["fetch_other"],
        "other_agent_tool": other_agent_tool,
        "request_state": primary_state,
        "calls": calls,
    }


def issue_network_approval(harness, *, expires_at=None):
    pending = harness["tool"].pending_approval_request
    assert pending is not None
    approval = harness["manager"].approve(
        pending,
        approved_by="network-reviewer",
        expires_at=expires_at or datetime.now(timezone.utc) + timedelta(hours=1),
    )
    harness["tool"].use_approval(approval)
    return approval


def freeze_approval_time(monkeypatch, instant):
    class FrozenDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return instant if tz is not None else instant.replace(tzinfo=None)

    monkeypatch.setattr(approval_module, "datetime", FrozenDateTime)


def test_framework_network_tool_uses_explicit_request_and_authoritative_boundary(monkeypatch):
    app, agent, kwargs = framework_app()
    monkeypatch.setattr(Network, "_resolve", lambda self, host, port: ("93.184.216.34",))
    monkeypatch.setattr(
        Network,
        "_request_once",
        lambda self, request, addresses, timeout: (200, {"content-type": "text/plain"}, b"network-ok"),
    )

    @agent.tool(**kwargs)
    def fetch():
        return make_request("https://example.com/data", classification=FlowClassification.PUBLIC)

    app.run()
    assert fetch()["body"] == "network-ok"


@pytest.mark.parametrize("result", ["ordinary string", b"bytes", {"url": "https://example.com"}, None, object()])
def test_network_tool_never_falls_back_to_ordinary_execution(monkeypatch, result):
    app, agent, kwargs = framework_app()
    connections = []
    monkeypatch.setattr(Network, "_resolve", lambda self, host, port: ("93.184.216.34",))
    monkeypatch.setattr(Network, "_request_once", lambda *args: connections.append(True))

    @agent.tool(**kwargs)
    def fetch():
        return result

    app.run()
    with pytest.raises(ToolExecutionDenied):
        fetch()
    assert connections == []


def test_framework_network_without_dataflow_blocks_before_dns(monkeypatch):
    app, agent, kwargs = framework_app(data_flow=False)
    dns_calls = []
    monkeypatch.setattr(Network, "_resolve", lambda *args: dns_calls.append(True))

    @agent.tool(**kwargs)
    def fetch():
        return make_request("https://example.com/")

    app.run()
    with pytest.raises(ToolExecutionDenied):
        fetch()
    assert dns_calls == []


def test_framework_network_permission_denial_opens_no_connection(monkeypatch):
    app, agent, kwargs = framework_app()
    app._engine._permissions = PermissionEngine([
        PermissionRule("call-tool", PermissionCategory.READ, action="fetch")
    ])
    calls = []
    monkeypatch.setattr(Network, "_resolve", lambda self, host, port: ("93.184.216.34",))
    monkeypatch.setattr(Network, "_request_once", lambda *args: calls.append(True))

    @agent.tool(**kwargs)
    def fetch():
        return make_request("https://example.com/")

    app.run()
    with pytest.raises(ToolExecutionDenied):
        fetch()
    assert calls == []


def test_framework_policy_denial_opens_no_connection(monkeypatch):
    app, agent, kwargs = framework_app()
    app._engine._policy = PolicyEngine([
        PolicyRule("deny-fetch", DecisionType.BLOCK, action="fetch")
    ])
    calls = []
    monkeypatch.setattr(Network, "_resolve", lambda self, host, port: ("93.184.216.34",))
    monkeypatch.setattr(Network, "_request_once", lambda *args: calls.append(True))

    @agent.tool(**kwargs)
    def fetch():
        return make_request("https://example.com/")

    app.run()
    with pytest.raises(ToolExecutionDenied):
        fetch()
    assert calls == []


def test_framework_authentication_denial_opens_no_connection(monkeypatch):
    app, _, kwargs = framework_app()
    authenticator = Authenticator()
    agent = Agent(
        "authenticated-network-agent",
        agent_id="authenticated-network-agent",
        credential=Credential("authenticated-network-agent", "secret"),
    )
    app = PreventApp(
        policy=PolicyEngine([PolicyRule("allow", DecisionType.ALLOW)]),
        permissions=app._engine.permissions,
        data_flow=app._data_flow,
        authenticator=authenticator,
    )
    agent = app.add_agent(agent)
    calls = []
    monkeypatch.setattr(Network, "_resolve", lambda self, host, port: ("93.184.216.34",))
    monkeypatch.setattr(Network, "_request_once", lambda *args: calls.append(True))

    @agent.tool(**kwargs)
    def fetch():
        return make_request("https://example.com/")

    app.run()
    authenticator.revoke(agent.agent_id)
    with pytest.raises(ToolExecutionDenied):
        fetch()
    assert calls == []


def test_framework_input_guard_denial_opens_no_connection(monkeypatch):
    app, agent, kwargs = framework_app()
    calls = []
    monkeypatch.setattr(Network, "_request_once", lambda *args: calls.append(True))

    @agent.tool(content_guard=ContentGuard(blocked_phrases=("BLOCK_NETWORK",)), **kwargs)
    def fetch(value):
        return make_request("https://example.com/")

    app.run()
    with pytest.raises(ToolExecutionDenied):
        fetch("BLOCK_NETWORK")
    assert calls == []


def test_framework_runtime_denial_opens_no_connection(monkeypatch):
    app, agent, kwargs = framework_app()
    calls = []
    monkeypatch.setattr(Network, "_request_once", lambda *args: calls.append(True))

    @agent.tool(**kwargs)
    def fetch():
        return make_request("https://example.com/")

    app.run()
    tool = agent.tools["fetch"]
    tool._runtime_controller = RuntimeController(
        RuntimePolicy(max_concurrent_tools=1), application_id="test-app"
    )
    assert tool._runtime_controller.acquire(
        agent.agent_id, tool.tool_id, agent.session_id
    ) is None
    try:
        with pytest.raises(ToolExecutionDenied):
            fetch()
    finally:
        tool._runtime_controller.release()
    assert calls == []


def test_framework_approval_denial_opens_no_connection(monkeypatch):
    app, agent, kwargs = framework_app()
    agent._approval_policy = ApprovalPolicy([
        ApprovalRule("approve-network", action="http.get", tool="fetch")
    ])
    calls = []
    monkeypatch.setattr(Network, "_request_once", lambda *args: calls.append(True))

    @agent.tool(**kwargs)
    def fetch():
        return make_request("https://example.com/")

    app.run()
    with pytest.raises(ToolApprovalRequired):
        fetch()
    assert calls == []


def test_approval_for_one_host_does_not_authorize_another_host(monkeypatch):
    approval_manager = ApprovalManager()
    flow = DataFlowEngine([
        DataFlowPolicy("allow-external", DecisionType.ALLOW, destination="external.*")
    ])
    permissions = PermissionEngine([
        PermissionRule("call-tool", PermissionCategory.READ, action="fetch"),
        PermissionRule(
            "network-read",
            PermissionCategory.NETWORK,
            action="http.get",
            resource="https://*",
        ),
    ])
    app = PreventApp(
        policy=PolicyEngine([PolicyRule("allow", DecisionType.ALLOW)]),
        permissions=permissions,
        data_flow=flow,
        approval_manager=approval_manager,
        approval_policy=ApprovalPolicy([
            ApprovalRule("approve-network", action="http.get", tool="fetch")
        ]),
    )
    agent = app.add_agent(Agent("network-approval-agent"))
    current_host = ["example.com"]
    calls = []
    monkeypatch.setattr(Network, "_request_once", lambda *args: calls.append(True))

    @agent.tool(
        network_policy=NetworkPolicy(
            allowed_hosts=("example.com", "other.example.com")
        ),
        data_source=DataSource(
            "public", FlowClassification.PUBLIC, FlowTrust.TRUSTED
        ),
        data_destination=DataFlowDestination("external.api"),
    )
    def fetch():
        return make_request(f"https://{current_host[0]}/")

    app.run()
    with pytest.raises(ToolApprovalRequired):
        fetch()
    tool = agent.tools["fetch"]
    pending = tool.pending_approval_request
    assert pending is not None
    tool.use_approval(approval_manager.approve(pending, approved_by="reviewer"))
    current_host[0] = "other.example.com"
    with pytest.raises(ToolExecutionDenied):
        fetch()
    assert calls == []


def test_network_approval_valid_before_expiry_allows_exact_request(monkeypatch):
    harness = approval_harness(monkeypatch)
    issue_network_approval(harness)
    assert harness["primary"]()["body"] == "approved"
    assert harness["calls"] == [
        ("example.com", 443, "https", "/data")
    ]


def test_expired_network_approval_blocks_before_connection(monkeypatch):
    harness = approval_harness(monkeypatch)
    expiry = datetime.now(timezone.utc) + timedelta(minutes=1)
    issue_network_approval(harness, expires_at=expiry)
    freeze_approval_time(monkeypatch, expiry + timedelta(seconds=1))
    with pytest.raises(ToolExecutionDenied):
        harness["primary"]()
    assert harness["calls"] == []


def test_network_approval_expiring_after_creation_blocks_final_execution(monkeypatch):
    harness = approval_harness(monkeypatch)
    expiry = datetime.now(timezone.utc) + timedelta(minutes=1)
    issue_network_approval(harness, expires_at=expiry)
    original_verify = harness["manager"].verify
    calls_to_verify = []

    def verify_then_advance(*args, **kwargs):
        decision = original_verify(*args, **kwargs)
        calls_to_verify.append(kwargs.get("consume", True))
        if len(calls_to_verify) == 1:
            freeze_approval_time(monkeypatch, expiry + timedelta(seconds=1))
        return decision

    monkeypatch.setattr(harness["manager"], "verify", verify_then_advance)
    with pytest.raises(ToolExecutionDenied):
        harness["primary"]()
    assert calls_to_verify == [False, True]
    assert harness["calls"] == []


def test_expired_approval_remains_invalid_for_its_exact_request(monkeypatch):
    harness = approval_harness(monkeypatch)
    expiry = datetime.now(timezone.utc) + timedelta(minutes=1)
    issue_network_approval(harness, expires_at=expiry)
    freeze_approval_time(monkeypatch, expiry + timedelta(seconds=1))
    with pytest.raises(ToolExecutionDenied):
        harness["primary"]()
    assert harness["calls"] == []


def test_modified_request_with_expired_approval_blocks_before_connection(monkeypatch):
    harness = approval_harness(monkeypatch)
    expiry = datetime.now(timezone.utc) + timedelta(minutes=1)
    issue_network_approval(harness, expires_at=expiry)
    harness["request_state"]["value"] = replace(
        harness["request_state"]["value"],
        url="https://example.com/modified",
    )
    freeze_approval_time(monkeypatch, expiry + timedelta(seconds=1))
    with pytest.raises(ToolExecutionDenied):
        harness["primary"]()
    assert harness["calls"] == []


def test_revoked_network_approval_blocks_before_connection_and_on_reuse(monkeypatch):
    harness = approval_harness(monkeypatch)
    approval = issue_network_approval(harness)
    harness["manager"].revoke(approval.approval_id)
    for _ in range(2):
        with pytest.raises(ToolExecutionDenied):
            harness["primary"]()
        assert harness["calls"] == []


def test_network_approval_revoked_between_request_and_connection_blocks(monkeypatch):
    harness = approval_harness(monkeypatch)
    approval = issue_network_approval(harness)
    original_verify = harness["manager"].verify
    verification_count = []

    def verify_then_revoke(*args, **kwargs):
        decision = original_verify(*args, **kwargs)
        verification_count.append(kwargs.get("consume", True))
        if len(verification_count) == 1:
            harness["manager"].revoke(approval.approval_id)
        return decision

    monkeypatch.setattr(harness["manager"], "verify", verify_then_revoke)
    with pytest.raises(ToolExecutionDenied):
        harness["primary"]()
    assert verification_count == [False, True]
    assert harness["calls"] == []


@pytest.mark.parametrize(
    "mismatch",
    [
        "another_agent",
        "another_session",
        "another_tool",
        "another_hostname",
        "another_port",
        "another_scheme",
        "modified_url",
        "modified_method",
        "modified_body",
    ],
)
def test_network_approval_is_bound_to_exact_identity_and_request(
    monkeypatch, mismatch
):
    harness = approval_harness(monkeypatch)
    approval = issue_network_approval(harness)
    target = harness["tool"]
    if mismatch == "another_agent":
        target = harness["other_agent_tool"]
    elif mismatch == "another_tool":
        target = harness["other_tool"]
    elif mismatch == "another_session":
        harness["agent"].session_id = "different-session"
    elif mismatch == "another_hostname":
        harness["request_state"]["value"] = replace(
            harness["request_state"]["value"],
            url="https://other.example.com/data",
        )
    elif mismatch == "another_port":
        harness["request_state"]["value"] = replace(
            harness["request_state"]["value"],
            url="https://example.com:8443/data",
        )
    elif mismatch == "another_scheme":
        harness["request_state"]["value"] = replace(
            harness["request_state"]["value"],
            url="http://example.com/data",
        )
    elif mismatch == "modified_url":
        harness["request_state"]["value"] = replace(
            harness["request_state"]["value"],
            url="https://example.com/other",
        )
    elif mismatch == "modified_method":
        harness["request_state"]["value"] = replace(
            harness["request_state"]["value"],
            method="POST",
        )
    elif mismatch == "modified_body":
        harness["request_state"]["value"] = replace(
            harness["request_state"]["value"],
            body=b"changed request body",
        )
    target.use_approval(approval)
    with pytest.raises(ToolExecutionDenied):
        target()
    assert harness["calls"] == []


@pytest.mark.parametrize(
    "layer",
    [
        "authentication",
        "delegation",
        "policy",
        "permission",
        "guard",
        "dataflow",
        "runtime",
        "network_policy",
    ],
)
def test_valid_approval_cannot_bypass_other_security_layers(monkeypatch, layer):
    harness = approval_harness(monkeypatch)
    approval = issue_network_approval(harness)
    tool = harness["tool"]
    app = harness["app"]
    agent = harness["agent"]

    if layer == "authentication":
        authenticator = Authenticator()
        credential = Credential(agent.agent_id, "network-test-secret")
        authenticator.register(agent.identity, credential)
        tool._authenticator = authenticator
        agent._credential = credential
        authenticator.revoke(agent.agent_id)
    elif layer == "delegation":
        grant = app._delegation_registry.create(
            "issuer",
            agent.agent_id,
            (Capability("fetch", resource=tool.tool_id, tool="fetch"),),
            expires_at=datetime.now(timezone.utc) + timedelta(minutes=5),
            issuer_permissions=app._engine.permissions,
        )
        agent.use_delegation(grant)
    elif layer == "policy":
        app._engine._policy = PolicyEngine([
            PolicyRule("allow-tools", DecisionType.ALLOW),
            PolicyRule(
                "deny-network",
                DecisionType.BLOCK,
                action="http.*",
                priority=1,
            ),
        ])
    elif layer == "permission":
        app._engine._permissions = PermissionEngine([
            PermissionRule("read-tools", PermissionCategory.READ, action="*")
        ])
    elif layer == "guard":
        from preventai.core import Decision, DecisionReason, Severity

        monkeypatch.setattr(
            tool._content_guard,
            "scan",
            lambda value: Decision(
                DecisionType.BLOCK,
                DecisionReason.CONTENT_POLICY_VIOLATION,
                severity=Severity.HIGH,
            ),
        )
    elif layer == "dataflow":
        denied = DataFlowEngine([
            DataFlowPolicy(
                "deny-external",
                DecisionType.BLOCK,
                destination="external.*",
            )
        ])
        tool._data_flow_engine = denied
        agent._data_flow_engine = denied
    elif layer == "runtime":
        controller = RuntimeController(
            RuntimePolicy(max_concurrent_tools=1),
            application_id="approval-runtime-test",
        )
        tool._runtime_controller = controller
        assert controller.acquire(agent.agent_id, tool.tool_id, agent.session_id) is None
    elif layer == "network_policy":
        tool._network = bind_test_boundary(Network(NetworkPolicy(allowed_hosts=("blocked.test",))))

    tool.use_approval(approval)
    try:
        with pytest.raises(ToolExecutionDenied):
            harness["primary"]()
        assert harness["calls"] == []
    finally:
        if layer == "runtime":
            controller.release()


@pytest.mark.parametrize(
    "location",
    [
        "https://other.example.com/redirected",
        "https://example.com:8443/redirected",
        "http://example.com/redirected",
        "/different-path",
    ],
)
def test_approved_request_cannot_authorize_a_redirect_target(monkeypatch, location):
    harness = approval_harness(
        monkeypatch,
        responses=[(302, {"Location": location}, b"")],
    )
    issue_network_approval(harness)
    with pytest.raises(ToolExecutionDenied):
        harness["primary"]()
    assert harness["calls"] == [
        ("example.com", 443, "https", "/data")
    ]


def test_valid_approval_does_not_replace_missing_network_authorization(monkeypatch):
    harness = approval_harness(monkeypatch)
    issue_network_approval(harness)
    monkeypatch.setattr(harness["tool"]._network, "_issue_authorization", lambda *args, **kwargs: None)
    with pytest.raises(ToolExecutionDenied):
        harness["primary"]()
    assert harness["calls"] == []


def test_valid_approval_does_not_replace_wrong_network_authorization(monkeypatch):
    harness = approval_harness(monkeypatch)
    issue_network_approval(harness)
    network = harness["tool"]._network
    issue = network._issue_authorization

    def wrong_scope(request, **kwargs):
        wrong_request = replace(request, method="POST")
        return issue(
            wrong_request,
            resolved_addresses=("93.184.216.34",),
            allow_redirects=kwargs.get("allow_redirects", True),
            _issuer=_FRAMEWORK_ISSUER,
        )

    monkeypatch.setattr(network, "_issue_authorization", wrong_scope)
    with pytest.raises(ToolExecutionDenied):
        harness["primary"]()
    assert harness["calls"] == []


def test_valid_approval_and_exact_network_authorization_allow_request(monkeypatch):
    harness = approval_harness(monkeypatch)
    issue_network_approval(harness)
    assert harness["primary"]()["body"] == "approved"
    assert harness["calls"] == [("example.com", 443, "https", "/data")]


def test_delegation_cannot_expand_from_tool_to_network_destination(monkeypatch):
    app, agent, kwargs = framework_app()
    calls = []
    monkeypatch.setattr(Network, "_request_once", lambda *args: calls.append(True))

    @agent.tool(**kwargs)
    def fetch():
        return make_request("https://example.com/")

    app.run()
    grant = app._delegation_registry.create(
        "issuer",
        agent.agent_id,
        (Capability("fetch", resource=agent.tools["fetch"].tool_id, tool="fetch"),),
        expires_at=datetime.now(timezone.utc) + timedelta(minutes=1),
        issuer_permissions=app._engine.permissions,
    )
    agent.use_delegation(grant)
    with pytest.raises(ToolExecutionDenied):
        fetch()
    assert calls == []


def test_secret_classification_dataflow_denial_opens_no_connection(monkeypatch):
    app, agent, kwargs = framework_app()
    restrictive_flow = DataFlowEngine([
        DataFlowPolicy("allow-external", DecisionType.ALLOW, destination="external.*"),
        DataFlowPolicy(
            "deny-secret-external",
            DecisionType.BLOCK,
            classification=FlowClassification.SECRET,
            destination="external.*",
            priority=1,
        ),
    ])
    app._data_flow = restrictive_flow
    agent._data_flow_engine = restrictive_flow
    calls = []
    monkeypatch.setattr(Network, "_resolve", lambda self, host, port: ("93.184.216.34",))
    monkeypatch.setattr(Network, "_request_once", lambda *args: calls.append(True))

    @agent.tool(**kwargs)
    def fetch():
        return make_request(
            "https://example.com/",
            classification=FlowClassification.SECRET,
            source_trust=FlowTrust.UNTRUSTED,
        )

    app.run()
    with pytest.raises(ToolExecutionDenied):
        fetch()
    assert calls == []


def test_trusted_source_can_follow_explicit_external_dataflow_policy(monkeypatch):
    app, agent, kwargs = framework_app()
    trusted_flow = DataFlowEngine([
        DataFlowPolicy(
            "trusted-external",
            DecisionType.ALLOW,
            destination="external.*",
            trust=FlowTrust.TRUSTED,
        ),
        DataFlowPolicy(
            "deny-untrusted-external",
            DecisionType.BLOCK,
            destination="external.*",
            trust=FlowTrust.UNTRUSTED,
            priority=1,
        ),
    ])
    app._data_flow = trusted_flow
    agent._data_flow_engine = trusted_flow
    monkeypatch.setattr(Network, "_resolve", lambda self, host, port: ("93.184.216.34",))
    monkeypatch.setattr(
        Network,
        "_request_once",
        lambda *args: (200, {}, b"trusted"),
    )

    @agent.tool(**kwargs)
    def fetch():
        return make_request(
            "https://example.com/",
            source_trust=FlowTrust.TRUSTED,
            classification=FlowClassification.PUBLIC,
        )

    app.run()
    assert fetch()["body"] == "trusted"


def test_framework_network_policy_denial_opens_no_connection(monkeypatch):
    app, agent, kwargs = framework_app(
        network_policy=NetworkPolicy(allowed_hosts=("allowed.test",))
    )
    calls = []
    monkeypatch.setattr(Network, "_resolve", lambda self, host, port: ("93.184.216.34",))
    monkeypatch.setattr(Network, "_request_once", lambda *args: calls.append(True))

    @agent.tool(**kwargs)
    def fetch():
        return make_request("https://example.com/")

    app.run()
    with pytest.raises(ToolExecutionDenied):
        fetch()
    assert calls == []


def test_framework_dataflow_denial_opens_no_connection(monkeypatch):
    app, agent, kwargs = framework_app()
    app._data_flow = DataFlowEngine([
        DataFlowPolicy(
            "block-external",
            DecisionType.BLOCK,
            classification=FlowClassification.PUBLIC,
            destination="external.*",
        )
    ])
    agent._data_flow_engine = app._data_flow
    for tool in agent.tools.values():
        tool._data_flow_engine = app._data_flow
    calls = []
    monkeypatch.setattr(Network, "_resolve", lambda self, host, port: ("93.184.216.34",))
    monkeypatch.setattr(Network, "_request_once", lambda *args: calls.append(True))

    @agent.tool(**kwargs)
    def fetch():
        return make_request("https://example.com/")

    app.run()
    with pytest.raises(ToolExecutionDenied):
        fetch()
    assert calls == []


def test_authorized_private_destination_still_requires_internal_dataflow_policy(monkeypatch):
    app, agent, kwargs = framework_app(
        network_policy=NetworkPolicy(
            allowed_hosts=("example.com",),
            allowed_ip_ranges=("10.0.0.0/8",),
        )
    )
    calls = []
    monkeypatch.setattr(Network, "_resolve", lambda self, host, port: ("10.2.3.4",))
    monkeypatch.setattr(Network, "_request_once", lambda *args: calls.append(True))

    @agent.tool(**kwargs)
    def fetch():
        return make_request("https://example.com/")

    app.run()
    with pytest.raises(ToolExecutionDenied):
        fetch()
    assert calls == []


@pytest.mark.parametrize(
    "classification",
    [
        FlowClassification.INTERNAL,
        FlowClassification.PRIVATE,
        FlowClassification.SENSITIVE,
        FlowClassification.SECRET,
    ],
)
def test_restricted_data_classifications_cannot_flow_to_public_network(
    monkeypatch, classification
):
    app, agent, kwargs = framework_app()
    restrictive_flow = DataFlowEngine([
        DataFlowPolicy(
            "public-external",
            DecisionType.ALLOW,
            classification=FlowClassification.PUBLIC,
            destination="external.*",
        ),
        DataFlowPolicy(
            "restricted-external",
            DecisionType.BLOCK,
            classification=classification,
            destination="external.*",
            priority=1,
        ),
    ])
    app._data_flow = restrictive_flow
    agent._data_flow_engine = restrictive_flow
    calls = []
    monkeypatch.setattr(Network, "_resolve", lambda self, host, port: ("93.184.216.34",))
    monkeypatch.setattr(Network, "_request_once", lambda *args: calls.append(True))

    @agent.tool(**kwargs)
    def fetch():
        return make_request("https://example.com/", classification=classification)

    app.run()
    with pytest.raises(ToolExecutionDenied):
        fetch()
    assert calls == []


def test_untrusted_source_dataflow_denial_opens_no_connection(monkeypatch):
    app, agent, kwargs = framework_app()
    restrictive_flow = DataFlowEngine([
        DataFlowPolicy(
            "trusted-external",
            DecisionType.ALLOW,
            destination="external.*",
            trust=FlowTrust.TRUSTED,
        ),
        DataFlowPolicy(
            "deny-untrusted-external",
            DecisionType.BLOCK,
            destination="external.*",
            trust=FlowTrust.UNTRUSTED,
            priority=1,
        ),
    ])
    app._data_flow = restrictive_flow
    agent._data_flow_engine = restrictive_flow
    calls = []
    monkeypatch.setattr(Network, "_resolve", lambda self, host, port: ("93.184.216.34",))
    monkeypatch.setattr(Network, "_request_once", lambda *args: calls.append(True))

    @agent.tool(**kwargs)
    def fetch():
        return make_request("https://example.com/")

    app.run()
    with pytest.raises(ToolExecutionDenied):
        fetch()
    assert calls == []


def test_framework_output_guard_blocks_untrusted_secret_response(monkeypatch):
    app, agent, kwargs = framework_app(
        output_guard=OutputGuard(custom_patterns={"test_secret": r"NET_SECRET_[A-Za-z0-9]{20}"})
    )
    monkeypatch.setattr(Network, "_resolve", lambda self, host, port: ("93.184.216.34",))
    monkeypatch.setattr(
        Network,
        "_request_once",
        lambda *args: (200, {}, b"NET_SECRET_01234567890123456789"),
    )

    @agent.tool(**kwargs)
    def fetch():
        return make_request("https://example.com/")

    app.run()
    with pytest.raises(ToolExecutionDenied):
        fetch()


def test_framework_output_guard_exception_blocks_response(monkeypatch):
    app, agent, kwargs = framework_app()
    monkeypatch.setattr(Network, "_resolve", lambda self, host, port: ("93.184.216.34",))
    monkeypatch.setattr(
        Network,
        "_request_once",
        lambda *args: (200, {}, b"safe"),
    )

    @agent.tool(**kwargs)
    def fetch():
        return make_request("https://example.com/")

    app.run()
    tool = agent.tools["fetch"]
    monkeypatch.setattr(
        tool._output_guard,
        "validate",
        lambda *args: (_ for _ in ()).throw(RuntimeError("guard failed")),
    )
    with pytest.raises(ToolExecutionDenied):
        fetch()


def test_network_policy_and_authorization_failures_do_not_open_connector(monkeypatch):
    network = bind_test_boundary(Network(NetworkPolicy(allowed_hosts=("example.com",))))
    calls = []
    monkeypatch.setattr(network, "_request_once", lambda *args: calls.append(True))
    request = replace(make_request(), agent_id="agent", session_id="session", tool_id="tool")
    with pytest.raises(NetworkExecutionDenied):
        network.execute(request)
    assert calls == []


def test_request_and_response_structures_are_frozen():
    request = make_request()
    response = NetworkResponse(200, {}, b"ok", "https://example.com/")
    with pytest.raises((AttributeError, TypeError)):
        request.url = "https://other.example.com/"
    with pytest.raises((AttributeError, TypeError)):
        response.status = 201


def test_authorization_type_is_immutable_and_exact():
    authorization = NetworkAuthorization(
        "agent", "session", "tool", "http.get", "https", "example.com",
        443, "digest", ("93.184.216.34",), True, 100.0, "proof",
    )
    with pytest.raises((AttributeError, TypeError)):
        authorization.host = "other.example.com"
