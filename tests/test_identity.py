import secrets

import pytest

from preventai import (
    Agent,
    Authenticator,
    Credential,
    DecisionReason,
    PreventApp,
    RegisteredAgentIdentity,
    ToolExecutionDenied,
)


def make_auth():
    auth = Authenticator()
    agent_id = "identity-test-agent"
    secret = secrets.token_urlsafe(24)
    credential = Credential(agent_id, secret)
    auth.register(RegisteredAgentIdentity(agent_id, "test"), credential)
    return auth, agent_id, secret


def test_authentication_success_and_session_binding():
    auth, agent_id, secret = make_auth()
    result = auth.authenticate(agent_id, secret, session_id="session-1")
    assert result.authenticated
    assert result.session_id == "session-1"
    assert auth.authenticate(agent_id, "wrong").authenticated is False


def test_rotation_revocation_and_disablement():
    auth, agent_id, secret = make_auth()
    replacement = secrets.token_urlsafe(24)
    auth.rotate(agent_id, Credential(agent_id, replacement))
    assert not auth.authenticate(agent_id, secret).authenticated
    assert auth.authenticate(agent_id, replacement).authenticated
    auth.revoke(agent_id)
    assert not auth.authenticate(agent_id, replacement).authenticated
    auth.enable(agent_id)
    auth.rotate(agent_id, Credential(agent_id, replacement))
    auth.disable(agent_id)
    assert not auth.authenticate(agent_id, replacement).authenticated


def test_authenticated_app_blocks_before_callable_on_bad_credential():
    auth, agent_id, _ = make_auth()
    agent = Agent(
        "test",
        agent_id=agent_id,
        credential=Credential(agent_id, secrets.token_urlsafe(24)),
    )
    called = []

    @agent.tool()
    def protected():
        called.append(True)
        return "ok"

    app = PreventApp(authenticator=auth)
    app.add_agent(agent)
    app.run()
    with pytest.raises(ToolExecutionDenied) as error:
        protected()
    assert error.value.decision.reason is DecisionReason.AUTHENTICATION_FAILED
    assert called == []


def test_authenticated_app_accepts_matching_credential():
    auth, agent_id, secret = make_auth()
    agent = Agent(agent_id, agent_id=agent_id, credential=Credential(agent_id, secret))

    @agent.tool()
    def protected():
        return "ok"

    app = PreventApp(authenticator=auth)
    app.add_agent(agent)
    app.run()
    assert auth.authenticate(agent_id, secret, session_id=agent.session_id).authenticated
