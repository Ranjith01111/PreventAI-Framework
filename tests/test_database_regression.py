import pytest
from preventai import DatabaseExecutionDenied
from test_database import make_db, request, auth


@pytest.mark.parametrize("payload", ["SELECT * FROM users", "users WHERE id=1; DROP TABLE users", "x' OR '1'='1"])
def test_raw_and_injection_payloads_never_execute(payload):
    db = make_db(); req = request(raw_sql=payload)
    with pytest.raises(DatabaseExecutionDenied): db.execute(req, authorization=auth(db, req), agent_id="a", session_id="s", tool_id="t")
    assert db._backend.execution_count == 0


def test_canonical_operation_is_uppercase():
    assert request("select").operation == "SELECT"


def test_canonical_digest_is_stable():
    assert request().digest() == request().digest()


def test_sensitive_parameters_are_not_in_response():
    db = make_db(); req = request(parameters=("password",))
    result = db.execute(req, authorization=auth(db, req), agent_id="a", session_id="s", tool_id="t")
    assert "password" not in repr(result.output_value())


def test_revocation_does_not_execute():
    db = make_db(); req = request(); token = auth(db, req); db.revoke(token)
    with pytest.raises(DatabaseExecutionDenied): db.execute(req, authorization=token, agent_id="a", session_id="s", tool_id="t")
    assert db._backend.execution_count == 0


def test_wrong_database_authorization_does_not_execute():
    db = make_db(); req = request(); token = auth(db, req)
    changed = request(database="other")
    with pytest.raises(DatabaseExecutionDenied): db.execute(changed, authorization=token, agent_id="a", session_id="s", tool_id="t")
    assert db._backend.execution_count == 0


def test_wrong_table_authorization_does_not_execute():
    db = make_db(); req = request(); token = auth(db, req)
    changed = request(table="orders")
    with pytest.raises(DatabaseExecutionDenied): db.execute(changed, authorization=token, agent_id="a", session_id="s", tool_id="t")
    assert db._backend.execution_count == 0


def test_unsupported_object_does_not_execute():
    db = make_db()
    with pytest.raises(DatabaseExecutionDenied): db.execute(object())
    assert db._backend.execution_count == 0
