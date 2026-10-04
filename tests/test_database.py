import time
import pytest

from preventai import (
    Database, DatabaseAuthorization, DatabaseExecutionDenied, DatabasePolicy,
    DatabaseRequest,
)
from _boundary_test_support import bind_test_boundary


def make_db(**kwargs):
    options = dict(
        allowed_databases=("app",), allowed_schemas=("public",), allowed_tables=("users",),
        allowed_columns=("id", "name", "tenant_id"), allowed_operations=("SELECT", "INSERT", "UPDATE", "DELETE"),
    )
    options.update(kwargs)
    return bind_test_boundary(Database(DatabasePolicy(**options)))


def request(operation="SELECT", **kwargs):
    values = dict(operation=operation, database="app", schema="public", table="users",
                  columns=("id",), template=f"{operation.lower()} users where id = ?",
                  parameters=(1,))
    values.update(kwargs)
    return DatabaseRequest(**values)


def auth(db, req, **kwargs):
    return db._issue_authorization(req, agent_id=kwargs.get("agent_id", "a"),
                                    session_id=kwargs.get("session_id", "s"),
                                    tool_id=kwargs.get("tool_id", "t"),
                                    expires_at=kwargs.get("expires_at", time.time() + 30),
                                    _capability=db._issuance_capability)


def test_structured_select_executes():
    db = make_db()
    req = request()
    result = db.execute(req, authorization=auth(db, req), agent_id="a", session_id="s", tool_id="t")
    assert result.operation == "SELECT"


@pytest.mark.parametrize("operation", ["SELECT", "INSERT", "UPDATE", "DELETE"])
def test_supported_operations(operation):
    db = make_db()
    req = request(operation)
    assert db.execute(req, authorization=auth(db, req), agent_id="a", session_id="s", tool_id="t").operation == operation


@pytest.mark.parametrize("operation", ["CREATE", "ALTER", "DROP", "TRUNCATE", "GRANT", "REVOKE", "BEGIN", "CALL"])
def test_unsupported_operations_fail_closed(operation):
    with pytest.raises(ValueError):
        request(operation)


@pytest.mark.parametrize("field,value", [
    ("database", "other"), ("schema", "private"), ("table", "orders"),
])
def test_allowlists_are_exact(field, value):
    db = make_db()
    req = request(**{field: value})
    with pytest.raises(DatabaseExecutionDenied):
        db.execute(req, authorization=auth(db, req), agent_id="a", session_id="s", tool_id="t")
    assert db._backend.execution_count == 0


@pytest.mark.parametrize("column", ["password", "secret", "email"])
def test_column_allowlist_blocks(column):
    db = make_db()
    req = request(columns=(column,))
    with pytest.raises(DatabaseExecutionDenied):
        db.execute(req, authorization=auth(db, req), agent_id="a", session_id="s", tool_id="t")
    assert db._backend.execution_count == 0


def test_deny_overrides_allow():
    db = make_db(denied_tables=("users",), allowed_tables=("users",))
    req = request()
    with pytest.raises(DatabaseExecutionDenied):
        db.execute(req, authorization=auth(db, req), agent_id="a", session_id="s", tool_id="t")
    assert db._backend.execution_count == 0


def test_raw_sql_denied():
    db = make_db()
    req = request(raw_sql="SELECT * FROM users")
    with pytest.raises(DatabaseExecutionDenied):
        db.execute(req, authorization=auth(db, req), agent_id="a", session_id="s", tool_id="t")
    assert db._backend.execution_count == 0


@pytest.mark.parametrize("template", ["select users; drop users", "select users -- x", "select users /* x */"])
def test_injection_templates_denied(template):
    db = make_db()
    req = request(template=template)
    with pytest.raises(DatabaseExecutionDenied):
        db.execute(req, authorization=auth(db, req), agent_id="a", session_id="s", tool_id="t")
    assert db._backend.execution_count == 0


def test_parameters_are_recorded_without_sql_interpolation():
    db = make_db()
    req = request(parameters=("1 OR 1=1",))
    db.execute(req, authorization=auth(db, req), agent_id="a", session_id="s", tool_id="t")
    assert db._backend.executions[0]["parameters"] == ("1 OR 1=1",)


def test_tenant_scope():
    db = make_db(allowed_tenants=("tenant-a",))
    req = request(tenant_id="tenant-b")
    with pytest.raises(DatabaseExecutionDenied):
        db.execute(req, authorization=auth(db, req), agent_id="a", session_id="s", tool_id="t")
    assert db._backend.execution_count == 0


def test_request_digest_changes_with_parameters():
    assert request(parameters=(1,)).digest() != request(parameters=(2,)).digest()


def test_malformed_request_fails_closed():
    with pytest.raises(ValueError):
        DatabaseRequest("SELECT", "app", "public", "users", columns=("bad-name",))


def test_no_fallback_for_arbitrary_values():
    db = make_db()
    with pytest.raises(DatabaseExecutionDenied):
        db.execute({"operation": "SELECT"})  # type: ignore[arg-type]
    assert db._backend.execution_count == 0
