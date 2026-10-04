import pytest
from preventai import (
    Database, DatabaseExecutionDenied, DatabasePolicy, DatabaseResponse,
    InstrumentedDatabaseBackend,
)
from test_database import make_db, request, auth
from _boundary_test_support import bind_test_boundary


@pytest.mark.parametrize("operation", ["CREATE", "ALTER", "DROP", "TRUNCATE", "GRANT", "REVOKE"])
def test_administrative_operations_are_denied(operation):
    db = make_db()
    with pytest.raises(ValueError): request(operation)


def test_procedure_and_function_identifiers_are_not_supported():
    db = make_db()
    with pytest.raises(ValueError): request("CALL")
    assert db._backend.execution_count == 0


def test_database_allowlist_empty_denies():
    db = bind_test_boundary(Database(DatabasePolicy()))
    req = request()
    with pytest.raises(DatabaseExecutionDenied): db.execute(req, authorization=auth(db, req), agent_id="a", session_id="s", tool_id="t")
    assert db._backend.execution_count == 0


def test_parameter_count_denial_precedes_backend():
    db = make_db(max_parameters=1); req = request(parameters=(1, 2))
    with pytest.raises(DatabaseExecutionDenied): db.execute(req, authorization=auth(db, req), agent_id="a", session_id="s", tool_id="t")
    assert db._backend.execution_count == 0


def test_column_count_denial_precedes_backend():
    db = make_db(max_columns=1); req = request(columns=("id", "name"))
    with pytest.raises(DatabaseExecutionDenied): db.execute(req, authorization=auth(db, req), agent_id="a", session_id="s", tool_id="t")
    assert db._backend.execution_count == 0


def test_query_size_denial_precedes_backend():
    db = make_db(max_request_bytes=4); req = request()
    with pytest.raises(DatabaseExecutionDenied): db.execute(req, authorization=auth(db, req), agent_id="a", session_id="s", tool_id="t")
    assert db._backend.execution_count == 0


def test_concurrency_limit_is_deterministic():
    db = make_db(max_concurrent=1); req = request()
    assert db.execute(req, authorization=auth(db, req), agent_id="a", session_id="s", tool_id="t").operation == "SELECT"


def test_backend_error_is_sanitized(monkeypatch):
    db = make_db(); req = request()
    def fail(_): raise RuntimeError("password=secret connection_string=hidden")
    monkeypatch.setattr(db._backend, "execute", fail)
    with pytest.raises(DatabaseExecutionDenied) as exc: db.execute(req, authorization=auth(db, req), agent_id="a", session_id="s", tool_id="t")
    assert "secret" not in str(exc.value) and "hidden" not in str(exc.value)


def test_authorization_minting_requires_framework_issuer():
    db = make_db(); req = request()
    with pytest.raises(PermissionError):
        db._issue_authorization(req, agent_id="a", session_id="s", tool_id="t", expires_at=1, _capability=object())


def test_no_public_framework_issuer_accessor():
    db = make_db()
    assert not hasattr(db, "_framework_issuer")


def test_template_column_expansion_is_blocked():
    db = make_db()
    req = request(columns=("id",), template="SELECT password FROM users WHERE id = ?")
    with pytest.raises(DatabaseExecutionDenied):
        db.execute(req, authorization=auth(db, req), agent_id="a", session_id="s", tool_id="t")
    assert db._backend.execution_count == 0


def test_template_cross_table_is_blocked():
    db = make_db()
    req = request(template="SELECT id FROM orders WHERE id = ?")
    with pytest.raises(DatabaseExecutionDenied):
        db.execute(req, authorization=auth(db, req), agent_id="a", session_id="s", tool_id="t")
    assert db._backend.execution_count == 0


def test_template_operation_mismatch_is_blocked():
    db = make_db()
    req = request(template="UPDATE users SET id = ? WHERE id = ?")
    with pytest.raises(DatabaseExecutionDenied):
        db.execute(req, authorization=auth(db, req), agent_id="a", session_id="s", tool_id="t")
    assert db._backend.execution_count == 0


def test_tenant_predicate_is_mandatory():
    db = make_db(allowed_tenants=("tenant-a",))
    req = request(tenant_id="tenant-a")
    with pytest.raises(DatabaseExecutionDenied):
        db.execute(req, authorization=auth(db, req), agent_id="a", session_id="s", tool_id="t")
    assert db._backend.execution_count == 0


def test_tenant_injection_value_remains_bound_data():
    db = make_db(allowed_tenants=("tenant-a",))
    tenant = "tenant-a' OR '1'='1"
    req = request(tenant_id=tenant, template="SELECT id FROM users WHERE id = ? AND tenant_id = ?")
    with pytest.raises(DatabaseExecutionDenied):
        db.execute(req, authorization=auth(db, req), agent_id="a", session_id="s", tool_id="t")
    assert db._backend.execution_count == 0


def test_omitted_row_limit_uses_policy():
    backend = InstrumentedDatabaseBackend()
    db = bind_test_boundary(Database(DatabasePolicy(
        allowed_databases=("app",), allowed_schemas=("public",), allowed_tables=("users",),
        allowed_columns=("id", "name", "tenant_id"), allowed_operations=("SELECT", "INSERT", "UPDATE", "DELETE"),
        max_rows=1,
    ), backend=backend))
    req = request()
    calls = []
    backend.execute = lambda r: (
        calls.append(r),
        DatabaseResponse(
            r.operation, r.database, r.schema, r.table,
            rows=({"id": 1}, {"id": 2}),
        ),
    )[1]
    with pytest.raises(DatabaseExecutionDenied):
        db.execute(req, authorization=auth(db, req), agent_id="a", session_id="s", tool_id="t")
    assert len(calls) == 1


def test_result_bytes_are_enforced_after_backend():
    backend = InstrumentedDatabaseBackend()
    db = bind_test_boundary(Database(DatabasePolicy(
        allowed_databases=("app",), allowed_schemas=("public",), allowed_tables=("users",),
        allowed_columns=("id", "name", "tenant_id"), allowed_operations=("SELECT", "INSERT", "UPDATE", "DELETE"),
        max_result_bytes=32,
    ), backend=backend))
    req = request()
    calls = []
    backend.execute = lambda r: (
        calls.append(r),
        DatabaseResponse(
            r.operation, r.database, r.schema, r.table,
            rows=({"id": "x" * 100},),
        ),
    )[1]
    with pytest.raises(DatabaseExecutionDenied):
        db.execute(req, authorization=auth(db, req), agent_id="a", session_id="s", tool_id="t")
    assert len(calls) == 1
