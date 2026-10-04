import time
import threading
import pytest
from preventai import Database, DatabaseExecutionDenied, DatabasePolicy, DatabaseRequest
from test_database import make_db, request, auth


def test_missing_authorization():
    db = make_db(); req = request()
    with pytest.raises(DatabaseExecutionDenied): db.execute(req)
    assert db._backend.execution_count == 0


def test_forged_authorization():
    db = make_db(); req = request(); token = auth(db, req)
    forged = type(token)(**{**token.__dict__, "signature": "x"}) if hasattr(token, "__dict__") else token
    if forged is token:
        object.__setattr__(token, "signature", "x")
    with pytest.raises(DatabaseExecutionDenied): db.execute(req, authorization=forged, agent_id="a", session_id="s", tool_id="t")
    assert db._backend.execution_count == 0


def test_cross_instance_isolation():
    first = make_db(); second = make_db(); req = request(); token = auth(first, req)
    with pytest.raises(DatabaseExecutionDenied): second.execute(req, authorization=token, agent_id="a", session_id="s", tool_id="t")
    assert second._backend.execution_count == 0


@pytest.mark.parametrize("field,value", [("agent_id", "other"), ("session_id", "other"), ("tool_id", "other")])
def test_identity_scope(field, value):
    db = make_db(); req = request(); token = auth(db, req)
    ids = {"agent_id": "a", "session_id": "s", "tool_id": "t"}; ids[field] = value
    with pytest.raises(DatabaseExecutionDenied): db.execute(req, authorization=token, **ids)
    assert db._backend.execution_count == 0


def test_expired_authorization():
    db = make_db(); req = request(); token = auth(db, req, expires_at=time.time() - 1)
    with pytest.raises(DatabaseExecutionDenied): db.execute(req, authorization=token, agent_id="a", session_id="s", tool_id="t")
    assert db._backend.execution_count == 0


def test_revoked_authorization():
    db = make_db(); req = request(); token = auth(db, req); db.revoke(token)
    with pytest.raises(DatabaseExecutionDenied): db.execute(req, authorization=token, agent_id="a", session_id="s", tool_id="t")
    assert db._backend.execution_count == 0


def test_replay_is_denied():
    db = make_db(); req = request(); token = auth(db, req)
    db.execute(req, authorization=token, agent_id="a", session_id="s", tool_id="t")
    with pytest.raises(DatabaseExecutionDenied): db.execute(req, authorization=token, agent_id="a", session_id="s", tool_id="t")
    assert db._backend.execution_count == 1


def test_scope_mismatch():
    db = make_db(); req = request(); token = auth(db, req)
    changed = request(columns=("name",))
    with pytest.raises(DatabaseExecutionDenied): db.execute(changed, authorization=token, agent_id="a", session_id="s", tool_id="t")
    assert db._backend.execution_count == 0


@pytest.mark.parametrize("kwargs", [
    {"max_parameters": 1}, {"max_columns": 1}, {"max_batch_size": 1},
])
def test_resource_limits(kwargs):
    db = make_db(**kwargs)
    req = request(parameters=(1, 2) if "max_parameters" in kwargs else (1,),
                  columns=("id", "name") if "max_columns" in kwargs else ("id",))
    if "max_batch_size" in kwargs:
        req = request(parameters=(1, 2))
    with pytest.raises(DatabaseExecutionDenied): db.execute(req, authorization=auth(db, req), agent_id="a", session_id="s", tool_id="t")
    assert db._backend.execution_count == 0


def test_result_row_limit():
    db = make_db(max_rows=1); req = request(max_rows=2)
    with pytest.raises(DatabaseExecutionDenied): db.execute(req, authorization=auth(db, req), agent_id="a", session_id="s", tool_id="t")
    assert db._backend.execution_count == 0


def test_timeout_configuration_is_bounded():
    with pytest.raises(ValueError): DatabasePolicy(timeout=0)


def test_slow_backend_times_out_without_exposing_result():
    db = make_db(timeout=0.01)
    finished = threading.Event()
    def slow(_):
        time.sleep(0.05)
        finished.set()
        return None
    db._backend.execute = slow
    req = request(timeout=0.01)
    with pytest.raises(DatabaseExecutionDenied) as exc:
        db.execute(req, authorization=auth(db, req), agent_id="a", session_id="s", tool_id="t")
    assert exc.value.decision.metadata["error"] == "timeout"
    assert not finished.is_set()
