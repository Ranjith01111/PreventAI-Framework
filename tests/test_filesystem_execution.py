import pytest

from _filesystem_test_support import instrument_filesystem, make_filesystem_tool
from preventai import ToolExecutionDenied


@pytest.mark.parametrize(
    ("operation", "args"),
    [
        ("read", {}),
        ("create", {"data": b"data"}),
        ("append", {"data": b"data"}),
        ("write", {"data": b"data", "overwrite": True}),
        ("delete", {}),
        ("rename", {"destination_path": "elsewhere"}),
        ("list", {}),
        ("stat", {}),
    ],
)
def test_framework_executes_explicit_filesystem_requests(tmp_path, operation, args):
    path = tmp_path / "item"
    path.write_text("old", encoding="utf-8")
    _, _, tool = make_filesystem_tool(tmp_path)
    request_path = str(tmp_path) if operation == "list" else str(path)
    if operation == "create":
        request_path = str(tmp_path / "created")
    if operation == "rename":
        args = {**args, "destination_path": str(tmp_path / "renamed")}
    result = tool(request_path, operation, **args)
    assert result["operation"] == operation


def test_wrong_callable_result_is_blocked_without_filesystem_access(tmp_path, monkeypatch):
    _, _, tool = make_filesystem_tool(tmp_path)
    operations = instrument_filesystem(tool, monkeypatch)
    tool._function = lambda *args, **kwargs: {"path": str(tmp_path)}
    with pytest.raises(ToolExecutionDenied):
        tool(str(tmp_path / "item"))
    assert operations == []


def test_invalid_path_is_blocked_without_filesystem_operation(tmp_path, monkeypatch):
    _, _, tool = make_filesystem_tool(tmp_path)
    operations = instrument_filesystem(tool, monkeypatch)
    with pytest.raises(ToolExecutionDenied):
        tool(str(tmp_path.parent / "outside"), "create", data=b"no")
    assert operations == []


def test_missing_permission_blocks_before_operation(tmp_path, monkeypatch):
    _, _, tool = make_filesystem_tool(
        tmp_path,
        permission_rules=[],
    )
    operations = instrument_filesystem(tool, monkeypatch)
    with pytest.raises(ToolExecutionDenied):
        tool(str(tmp_path / "item"), "create", data=b"no")
    assert operations == []


def test_policy_denial_blocks_before_operation(tmp_path, monkeypatch):
    from preventai import DecisionType
    from preventai.core import PolicyRule

    _, _, tool = make_filesystem_tool(
        tmp_path,
        policy_rules=[PolicyRule("deny", DecisionType.BLOCK, action="filesystem.*")],
    )
    operations = instrument_filesystem(tool, monkeypatch)
    with pytest.raises(ToolExecutionDenied):
        tool(str(tmp_path / "item"), "create", data=b"no")
    assert operations == []


def test_denied_sensitive_input_is_blocked_by_guard_before_operation(tmp_path, monkeypatch):
    _, _, tool = make_filesystem_tool(tmp_path)
    operations = instrument_filesystem(tool, monkeypatch)
    with pytest.raises(ToolExecutionDenied):
        tool(str(tmp_path / "id_rsa"), "create", data=b"secret")
    assert operations == []


def test_successful_read_returns_output_for_output_guard(tmp_path):
    path = tmp_path / "item"
    path.write_text("hello", encoding="utf-8")
    _, _, tool = make_filesystem_tool(tmp_path)
    assert tool(str(path))["content"] == "hello"
