from dataclasses import replace
import threading
import time
from weakref import WeakKeyDictionary

import pytest

from preventai.tools import (
    FileSystem,
    FileSystemAuthorization,
    FileSystemExecutionDenied,
    FileSystemPolicy,
    FileSystemRequest,
)
from _filesystem_test_support import make_filesystem_tool

_AUTHORIZERS = WeakKeyDictionary()


def filesystem(root, **kwargs):
    allowed_operations = kwargs.pop(
        "allowed_operations",
        ("read", "write", "create", "append", "delete", "rename", "list", "stat"),
    )
    _, _, tool = make_filesystem_tool(
        root,
        allowed_operations=allowed_operations,
        filesystem_policy_options=kwargs,
    )
    _AUTHORIZERS[tool._filesystem] = tool._filesystem_authorizer
    return tool._filesystem


def _authorize(fs, request, plan=None, *, ttl=30.0):
    authorizer = _AUTHORIZERS.get(fs)
    if authorizer is None:
        raise AssertionError("test filesystem must be bound to a framework tool")
    return authorizer(request, plan or fs._prepare(request), ttl=ttl)


def execute(fs, request):
    token = _authorize(fs, request)
    return fs.execute(request, authorization=token)


@pytest.mark.parametrize(
    "operation",
    ["read", "write", "create", "append", "delete", "rename", "list", "stat"],
)
def test_filesystem_policy_requires_explicit_operation_allowlist(tmp_path, operation):
    fs = FileSystem(FileSystemPolicy(allowed_roots=(str(tmp_path),)))
    with pytest.raises(FileSystemExecutionDenied):
        fs._prepare(FileSystemRequest(
            operation,
            str(tmp_path / "item"),
            destination_path=str(tmp_path / "destination") if operation == "rename" else None,
        ))


@pytest.mark.parametrize(
    ("operation", "expected"),
    [
        ("read", "existing"),
        ("create", None),
        ("append", None),
        ("write", None),
        ("list", None),
        ("stat", None),
        ("delete", None),
        ("rename", None),
    ],
)
def test_allowed_file_operations_are_bounded_and_functional(tmp_path, operation, expected):
    path = tmp_path / "item"
    path.write_text("existing", encoding="utf-8")
    fs = filesystem(tmp_path)
    if operation == "read":
        assert execute(fs, FileSystemRequest(operation, str(path))).content == expected
    elif operation == "create":
        target = tmp_path / "created"
        execute(fs, FileSystemRequest(operation, str(target), data=b"created"))
        assert target.read_text(encoding="utf-8") == "created"
    elif operation == "append":
        execute(fs, FileSystemRequest(operation, str(path), data=b"-more"))
        assert path.read_text(encoding="utf-8") == "existing-more"
    elif operation == "write":
        execute(fs, FileSystemRequest(operation, str(path), data=b"replaced", overwrite=True))
        assert path.read_text(encoding="utf-8") == "replaced"
    elif operation == "list":
        assert any(item["name"] == "item" for item in execute(fs, FileSystemRequest(operation, str(tmp_path))).entries)
    elif operation == "stat":
        assert execute(fs, FileSystemRequest(operation, str(path))).metadata["is_file"]
    elif operation == "delete":
        execute(fs, FileSystemRequest(operation, str(path)))
        assert not path.exists()
    elif operation == "rename":
        destination = tmp_path / "renamed"
        execute(fs, FileSystemRequest(operation, str(path), destination_path=str(destination)))
        assert destination.read_text(encoding="utf-8") == "existing"


@pytest.mark.parametrize(
    "operation",
    ["read", "write", "create", "append", "delete", "rename", "list", "stat"],
)
def test_paths_outside_allowed_roots_are_denied(tmp_path, operation):
    root = tmp_path / "allowed"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    item = outside / "item"
    item.write_text("preserve", encoding="utf-8")
    fs = filesystem(root)
    request = FileSystemRequest(
        operation,
        str(item),
        destination_path=str(item.with_name("renamed")) if operation == "rename" else None,
        data=b"x" if operation in {"write", "create", "append"} else b"",
    )
    with pytest.raises(FileSystemExecutionDenied):
        fs._prepare(request)
    assert item.read_text(encoding="utf-8") == "preserve"


@pytest.mark.parametrize("name", [".env", "id_rsa", "credentials", "secrets.json", ".ssh"])
def test_default_sensitive_path_components_are_denied(tmp_path, name):
    path = tmp_path / name
    path.write_text("protected", encoding="utf-8") if name != ".ssh" else path.mkdir()
    fs = filesystem(tmp_path)
    with pytest.raises(FileSystemExecutionDenied):
        fs._prepare(FileSystemRequest("read" if name != ".ssh" else "list", str(path)))


def test_symlink_escape_is_denied(tmp_path):
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    outside = tmp_path / "outside"
    outside.write_text("protected", encoding="utf-8")
    link = allowed / "shortcut"
    try:
        link.symlink_to(outside)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"symbolic link creation unavailable: {exc}")
    with pytest.raises(FileSystemExecutionDenied):
        filesystem(allowed)._prepare(FileSystemRequest("read", str(link)))


@pytest.mark.parametrize(
    ("policy_options", "request_factory"),
    [
        ({"max_read_bytes": 3}, lambda path: FileSystemRequest("read", str(path))),
        ({"max_write_bytes": 3}, lambda path: FileSystemRequest("create", str(path), data=b"four")),
        ({"max_path_length": 4}, lambda path: FileSystemRequest("stat", str(path))),
    ],
)
def test_policy_size_limits_deny_before_operation(tmp_path, policy_options, request_factory):
    path = tmp_path / "item"
    path.write_text("four", encoding="utf-8")
    fs = filesystem(tmp_path, **policy_options)
    with pytest.raises(FileSystemExecutionDenied):
        fs._prepare(request_factory(path))


def test_recursive_and_binary_requests_are_rejected(tmp_path):
    with pytest.raises(ValueError):
        FileSystemRequest("delete", str(tmp_path), recursive=True)
    with pytest.raises(ValueError):
        FileSystemRequest("create", str(tmp_path / "data"), data=b"\xff")


def test_revoked_authorization_denies_before_filesystem_operation(tmp_path, monkeypatch):
    fs = filesystem(tmp_path)
    request = FileSystemRequest("create", str(tmp_path / "item"), data=b"x")
    plan = fs._prepare(request)
    token = _authorize(fs, request, plan)
    fs.revoke_authorization(token)
    backend_calls = []
    monkeypatch.setattr(
        fs._backend, "write", lambda *args, **kwargs: backend_calls.append(True)
    )
    with pytest.raises(FileSystemExecutionDenied):
        fs.execute(request, authorization=token)
    assert backend_calls == []
    assert not (tmp_path / "item").exists()


def test_malformed_authorization_denies_before_path_inspection(tmp_path, monkeypatch):
    fs = filesystem(tmp_path)
    request = FileSystemRequest("stat", str(tmp_path))
    malformed = FileSystemAuthorization(
        request.digest(), str(tmp_path), None, "never", "nonce", "signature"
    )
    backend_calls = []
    monkeypatch.setattr(
        fs._backend, "stat", lambda *args, **kwargs: backend_calls.append(True)
    )
    monkeypatch.setattr(fs, "_prepare", lambda request: pytest.fail("path inspected"))
    with pytest.raises(FileSystemExecutionDenied):
        fs.execute(request, authorization=malformed)
    assert backend_calls == []


def test_concurrency_limit_rejects_second_operation_until_first_finishes(
    tmp_path, monkeypatch
):
    fs = filesystem(tmp_path, max_concurrent_operations=1)
    first = FileSystemRequest("create", str(tmp_path / "first"), data=b"one")
    second = FileSystemRequest("create", str(tmp_path / "second"), data=b"two")
    first_token = _authorize(fs, first)
    second_token = _authorize(fs, second)
    entered = threading.Event()
    release = threading.Event()
    original_write = fs._backend.write
    result = []

    def blocked_write(*args, **kwargs):
        entered.set()
        if not release.wait(2):
            raise TimeoutError("test did not release the first operation")
        return original_write(*args, **kwargs)

    monkeypatch.setattr(fs._backend, "write", blocked_write)

    def run_first():
        try:
            fs.execute(first, authorization=first_token)
        except Exception as exc:
            result.append(exc)

    worker = threading.Thread(target=run_first)
    worker.start()
    assert entered.wait(1)
    with pytest.raises(FileSystemExecutionDenied):
        fs.execute(second, authorization=second_token)
    release.set()
    worker.join(2)
    assert not worker.is_alive()
    assert result == []
    assert (tmp_path / "first").read_bytes() == b"one"
    assert not (tmp_path / "second").exists()


def test_modified_request_invalidates_exact_authorization(tmp_path, monkeypatch):
    fs = filesystem(tmp_path)
    first = FileSystemRequest("create", str(tmp_path / "one"), data=b"first")
    plan = fs._prepare(first)
    token = _authorize(fs, first, plan)
    modified = replace(first, data=b"modified")
    backend_calls = []
    monkeypatch.setattr(
        fs._backend, "write", lambda *args, **kwargs: backend_calls.append(True)
    )
    with pytest.raises(FileSystemExecutionDenied):
        fs.execute(modified, authorization=token)
    assert backend_calls == []
    assert not (tmp_path / "one").exists()


@pytest.mark.parametrize(
    "changes",
    [
        {"operation": "write"},
        {"path": "different"},
        {"data": b"changed"},
        {"overwrite": True},
        {"agent_id": "another-agent"},
        {"session_id": "another-session"},
        {"tool_id": "another-tool"},
    ],
)
def test_exact_authorization_rejects_modified_request_without_backend_access(
    tmp_path, monkeypatch, changes
):
    fs = filesystem(tmp_path)
    path = tmp_path / "item"
    request = FileSystemRequest(
        "create", str(path), data=b"original",
        agent_id="agent", session_id="session", tool_id="tool",
    )
    token = _authorize(fs, request)
    backend_calls = []
    monkeypatch.setattr(
        fs._backend,
        "write",
        lambda *args, **kwargs: backend_calls.append(True),
    )
    modified = replace(request, **changes)
    with pytest.raises(FileSystemExecutionDenied):
        fs.execute(modified, authorization=token)
    assert backend_calls == []
    assert not path.exists()


def test_exact_authorization_binds_rename_destination(tmp_path, monkeypatch):
    fs = filesystem(tmp_path)
    source = tmp_path / "source"
    source.write_text("payload", encoding="utf-8")
    first_destination = tmp_path / "first"
    request = FileSystemRequest(
        "rename", str(source), destination_path=str(first_destination)
    )
    token = _authorize(fs, request)
    backend_calls = []
    monkeypatch.setattr(
        fs._backend,
        "rename",
        lambda *args, **kwargs: backend_calls.append(True),
    )
    modified = replace(request, destination_path=str(tmp_path / "second"))
    with pytest.raises(FileSystemExecutionDenied):
        fs.execute(modified, authorization=token)
    assert backend_calls == []
    assert source.read_text(encoding="utf-8") == "payload"


def test_public_authorization_minting_is_unavailable_and_fake_tokens_fail(tmp_path, monkeypatch):
    fs = filesystem(tmp_path)
    request = FileSystemRequest("create", str(tmp_path / "item"), data=b"x")
    plan = fs._prepare(request)
    assert not hasattr(fs, "authorize")
    assert not hasattr(fs, "_issue_authorization")
    fake = FileSystemAuthorization(
        request.digest(), str(plan.path), None, time.time() + 30, "fake", "fake"
    )
    backend_calls = []
    monkeypatch.setattr(
        fs._backend, "write", lambda *args, **kwargs: backend_calls.append(True)
    )
    with pytest.raises(FileSystemExecutionDenied):
        fs.execute(request, authorization=fake)
    assert backend_calls == []


def test_filesystem_module_exposes_no_issuer_capability_and_manual_binding_fails(
    tmp_path,
):
    import preventai.tools.filesystem as filesystem_module

    fs = FileSystem(
        FileSystemPolicy(
            allowed_roots=(str(tmp_path),),
            allowed_operations=("create",),
        )
    )
    assert not hasattr(filesystem_module, "_FRAMEWORK_ISSUER")
    assert not hasattr(fs, "authorize")
    assert not hasattr(fs, "_issue_authorization")
    with pytest.raises(TypeError):
        fs._bind_framework_authorizer()
    with pytest.raises(PermissionError):
        fs._bind_framework_authorizer(object())


@pytest.mark.parametrize("caller_value", [True, "framework", object(), lambda: True])
def test_caller_controlled_values_cannot_bind_or_authorize_filesystem(
    tmp_path, monkeypatch, caller_value
):
    fs = FileSystem(
        FileSystemPolicy(
            allowed_roots=(str(tmp_path),),
            allowed_operations=("create",),
        )
    )
    request = FileSystemRequest("create", str(tmp_path / "item"), data=b"x")
    backend_calls = []
    monkeypatch.setattr(
        fs._backend, "write", lambda *args, **kwargs: backend_calls.append(True)
    )
    with pytest.raises(PermissionError):
        fs._bind_framework_authorizer(caller_value)
    with pytest.raises(FileSystemExecutionDenied):
        fs.execute(request, authorization=caller_value)
    assert backend_calls == []
    assert not (tmp_path / "item").exists()


def test_direct_authorization_construction_and_framework_authorizer_are_instance_scoped(
    tmp_path, monkeypatch
):
    fs_a = filesystem(tmp_path)
    fs_b = filesystem(tmp_path)
    request = FileSystemRequest("create", str(tmp_path / "item"), data=b"x")
    token_a = _authorize(fs_a, request)
    token_b = _authorize(fs_b, request)
    assert token_a.signature != token_b.signature

    backend_calls = []
    monkeypatch.setattr(
        fs_b._backend, "write", lambda *args, **kwargs: backend_calls.append(True)
    )
    with pytest.raises(FileSystemExecutionDenied):
        fs_b.execute(request, authorization=token_a)
    assert backend_calls == []
    assert not (tmp_path / "item").exists()


@pytest.mark.parametrize(
    "authorization",
    [True, "approved", object(), lambda: True],
)
def test_non_authorization_values_block_before_backend_operation(
    tmp_path, monkeypatch, authorization
):
    fs = filesystem(tmp_path)
    request = FileSystemRequest("create", str(tmp_path / "item"), data=b"x")
    backend_calls = []
    monkeypatch.setattr(
        fs._backend, "write", lambda *args, **kwargs: backend_calls.append(True)
    )
    with pytest.raises(FileSystemExecutionDenied):
        fs.execute(request, authorization=authorization)
    assert backend_calls == []


def test_missing_authorization_denies_before_backend_operation(tmp_path, monkeypatch):
    fs = filesystem(tmp_path)
    request = FileSystemRequest("create", str(tmp_path / "item"), data=b"x")
    backend_calls = []
    monkeypatch.setattr(
        fs._backend, "write", lambda *args, **kwargs: backend_calls.append(True)
    )
    with pytest.raises(FileSystemExecutionDenied):
        fs.execute(request, authorization=None)
    assert backend_calls == []


def test_exact_authorization_is_single_use(tmp_path):
    fs = filesystem(tmp_path)
    path = tmp_path / "item"
    request = FileSystemRequest("create", str(path), data=b"one")
    plan = fs._prepare(request)
    token = _authorize(fs, request, plan)
    fs.execute(request, authorization=token)
    backend_calls = []
    original_write = fs._backend.write

    def count_write(*args, **kwargs):
        backend_calls.append(True)
        return original_write(*args, **kwargs)

    fs._backend.write = count_write
    with pytest.raises(FileSystemExecutionDenied):
        fs.execute(request, authorization=token)
    assert backend_calls == []
    assert path.read_bytes() == b"one"


def test_expired_authorization_denies_before_filesystem_operation(tmp_path, monkeypatch):
    fs = filesystem(tmp_path)
    request = FileSystemRequest("create", str(tmp_path / "item"), data=b"x")
    plan = fs._prepare(request)
    token = _authorize(fs, request, plan, ttl=0.0001)
    backend_calls = []
    monkeypatch.setattr(
        fs._backend, "write", lambda *args, **kwargs: backend_calls.append(True)
    )
    monkeypatch.setattr("preventai.tools.filesystem.time.time", lambda: token.expires_at + 1)
    with pytest.raises(FileSystemExecutionDenied):
        fs.execute(request, authorization=token)
    assert backend_calls == []
    assert not (tmp_path / "item").exists()


def test_caller_cannot_forge_authorization(tmp_path, monkeypatch):
    fs = filesystem(tmp_path)
    request = FileSystemRequest("create", str(tmp_path / "item"), data=b"x")
    assert not hasattr(fs, "authorize")
    assert not hasattr(fs, "_issue_authorization")
    backend_calls = []
    monkeypatch.setattr(
        fs._backend, "write", lambda *args, **kwargs: backend_calls.append(True)
    )
    with pytest.raises(FileSystemExecutionDenied):
        fs.execute(request, authorization=FileSystemAuthorization(
            request.digest(), str(tmp_path / "item"), None,
            9999999999.0, "n", "invalid"
        ))
    assert backend_calls == []
