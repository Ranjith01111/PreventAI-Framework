from pathlib import Path
import os
import socket
import stat

import pytest

from _filesystem_test_support import instrument_filesystem, make_filesystem_tool
from preventai import ToolExecutionDenied
from preventai.tools import FileSystemRequest


def _map_resolved_paths(monkeypatch, path_map):
    original_resolve = Path.resolve

    def resolve(path, strict=False):
        target = path_map.get(path)
        if target is not None:
            return original_resolve(target, strict=strict)
        return original_resolve(path, strict=strict)

    monkeypatch.setattr(Path, "resolve", resolve)


def test_parent_traversal_is_denied_before_operation(tmp_path, monkeypatch):
    root = tmp_path / "root"
    root.mkdir()
    _, _, tool = make_filesystem_tool(root)
    operations = instrument_filesystem(tool, monkeypatch)
    outside = root.parent / "outside"
    with pytest.raises(ToolExecutionDenied):
        tool(str(root / ".." / outside.name), "create", data=b"denied")
    assert operations == []
    assert not outside.exists()


@pytest.mark.parametrize(
    "suffix",
    [
        "..\\outside\\item",
        ".\\..\\outside\\item",
        "nested\\..\\..\\outside\\item",
    ],
)
def test_traversal_variants_are_denied_before_backend_operation(
    tmp_path, monkeypatch, suffix
):
    root = tmp_path / "root"
    root.mkdir()
    _, _, tool = make_filesystem_tool(root)
    operations = instrument_filesystem(tool, monkeypatch)
    candidate = root / suffix
    with pytest.raises(ToolExecutionDenied):
        tool(str(candidate), "create", data=b"blocked")
    assert operations == []


def test_relative_path_is_canonicalized_from_current_directory(tmp_path, monkeypatch):
    root = tmp_path / "root"
    root.mkdir()
    (root / "nested").mkdir()
    monkeypatch.chdir(root)
    _, _, tool = make_filesystem_tool(root)
    result = tool("nested\\..\\item", "create", data=b"relative")
    assert result["operation"] == "create"
    assert (root / "item").read_bytes() == b"relative"


@pytest.mark.parametrize("variant", ["dot", "repeated-separators", "trailing-separator"])
def test_path_normalization_variants_stay_within_allowed_root(
    tmp_path, variant
):
    root = tmp_path / "root"
    root.mkdir()
    item = root / "item"
    item.write_text("inside", encoding="utf-8")
    if variant == "dot":
        supplied = root / "." / "item"
    elif variant == "repeated-separators":
        supplied = str(root) + os.sep * 2 + "item"
    else:
        supplied = str(item) + os.sep
    _, _, tool = make_filesystem_tool(root)
    assert tool(str(supplied), "read")["content"] == "inside"


@pytest.mark.skipif(os.name != "nt", reason="Windows drive and UNC path semantics")
def test_windows_case_variant_is_canonicalized_and_unc_path_is_denied(tmp_path, monkeypatch):
    root = tmp_path / "root"
    root.mkdir()
    item = root / "item"
    item.write_text("inside", encoding="utf-8")
    _, _, tool = make_filesystem_tool(root)
    upper_case_path = str(item).swapcase()
    assert tool(upper_case_path, "read")["content"] == "inside"
    operations = instrument_filesystem(tool, monkeypatch)
    with pytest.raises(ToolExecutionDenied):
        tool(r"\\server\share\outside.txt", "read")
    assert operations == []


def test_root_prefix_sibling_is_outside_allowed_root(tmp_path, monkeypatch):
    root = tmp_path / "root"
    sibling = tmp_path / "root-sibling"
    root.mkdir()
    sibling.mkdir()
    _, _, tool = make_filesystem_tool(root)
    operations = instrument_filesystem(tool, monkeypatch)
    with pytest.raises(ToolExecutionDenied):
        tool(str(sibling / "item"), "create", data=b"blocked")
    assert operations == []


def test_rename_checks_both_paths_before_any_operation(tmp_path, monkeypatch):
    root = tmp_path / "root"
    root.mkdir()
    source = root / "source"
    source.write_text("content", encoding="utf-8")
    _, _, tool = make_filesystem_tool(root)
    operations = instrument_filesystem(tool, monkeypatch)
    with pytest.raises(ToolExecutionDenied):
        tool(str(source), "rename", destination_path=str(root.parent / "escaped"))
    assert operations == []
    assert source.read_text(encoding="utf-8") == "content"


def test_special_file_is_denied_before_operation(tmp_path, monkeypatch):
    _, _, tool = make_filesystem_tool(tmp_path)
    operations = instrument_filesystem(tool, monkeypatch)
    with pytest.raises(ToolExecutionDenied):
        tool(str(tmp_path), "read")
    assert operations == []


def test_directory_is_not_opened_as_a_regular_file(tmp_path, monkeypatch):
    directory = tmp_path / "directory"
    directory.mkdir()
    _, _, tool = make_filesystem_tool(tmp_path)
    operations = instrument_filesystem(tool, monkeypatch)
    with pytest.raises(ToolExecutionDenied):
        tool(str(directory), "read")
    assert operations == []


@pytest.mark.skipif(os.name != "nt", reason="Windows path semantics")
@pytest.mark.parametrize("name", ["NUL", "CON.txt", "item:stream"])
def test_windows_device_and_alternate_stream_paths_are_denied(tmp_path, name):
    _, _, tool = make_filesystem_tool(tmp_path)
    with pytest.raises(ToolExecutionDenied):
        tool(str(tmp_path / name), "create", data=b"no")


def test_symlink_parent_escape_is_denied(tmp_path, monkeypatch):
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    link = root / "link"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"symbolic link creation unavailable: {exc}")
    _, _, tool = make_filesystem_tool(root)
    operations = instrument_filesystem(tool, monkeypatch)
    with pytest.raises(ToolExecutionDenied):
        tool(str(link / "new"), "create", data=b"no")
    assert operations == []
    assert list(outside.iterdir()) == []


@pytest.mark.parametrize("operation", ["read", "write", "rename"])
def test_fake_direct_symlink_path_denies_without_backend_operation(
    tmp_path, monkeypatch, operation
):
    root = tmp_path / "root"
    root.mkdir()
    source = root / "source"
    source.write_text("source", encoding="utf-8")
    virtual_link = root / "link"
    _, _, tool = make_filesystem_tool(root)
    operations = instrument_filesystem(tool, monkeypatch)
    original_is_symlink = Path.is_symlink

    def is_symlink(path):
        if path == virtual_link:
            return True
        return original_is_symlink(path)

    monkeypatch.setattr(Path, "is_symlink", is_symlink)
    if operation == "rename":
        args = {"destination_path": str(virtual_link)}
        path = source
    else:
        args = {"data": b"x"} if operation == "write" else {}
        path = virtual_link
    with pytest.raises(ToolExecutionDenied):
        tool(str(path), operation, **args)
    assert operations == []


def test_fake_parent_symlink_resolving_to_allowed_target_is_policy_allowed(
    tmp_path, monkeypatch
):
    root = tmp_path / "root"
    root.mkdir()
    actual_parent = root / "actual"
    actual_parent.mkdir()
    actual_file = actual_parent / "item"
    actual_file.write_text("allowed target", encoding="utf-8")
    virtual_parent = root / "virtual-parent"
    virtual_file = virtual_parent / "item"
    _map_resolved_paths(monkeypatch, {virtual_file: actual_file})
    _, _, tool = make_filesystem_tool(root)
    operations = instrument_filesystem(tool, monkeypatch)
    assert tool(str(virtual_file), "read")["content"] == "allowed target"
    assert operations == ["read"]


@pytest.mark.parametrize("operation", ["read", "write", "rename"])
def test_fake_parent_symlink_escape_is_denied_before_operation(
    tmp_path, monkeypatch, operation
):
    root = tmp_path / "root"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    source = root / "source"
    source.write_text("source", encoding="utf-8")
    escaped_file = outside / "item"
    escaped_file.write_text("outside", encoding="utf-8")
    virtual_file = root / "virtual-parent" / "item"
    _map_resolved_paths(monkeypatch, {virtual_file: escaped_file})
    _, _, tool = make_filesystem_tool(root)
    operations = instrument_filesystem(tool, monkeypatch)
    if operation == "rename":
        with pytest.raises(ToolExecutionDenied):
            tool(str(source), "rename", destination_path=str(virtual_file))
    else:
        kwargs = {"data": b"changed"} if operation == "write" else {}
        with pytest.raises(ToolExecutionDenied):
            tool(str(virtual_file), operation, **kwargs)
    assert operations == []
    assert escaped_file.read_text(encoding="utf-8") == "outside"
    assert source.read_text(encoding="utf-8") == "source"


def test_fake_nested_parent_symlink_escape_is_denied(tmp_path, monkeypatch):
    root = tmp_path / "root"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    outside_file = outside / "item"
    outside_file.write_text("outside", encoding="utf-8")
    nested_virtual = root / "link-one" / "link-two" / "item"
    _map_resolved_paths(monkeypatch, {nested_virtual: outside_file})
    _, _, tool = make_filesystem_tool(root)
    operations = instrument_filesystem(tool, monkeypatch)
    with pytest.raises(ToolExecutionDenied):
        tool(str(nested_virtual), "read")
    assert operations == []


def test_fake_symlink_target_inside_root_is_rejected_when_leaf_is_link(
    tmp_path, monkeypatch
):
    root = tmp_path / "root"
    root.mkdir()
    target = root / "target"
    target.write_text("target", encoding="utf-8")
    virtual_link = root / "link"
    _, _, tool = make_filesystem_tool(root)
    operations = instrument_filesystem(tool, monkeypatch)
    original_is_symlink = Path.is_symlink
    monkeypatch.setattr(
        Path,
        "is_symlink",
        lambda path: True if path == virtual_link else original_is_symlink(path),
    )
    with pytest.raises(ToolExecutionDenied):
        tool(str(virtual_link), "read")
    assert operations == []


def test_fake_direct_destination_symlink_is_denied_before_rename(
    tmp_path, monkeypatch
):
    root = tmp_path / "root"
    root.mkdir()
    source = root / "source"
    source.write_text("source", encoding="utf-8")
    link = root / "link"
    _, _, tool = make_filesystem_tool(root)
    operations = instrument_filesystem(tool, monkeypatch)
    original_is_symlink = Path.is_symlink
    monkeypatch.setattr(
        Path,
        "is_symlink",
        lambda path: True if path == link else original_is_symlink(path),
    )
    with pytest.raises(ToolExecutionDenied):
        tool(str(source), "rename", destination_path=str(link))
    assert operations == []
    assert source.read_text(encoding="utf-8") == "source"


def test_fifo_is_denied_without_backend_operation(tmp_path, monkeypatch):
    if not hasattr(os, "mkfifo"):
        pytest.skip("FIFO creation is unavailable")
    fifo = tmp_path / "pipe"
    try:
        os.mkfifo(fifo)
    except OSError as exc:
        pytest.skip(f"FIFO creation unavailable: {exc}")
    _, _, tool = make_filesystem_tool(tmp_path)
    operations = instrument_filesystem(tool, monkeypatch)
    with pytest.raises(ToolExecutionDenied):
        tool(str(fifo), "read")
    assert operations == []


def test_socket_file_is_denied_without_backend_operation(tmp_path, monkeypatch):
    if os.name == "nt":
        pytest.skip("filesystem Unix-domain sockets are unavailable on this platform")
    socket_path = tmp_path / "socket"
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        listener.bind(str(socket_path))
        _, _, tool = make_filesystem_tool(tmp_path)
        operations = instrument_filesystem(tool, monkeypatch)
        with pytest.raises(ToolExecutionDenied):
            tool(str(socket_path), "read")
        assert operations == []
    finally:
        listener.close()


@pytest.mark.parametrize(
    "mode",
    [
        stat.S_IFIFO,
        stat.S_IFSOCK,
        stat.S_IFCHR,
        getattr(stat, "S_IFBLK", stat.S_IFCHR),
    ],
)
def test_simulated_special_file_metadata_is_denied(tmp_path, monkeypatch, mode):
    path = tmp_path / "special"
    _, _, tool = make_filesystem_tool(tmp_path)
    operations = instrument_filesystem(tool, monkeypatch)
    original_stat = Path.stat

    def fake_stat(candidate, *args, **kwargs):
        if candidate == path:
            return os.stat_result((mode, 0, 0, 0, 0, 0, 0, 0, 0, 0))
        return original_stat(candidate, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", fake_stat)
    with pytest.raises(ToolExecutionDenied):
        tool(str(path), "read")
    assert operations == []


def test_hard_link_remains_inside_allowed_root(tmp_path):
    source = tmp_path / "source"
    source.write_text("same inode", encoding="utf-8")
    alias = tmp_path / "alias"
    try:
        alias.hardlink_to(source)
    except OSError:
        pytest.skip("hard links are not available")
    _, _, tool = make_filesystem_tool(tmp_path)
    assert tool(str(alias))["content"] == "same inode"


def test_framework_never_exposes_canonical_parent_on_output(tmp_path):
    path = tmp_path / "item"
    path.write_text("visible", encoding="utf-8")
    _, _, tool = make_filesystem_tool(tmp_path)
    result = tool(str(path))
    assert result["path"] == str(path)


def test_write_denial_does_not_modify_existing_file(tmp_path, monkeypatch):
    path = tmp_path / "item"
    path.write_text("preserved", encoding="utf-8")
    _, _, tool = make_filesystem_tool(tmp_path)
    operations = instrument_filesystem(tool, monkeypatch)
    with pytest.raises(ToolExecutionDenied):
        tool(str(path), "write", data=b"changed")
    assert operations == []
    assert path.read_text(encoding="utf-8") == "preserved"


def test_framework_return_type_must_be_filesystem_request(tmp_path, monkeypatch):
    _, _, tool = make_filesystem_tool(tmp_path)
    operations = instrument_filesystem(tool, monkeypatch)
    tool._function = lambda *args, **kwargs: {"operation": "stat", "path": str(tmp_path)}
    with pytest.raises(ToolExecutionDenied):
        tool(str(tmp_path / "other"), "stat")
    assert operations == []
