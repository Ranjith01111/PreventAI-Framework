import pytest

from preventai import (
    Agent,
    DatabasePolicy,
    FileSystemPolicy,
    NetworkPolicy,
    PreventApp,
    SandboxPolicy,
    ToolExecutionDenied,
)


@pytest.mark.parametrize(
    ("keyword", "policy"),
    [
        ("sandbox_policy", SandboxPolicy(allowed_commands=("echo",))),
        ("network_policy", NetworkPolicy(allowed_hosts=("example.com",))),
        (
            "filesystem_policy",
            FileSystemPolicy(allowed_roots=(".",), allowed_operations=("read",)),
        ),
        (
            "database_policy",
            DatabasePolicy(allowed_databases=("db",), allowed_tables=("table",)),
        ),
    ],
)
def test_untrusted_framework_boundary_tool_blocks_before_dispatch(keyword, policy):
    app = PreventApp()
    agent = app.add_agent(Agent("boundary-agent"))
    calls = []

    @agent.tool("boundary", **{keyword: policy})
    def boundary():
        calls.append(True)
        return None

    app.run()
    with pytest.raises(ToolExecutionDenied):
        boundary()
    assert calls == []


def test_untrusted_mcp_capable_tool_is_blocked_before_execution():
    app = PreventApp()
    agent = app.add_agent(Agent("mcp-agent"))
    calls = []

    @agent.tool("mcp")
    def mcp():
        calls.append(True)
        return "not trusted"

    app.run()
    with pytest.raises(ToolExecutionDenied):
        mcp()
    assert calls == []
