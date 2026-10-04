from preventai import (
    Agent,
    ApprovalManager,
    ApprovalPolicy,
    DataFlowEngine,
    DataFlowPolicy,
    DataFlowDestination,
    DataSource,
    DecisionType,
    Credential,
    FlowClassification,
    FlowTrust,
    PreventApp,
    RuntimePolicy,
    AuditLogger,
    DataReleasePolicy,
)
from preventai.core import (
    PermissionCategory,
    PermissionEngine,
    PermissionRule,
    PolicyEngine,
    PolicyRule,
)
from preventai.tools import FileSystemPolicy, FileSystemRequest


def make_filesystem_tool(
    root,
    *,
    operation=None,
    allowed_operations=("read", "write", "create", "append", "delete", "rename", "list", "stat"),
    policy_rules=None,
    permission_rules=None,
    flow_rules=None,
    approval_policy=None,
    approval_manager=None,
    runtime_policy=None,
    source_classification=FlowClassification.INTERNAL,
    source_trust=FlowTrust.TRUSTED,
    authenticator=None,
    agent_id=None,
    credential=None,
    audit_logger=None,
    filesystem_policy_options=None,
    callable_failure=False,
    request_path=None,
    request_data=None,
    data_release_policy=None,
):
    rules = [
        PolicyRule("allow-all", DecisionType.ALLOW, action="*")
    ] if policy_rules is None else policy_rules
    permissions = [
        PermissionRule("tool-read", PermissionCategory.READ, action="operate"),
        PermissionRule("filesystem", PermissionCategory.FILE, action="filesystem.*"),
    ] if permission_rules is None else permission_rules
    flows = [DataFlowPolicy("allow-flow", DecisionType.ALLOW)] if flow_rules is None else flow_rules
    app = PreventApp(
        policy=PolicyEngine(rules),
        permissions=PermissionEngine(permissions),
        data_flow=DataFlowEngine(flows),
        approval_policy=approval_policy,
        approval_manager=approval_manager,
        runtime_policy=runtime_policy,
        authenticator=authenticator,
        audit_logger=audit_logger,
        data_release_policy=data_release_policy,
    )
    agent = app.add_agent(
        Agent("filesystem-agent", agent_id=agent_id, credential=credential)
    )
    fs_policy = FileSystemPolicy(
        allowed_roots=(str(root),),
        allowed_operations=tuple(allowed_operations),
        **(filesystem_policy_options or {}),
    )

    @agent.tool(
        "operate",
        filesystem_policy=fs_policy,
        data_source=DataSource("source", source_classification, source_trust),
        data_destination=DataFlowDestination("sink"),
        runtime_policy=runtime_policy,
    )
    def operate(
        path, op="read", data=b"", destination_path=None, overwrite=False,
        classification=FlowClassification.INTERNAL,
        request_trust=FlowTrust.UNTRUSTED,
    ):
        if callable_failure:
            raise RuntimeError("callable failure")
        return FileSystemRequest(
            operation=op,
            path=str(request_path if request_path is not None else path),
            data=request_data if request_data is not None else data,
            destination_path=destination_path,
            overwrite=overwrite,
            classification=classification,
            source_trust=request_trust,
        )

    app.run()
    return app, agent, operate


def instrument_filesystem(tool, monkeypatch):
    operations = []
    backend = tool._filesystem._backend
    for name in ("read", "write", "delete", "rename", "list", "stat"):
        original = getattr(backend, name)

        def wrapped(*args, __original=original, __name=name, **kwargs):
            operations.append(__name)
            return __original(*args, **kwargs)

        monkeypatch.setattr(backend, name, wrapped)
    return operations
