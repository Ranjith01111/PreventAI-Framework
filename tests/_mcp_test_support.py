from preventai import MCPBoundary as _MCPBoundary
from _boundary_test_support import bind_test_boundary


def MCPBoundary(policy, downstream, **kwargs):
    checks = {
        "policy_check": lambda request: True,
        "permission_check": lambda request: True,
        "delegation_check": lambda request: True,
        "dataflow_check": lambda request: True,
        "runtime_check": lambda request: True,
        "approval_check": lambda request: True,
    }
    checks.update(kwargs)
    return bind_test_boundary(_MCPBoundary(policy, downstream, **checks))
