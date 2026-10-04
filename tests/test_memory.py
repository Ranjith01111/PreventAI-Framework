from datetime import datetime, timedelta, timezone
from dataclasses import FrozenInstanceError
from types import MappingProxyType

import pytest

from preventai import (
    DataReleasePolicy,
    DataReleaseRequest,
    DestinationTrust,
    DecisionType,
    PreventApp,
)
from preventai.core.dataflow import FlowClassification, FlowTrust
from preventai.memory import (
    MemoryAccessContext,
    MemoryAccessDenied,
    MemoryOperation,
    MemoryPolicy,
    MemoryPolicyRule,
    MemoryStore,
)
from preventai.memory.memory import MemoryEntry, MemoryStatus


def store(*, allow=MemoryOperation) -> MemoryStore:
    rules = [
        MemoryPolicyRule(operation.value, operation, DecisionType.ALLOW)
        for operation in allow
    ]
    return MemoryStore(policy=MemoryPolicy(rules))


def context(agent="agent-a", session="session-a", namespace="customer"):
    return MemoryAccessContext(agent, agent, session, namespace)


def test_memory_crud_and_optimistic_versioning():
    memory = store()
    entry = memory.write(context(), {"value": "safe"})
    assert memory.read(context(), entry.memory_id).payload == {"value": "safe"}
    updated = memory.update(context(), entry.memory_id, {"value": "new"}, expected_version=1)
    assert updated.version == 2
    with pytest.raises(MemoryAccessDenied):
        memory.update(context(), entry.memory_id, {"value": "stale"}, expected_version=1)
    memory.delete(context(), entry.memory_id)


def test_broader_memory_scope_requires_release_authorization():
    memory = MemoryStore(
        policy=MemoryPolicy([
            MemoryPolicyRule("write", MemoryOperation.WRITE, DecisionType.ALLOW),
        ]),
        application_id="app-a",
    )
    with pytest.raises(MemoryAccessDenied) as denied:
        memory.write(context(namespace="private"), {"secret": "value"}, namespace="shared")
    assert denied.value.decision.metadata["category"] == "memory_release_authorization_required"


def test_framework_authorized_broader_memory_scope_is_permitted():
    app = PreventApp(
        data_release_policy=DataReleasePolicy(
            allowed_destinations=("shared",),
            allowed_destination_trust=(DestinationTrust.INTERNAL,),
            allowed_classifications=(FlowClassification.PRIVATE,),
        )
    )
    memory = MemoryStore(
        policy=MemoryPolicy([
            MemoryPolicyRule("write", MemoryOperation.WRITE, DecisionType.ALLOW),
        ]),
        application_id=app._application_id,
        release_issuer=app.authorize_data_release,
    )
    owner = context(namespace="private")
    release_request = DataReleaseRequest(
        agent_id=owner.agent_id,
        session_id=owner.session_id,
        source="private",
        source_classification=FlowClassification.PRIVATE,
        lineage=("private",),
        destination="shared",
        destination_trust=DestinationTrust.INTERNAL,
        operation="memory.write",
        tool_id="memory",
        tool_name="memory",
    )
    entry = memory.write(
        owner,
        {"value": "approved"},
        namespace="shared",
        classification=FlowClassification.PRIVATE,
        release_request=release_request,
    )
    assert entry.namespace == "shared"


def test_memory_isolation_and_explicit_sharing():
    memory = store()
    owner = context()
    other = context("agent-b", "session-b")
    entry = memory.write(owner, {"value": 1})
    with pytest.raises(MemoryAccessDenied):
        memory.read(other, entry.memory_id)
    share = memory.share(owner, entry.memory_id, "agent-b", grantee_session_id="session-b")
    assert memory.read(other, entry.memory_id).payload == {"value": 1}
    memory.revoke_share(share.share_id)
    with pytest.raises(MemoryAccessDenied):
        memory.read(other, entry.memory_id)


def test_untrusted_and_classification_labels_are_not_downgraded():
    memory = store()
    entry = memory.write(
        context(), {"value": 1},
        classification=FlowClassification.SECRET,
        trust=FlowTrust.UNTRUSTED,
    )
    with pytest.raises(MemoryAccessDenied):
        memory.update(
            context(), entry.memory_id, {"value": 2}, expected_version=1,
            classification=FlowClassification.PUBLIC, trust=FlowTrust.TRUSTED,
        )
    updated = memory.update(
        context(), entry.memory_id, {"value": 2}, expected_version=1,
        trust=FlowTrust.TRUSTED,
    )
    assert updated.classification is FlowClassification.SECRET
    assert updated.trust is FlowTrust.UNTRUSTED


def test_expiration_and_limits_fail_closed():
    memory = MemoryStore(
        policy=MemoryPolicy([
            MemoryPolicyRule("write", MemoryOperation.WRITE, DecisionType.ALLOW),
            MemoryPolicyRule("read", MemoryOperation.READ, DecisionType.ALLOW),
        ]),
        max_entry_bytes=5,
    )
    with pytest.raises(MemoryAccessDenied):
        memory.write(context(), {"too": "large"})
    expired = store()
    expired_entry = expired.write(
        context(), {"value": 1},
        expires_at=datetime.now(timezone.utc) - timedelta(seconds=1),
    )
    with pytest.raises(MemoryAccessDenied):
        expired.read(context(), expired_entry.memory_id)


@pytest.mark.parametrize("classification", list(FlowClassification))
def test_all_classifications_are_preserved(classification):
    memory = store()
    entry = memory.write(context(), {"classification": classification.value}, classification=classification)
    assert memory.read(context(), entry.memory_id).classification is classification


def test_entry_validation_and_immutable_metadata():
    now = datetime.now(timezone.utc)
    with pytest.raises(ValueError):
        MemoryEntry("", "agent", "session", "namespace", FlowClassification.PRIVATE,
                    FlowTrust.TRUSTED, now, now, None, 1, MemoryStatus.ACTIVE, {}, "payload")
    with pytest.raises(ValueError):
        MemoryEntry("id", "agent", "session", "", FlowClassification.PRIVATE,
                    FlowTrust.TRUSTED, now, now, None, 1, MemoryStatus.ACTIVE, {}, "payload")
    with pytest.raises(ValueError):
        MemoryEntry("id", "agent", "session", "namespace", "private",
                    FlowTrust.TRUSTED, now, now, None, 1, MemoryStatus.ACTIVE, {}, "payload")
    with pytest.raises(ValueError):
        MemoryEntry("id", "agent", "session", "namespace", FlowClassification.PRIVATE,
                    "trusted", now, now, None, 1, MemoryStatus.ACTIVE, {}, "payload")
    with pytest.raises(ValueError):
        MemoryEntry("id", "agent", "session", "namespace", FlowClassification.PRIVATE,
                    FlowTrust.TRUSTED, now.replace(tzinfo=None), now, None, 1,
                    MemoryStatus.ACTIVE, {}, "payload")
    with pytest.raises(ValueError):
        MemoryEntry("id", "agent", "session", "namespace", FlowClassification.PRIVATE,
                    FlowTrust.TRUSTED, now, now, None, 0, MemoryStatus.ACTIVE, {}, "payload")
    entry = MemoryEntry("id", "agent", "session", "namespace", FlowClassification.PRIVATE,
                        FlowTrust.TRUSTED, now, now, None, 1, MemoryStatus.ACTIVE,
                        {"label": "safe"}, {"value": 1})
    assert isinstance(entry.metadata, MappingProxyType)
    with pytest.raises(TypeError):
        entry.metadata["label"] = "changed"
    with pytest.raises(FrozenInstanceError):
        entry.version = 2


@pytest.mark.parametrize("operation", [
    MemoryOperation.READ, MemoryOperation.UPDATE, MemoryOperation.DELETE,
])
def test_missing_memory_operations_fail_closed(operation):
    memory = store()
    with pytest.raises(MemoryAccessDenied) as error:
        if operation is MemoryOperation.READ:
            memory.read(context(), "missing")
        elif operation is MemoryOperation.UPDATE:
            memory.update(context(), "missing", {"x": 1}, expected_version=1)
        else:
            memory.delete(context(), "missing")
    assert error.value.decision.decision_type is DecisionType.BLOCK


def test_duplicate_memory_id_does_not_replace_existing_entry():
    memory = store()
    first = memory.write(context(), {"value": "first"}, memory_id="fixed-id")
    with pytest.raises(MemoryAccessDenied):
        memory.write(context(), {"value": "second"}, memory_id="fixed-id")
    assert memory.read(context(), first.memory_id).payload == {"value": "first"}


def test_sensitive_metadata_and_payload_are_not_disclosed_in_denial():
    memory = store()
    with pytest.raises(ValueError) as error:
        memory.write(context(), {"api_key": "synthetic-secret"}, metadata={"api_key": "synthetic-secret"})
    assert "synthetic-secret" not in str(error.value)
    entry = memory.write(context(), {"api_key": "synthetic-secret"})
    with pytest.raises(MemoryAccessDenied) as denied:
        memory.read(context("other", "other-session"), entry.memory_id)
    assert "synthetic-secret" not in str(denied.value)
    assert "synthetic-secret" not in repr(denied.value.decision.to_dict())


def test_failed_limit_write_does_not_change_existing_state():
    memory = MemoryStore(
        policy=MemoryPolicy([
            MemoryPolicyRule("write", MemoryOperation.WRITE, DecisionType.ALLOW),
            MemoryPolicyRule("read", MemoryOperation.READ, DecisionType.ALLOW),
        ]),
        max_total_bytes=10,
    )
    first = memory.write(context(), "one")
    with pytest.raises(MemoryAccessDenied):
        memory.write(context(), "this payload exceeds the total limit")
    assert memory.read(context(), first.memory_id).payload == "one"


def test_invalid_memory_configuration_is_rejected():
    with pytest.raises(ValueError):
        MemoryStore(policy=MemoryPolicy(), max_entry_bytes=0)
    with pytest.raises(ValueError):
        MemoryStore(policy=MemoryPolicy(), max_entries_per_namespace=False)
    with pytest.raises(ValueError):
        MemoryStore(policy=MemoryPolicy(), max_total_bytes=-1)
