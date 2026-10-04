"""Structured, policy-enforced database execution over an offline backend."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import hmac
import json
import math
import re
import secrets
import threading
import time
from types import MappingProxyType
from typing import Any, Mapping

from ..core.decision import Decision, DecisionReason, DecisionType, Severity
from ..core.dataflow import FlowClassification
from ..core.data_release import DataReleaseAuthorization, DataReleaseRequest, verify_release_authorization

_OPERATIONS = frozenset({"SELECT", "INSERT", "UPDATE", "DELETE"})
_DENIED_OPERATIONS = frozenset(
    {"CREATE", "ALTER", "DROP", "TRUNCATE", "GRANT", "REVOKE", "BEGIN",
     "COMMIT", "ROLLBACK", "SAVEPOINT", "CALL", "EXECUTE", "FUNCTION",
     "PROCEDURE"}
)
_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,127}$")


def _identifier(value: str, field: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise ValueError(f"{field} is invalid")
    return value


def _digest(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class DatabasePolicy:
    """Exact database/schema/table/column allowlists with deny precedence."""

    allowed_databases: tuple[str, ...] = ()
    denied_databases: tuple[str, ...] = ()
    allowed_schemas: tuple[str, ...] = ()
    denied_schemas: tuple[str, ...] = ()
    allowed_tables: tuple[str, ...] = ()
    denied_tables: tuple[str, ...] = ()
    allowed_columns: tuple[str, ...] = ()
    denied_columns: tuple[str, ...] = ()
    allowed_operations: tuple[str, ...] = ("SELECT",)
    denied_operations: tuple[str, ...] = ()
    max_request_bytes: int = 64 * 1024
    max_parameters: int = 100
    max_columns: int = 100
    max_batch_size: int = 100
    max_rows: int = 1000
    max_result_bytes: int = 1_000_000
    max_concurrent: int = 1
    timeout: float = 10.0
    allow_raw_sql: bool = False
    allowed_tenants: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for field in ("allowed_databases", "denied_databases", "allowed_schemas",
                      "denied_schemas", "allowed_tables", "denied_tables",
                      "allowed_columns", "denied_columns", "allowed_tenants"):
            values = getattr(self, field)
            if not isinstance(values, tuple) or any(not isinstance(v, str) or not v for v in values):
                raise ValueError(f"{field} must contain strings")
        for field in ("allowed_operations", "denied_operations"):
            values = getattr(self, field)
            if not isinstance(values, tuple) or any(not isinstance(v, str) or v.upper() not in _OPERATIONS for v in values):
                raise ValueError(f"{field} contains unsupported operation")
            object.__setattr__(self, field, tuple(v.upper() for v in values))
        for field in ("max_request_bytes", "max_parameters", "max_columns",
                      "max_batch_size", "max_rows", "max_result_bytes", "max_concurrent"):
            value = getattr(self, field)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{field} must be positive")
        if not isinstance(self.timeout, (int, float)) or isinstance(self.timeout, bool) or not math.isfinite(self.timeout) or self.timeout <= 0:
            raise ValueError("timeout must be positive")


@dataclass(frozen=True, slots=True)
class DatabaseRequest:
    operation: str
    database: str
    schema: str
    table: str
    columns: tuple[str, ...] = ()
    parameters: tuple[Any, ...] = ()
    template: str = ""
    raw_sql: str | None = None
    values: Mapping[str, Any] | None = None
    tenant_id: str | None = None
    max_rows: int | None = None
    timeout: float | None = None
    agent_id: str | None = None
    session_id: str | None = None
    tool_id: str | None = None
    classification: FlowClassification | None = None

    def __post_init__(self) -> None:
        operation = self.operation.upper() if isinstance(self.operation, str) else self.operation
        if operation not in _OPERATIONS:
            raise ValueError("unsupported database operation")
        object.__setattr__(self, "operation", operation)
        for field in ("database", "schema", "table"):
            _identifier(getattr(self, field), field)
        if not isinstance(self.columns, tuple) or any(not isinstance(c, str) or not _IDENTIFIER.fullmatch(c) for c in self.columns):
            raise ValueError("columns must contain identifiers")
        if not isinstance(self.parameters, tuple):
            raise ValueError("parameters must be a tuple")
        if not isinstance(self.template, str) or len(self.template) > 4096:
            raise ValueError("template is invalid")
        if self.raw_sql is not None and (not isinstance(self.raw_sql, str) or len(self.raw_sql) > 4096):
            raise ValueError("raw_sql is invalid")
        if self.values is not None and not isinstance(self.values, Mapping):
            raise ValueError("values must be an object")
        if self.max_rows is not None and (not isinstance(self.max_rows, int) or self.max_rows <= 0):
            raise ValueError("max_rows is invalid")
        if self.timeout is not None and (not isinstance(self.timeout, (int, float)) or self.timeout <= 0):
            raise ValueError("timeout is invalid")
        if self.classification is not None and not isinstance(self.classification, FlowClassification):
            raise ValueError("classification must be a FlowClassification")

    @property
    def action(self) -> str:
        return f"database.{self.operation.lower()}"

    def digest(self) -> str:
        return _digest({
            "operation": self.operation, "database": self.database, "schema": self.schema,
            "table": self.table, "columns": self.columns, "parameters": self.parameters,
            "template": self.template, "raw_sql": self.raw_sql, "values": dict(self.values or {}),
            "tenant_id": self.tenant_id, "max_rows": self.max_rows, "timeout": self.timeout,
            "classification": self.classification.value if self.classification is not None else None,
            "agent_id": self.agent_id, "session_id": self.session_id, "tool_id": self.tool_id,
        })

    def canonical_template(self) -> str:
        """Return the only executable form permitted by the structured request."""
        operation = self.operation
        qualified = f"{self.database}.{self.schema}.{self.table}"
        columns = ", ".join(self.columns) if self.columns else "*"
        if operation == "SELECT":
            tenant = " AND tenant_id = ?" if self.tenant_id is not None else ""
            return f"SELECT {columns} FROM {qualified} WHERE id = ?{tenant}"
        if operation == "INSERT":
            return f"INSERT INTO {qualified} ({', '.join(self.columns)}) VALUES ({', '.join('?' for _ in self.columns)})"
        if operation == "UPDATE":
            return f"UPDATE {qualified} SET {', '.join(f'{c} = ?' for c in self.columns)} WHERE id = ?"
        return f"DELETE FROM {qualified} WHERE id = ?"


@dataclass(frozen=True, slots=True)
class DatabaseResponse:
    operation: str
    database: str
    schema: str
    table: str
    rows: tuple[Mapping[str, Any], ...] = ()
    affected_rows: int = 0
    query_id: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "rows", tuple(MappingProxyType(dict(row)) for row in self.rows))

    def output_value(self) -> dict[str, Any]:
        return {"operation": self.operation, "database": self.database, "schema": self.schema,
                "table": self.table, "rows": [dict(r) for r in self.rows],
                "affected_rows": self.affected_rows, "query_id": self.query_id}


@dataclass(frozen=True, slots=True)
class DatabaseAuthorization:
    request_digest: str
    agent_id: str
    session_id: str
    tool_id: str
    operation: str
    database: str
    schema: str
    table: str
    columns: tuple[str, ...]
    expires_at: float
    nonce: str
    signature: str
    used: bool = False


class DatabaseExecutionDenied(PermissionError):
    def __init__(self, decision: Decision):
        self.decision = decision
        super().__init__(decision.metadata.get("error", "database execution blocked"))


class InstrumentedDatabaseBackend:
    def __init__(self) -> None:
        self.execution_count = 0
        self.executions: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    def execute(self, request: DatabaseRequest) -> DatabaseResponse:
        with self._lock:
            self.execution_count += 1
            self.executions.append({
                "operation": request.operation, "database": request.database,
                "schema": request.schema, "table": request.table,
                "template": request.template, "parameters": request.parameters,
            })
        if request.operation == "SELECT":
            return DatabaseResponse(request.operation, request.database, request.schema, request.table,
                                     rows=({"tenant_id": request.tenant_id or "tenant", "value": 1},),
                                     query_id=secrets.token_hex(8))
        return DatabaseResponse(request.operation, request.database, request.schema, request.table,
                                affected_rows=1, query_id=secrets.token_hex(8))


class Database:
    def __init__(self, policy: DatabasePolicy, *, backend: InstrumentedDatabaseBackend | None = None) -> None:
        if not isinstance(policy, DatabasePolicy):
            raise TypeError("policy must be a DatabasePolicy")
        self.policy = policy
        self._backend = backend or InstrumentedDatabaseBackend()
        self._secret = secrets.token_bytes(32)
        self._issuance_capability = object()
        self._revoked: set[str] = set()
        self._lock = threading.BoundedSemaphore(policy.max_concurrent)
        self._authorization_lock = threading.Lock()
        self._active = 0
        self._mission_registry = None
        self._mission_agent_id: str | None = None
        self._mission_tool: str | None = None

    def bind_mission_context(
        self, mission_registry: object, agent_id: str, *, tool: str | None = None
    ) -> None:
        from ..identity.mission import MissionRegistry
        if (
            not isinstance(mission_registry, MissionRegistry)
            or not mission_registry.application_owned
            or not isinstance(agent_id, str)
        ):
            raise TypeError("authoritative mission context is required")
        self._mission_registry = mission_registry
        self._mission_agent_id = agent_id
        self._mission_tool = tool

    @staticmethod
    def _blocked(error: str) -> Decision:
        return Decision(DecisionType.BLOCK, DecisionReason.INVALID_REQUEST, severity=Severity.HIGH, metadata={"error": error})

    def _prepare(self, request: DatabaseRequest) -> None:
        p = self.policy
        if request.raw_sql is not None or not request.template or any(x in request.template.upper() for x in (";", "--", "/*", "*/")):
            raise DatabaseExecutionDenied(self._blocked("raw_sql_or_unsafe_template"))
        normalized = " ".join(request.template.strip().split())
        structure = re.compile(
            rf"^(SELECT|INSERT|UPDATE|DELETE)\b.*\b{re.escape(request.table)}\b.*$",
            re.IGNORECASE,
        )
        if not structure.fullmatch(normalized):
            raise DatabaseExecutionDenied(self._blocked("invalid_query_structure"))
        upper = normalized.upper()
        if not upper.startswith(request.operation + " "):
            raise DatabaseExecutionDenied(self._blocked("query_operation_mismatch"))
        if request.operation == "SELECT":
            selected = re.search(r"^SELECT\s+(.+?)\s+FROM\s+([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*){0,2})\b", normalized, re.IGNORECASE)
            if selected:
                selected_columns = tuple(c.strip() for c in selected.group(1).split(","))
                if selected_columns != request.columns:
                    raise DatabaseExecutionDenied(self._blocked("query_columns_mismatch"))
                target = selected.group(2).split(".")
                if target[-1] != request.table or (len(target) == 2 and target[0] != request.schema) or (len(target) == 3 and tuple(target[:2]) != (request.database, request.schema)):
                    raise DatabaseExecutionDenied(self._blocked("query_scope_mismatch"))
            if self.policy.allowed_tenants:
                if not request.tenant_id or not re.search(r"\btenant_id\s*=\s*\?", normalized, re.IGNORECASE):
                    raise DatabaseExecutionDenied(self._blocked("tenant_predicate_required"))
                if request.tenant_id not in self.policy.allowed_tenants:
                    raise DatabaseExecutionDenied(self._blocked("tenant_denied"))
        elif request.operation == "INSERT":
            inserted = re.search(r"^INSERT\s+INTO\s+([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*){0,2})\s*\(([^)]*)\)", normalized, re.IGNORECASE)
            if inserted and (inserted.group(1).split(".")[-1] != request.table or tuple(c.strip() for c in inserted.group(2).split(",")) != request.columns):
                raise DatabaseExecutionDenied(self._blocked("query_scope_mismatch"))
        elif request.operation == "UPDATE":
            updated = re.search(r"^UPDATE\s+([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*){0,2})\s+SET\s+(.+?)\s+WHERE\b", normalized, re.IGNORECASE)
            if updated and (updated.group(1).split(".")[-1] != request.table or tuple(part.split("=", 1)[0].strip() for part in updated.group(2).split(",")) != request.columns):
                raise DatabaseExecutionDenied(self._blocked("query_scope_mismatch"))
        elif request.operation == "DELETE":
            deleted = re.search(r"^DELETE\s+FROM\s+([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*){0,2})\b", normalized, re.IGNORECASE)
            if deleted and deleted.group(1).split(".")[-1] != request.table:
                raise DatabaseExecutionDenied(self._blocked("query_scope_mismatch"))
        if request.operation in p.denied_operations or request.operation not in p.allowed_operations:
            raise DatabaseExecutionDenied(self._blocked("operation_denied"))
        for field, value, allowed, denied in (
            ("database", request.database, p.allowed_databases, p.denied_databases),
            ("schema", request.schema, p.allowed_schemas, p.denied_schemas),
            ("table", request.table, p.allowed_tables, p.denied_tables),
        ):
            if value in denied or value not in allowed:
                raise DatabaseExecutionDenied(self._blocked(f"{field}_denied"))
        if any(c in p.denied_columns or c not in p.allowed_columns for c in request.columns):
            raise DatabaseExecutionDenied(self._blocked("column_denied"))
        if len(request.parameters) > p.max_parameters or len(request.parameters) > p.max_batch_size or len(request.columns) > p.max_columns:
            raise DatabaseExecutionDenied(self._blocked("resource_limit"))
        if len(request.parameters) + len(request.columns) + len(request.template) > p.max_request_bytes:
            raise DatabaseExecutionDenied(self._blocked("request_size_limit"))
        if request.max_rows is not None and request.max_rows > p.max_rows:
            raise DatabaseExecutionDenied(self._blocked("row_limit"))
        if request.tenant_id is not None and p.allowed_tenants and request.tenant_id not in p.allowed_tenants:
            raise DatabaseExecutionDenied(self._blocked("tenant_denied"))

    def _issue_authorization(self, request: DatabaseRequest, *, agent_id: str, session_id: str,
                             tool_id: str, expires_at: float, _capability: object) -> DatabaseAuthorization:
        if _capability is not self._issuance_capability:
            raise PermissionError("framework authorization required")
        payload = f"{agent_id}|{session_id}|{tool_id}|{request.action}|{request.digest()}|{expires_at}"
        return DatabaseAuthorization(request.digest(), agent_id, session_id, tool_id, request.operation,
                                      request.database, request.schema, request.table, request.columns,
                                      expires_at, secrets.token_hex(16),
                                      hmac.new(self._secret, payload.encode(), hashlib.sha256).hexdigest())

    def revoke(self, authorization: DatabaseAuthorization) -> None:
        self._revoked.add(authorization.nonce)

    def execute(self, request: DatabaseRequest, *, authorization: DatabaseAuthorization | None = None,
                agent_id: str | None = None, session_id: str | None = None, tool_id: str | None = None,
                release_authorization: DataReleaseAuthorization | None = None,
                release_request: DataReleaseRequest | None = None,
                application_id: str | None = None) -> DatabaseResponse:
        if self._mission_registry is None:
            raise DatabaseExecutionDenied(self._blocked("mission_context_missing"))
        decision = self._mission_registry.check(
            self._mission_agent_id,
            action=request.action if isinstance(request, DatabaseRequest) else "",
            resource=(
                f"{request.database}:{request.schema}:{request.table}"
                if isinstance(request, DatabaseRequest) else ""
            ),
            tool=self._mission_tool or "",
            data=(
                request.classification.value
                if isinstance(request, DatabaseRequest)
                and request.classification is not None
                else None
            ),
            context=(
                {
                    "database": request.database,
                    "schema": request.schema,
                    "table": request.table,
                    "columns": request.columns,
                    "tenant": request.tenant_id,
                    "operation": request.operation,
                }
                if isinstance(request, DatabaseRequest)
                else None
            ),
            operation="database_execution",
        )
        if decision.decision_type is not DecisionType.ALLOW:
            raise DatabaseExecutionDenied(decision)
        if not isinstance(request, DatabaseRequest):
            raise DatabaseExecutionDenied(self._blocked("structured_request_required"))
        if application_id is not None and not verify_release_authorization(
            release_authorization, release_request, application_id=application_id
        ):
            raise DatabaseExecutionDenied(self._blocked("release_authorization_invalid"))
        current_policy_version = getattr(
            self, "_release_policy_version_provider", lambda: None
        )()
        if (
            current_policy_version is not None
            and release_authorization is not None
            and release_authorization.policy_version != current_policy_version
        ):
            raise DatabaseExecutionDenied(self._blocked("release_policy_version_changed"))
        self._prepare(request)
        if not isinstance(authorization, DatabaseAuthorization):
            raise DatabaseExecutionDenied(self._blocked("authorization_required"))
        now = time.time()
        with self._authorization_lock:
            if authorization.nonce in self._revoked or authorization.used or authorization.expires_at <= now:
                raise DatabaseExecutionDenied(self._blocked("authorization_invalid"))
        if (authorization.request_digest != request.digest() or
            (agent_id, session_id, tool_id) != (authorization.agent_id, authorization.session_id, authorization.tool_id) or
            (authorization.operation, authorization.database, authorization.schema, authorization.table, authorization.columns) !=
            (request.operation, request.database, request.schema, request.table, request.columns)):
            raise DatabaseExecutionDenied(self._blocked("authorization_scope_mismatch"))
        payload = f"{authorization.agent_id}|{authorization.session_id}|{authorization.tool_id}|{request.action}|{request.digest()}|{authorization.expires_at}"
        expected = hmac.new(self._secret, payload.encode(), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(expected, authorization.signature):
            raise DatabaseExecutionDenied(self._blocked("authorization_invalid"))
        with self._authorization_lock:
            if authorization.used or authorization.nonce in self._revoked:
                raise DatabaseExecutionDenied(self._blocked("authorization_invalid"))
            object.__setattr__(authorization, "used", True)
        if not self._lock.acquire(blocking=False):
            raise DatabaseExecutionDenied(self._blocked("concurrency_limit"))
        execution_parameters = request.parameters + ((request.tenant_id,) if request.tenant_id is not None else ())
        execution_request = replace(
            request,
            template=request.canonical_template(),
            parameters=execution_parameters,
        )
        result: list[DatabaseResponse] = []
        error: list[BaseException] = []
        def run_backend() -> None:
            try:
                result.append(self._backend.execute(execution_request))
            except BaseException as exc:
                error.append(exc)
        worker = threading.Thread(target=run_backend, daemon=True)
        worker.start()
        worker.join(min(float(request.timeout or self.policy.timeout), self.policy.timeout))
        if worker.is_alive():
            self._lock.release()
            raise DatabaseExecutionDenied(self._blocked("timeout"))
        if error:
            self._lock.release()
            raise DatabaseExecutionDenied(self._blocked("backend execution failed")) from error[0]
        response = result[0]
        effective_rows = min(request.max_rows or self.policy.max_rows, self.policy.max_rows)
        if len(response.rows) > effective_rows:
            self._lock.release()
            raise DatabaseExecutionDenied(self._blocked("result_row_limit"))
        encoded = json.dumps(response.output_value(), sort_keys=True, separators=(",", ":"), default=str).encode()
        if len(encoded) > self.policy.max_result_bytes:
            self._lock.release()
            raise DatabaseExecutionDenied(self._blocked("result_size_limit"))
        self._lock.release()
        return response


DatabaseTool = Database

__all__ = ["Database", "DatabaseTool", "DatabasePolicy", "DatabaseRequest",
           "DatabaseResponse", "DatabaseAuthorization", "DatabaseExecutionDenied",
           "InstrumentedDatabaseBackend"]
