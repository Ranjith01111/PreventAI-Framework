"""Policy-enforced, bounded filesystem operations behind the framework."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import hmac
import json
import math
import os
from pathlib import Path
import secrets
import stat
import threading
import time
from types import MappingProxyType
from typing import Any, Callable, Mapping

from ..audit.event import AuditEventType
from ..audit.logger import AuditLogger
from ..core.dataflow import FlowClassification, FlowTrust
from ..core.data_release import DataReleaseAuthorization, DataReleaseRequest, verify_release_authorization
from ..core.decision import Decision, DecisionReason, DecisionType, Severity

_DEFAULT_SENSITIVE_NAMES = (
    ".ssh", ".aws", ".azure", ".env", "id_rsa", "id_ed25519",
    "credentials", "secrets.json",
)
_WINDOWS_DEVICE_NAMES = frozenset(
    {"con", "prn", "aux", "nul"}
    | {f"com{index}" for index in range(1, 10)}
    | {f"lpt{index}" for index in range(1, 10)}
    | {f"com{digit}" for digit in "¹²³"}
    | {f"lpt{digit}" for digit in "¹²³"}
)
_VALID_OPERATIONS = frozenset(
    {"read", "write", "create", "append", "delete", "rename", "list", "stat"}
)
_MAX_CONSTRUCTION_BYTES = 32 * 1024 * 1024


def _text(value: str, field: str, limit: int = 4096) -> None:
    if not isinstance(value, str) or not value or len(value) > limit or "\x00" in value:
        raise ValueError(f"{field} must be a non-empty path string")


def _path_key(path: str) -> str:
    return os.path.normcase(os.path.normpath(os.path.abspath(path)))


def _digest(value: Mapping[str, Any]) -> str:
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class FileSystemPolicy:
    """An explicit path and operation allowlist; empty roots deny every path."""

    allowed_roots: tuple[str, ...] = ()
    denied_roots: tuple[str, ...] = ()
    allowed_operations: tuple[str, ...] = ()
    denied_names: tuple[str, ...] = ()
    max_read_bytes: int = 1_000_000
    max_write_bytes: int = 1_000_000
    max_entries: int = 1000
    max_path_length: int = 4096
    max_concurrent_operations: int = 1

    def __post_init__(self) -> None:
        for field in ("allowed_roots", "denied_roots"):
            roots = getattr(self, field)
            if not isinstance(roots, tuple) or any(
                not isinstance(root, str) or not root.strip() for root in roots
            ):
                raise ValueError(f"{field} must contain non-empty paths")
            object.__setattr__(
                self,
                field,
                tuple(str(Path(root).expanduser().resolve(strict=False)) for root in roots),
            )
        if not isinstance(self.allowed_operations, tuple) or any(
            not isinstance(op, str) or op not in _VALID_OPERATIONS
            for op in self.allowed_operations
        ):
            raise ValueError("allowed_operations contains an unsupported operation")
        if len(set(self.allowed_operations)) != len(self.allowed_operations):
            raise ValueError("allowed_operations cannot contain duplicates")
        if not isinstance(self.denied_names, tuple) or any(
            not isinstance(name, str) or not name or name in {".", ".."}
            for name in self.denied_names
        ):
            raise ValueError("denied_names must contain path component names")
        object.__setattr__(
            self,
            "denied_names",
            tuple(dict.fromkeys((*_DEFAULT_SENSITIVE_NAMES, *self.denied_names))),
        )
        for field in (
            "max_read_bytes", "max_write_bytes", "max_entries",
            "max_path_length", "max_concurrent_operations",
        ):
            value = getattr(self, field)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{field} must be a positive integer")


@dataclass(frozen=True, slots=True)
class FileSystemRequest:
    """One exact filesystem operation. Writes accept bounded UTF-8 text only."""

    operation: str
    path: str
    destination_path: str | None = None
    data: bytes = b""
    overwrite: bool = False
    recursive: bool = False
    classification: FlowClassification = FlowClassification.INTERNAL
    source_trust: FlowTrust = FlowTrust.UNTRUSTED
    agent_id: str | None = None
    session_id: str | None = None
    tool_id: str | None = None

    def __post_init__(self) -> None:
        if self.operation not in _VALID_OPERATIONS:
            raise ValueError("unsupported filesystem operation")
        _text(self.path, "path")
        if self.destination_path is not None:
            _text(self.destination_path, "destination_path")
        if self.operation == "rename" and self.destination_path is None:
            raise ValueError("rename requires destination_path")
        if self.operation != "rename" and self.destination_path is not None:
            raise ValueError("destination_path is only valid for rename")
        if not isinstance(self.data, bytes) or len(self.data) > _MAX_CONSTRUCTION_BYTES:
            raise ValueError("data must be bounded bytes")
        try:
            self.data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("filesystem content must be UTF-8 text") from exc
        if self.operation not in {"write", "create", "append"} and self.data:
            raise ValueError("data is only valid for write, create, or append")
        if not isinstance(self.overwrite, bool) or not isinstance(self.recursive, bool):
            raise ValueError("overwrite and recursive must be booleans")
        if self.recursive:
            raise ValueError("recursive filesystem operations are unsupported")
        if not isinstance(self.classification, FlowClassification):
            raise ValueError("classification must be a FlowClassification")
        if not isinstance(self.source_trust, FlowTrust):
            raise ValueError("source_trust must be a FlowTrust")
        for field in ("agent_id", "session_id", "tool_id"):
            value = getattr(self, field)
            if value is not None:
                _text(value, field, 256)

    def digest(self) -> str:
        return _digest({
            "operation": self.operation,
            "path": _path_key(self.path),
            "destination_path": _path_key(self.destination_path) if self.destination_path else None,
            "data_sha256": hashlib.sha256(self.data).hexdigest(),
            "data_length": len(self.data),
            "overwrite": self.overwrite,
            "recursive": self.recursive,
            "classification": self.classification.value,
            "source_trust": self.source_trust.value,
            "agent_id": self.agent_id,
            "session_id": self.session_id,
            "tool_id": self.tool_id,
        })

    @property
    def action(self) -> str:
        return f"filesystem.{self.operation}"


@dataclass(frozen=True, slots=True)
class FileSystemResponse:
    operation: str
    path: str
    content: str | None = None
    entries: tuple[Mapping[str, Any], ...] = ()
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "entries",
            tuple(MappingProxyType(dict(item)) for item in self.entries),
        )
        object.__setattr__(
            self, "metadata", MappingProxyType(dict(self.metadata or {})),
        )

    def output_value(self) -> dict[str, Any]:
        return {
            "operation": self.operation,
            "path": self.path,
            "content": self.content,
            "entries": [dict(item) for item in self.entries],
            "metadata": dict(self.metadata or {}),
        }


@dataclass(frozen=True, slots=True)
class FileSystemAuthorization:
    request_digest: str
    canonical_path: str
    canonical_destination: str | None
    expires_at: float
    nonce: str
    signature: str


@dataclass(frozen=True, slots=True)
class _Plan:
    path: Path
    destination: Path | None


class _FilesystemBackend:
    """Small syscall seam used by deterministic tests; all access stays here."""

    @staticmethod
    def _open_regular(path: Path, flags: int, mode: int = 0o600) -> int:
        safe_flags = (
            flags
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_BINARY", 0)
            | getattr(os, "O_NONBLOCK", 0)
        )
        descriptor = os.open(path, safe_flags, mode)
        try:
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                raise OSError("only regular files are supported")
            return descriptor
        except Exception:
            os.close(descriptor)
            raise

    def read(self, path: Path, limit: int) -> bytes:
        descriptor = self._open_regular(path, os.O_RDONLY)
        try:
            chunks = bytearray()
            while len(chunks) <= limit:
                chunk = os.read(descriptor, min(65536, limit + 1 - len(chunks)))
                if not chunk:
                    break
                chunks.extend(chunk)
            return bytes(chunks)
        finally:
            os.close(descriptor)

    def write(self, path: Path, data: bytes, *, create: bool, append: bool, overwrite: bool) -> int:
        flags = os.O_WRONLY
        if create:
            flags |= os.O_CREAT | os.O_EXCL
        elif append:
            flags |= os.O_CREAT | os.O_APPEND
        elif overwrite:
            flags |= os.O_CREAT | os.O_TRUNC
        else:
            flags |= os.O_CREAT | os.O_EXCL
        descriptor = self._open_regular(path, flags)
        written = 0
        try:
            view = memoryview(data)
            while written < len(view):
                amount = os.write(descriptor, view[written:])
                if amount <= 0:
                    raise OSError("short filesystem write")
                written += amount
            return written
        finally:
            os.close(descriptor)

    def delete(self, path: Path) -> None:
        if path.is_dir():
            os.rmdir(path)
        else:
            os.unlink(path)

    def rename(self, source: Path, destination: Path, *, overwrite: bool) -> None:
        if not overwrite and destination.exists():
            raise FileExistsError(str(destination))
        if destination.is_dir() or destination.is_symlink():
            raise OSError("rename destination is not a regular file")
        os.replace(source, destination) if overwrite else os.rename(source, destination)

    def list(self, path: Path, limit: int) -> tuple[Mapping[str, Any], ...]:
        rows = []
        with os.scandir(path) as entries:
            for index, entry in enumerate(entries):
                if index >= limit:
                    raise OSError("directory entry limit exceeded")
                info = entry.stat(follow_symlinks=False)
                rows.append({
                    "name": entry.name,
                    "is_file": stat.S_ISREG(info.st_mode),
                    "is_directory": stat.S_ISDIR(info.st_mode),
                    "size": info.st_size if stat.S_ISREG(info.st_mode) else None,
                })
        return tuple(sorted(rows, key=lambda row: str(row["name"]).casefold()))

    def stat(self, path: Path) -> Mapping[str, Any]:
        info = path.stat()
        if not (stat.S_ISREG(info.st_mode) or stat.S_ISDIR(info.st_mode)):
            raise OSError("special files are not supported")
        return {
            "is_file": stat.S_ISREG(info.st_mode),
            "is_directory": stat.S_ISDIR(info.st_mode),
            "size": info.st_size,
        }


class FileSystem:
    """Exact-request filesystem executor; framework callers alone can issue tokens."""

    def __init__(
        self,
        policy: FileSystemPolicy,
        *,
        audit_logger: AuditLogger | None = None,
        backend: _FilesystemBackend | None = None,
    ) -> None:
        if not isinstance(policy, FileSystemPolicy):
            raise TypeError("policy must be a FileSystemPolicy")
        if audit_logger is not None and not isinstance(audit_logger, AuditLogger):
            raise TypeError("audit_logger must be an AuditLogger")
        self.policy = policy
        self._audit_logger = audit_logger
        self._backend = backend if backend is not None else _FilesystemBackend()
        self._secret = secrets.token_bytes(32)
        self._used: set[str] = set()
        self._revoked: set[str] = set()
        self._lock = threading.RLock()
        self._slots = threading.BoundedSemaphore(policy.max_concurrent_operations)
        self._authorizer_bound = False
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

    def _prepare(self, request: FileSystemRequest) -> _Plan:
        if not isinstance(request, FileSystemRequest):
            raise TypeError("request must be a FileSystemRequest")
        if request.operation not in self.policy.allowed_operations:
            raise self._deny("operation_not_allowed")
        if len(request.path) > self.policy.max_path_length or (
            request.destination_path and len(request.destination_path) > self.policy.max_path_length
        ):
            raise self._deny("path_too_long")
        path = self._resolve(request.path)
        destination = self._resolve(request.destination_path) if request.destination_path else None
        self._check_path(path)
        if destination is not None:
            self._check_path(destination)
        if request.operation in {"write", "create", "append"}:
            if len(request.data) > self.policy.max_write_bytes:
                raise self._deny("write_limit_exceeded")
            try:
                request.data.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise self._deny("binary_content_not_supported") from exc
        if request.operation == "read" and path.exists():
            self._check_regular(path)
            if path.stat().st_size > self.policy.max_read_bytes:
                raise self._deny("read_limit_exceeded")
        if request.operation in {"write", "create", "append"} and path.exists():
            self._check_regular(path)
            if request.operation == "append" and path.stat().st_size + len(request.data) > self.policy.max_write_bytes:
                raise self._deny("write_limit_exceeded")
            if request.operation == "write" and not request.overwrite:
                raise self._deny("overwrite_not_allowed")
            if request.operation == "write" and request.overwrite and not path.exists():
                raise self._deny("overwrite_target_missing")
        if request.overwrite and request.operation != "write" and request.operation != "rename":
            raise self._deny("invalid_overwrite")
        if request.operation in {"delete", "rename"} and not path.exists():
            raise self._deny("source_missing")
        if request.operation == "delete" and not path.is_dir():
            self._check_regular(path)
        if request.operation == "rename":
            self._check_regular(path)
        if request.operation == "rename" and destination is not None and destination.exists():
            if not request.overwrite:
                raise self._deny("destination_exists")
            self._check_regular(destination)
        return _Plan(path, destination)

    def _resolve(self, value: str | None) -> Path:
        if value is None:
            raise self._deny("path_missing")
        try:
            supplied = Path(value).expanduser()
            if supplied.is_symlink():
                raise self._deny("symbolic_link_denied")
            return supplied.resolve(strict=False)
        except FileSystemExecutionDenied:
            raise
        except (OSError, RuntimeError, ValueError) as exc:
            raise self._deny("path_resolution_failed") from exc

    def _check_path(self, path: Path) -> None:
        if os.name == "nt":
            for part in path.parts:
                if part == path.anchor:
                    continue
                if ":" in part:
                    raise self._deny("alternate_stream_denied")
                device_name = part.rstrip(" .").split(".", 1)[0].casefold()
                if device_name in _WINDOWS_DEVICE_NAMES:
                    raise self._deny("device_path_denied")
        if any(part.casefold() in {name.casefold() for name in self.policy.denied_names} for part in path.parts):
            raise self._deny("sensitive_path_denied")
        if any(self._within(path, Path(root)) for root in self.policy.denied_roots):
            raise self._deny("path_denied")
        if not self.policy.allowed_roots or not any(
            self._within(path, Path(root)) for root in self.policy.allowed_roots
        ):
            raise self._deny("path_outside_allowed_roots")

    @staticmethod
    def _within(path: Path, root: Path) -> bool:
        try:
            path.relative_to(root)
            return True
        except ValueError:
            return False

    def _check_regular(self, path: Path) -> None:
        try:
            if path.is_symlink() or not stat.S_ISREG(path.stat().st_mode):
                raise self._deny("special_file_denied")
        except OSError as exc:
            raise self._deny("path_inspection_failed") from exc

    def _bind_framework_authorizer(
        self, owner: object
    ) -> Callable[..., FileSystemAuthorization]:
        from ..developer.framework import Tool

        if (
            not isinstance(owner, Tool)
            or owner._filesystem is not self
            or owner._agent is None
            or owner._engine is None
        ):
            raise PermissionError("filesystem authorizer requires a bound framework tool")
        with self._lock:
            if self._authorizer_bound:
                raise RuntimeError("filesystem is already bound to a framework tool")
            self._authorizer_bound = True

        def authorize(
            request: FileSystemRequest, plan: _Plan, *, ttl: float = 30.0
        ) -> FileSystemAuthorization:
            if not isinstance(request, FileSystemRequest) or not isinstance(plan, _Plan):
                raise PermissionError("framework authorization requires a validated request")
            if not isinstance(ttl, (int, float)) or isinstance(ttl, bool):
                raise ValueError("ttl must be a positive finite number")
            try:
                ttl_seconds = float(ttl)
            except (OverflowError, ValueError) as exc:
                raise ValueError("ttl must be a positive finite number") from exc
            if not math.isfinite(ttl_seconds) or ttl_seconds <= 0:
                raise ValueError("ttl must be a positive finite number")
            expected_plan = self._prepare(request)
            if plan != expected_plan:
                raise PermissionError("framework authorization plan does not match request")
            expiry = time.time() + min(ttl_seconds, 30.0)
            nonce = secrets.token_urlsafe(24)
            body = self._authorization_body(request, plan, expiry, nonce)
            return FileSystemAuthorization(
                request.digest(),
                str(plan.path),
                str(plan.destination) if plan.destination else None,
                expiry,
                nonce,
                self._sign(body),
            )

        return authorize

    def revoke_authorization(self, authorization: FileSystemAuthorization) -> None:
        if not isinstance(authorization, FileSystemAuthorization):
            raise TypeError("authorization must be a FileSystemAuthorization")
        if (
            not isinstance(authorization.request_digest, str)
            or not isinstance(authorization.canonical_path, str)
            or (
                authorization.canonical_destination is not None
                and not isinstance(authorization.canonical_destination, str)
            )
            or not isinstance(authorization.expires_at, float)
            or not math.isfinite(authorization.expires_at)
            or not isinstance(authorization.nonce, str)
            or not authorization.nonce
            or not isinstance(authorization.signature, str)
            or not authorization.signature
        ):
            raise ValueError("authorization fields are invalid")
        body = {
            "request_digest": authorization.request_digest,
            "canonical_path": authorization.canonical_path,
            "canonical_destination": authorization.canonical_destination,
            "expires_at": authorization.expires_at,
            "nonce": authorization.nonce,
        }
        if not hmac.compare_digest(authorization.signature, self._sign(body)):
            raise ValueError("authorization signature is invalid")
        with self._lock:
            self._revoked.add(authorization.nonce)

    def execute(
        self, request: FileSystemRequest, *, authorization: FileSystemAuthorization,
        release_authorization: DataReleaseAuthorization | None = None,
        release_request: DataReleaseRequest | None = None,
        application_id: str | None = None,
    ) -> FileSystemResponse:
        if self._mission_registry is None:
            raise self._deny("mission_context_missing")
        decision = self._mission_registry.check(
            self._mission_agent_id,
            action=request.operation if isinstance(request, FileSystemRequest) else "",
            resource=(
                str(Path(request.path).expanduser().resolve(strict=False))
                if isinstance(request, FileSystemRequest)
                else ""
            ),
            tool=self._mission_tool or "",
            data=request.classification.value if isinstance(request, FileSystemRequest) else None,
            operation="filesystem_execution",
        )
        if decision.decision_type is not DecisionType.ALLOW:
            raise FileSystemExecutionDenied(decision)
        if not isinstance(authorization, FileSystemAuthorization):
            raise self._deny("authorization_missing")
        if application_id is not None and not verify_release_authorization(
            release_authorization, release_request, application_id=application_id
        ):
            raise self._deny("release_authorization_invalid")
        current_policy_version = getattr(
            self, "_release_policy_version_provider", lambda: None
        )()
        if (
            current_policy_version is not None
            and release_authorization is not None
            and release_authorization.policy_version != current_policy_version
        ):
            raise self._deny("release_policy_version_changed")
        if not self._slots.acquire(blocking=False):
            raise self._deny("concurrency_limit_exceeded")
        try:
            if (
                not isinstance(request, FileSystemRequest)
                or not isinstance(authorization.request_digest, str)
                or not isinstance(authorization.canonical_path, str)
                or (
                    authorization.canonical_destination is not None
                    and not isinstance(authorization.canonical_destination, str)
                )
                or not isinstance(authorization.expires_at, float)
                or not math.isfinite(authorization.expires_at)
                or not isinstance(authorization.nonce, str)
                or not authorization.nonce
                or not isinstance(authorization.signature, str)
                or not authorization.signature
            ):
                raise self._deny("authorization_invalid")
            authorization_body = {
                "request_digest": authorization.request_digest,
                "canonical_path": authorization.canonical_path,
                "canonical_destination": authorization.canonical_destination,
                "expires_at": authorization.expires_at,
                "nonce": authorization.nonce,
            }
            if (
                authorization.request_digest != request.digest()
                or time.time() >= authorization.expires_at
                or not hmac.compare_digest(
                    authorization.signature, self._sign(authorization_body)
                )
            ):
                raise self._deny("authorization_invalid")
            with self._lock:
                if authorization.nonce in self._revoked:
                    raise self._deny("authorization_revoked")
                if authorization.nonce in self._used:
                    raise self._deny("authorization_already_used")
            plan = self._prepare(request)
            current_path = str(plan.path)
            current_destination = str(plan.destination) if plan.destination else None
            if (
                authorization.canonical_path != current_path
                or authorization.canonical_destination != current_destination
            ):
                raise self._deny("authorization_invalid")
            with self._lock:
                if authorization.nonce in self._revoked:
                    raise self._deny("authorization_revoked")
                if authorization.nonce in self._used:
                    raise self._deny("authorization_already_used")
                self._used.add(authorization.nonce)
            return self._perform(request, plan)
        except FileSystemExecutionDenied as exc:
            self._audit(AuditEventType.FILESYSTEM_BLOCKED, request, exc.decision)
            raise
        except (OSError, UnicodeError, ValueError) as exc:
            decision = self._blocked("filesystem_operation_failed")
            self._audit(AuditEventType.FILESYSTEM_BLOCKED, request, decision)
            raise FileSystemExecutionDenied(decision) from exc
        finally:
            self._slots.release()

    def _perform(self, request: FileSystemRequest, plan: _Plan) -> FileSystemResponse:
        path = plan.path
        self._audit(AuditEventType.FILESYSTEM_OPERATION_STARTED, request, self._allowed())
        if request.operation == "read":
            raw = self._backend.read(path, self.policy.max_read_bytes)
            if len(raw) > self.policy.max_read_bytes:
                raise self._deny("read_limit_exceeded")
            content = raw.decode("utf-8")
            response = FileSystemResponse("read", request.path, content=content)
        elif request.operation in {"write", "create", "append"}:
            count = self._backend.write(
                path, request.data,
                create=request.operation == "create",
                append=request.operation == "append",
                overwrite=request.operation == "write" and request.overwrite,
            )
            response = FileSystemResponse(
                request.operation, request.path, metadata={"bytes_written": count}
            )
        elif request.operation == "delete":
            self._backend.delete(path)
            response = FileSystemResponse("delete", request.path)
        elif request.operation == "rename":
            assert plan.destination is not None
            self._backend.rename(path, plan.destination, overwrite=request.overwrite)
            response = FileSystemResponse("rename", request.destination_path or "")
        elif request.operation == "list":
            if not path.is_dir():
                raise self._deny("directory_required")
            entries = self._backend.list(path, self.policy.max_entries)
            response = FileSystemResponse("list", request.path, entries=entries)
        elif request.operation == "stat":
            response = FileSystemResponse("stat", request.path, metadata=self._backend.stat(path))
        else:
            raise self._deny("operation_not_allowed")
        self._audit(AuditEventType.FILESYSTEM_OPERATION_COMPLETED, request, self._allowed())
        return response

    def _authorization_body(
        self, request: FileSystemRequest, plan: _Plan, expiry: float, nonce: str
    ) -> Mapping[str, Any]:
        return {
            "request_digest": request.digest(),
            "canonical_path": str(plan.path),
            "canonical_destination": str(plan.destination) if plan.destination else None,
            "expires_at": expiry,
            "nonce": nonce,
        }

    def _sign(self, value: Mapping[str, Any]) -> str:
        return hmac.new(
            self._secret,
            json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

    def _audit(self, event_type: AuditEventType, request: FileSystemRequest, decision: Decision) -> None:
        if self._audit_logger is None:
            return
        self._audit_logger.record(
            event_type,
            decision=decision.decision_type.value,
            reason=decision.reason.value,
            severity=decision.severity.value if decision.severity else None,
            rule_id=decision.rule_id,
            agent_id=request.agent_id,
            session_id=request.session_id,
            tool_id=request.tool_id,
            action=request.action,
            resource="filesystem",
            metadata={"operation": request.operation},
        )

    @staticmethod
    def _allowed() -> Decision:
        return Decision(DecisionType.ALLOW, DecisionReason.REQUEST_ALLOWED)

    @staticmethod
    def _blocked(category: str) -> Decision:
        return Decision(
            DecisionType.BLOCK,
            DecisionReason.FILESYSTEM_POLICY_VIOLATION,
            severity=Severity.HIGH,
            metadata={"category": category, "stage": "filesystem"},
        )

    def _deny(self, category: str) -> "FileSystemExecutionDenied":
        return FileSystemExecutionDenied(self._blocked(category))


class FileSystemExecutionDenied(PermissionError):
    def __init__(self, decision: Decision) -> None:
        self.decision = decision
        super().__init__(decision.reason.value)
